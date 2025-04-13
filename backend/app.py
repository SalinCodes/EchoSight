import os
import tempfile
import time
import io
import traceback
import uuid
from collections import defaultdict
from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn.functional as F
import speech_recognition as sr
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from pydub import AudioSegment
from pyannote.audio import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from speechbrain.inference.speaker import EncoderClassifier
import requests
from msclap import CLAP
import io

load_dotenv()

# --- Configuration ---

# MODELS
PYANNOTE_PIPELINE = "pyannote/speaker-diarization-3.1"
# Using a standard SpeechBrain ECAPA-TDNN model for speaker embeddings
EMBEDDING_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
# CLAP model configuration
CLAP_VERSION = "2023"
CLAP_CONFIDENCE_THRESHOLD = 0.75  # Updated threshold to 75% as requested
ENVIRONMENT_SOUND_CLASSES = [
    "dog", "crackling fire", "clapping", "siren",
    "cow", "door, wood creaks", "car horn",
    "frog", "chirping birds", "coughing", "engine",
    "cat", "footsteps", "washing machine", "train",
    "hen", "wind", "laughing", "rain",
    "pouring water", "airplane", "sheep",
    "fireworks", "crow", "thunderstorm", "glass breaking"
]
SOUND_PROMPT = 'this is a sound of '

# Debug flag for verbose similarity logs
DEBUG_SIMILARITY = True

# Device Setup
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"

# Enrollment Configuration
MIN_ENROLL_CLIP_DURATION_MS = 5000 
MIN_SUCCESSFUL_CLIPS = 3         

# Improved Similarity Configuration
SIMILARITY_CONFIDENCE_LEVELS = {
    "strict": 0.50,     # Very high confidence
    "normal": 0.40,     # Default setting
    "relaxed": 0.35,    # More lenient matching for challenging audio
    "very_relaxed": 0.30  # Use only when speaker separation is critical
}

# Default similarity configuration
SIMILARITY_MODE = "normal"
PRIMARY_USER_THRESHOLD = SIMILARITY_CONFIDENCE_LEVELS[SIMILARITY_MODE]
OTHER_SPEAKER_THRESHOLD = SIMILARITY_CONFIDENCE_LEVELS[SIMILARITY_MODE]

#----------------------------#
#   ESP-32 CONFIGURATION
#----------------------------#

ESP_IP = "172.20.10.6"
ESP32_URL = f"http://{ESP_IP}/display"


session_storage = defaultdict(lambda: {
    "primary_embeddings": [], # List of numpy arrays of the USER embeddings
    "speaker_history": {},    # { "User_1": [np.array, ...], "User_2": [...] }
    "next_user_id": 1
})

# Environment variable setup for pyannote/speechbrain
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SPEECHBRAIN_DISABLE_SYMLINKS"] = "1"

# --- Initialize Flask App ---
app = Flask(__name__)

app.secret_key = os.environ.get("FLASK_SECRET_KEY")
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


# --- Load Models Globally ---
diarization_pipeline_instance = None
embedding_classifier = None
clap_model = None

# Transcribe using Google Speech Recognition API
recognizer = sr.Recognizer()

print(f"Device selected: {DEVICE}")

print(f"Loading Pyannote pipeline: {PYANNOTE_PIPELINE}...")
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    print("Error: HUGGINGFACE_TOKEN environment variable not set. Pyannote requires it.")
else:
    try:
        diarization_pipeline_instance = Pipeline.from_pretrained(PYANNOTE_PIPELINE, use_auth_token=HF_TOKEN)
        if USE_GPU:
            diarization_pipeline_instance.to(torch.device(DEVICE))
        print("Pyannote pipeline loaded.")
    except Exception as e:
        print(f"Error initializing Pyannote pipeline: {e}")
        traceback.print_exc()

print(f"Loading Embedding model: {EMBEDDING_MODEL_SOURCE}...")
try:
    embedding_classifier = EncoderClassifier.from_hparams(
        source=EMBEDDING_MODEL_SOURCE,
        run_opts={"device": DEVICE}
    )
    embedding_classifier.eval()
    print("Embedding model loaded.")
except Exception as e:
    print(f"Error initializing Embedding model: {e}")
    traceback.print_exc()

# Load CLAP model for environmental sound detection
print(f"Loading CLAP model version: {CLAP_VERSION}...")
try:
    clap_model = CLAP(version=CLAP_VERSION, use_cuda=USE_GPU)
    # Prepare class prompts
    class_prompts = [SOUND_PROMPT + x for x in ENVIRONMENT_SOUND_CLASSES]
    # Pre-compute text embeddings for efficiency
    environment_text_embeddings = clap_model.get_text_embeddings(class_prompts)
    print("CLAP model loaded successfully.")
except Exception as e:
    print(f"Error initializing CLAP model: {e}")
    traceback.print_exc()


def get_session_id():
    """Gets or creates a basic session identifier."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        print(f"New session created: {session['session_id']}")
    # Ensuring the session is saved if modified
    session.modified = True
    return session['session_id']

def load_audio_for_embedding(audio_segment: AudioSegment, target_sr=16000):
    """Loads pydub segment, resamples, converts to tensor for SpeechBrain."""
    if audio_segment.frame_rate != target_sr:
        audio_segment = audio_segment.set_frame_rate(target_sr)
    if audio_segment.channels > 1:
        audio_segment = audio_segment.set_channels(1)
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(np.int16).max # Normalizing
    audio_tensor = torch.from_numpy(samples).unsqueeze(0).to(DEVICE) # Add batch dim, move to device
    return audio_tensor

def update_speaker_embedding(history_embeddings, new_embedding, max_embeddings=7):
    """
    Updates speaker embeddings by adding new embeddings
    Keeps only the most recent max_embeddings for each speaker.
    """
    history_embeddings.append(new_embedding)
    
    # Limit the number of embeddings we keep (FIFO)
    if len(history_embeddings) > max_embeddings:
        history_embeddings = history_embeddings[-max_embeddings:]
    
    return history_embeddings

def extract_embedding(audio_tensor):
    """Extracts speaker embedding using the loaded SpeechBrain model."""
    if embedding_classifier is None: return None
    try:
        with torch.no_grad(): # Disable gradient calculation for inference
            # The output is usually (batch, time, embed_dim), we need embedding vector
            embeddings = embedding_classifier.encode_batch(audio_tensor)
            # Pooling strategy depends on model, often mean pooling over time dim
            embedding_vector = torch.mean(embeddings, dim=1).squeeze(0) # Squeeze batch dim
            return embedding_vector.cpu().numpy() # Return as numpy array on CPU
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        traceback.print_exc()
        return None

def get_similarity(emb1, emb2):
    """Calculates cosine similarity between two embeddings."""
    if not isinstance(emb1, np.ndarray) or not isinstance(emb2, np.ndarray):
        print(f"Error: Invalid embedding types: {type(emb1)}, {type(emb2)}")
        return 0.0
        
    try:
        # Check for zero vectors
        if np.all(emb1 == 0) or np.all(emb2 == 0):
            print("Warning: Zero vector in similarity calculation")
            return 0.0
            
        # Ensure embeddings are 2D arrays for cosine_similarity
        emb1_2d = emb1.reshape(1, -1)
        emb2_2d = emb2.reshape(1, -1)
        
        # Compute cosine similarity
        sim = cosine_similarity(emb1_2d, emb2_2d)[0][0]
        
        # Check for NaN results
        if np.isnan(sim):
            print("Warning: NaN similarity result, defaulting to 0")
            return 0.0
            
        return sim
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        traceback.print_exc()
        return 0.0

def ensure_list(value):
    """Ensures a value is a list. Useful for handling speaker history embeddings."""
    if value is None:
        return []
    elif isinstance(value, (list, tuple)):
        return list(value)
    else:
        # If it's a single value (e.g., np.ndarray), wrap it in a list
        return [value]

def dump_embeddings_info(label, embeddings):
    """Debug helper to dump information about embeddings."""
    if not DEBUG_SIMILARITY:
        return
    
    print(f"=== DEBUG: {label} ===")
    print(f"Type: {type(embeddings)}")
    if isinstance(embeddings, list):
        print(f"Length: {len(embeddings)}")
        for i, emb in enumerate(embeddings):
            print(f"  [{i}] Type: {type(emb)}")
            if isinstance(emb, np.ndarray):
                print(f"      Shape: {emb.shape}, Mean: {np.mean(emb):.4f}, Std: {np.std(emb):.4f}")
    elif isinstance(embeddings, np.ndarray):
        print(f"Shape: {embeddings.shape}, Mean: {np.mean(embeddings):.4f}, Std: {np.std(embeddings):.4f}")
    print("=" * 40)

def detect_environment_sound(audio_path):
    """
    Detects environmental sounds using the CLAP model.
    Returns the top prediction and its confidence score.
    """
    if clap_model is None:
        print("CLAP model not loaded, skipping environment sound detection")
        return None, 0.0
    
    try:
        # Get audio embeddings
        audio_embeddings = clap_model.get_audio_embeddings([audio_path], resample=True)
        
        # Compute similarity with pre-computed text embeddings
        similarity = clap_model.compute_similarity(audio_embeddings, environment_text_embeddings)
        
        # Apply softmax to get probabilities
        similarity = F.softmax(similarity, dim=1)
        
        # Get top prediction
        values, indices = similarity[0].topk(1)
        top_value = values[0].item()
        top_class = ENVIRONMENT_SOUND_CLASSES[indices[0].item()]
        
        print(f"Environment sound detection: {top_class} with confidence {top_value:.4f}")
        return top_class, top_value
    except Exception as e:
        print(f"Error in environment sound detection: {e}")
        traceback.print_exc()
        return None, 0.0

# --- API Endpoints ---

@app.route('/health', methods=['GET'])
def health_check():
    models_loaded = all([diarization_pipeline_instance, embedding_classifier])
    return jsonify({'status': 'ok', 'models_loaded': models_loaded})

@app.route('/enroll', methods=['POST'])
def enroll_primary_user():
    """
    Endpoint to enroll the primary user's voice using multiple audio clips.
    Expects multiple files under the 'audio' field name in FormData.
    """
    session_id = get_session_id()
    # Use getlist to retrieve all files sent with the name 'audio'
    audio_files = request.files.getlist('audio')

    if not audio_files:
        return jsonify({'success': False, 'error': "No audio files found in request"}), 400

    print(f"Enrollment request for session: {session_id} with {len(audio_files)} clips.")

    extracted_embeddings = []
    clip_errors = []

    for i, audio_file in enumerate(audio_files):
        clip_number = i + 1
        print(f"Processing enrollment clip {clip_number}/{len(audio_files)}: {audio_file.filename}")
        if audio_file.filename == '':
            msg = f"Clip {clip_number}: No filename provided (Skipped)"
            print(msg)
            clip_errors.append(msg)
            continue # Skip this file

        try:
            audio_data = audio_file.read()
            # Load audio using pydub
            audio = AudioSegment.from_file(io.BytesIO(audio_data))

            # --- Validation per clip ---
            if len(audio) < MIN_ENROLL_CLIP_DURATION_MS:
                msg = f"Clip {clip_number} ({audio_file.filename}): Audio too short ({len(audio)}ms). Min required: {MIN_ENROLL_CLIP_DURATION_MS}ms."
                print(f"  Error: {msg}")
                clip_errors.append(msg)
                continue # Skip this clip

            #---- Extract embedding per clip ----
            audio_tensor = load_audio_for_embedding(audio)
            embedding = extract_embedding(audio_tensor)

            if embedding is None:
                msg = f"Clip {clip_number} ({audio_file.filename}): Failed to extract embedding."
                print(f"  Error: {msg}")
                clip_errors.append(msg)
                continue # Skip this clip

            # --- Embedding seems valid, add to list ---
            extracted_embeddings.append(embedding)
            print(f"  Successfully extracted embedding for clip {clip_number}.")
            if DEBUG_SIMILARITY:
                dump_embeddings_info(f"Enrollment Clip {clip_number} Embedding", embedding)

        except Exception as e:
            msg = f"Clip {clip_number} ({audio_file.filename}): Error during processing: {str(e)}"
            print(f"  Error: {msg}")
            traceback.print_exc()
            clip_errors.append(msg)
            continue # Skip this clip on error

    # --- Check if enough clips were processed successfully ---
    if len(extracted_embeddings) < MIN_SUCCESSFUL_CLIPS:
        error_message = f"Enrollment failed. Only {len(extracted_embeddings)}/{len(audio_files)} clips processed successfully (minimum {MIN_SUCCESSFUL_CLIPS} required). "
        error_message += "Errors: " + "; ".join(clip_errors)
        print(f"Enrollment failed for session {session_id}: {error_message}")
        return jsonify({'success': False, 'error': error_message}), 400

    # --- Store the collected embeddings for the session ---
    # Replace the entire list with the newly extracted valid embeddings
    session_storage[session_id]["primary_embeddings"] = extracted_embeddings
    session.modified = True

    success_message = f"Primary user enrolled successfully with {len(extracted_embeddings)} embeddings from {len(audio_files)} clips."
    print(f"{success_message} for session: {session_id}")
    response_data = {'success': True, 'message': success_message}
    if clip_errors:
        response_data['warnings'] = clip_errors
    return jsonify(response_data)

@app.route('/transcribe', methods=['POST'])

def transcribe():
    global PRIMARY_USER_THRESHOLD, OTHER_SPEAKER_THRESHOLD

    """Main endpoint for diarization, transcription, and speaker matching."""
    start_time = time.time()
    session_id = get_session_id()

    # ----Check models loaded----#
    if not all([diarization_pipeline_instance, embedding_classifier]):
        return jsonify({'success': False, 'error': "Service unavailable (models not loaded)"}), 503

    #----Get session data----#
    session_data = session_storage.get(session_id, {
        "primary_embeddings": [],
        "speaker_history": {},
        "next_user_id": 1
    })
    
    primary_embeddings = session_data.get("primary_embeddings", [])
    speaker_history = session_data.get("speaker_history", {})
    next_user_id = session_data.get("next_user_id", 1)

    # Debug info about stored primary embeddings
    if DEBUG_SIMILARITY and primary_embeddings:
        dump_embeddings_info("Stored Primary Embeddings", primary_embeddings)

    # Ensure speaker_history has proper format - values should be lists
    for speaker_id, embeddings in speaker_history.items():
        speaker_history[speaker_id] = ensure_list(embeddings)

    if not primary_embeddings:
        # Allow transcription even without enrollment, but primary user won't be filtered
        print(f"Warning: Primary user not enrolled for session {session_id}. Transcription will include all speakers.")

    # --- Get audio file ---
    if 'audio' not in request.files: return jsonify({'success': False, 'error': "No audio file part"}), 400
    audio_file = request.files['audio']
    if audio_file.filename == '': return jsonify({'success': False, 'error': "No selected file"}), 400

    print(f"\n--- Processing request for session: {session_id} ---")
    print(f"Received audio file: {audio_file.filename} ({audio_file.content_type})")

    temp_file_path = None
    try:
        audio_data = audio_file.read()
        if len(audio_data) < 1000: # Basic check
             print("Audio data too short, skipping.")
             return jsonify({'success': True, 'transcription': ""}), 200

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(audio_data)
        print(f"Audio saved to temp file: {temp_file_path}")

        # Load audio with pydub for processing
        try:
            audio = AudioSegment.from_file(temp_file_path)
            audio_duration_ms = len(audio)
            print(f"Audio loaded successfully. Duration: {audio_duration_ms}ms")
        except Exception as e:
            print(f"Error loading audio with pydub: {e}")
            return jsonify({'success': False, 'error': "Failed to load audio data"}), 500

        # --- Check for environmental sounds first ---
        env_sound, env_confidence = detect_environment_sound(temp_file_path)
        
        # If we detected an environmental sound with high confidence, skip speech processing
        if env_sound and env_confidence >= CLAP_CONFIDENCE_THRESHOLD:
            print(f"Environmental sound detected with high confidence: {env_sound} ({env_confidence:.4f})")
            final_transcription = f"Environmental Sound Detected: {env_sound.title()}"
            
            # Send to ESP-32
            try:
                response = requests.post(ESP32_URL, data=final_transcription)
                print("Environmental sound notification sent to ESP-32 successfully.")
            except requests.exceptions.RequestException as e:
                print(f"Error sending to ESP-32: {e}")
                
            return jsonify({
                'success': True, 
                'transcription': final_transcription,
                'is_environmental_sound': True,
                'sound_type': env_sound,
                'confidence': env_confidence
            })

        # --- 1. Diarization (who spoke when) ---
        print("Running diarization...")
        diarization_start = time.time()
        try:
            diarization = diarization_pipeline_instance(temp_file_path)
            print(f"Diarization completed in {time.time() - diarization_start:.2f}s")
        except Exception as e:
            print(f"Error during diarization: {e}")
            traceback.print_exc()
            diarization = None
            # Continue with transcription even if diarization fails
            
        # --- 2. Transcription (Google Speech Recognition on the current chunk) ---
        recognizer = sr.Recognizer()
        transcription_text = ""
        transcribe_start = time.time()  # Add this line to track transcription time

        try:
            # Use the *correctly formatted* temporary WAV file as the source
            with sr.AudioFile(temp_file_path) as source:
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    print(f"Energy threshold set to: {recognizer.energy_threshold}")
                except Exception as noise_e:
                    print(f"Could not adjust for ambient noise: {noise_e}")

                # Record the audio data from the file
                print("Recording audio data with SpeechRecognition...")
                # Use audio_segment_sr to avoid shadowing the pydub object `audio`
                audio_segment_sr = recognizer.record(source)
                print(f"Audio data recorded by sr. Duration: {audio_segment_sr.duration if hasattr(audio_segment_sr, 'duration') else 'N/A'}")


                # Perform transcription using Google Web Speech API
                try:
                    print("Sending audio to Google Speech Recognition API...")
                    transcription_text = recognizer.recognize_google(audio_segment_sr, language="ne-NP")
                    print(f"Google Speech Recognition successful in {time.time() - transcribe_start:.2f}s")
                    # Log the actual result for debugging
                    print(f"Received Transcription: {transcription_text}")

                except sr.UnknownValueError:
                    # This is the error you were seeing
                    transcription_text = "{Audio not understood}"
                    print("Google Speech Recognition could not understand audio. Possible reasons: silence, noise, wrong language, too short?")
                except sr.RequestError as e:
                    # This handles API/network errors
                    error_msg = f"Could not request results from Google Speech Recognition service; {e}"
                    print(error_msg)
                    # Return an error instead of proceeding with bad text
                    return jsonify({'success': False, 'error': error_msg}), 503 # 503 Service Unavailable

        except FileNotFoundError:
             print(f"Error: Temporary audio file not found for transcription: {temp_file_path}")
             return jsonify({'success': False, 'error': "Internal processing error (audio file missing)"}), 500
        except sr.WaitTimeoutError:
             # Less common with files, but possible if record() has issues
             print("No speech detected within timeout by SpeechRecognition.")
             transcription_text = "{No speech detected}"
        except Exception as e:
            print(f"Error during speech recognition processing: {e}")
            traceback.print_exc()
            # Return an error as transcription failed
            return jsonify({'success': False, 'error': f"Speech recognition processing failed: {e}"}), 500

        # --- 3. Embedding Extraction, Matching & Labeling----
        # Check if diarization actually succeeded before trying to use it
        if diarization:
            print("Extracting embeddings and matching speakers...")
            chunk_speaker_map = {} # Maps Pyannote label (SPEAKER_XX) to consistent label (Primary User, User_Y)
            match_start = time.time()

            for turn, _, speaker_pyannote in diarization.itertracks(yield_label=True):
                if speaker_pyannote in chunk_speaker_map: continue

                start_ms = int(turn.start * 1000)
                end_ms = int(turn.end * 1000)
                if start_ms >= end_ms or end_ms > audio_duration_ms: # Basic bounds check
                     print(f"Skipping invalid diarization segment: {start_ms}ms - {end_ms}ms (Audio length: {audio_duration_ms}ms)")
                     continue

                # Extract segment for embedding using the pydub audio object
                segment_pydub = audio[start_ms:end_ms]
                if len(segment_pydub) < 150: # Skip very short segments
                    print(f"Skipping short segment for embedding: {len(segment_pydub)}ms")
                    continue

                segment_tensor = load_audio_for_embedding(segment_pydub)
                current_embedding = extract_embedding(segment_tensor)
                if current_embedding is None:
                    print(f"Failed to extract embedding for segment {speaker_pyannote} ({start_ms}-{end_ms}ms)")
                    continue

                if DEBUG_SIMILARITY:
                    print(f"\n--- Speaker segment: {speaker_pyannote}, {start_ms}ms to {end_ms}ms ---")
                    dump_embeddings_info("Current Segment Embedding", current_embedding)

                # --- Speaker Matching Logic (A, B, C) ---
                matched_label = None

                # A. Check against Primary User
                if primary_embeddings:
                    # ... (your primary user check code) ...
                    max_primary_sim = 0
                    # ... calculation ...
                    # print(f"MAX PRIMARY SIM: {max_primary_sim:.4f}, THRESHOLD: {PRIMARY_USER_THRESHOLD}") # Moved debug print
                    if max_primary_sim >= PRIMARY_USER_THRESHOLD:
                        matched_label = "Primary User"
                        print(f"  MATCH: {speaker_pyannote} as Primary User (Sim: {max_primary_sim:.3f})")
                # else: # No need for else print here, handled by outer check

                # B. Check against HISTORY USERS.
                if matched_label is None:
                    best_match_label = None
                    best_match_sim = 0
                    # ... (your history check code) ...
                    # print(f"BEST HISTORY MATCH: {best_match_label}, SIM: {best_match_sim:.4f}, THRESHOLD: {OTHER_SPEAKER_THRESHOLD}") # Moved debug print
                    if best_match_sim >= OTHER_SPEAKER_THRESHOLD:
                        matched_label = best_match_label
                        speaker_history[matched_label] = update_speaker_embedding(
                            ensure_list(speaker_history.get(matched_label, [])), current_embedding) # Ensure list before update
                        print(f"  MATCH: {speaker_pyannote} as existing {matched_label} (Sim: {best_match_sim:.3f})")
                    else:
                        # C. Assign New Speaker Label
                        matched_label = f"User_{next_user_id}"
                        speaker_history[matched_label] = [current_embedding] # Start history
                        next_user_id += 1
                        print(f"  NEW: Assigned new label {matched_label} to {speaker_pyannote} (Best sim: {best_match_sim:.3f})")

                chunk_speaker_map[speaker_pyannote] = matched_label

            print(f"Speaker matching finished in {time.time() - match_start:.2f}s")
        else:
            # Diarization failed or was skipped
            print("Skipping speaker matching as diarization was not successful.")
            chunk_speaker_map = {} # Ensure it's empty

        # --- 4. Combine Transcription with Consistent Labels ---
        print("Combining transcription with consistent labels...")
        combined_segments = []
        labeled_segments = []
        unlabeled_segments = []

        # Since we're using Google Speech Recognition instead of Whisper,
        # we need to handle the transcription differently
        if transcription_text:
            # Create a single segment with the entire transcription
            seg_text = transcription_text.strip()
            if seg_text and diarization:
                # Find the most likely speaker for the entire segment
                most_common_speaker = None
                max_count = 0
                
                # Count occurrences of each speaker in the diarization
                speaker_counts = {}
                for turn, _, speaker_pyannote in diarization.itertracks(yield_label=True):
                    speaker_label = chunk_speaker_map.get(speaker_pyannote, "User_1")
                    speaker_counts[speaker_label] = speaker_counts.get(speaker_label, 0) + 1
                    if speaker_counts[speaker_label] > max_count:
                        max_count = speaker_counts[speaker_label]
                        most_common_speaker = speaker_label
                
                # If no speaker was found, use a default
                if not most_common_speaker:
                    most_common_speaker = "User_1"
                
                # Create a single segment with the entire transcription
                labeled_segments.append({
                    "speaker": most_common_speaker,
                    "text": seg_text,
                    "start_s": 0,
                    "end_s": audio_duration_ms / 1000  # Convert ms to seconds
                })
            else:
                # If no diarization, just create an unlabeled segment
                unlabeled_segments.append({
                    "speaker": None,
                    "text": seg_text,
                    "start_s": 0,
                    "end_s": audio_duration_ms / 1000
                })

        # Sort all segments by start time
        all_segments = labeled_segments + unlabeled_segments
        all_segments.sort(key=lambda x: x["start_s"])

        # Now process segments and assign labels to unknown segments
        next_known_speaker = None
        processed_segments = []

        # First forward pass - assign based on next speaker
        for i, segment in enumerate(all_segments):
            if segment["speaker"] is not None:
                # This is a labeled segment, use it as reference for previous unlabeled ones
                next_known_speaker = segment["speaker"]
                # Process any accumulated unlabeled segments
                for unprocessed in processed_segments:
                    if unprocessed["speaker"] is None:
                        unprocessed["speaker"] = next_known_speaker
                processed_segments.append(segment)
            else:
                # Add to processing queue, will be assigned once we find the next speaker
                processed_segments.append(segment)

        # If we still have unprocessed segments at the end (no next speaker found)
        # And we have at least one identified speaker in the entire recording
        if any(seg["speaker"] is None for seg in processed_segments) and any(seg["speaker"] is not None for seg in processed_segments):
            # Find the first valid speaker in the recording
            first_valid_speaker = None
            for seg in processed_segments:
                if seg["speaker"] is not None:
                    first_valid_speaker = seg["speaker"]
                    break
                    
            # Assign remaining unprocessed segments
            for segment in processed_segments:
                if segment["speaker"] is None:
                    segment["speaker"] = first_valid_speaker

        # Final fallback - if we still have unassigned speakers
        for segment in processed_segments:
            if segment["speaker"] is None:
                segment["speaker"] = "User_1"

        # Now all segments should have a speaker assigned
        combined_segments = processed_segments

        # --- 5. Combine, Filter, and Format Output ---
        print("Combining and formatting results...")
        final_transcription = ""
        filtered_count = 0
        primary_count = 0

        # Group consecutive segments by speaker
        grouped_segments = []
        current_group = None

        for seg in combined_segments:
            speaker = seg["speaker"]
            text = seg["text"].strip()
            if not text:
                continue
                
            # Start a new group if:
            # 1. This is the first segment
            # 2. Speaker has changed
            # 3. There's a significant time gap (e.g., > 1.5 seconds)
            if (not current_group or 
                current_group["speaker"] != speaker or 
                (seg["start_s"] - current_group["end_s"]) > 1.5):
                
                # Add previous group to results if it exists
                if current_group:
                    grouped_segments.append(current_group)
                    
                # Start new group
                current_group = {
                    "speaker": speaker,
                    "text": text,
                    "start_s": seg["start_s"],
                    "end_s": seg["end_s"]
                }
            else:
                # Append to current group with proper spacing
                # Check if we need a space or punctuation 
                if text.startswith((".", ",", "!", "?", ":", ";", "'", "\"", ")", "]", "}")):
                    # No space needed before punctuation
                    current_group["text"] += text
                else:
                    # Add space between words
                    current_group["text"] += " " + text
                    
                # Update end time
                current_group["end_s"] = seg["end_s"]

        # Don't forget to add the last group
        if current_group:
            grouped_segments.append(current_group)

        # Count speakers for debugging
        speaker_counts = {}
        for group in grouped_segments:
            speaker = group["speaker"]
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            
            # Skip primary user segments
            if speaker == "Primary User":
                primary_count += 1
                filtered_count += 1
                continue
                
            # Simple formatting: Speaker: Text
            # Add basic sentence capitalization
            text = group["text"]
            if text and len(text) > 0:
                text = text[0].upper() + text[1:]
            
            # Add to final transcription
            final_transcription += f"{speaker}: {text}\n"

        # Add environmental sound info if detected but below threshold
        if env_sound and env_confidence > 0.3:
            print(f"Low confidence environmental sound detected: {env_sound.title()} ({env_confidence:.2%})")

        final_transcription = final_transcription.strip()
        
        #-------Sending to ESP-32-------#
        # In your transcribe function, replace the ESP32 sending part:
        print("Sending to ESP-32...")
        try:
            res = requests.post(ESP32_URL, data=final_transcription)
            print("Image sent to ESP-32 successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error sending to ESP-32: {e}")
        
        print("\n--- Speaker Statistics ---")
        for speaker, count in speaker_counts.items():
            print(f"{speaker}: {count} segments {'(filtered out)' if speaker == 'Primary User' else ''}")

        # Print environmental sound prediction if available
        if env_sound:
            print(f"\nEnvironmental Sound Prediction: {env_sound.title()} with {env_confidence:.2%} confidence")

        print(f"\nFormatted Transcription (excluding {filtered_count} primary user segments):\n{final_transcription}")
        response_data = {
            'success': True, 
            'transcription': final_transcription,
            'environmental_sound': {
                'type': env_sound,
                'confidence': env_confidence
            } if env_sound else None,
            'debug_info': {
                'primary_user_segments': primary_count,
                'filtered_segments': filtered_count,
                'speaker_counts': speaker_counts
            } if DEBUG_SIMILARITY else None
        }

        # --- Update session storage ---
        session_data["speaker_history"] = speaker_history
        session_data["next_user_id"] = next_user_id
        session_storage[session_id] = session_data
        session.modified = True

        end_time = time.time()
        print(f"Total processing time for request: {end_time - start_time:.2f}s")
        return jsonify(response_data)

    except Exception as e:
        print(f"Unhandled error in /transcribe endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': "An internal server error occurred."}), 500
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path); print(f"Deleted temporary file: {temp_file_path}")
            except Exception as e: print(f"Error deleting temp file {temp_file_path}: {e}")

# --- Run App ---
if __name__ == '__main__':
    print("Starting Flask server with diarization and embedding support...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)