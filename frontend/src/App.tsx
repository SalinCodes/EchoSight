import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './App.css';
import './accessibility.css';
import TranscriptionDisplay from './components/TranscriptionDisplay';
import MicrophoneButton from './components/MicrophoneButton';
import AudioProcessor from './components/AudioProcessor';
import RadialVisualizer from './components/RadialVisualizer';
import AccessibilitySettings from './components/AccessibilitySettings';
import StatusIndicator from './components/StatusIndicator';

// --- IMPORT FROM SERVICE FILE ---
import {
    checkWhisperBackendAvailability,
    transcribeAudio,
    enrollPrimaryUser
} from './services/WhisperService';

const App = () => {
  const [isListening, setIsListening] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [transcript, setTranscript] = useState('');
  const [backendAvailable, setBackendAvailable] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [fontSize, setFontSize] = useState('medium');
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [enrollmentStep, setEnrollmentStep] = useState(0); // 0: Idle, 1-4: Recording clip, 5: Processing
  const [enrollmentMessage, setEnrollmentMessage] = useState<string | null>(null);
  const enrollmentClips = useRef<Blob[]>([]); // Use ref to store blobs between steps
  const clipDuration = 10000; // 10 seconds per clip
  const totalClips = 4;


  useEffect(() => {

      // Check backend availability using the imported function
      const checkBackend = async () => {
          console.log("Checking backend availability...");

          const isAvailable = await checkWhisperBackendAvailability();
          setBackendAvailable(isAvailable);
          console.log(`Backend available: ${isAvailable}`);
      };
      checkBackend();
  }, []); // Empty dependency array ensures this runs once on mount

  // Implement handleFontSizeChange function
  const handleFontSizeChange = (size: string) => {
    setFontSize(size);
  };

  // --- Audio Processor Callbacks ---
  const handleAudioLevelChange = useCallback((level: number) => {
    setAudioLevel(level);
  }, []);

  const handleSpeechStart = useCallback(() => {
    console.log("App: Speech started");
    setIsSpeaking(true);
  }, []);

  // Triggered by AudioProcessor when VAD detects speech end AND provides the audio blob
  const handleSpeechEnd = useCallback(async (audioBlob: Blob) => {
    console.log("App: Speech ended, received audio blob.");
    setIsSpeaking(false)

    if (!backendAvailable) {
      setTranscript(prev => prev + "\n[Backend unavailable]");
      return;
    }
    if (!audioBlob || audioBlob.size === 0) {
        console.log("App: Received empty audio blob, skipping transcription.");
        return;
    }

    setIsTranscribing(true); // Show processing indicator
    try {
      console.log("Sending audio for transcription...");
      const result = await transcribeAudio(audioBlob); // Send to backend service

      if (result.transcription) {
        console.log("Received transcription:", result.transcription);
        // Update transcript state
        setTranscript(prev => {
          const newTranscript = prev + (prev ? '\n' : '') + result.transcription;
          console.log("Updated transcript:", newTranscript);
          return newTranscript;
        });
        
        // Use a more reliable approach with requestAnimationFrame to ensure UI has updated
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            setIsTranscribing(false);
            console.log("Transcription complete, setting isTranscribing to false");
          });
        });

      } else if (result.error) {
          console.error('Transcription failed:', result.error);
          setTranscript(prev => {
            const newTranscript = prev + `\n[Transcription Error: ${result.error}]`;
            return newTranscript;
          });
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              setIsTranscribing(false);
              console.log("Transcription error, setting isTranscribing to false");
            });
          });
      }
    } catch (error) {
      console.error('Error during transcription fetch:', error);
      setTranscript(prev => {
        const newTranscript = prev + "\n[Transcription Error]";
        return newTranscript;
      });
      requestAnimationFrame(() => {
        setIsTranscribing(false);
        console.log("Transcription exception, setting isTranscribing to false");
      });
    }
    }, [backendAvailable]);

  // --- Enrollment Logic ---
  const recordSingleClip = useCallback(async (clipNumber: number): Promise<Blob | null> => {
    return new Promise(async (resolve, reject) => {
      setEnrollmentMessage(`Recording Clip ${clipNumber}/${totalClips} (Speak naturally for ${clipDuration / 1000}s)...`);
      console.log(`Starting recording for clip ${clipNumber}`);

      let stream: MediaStream | null = null;
      let recorder: MediaRecorder | null = null;
      const audioChunks: Blob[] = [];

      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recorder = new MediaRecorder(stream);
        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunks.push(event.data);
          }
        };

        recorder.onstop = () => {
          console.log(`Recording stopped for clip ${clipNumber}.`);
          stream?.getTracks().forEach(track => track.stop()); // Stop microphone access
          // audioContext?.close(); // Close context if used

          if (audioChunks.length === 0) {
            console.error(`No audio data recorded for clip ${clipNumber}.`);
            reject(new Error(`No audio recorded for clip ${clipNumber}. Please ensure microphone is working.`));
            return;
          }

          const audioBlob = new Blob(audioChunks, { type: 'audio/wav' }); // Use WAV
          resolve(audioBlob);
        };

        recorder.onerror = (event) => {
            console.error("MediaRecorder error:", event);
            stream?.getTracks().forEach(track => track.stop());
            reject(new Error("Recording failed due to a recorder error."));
        };

        recorder.start();

        // Stop recording after the specified duration
        setTimeout(() => {
          if (recorder?.state === "recording") {
            recorder.stop();
          }
        }, clipDuration);

      } catch (error) {
        console.error(`Error during recording clip ${clipNumber}:`, error);
        stream?.getTracks().forEach(track => track.stop()); // Ensure tracks are stopped
        // audioContext?.close(); // Close context if used
        reject(error instanceof Error ? error : new Error('Could not start recording'));
      }
    });
  }, []); // No dependencies needed here


  const handleEnrollProcess = useCallback(async () => {
    if (isEnrolling) return;

    setIsEnrolling(true);
    setEnrollmentStep(1);
    enrollmentClips.current = []; // Clear previous attempts
    setEnrollmentMessage(null); // Clear previous messages

    try {
      for (let i = 1; i <= totalClips; i++) {
        setEnrollmentStep(i); // Update step for UI feedback
        const blob = await recordSingleClip(i);
        if (blob) {
          enrollmentClips.current.push(blob);
          console.log(`Clip ${i} recorded, size: ${blob.size}`);
        } else {
          // Should have been caught by reject, but as safety
          throw new Error(`Recording failed for clip ${i}.`);
        }
        // Optional small delay between recordings
        if (i < totalClips) {
            setEnrollmentMessage(`Prepare for Clip ${i + 1}/${totalClips}...`);
            await new Promise(resolve => setTimeout(resolve, 1500)); // 1.5s pause
        }
      }

      // All clips recorded, now send to backend
      setEnrollmentStep(totalClips + 1); // Processing step
      setEnrollmentMessage("Processing enrollment audio...");
      console.log(`Sending ${enrollmentClips.current.length} clips to backend...`);

      const result = await enrollPrimaryUser(enrollmentClips.current); // Pass array of blobs

      if (result.success) {
        console.log("Enrollment successful:", result.message);
        setEnrollmentMessage(result.message || "Enrollment successful!");
      } else {
        console.error("Enrollment failed:", result.error);
        setEnrollmentMessage(`Enrollment failed: ${result.error || 'Unknown error'}`);
      }

    } catch (error) {
      console.error("Error during enrollment process:", error);
      setEnrollmentMessage(`Enrollment cancelled: ${error instanceof Error ? error.message : 'An unknown error occurred'}`);
    } finally {
      // Reset state regardless of success or failure
      setIsEnrolling(false);
      setEnrollmentStep(0);
      enrollmentClips.current = []; // Clear blobs
      // Keep the final success/error message visible for a bit? Or clear it?
      // setTimeout(() => setEnrollmentMessage(null), 5000); // Example: clear after 5s
    }
  }, [isEnrolling, recordSingleClip]); // Add dependencies

  // Toggle listening state
  const toggleListening = useCallback(() => {
    setIsListening(prev => {
      const nextState = !prev;
      if (nextState) {
        console.log("App: Starting listening mode");
        setTranscript(''); // Clear transcript on start
      } else {
        console.log("App: Stopping listening mode");
        setIsSpeaking(false);
      }
      return nextState;
    });
  }, []);

  // --- UI Rendering Logic ---
  const getBackgroundAndTextColors = () => {
    let bgColor = '';
    let textColorClass = '';
    let gradientCenterColor = '';

    if (isSpeaking) { // Use isSpeaking for the "recording" visual cue
      bgColor = 'rgb(127 29 29 / 0.8)';
      gradientCenterColor = 'rgb(153 27 27 / 0.3)';
      textColorClass = 'text-red-100';
    } else if (isListening) { // Listening but not actively speaking
      bgColor = 'rgb(20 83 45 / 0.8)';
      gradientCenterColor = 'rgb(21 128 61 / 0.3)';
      textColorClass = 'text-green-100';
    } else { // Idle state
      bgColor = 'rgb(67 20 104 / 0.9)';
      gradientCenterColor = 'rgb(107 33 168 / 0.4)';
      textColorClass = 'text-purple-100';
    }
    return { bgColor, textColorClass, gradientCenterColor };
  };

  const { bgColor, textColorClass, gradientCenterColor } = getBackgroundAndTextColors();
  const radialGradientStyle = `radial-gradient(circle at center, ${gradientCenterColor} 0%, ${bgColor} 70%)`;

  const getEnrollButtonText = () => {
    if (enrollmentStep > 0 && enrollmentStep <= totalClips) {
      return `Recording Clip ${enrollmentStep}/${totalClips}...`;
    }
    if (enrollmentStep === totalClips + 1) {
      return 'Processing...';
    }
    return `Enroll Voice (${totalClips}x${clipDuration / 1000}s)`;
  };

  return (
    <div
      className={`relative min-h-screen transition-colors duration-500 ease-in-out overflow-hidden ${textColorClass}`}
      style={{ backgroundColor: bgColor }}
    >
      {/* Background effects */}
      <div className="absolute inset-0 z-0 pointer-events-none" style={{ background: radialGradientStyle }}></div>
      <div className="absolute inset-0 z-0 opacity-20 pointer-events-none bg-noise"></div>
      <div className="vignette"></div>
      <div className="absolute inset-x-0 top-1/3 transform -translate-y-1/2 z-5 pointer-events-none overflow-hidden w-full">
          <h1 className="text-center text-[20vw] font-black opacity-15 whitespace-nowrap tracking-tighter w-full" style={{ color: 'rgba(0, 0, 0, 0.25)', textShadow: '0 2px 30px rgba(0, 0, 0, 0.2)', letterSpacing: '-0.05em' }}>
              EchoSight
          </h1>
      </div>

      {/* Main Content Area */}
      <div className="relative z-10">
        <motion.div className="container mx-auto px-4 py-8 flex flex-col">
          <header>
             <div className="absolute top-0 right-0 flex items-center space-x-4">
               <AccessibilitySettings 
                 fontSize={fontSize} 
                 onFontSizeChange={handleFontSizeChange}
               />
             </div>
          </header>

          <main className="max-w-3xl mx-auto w-full flex flex-col items-center">
            <div className="flex flex-col items-center mb-8 w-full">
              <AnimatePresence mode="wait">
                <motion.div className="mt-10 relative flex justify-center items-center w-80 h-80">
                  {/* Visualizer */}
                  <div className="absolute inset-0 flex justify-center items-center"> 
                    <div className="w-full h-full"> 
                      <RadialVisualizer 
                        audioLevel={audioLevel} 
                        isListening={isListening} 
                        isRecording={isSpeaking} 
                      /> 
                    </div> 
                  </div>
                  {/* Button */}
                  <div className="relative z-10"> 
                    <MicrophoneButton 
                      isListening={isListening} 
                      toggleListening={toggleListening} 
                      audioLevel={audioLevel} 
                      isRecording={isSpeaking} 
                      size="large" 
                    /> 
                  </div>
                  {/* Status */}
                  {/* Status Indicator - Moved up by adding negative margin */}
                  <div className="absolute mb-10 -bottom-0 left-1/2 transform -translate-x-1/2"> 
                    <StatusIndicator 
                      isListening={isListening} 
                      isRecording={isSpeaking} 
                      isTranscribing={isTranscribing} 
                    /> 
                  </div>
                </motion.div>
              </AnimatePresence>

              {/* Enroll Button & Status */}
              <div className="mt-4 mb-4 flex flex-col items-center space-y-2 min-h-[60px]"> {/* Added min-height */}
                 <button
                   onClick={handleEnrollProcess} // Use the new handler
                   disabled={isEnrolling || isListening} // Disable if enrolling or already listening
                   className={`px-6 py-2 rounded-full text-white font-semibold shadow-md transition-colors duration-200 ${
                     isEnrolling
                       ? 'bg-yellow-600 cursor-not-allowed animate-pulse' // Keep pulse for visual feedback
                       : isListening
                       ? 'bg-gray-500 cursor-not-allowed'
                       : 'bg-blue-600 hover:bg-blue-700'
                   }`}
                 >
                   {getEnrollButtonText()} {/* Dynamic button text */}
                 </button>
                 {/* Display enrollment message */}
                 {enrollmentMessage && (
                   <p className={`text-sm text-center ${enrollmentMessage.toLowerCase().includes('error') || enrollmentMessage.toLowerCase().includes('failed') || enrollmentMessage.toLowerCase().includes('cancelled') ? 'text-red-400' : 'text-green-400'}`}>
                     {enrollmentMessage}
                   </p>
                 )}
              </div>
            </div>

            {/* Transcription display */}
            <motion.div
            className="relative backdrop-blur-md rounded-xl pt-6 px-6 pb-6 max-h-[200px] w-[400px] overflow-y-auto bg-slate-800/60 shadow-lg"
            style={{
              // Keep your custom border/shadow styles if you like them
              borderLeft: '1px solid rgba(51, 65, 85, 0.6)',
              borderRight: '1px solid rgba(51, 65, 85, 0.6)',
              borderBottom: '1px solid rgba(51, 65, 85, 0.6)',
              boxShadow: '-5px 5px 15px rgba(0, 0, 0, 0.3)',

              // --- THE MASK ---
              // Defining the mask image as a linear gradient
              // Starts fully transparent at the top, becomes fully opaque further down
              maskImage: 'linear-gradient(to bottom, transparent 0%, black 25%)',
              // Webkit prefix for broader browser compatibility
              WebkitMaskImage: 'linear-gradient(to bottom, transparent 0%, black 25%)',
            }}
          >
            {/* Display Component */}
            <TranscriptionDisplay
              transcript={transcript}
              fontSize={fontSize}
            />
          </motion.div>
          </main>
        </motion.div>
      </div>

      {/* AudioProcessor Component */}
      <AudioProcessor
        isListening={isListening}
        onAudioLevelChange={handleAudioLevelChange}
        onSpeechStart={handleSpeechStart}
        onSpeechEnd={handleSpeechEnd}
      />
    </div>
  );
};
    
export default App;
