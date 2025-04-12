import React, { useEffect, useRef, useCallback } from 'react';

interface AudioProcessorProps {
  isListening: boolean;
  onAudioLevelChange: (level: number) => void;
  onSpeechStart: () => void;
  onSpeechEnd: (audioData: Blob) => void;
}

const AudioProcessor: React.FC<AudioProcessorProps> = ({
  isListening,
  onAudioLevelChange,
  onSpeechStart,
  onSpeechEnd,
}) => {
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioProcessorNodeRef = useRef<ScriptProcessorNode | null>(null);

  const preBufferRef = useRef<Float32Array[]>([]);
  const recordingBufferRef = useRef<Float32Array[]>([]);
  const isCurrentlyRecordingRef = useRef<boolean>(false);
  const debugTimerRef = useRef<NodeJS.Timer | null>(null);

  // Settings for audio processing
  const SAMPLE_RATE = 16000;
  const FRAME_SIZE = 512;
  
  // Pre-buffer of 1 second
  const PRE_BUFFER_DURATION_MS = 1000;
  const MAX_PRE_BUFFER_FRAMES = Math.ceil((PRE_BUFFER_DURATION_MS / 1000) * SAMPLE_RATE / FRAME_SIZE);
  
  // Simplified speech detection parameters
  const SPEECH_THRESHOLD_DB = -14; // Fixed threshold in dB
  const framesAboveThresholdRef = useRef<number>(0);
  const framesBelowThresholdRef = useRef<number>(0);
  
  // Time constants
  const FRAMES_TO_START_RECORDING = Math.ceil(0.5 * SAMPLE_RATE / FRAME_SIZE); // 0.5s above threshold
  const FRAMES_TO_STOP_RECORDING = Math.ceil(1.25 * SAMPLE_RATE / FRAME_SIZE); // 1.25s below threshold

  const callbacksRef = useRef({ onAudioLevelChange, onSpeechStart, onSpeechEnd });
  useEffect(() => {
    callbacksRef.current = { onAudioLevelChange, onSpeechStart, onSpeechEnd };
  }, [onAudioLevelChange, onSpeechStart, onSpeechEnd]);

  const finalizeRecording = useCallback(() => {
    console.log("Finalizing recording buffer.");
    if (recordingBufferRef.current.length > 0) {
      const audioData = createAudioBlob(recordingBufferRef.current);
      console.log(`Sending audio blob of size: ${audioData.size} bytes`);
      if (audioData.size > 44) {
        callbacksRef.current.onSpeechEnd(audioData);
      } else {
        console.warn("Finalized recording is empty or too small, not sending.");
      }
      // Reset recording buffer
      recordingBufferRef.current = [];
    } else {
      console.log("Finalizing recording, but buffer is empty.");
    }
    
    // Reset counters
    framesAboveThresholdRef.current = 0;
    framesBelowThresholdRef.current = 0;
    isCurrentlyRecordingRef.current = false;
  }, []);

  // Decibel calculation
  const calculateDecibels = (dataArray: Uint8Array): number => {
    // Convert byte data to float between 0 and 1
    const floatData = Array.from(dataArray).map(byte => byte / 255);

    // Calculate RMS (Root Mean Square)
    const rms = Math.sqrt(
      floatData.reduce((sum, sample) => sum + (sample * sample), 0) / floatData.length
    );

    // Convert to decibels
    return rms > 0 ? 20 * Math.log10(rms) : -60;
  };

  const handleAudioProcess = useCallback((e: AudioProcessingEvent) => {
    const inputData = e.inputBuffer.getChannelData(0);
    const audioDataCopy = new Float32Array(inputData);

    // Always maintain the pre-buffer regardless of recording state
    preBufferRef.current.push(audioDataCopy);
    while (preBufferRef.current.length > MAX_PRE_BUFFER_FRAMES) {
      preBufferRef.current.shift();
    }

    if (analyserRef.current) {
      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
      analyserRef.current.getByteFrequencyData(dataArray);

      const decibelLevel = calculateDecibels(dataArray);
      
      // Report level to parent component
      callbacksRef.current.onAudioLevelChange(decibelLevel);

      // Debug logging
      if (!debugTimerRef.current) {
        debugTimerRef.current = setInterval(() => {
          const recordingStatus = isCurrentlyRecordingRef.current ? "RECORDING" : "LISTENING";
          console.log(`[${recordingStatus}] Current dB: ${decibelLevel.toFixed(2)}, Threshold: ${SPEECH_THRESHOLD_DB}, ` + 
            `Frames above: ${framesAboveThresholdRef.current}/${FRAMES_TO_START_RECORDING}, ` + 
            `Frames below: ${framesBelowThresholdRef.current}/${FRAMES_TO_STOP_RECORDING}`);
        }, 1000);
      }

      // Simple threshold-based speech detection
      if (decibelLevel >= SPEECH_THRESHOLD_DB) {
        framesAboveThresholdRef.current++;
        framesBelowThresholdRef.current = 0;
        
        // Start recording if we've been above threshold long enough
        if (!isCurrentlyRecordingRef.current && 
            framesAboveThresholdRef.current >= FRAMES_TO_START_RECORDING) {
          console.log(`SPEECH DETECTED! Level: ${decibelLevel.toFixed(2)} dB, Starting recording`);
          isCurrentlyRecordingRef.current = true;
          
          // Copy the pre-buffer to capture speech onset
          recordingBufferRef.current = [...preBufferRef.current];
          
          callbacksRef.current.onSpeechStart();
        }
      } else {
        // Below threshold
        framesAboveThresholdRef.current = 0;
        
        if (isCurrentlyRecordingRef.current) {
          framesBelowThresholdRef.current++;
          
          // Stop recording if we've been below threshold long enough
          if (framesBelowThresholdRef.current >= FRAMES_TO_STOP_RECORDING) {
            console.log(`SILENCE DETECTED! Level: ${decibelLevel.toFixed(2)} dB, Stopping recording`);
            finalizeRecording();
          }
        }
      }
      
      // If recording, add current frame to buffer
      if (isCurrentlyRecordingRef.current) {
        recordingBufferRef.current.push(audioDataCopy);
      }
    }
  }, [finalizeRecording]);

  useEffect(() => {
    let localStream: MediaStream | null = null;

    const setupAudio = async () => {
      try {
        console.log("Requesting microphone access...");
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            sampleRate: SAMPLE_RATE,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
        localStream = stream;
        mediaStreamRef.current = stream;
        console.log("Microphone access granted.");

        const context = new AudioContext({ sampleRate: SAMPLE_RATE });
        audioContextRef.current = context;
        if (context.sampleRate !== SAMPLE_RATE) {
          console.warn(`AudioContext rate is ${context.sampleRate}, expected ${SAMPLE_RATE}.`);
        }

        const source = context.createMediaStreamSource(stream);
        mediaStreamSourceRef.current = source;

        const analyser = context.createAnalyser();
        analyser.fftSize = 1024;
        
        // Set a lower smoothing constant for more responsive level readings
        analyser.smoothingTimeConstant = 0.3;
        analyserRef.current = analyser;

        const processor = context.createScriptProcessor(FRAME_SIZE, 1, 1);
        processor.onaudioprocess = handleAudioProcess;
        audioProcessorNodeRef.current = processor;

        source.connect(analyser);
        source.connect(processor);
        processor.connect(context.destination);

        console.log(`Audio setup complete. Using fixed threshold of ${SPEECH_THRESHOLD_DB} dB`);
        console.log(`Will start recording after ${FRAMES_TO_START_RECORDING} frames (${(FRAMES_TO_START_RECORDING * FRAME_SIZE / SAMPLE_RATE).toFixed(2)}s) above threshold`);
        console.log(`Will stop recording after ${FRAMES_TO_STOP_RECORDING} frames (${(FRAMES_TO_STOP_RECORDING * FRAME_SIZE / SAMPLE_RATE).toFixed(2)}s) below threshold`);
        
      } catch (error) {
        console.error("Error setting up audio:", error);
      }
    };

    const cleanupAudio = () => {
      console.log("Cleaning up audio resources...");
      if (audioProcessorNodeRef.current) {
        audioProcessorNodeRef.current.disconnect();
        audioProcessorNodeRef.current.onaudioprocess = null;
        audioProcessorNodeRef.current = null;
      }
      if (mediaStreamSourceRef.current) {
        mediaStreamSourceRef.current.disconnect();
        mediaStreamSourceRef.current = null;
      }
      if (analyserRef.current) {
        analyserRef.current.disconnect();
        analyserRef.current = null;
      }
      if (audioContextRef.current && audioContextRef.current.state !== "closed") {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
      if (localStream) {
        localStream.getTracks().forEach((track) => track.stop());
        localStream = null;
      }
      mediaStreamRef.current = null;

      preBufferRef.current = [];
      recordingBufferRef.current = [];
      isCurrentlyRecordingRef.current = false;
      framesAboveThresholdRef.current = 0;
      framesBelowThresholdRef.current = 0;

      if (debugTimerRef.current) {
        clearInterval(debugTimerRef.current);
        debugTimerRef.current = null;
      }

      console.log("Audio cleanup complete.");
    };

    if (isListening) {
      setupAudio();
    } else {
      cleanupAudio();
    }

    return cleanupAudio;
  }, [isListening, handleAudioProcess]);

  const createAudioBlob = (buffer: Float32Array[]): Blob => {
    const length = buffer.reduce((acc, curr) => acc + curr.length, 0);
    const mergedBuffer = new Float32Array(length);
    let offset = 0;
    for (const chunk of buffer) {
      mergedBuffer.set(chunk, offset);
      offset += chunk.length;
    }
    const wavBuffer = float32ToWav(mergedBuffer, SAMPLE_RATE);
    return new Blob([wavBuffer], { type: "audio/wav" });
  };

  const float32ToWav = (samples: Float32Array, sampleRate: number): ArrayBuffer => {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, "data");
    view.setUint32(40, samples.length * 2, true);
    floatTo16BitPCM(view, 44, samples);
    return buffer;
  };

  const writeString = (view: DataView, offset: number, string: string): void => {
    for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i));
  };

  const floatTo16BitPCM = (output: DataView, offset: number, input: Float32Array): void => {
    for (let i = 0; i < input.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, input[i]));
      output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
  };

  return null;
};

export default AudioProcessor;