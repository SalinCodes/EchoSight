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
} from './services/WhisperService';

const App = () => {
  const [isListening, setIsListening] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [transcript, setTranscript] = useState('');
  const [backendAvailable, setBackendAvailable] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [fontSize, setFontSize] = useState('medium');


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
      // For MVP, simulating a response instead of calling the backend
      setTimeout(() => {
        setTranscript(prev => {
          const newTranscript = prev + (prev ? '\n' : '') + "This is a simulated response";
          console.log("Updated transcript:", newTranscript);
          return newTranscript;
        });
        
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            setIsTranscribing(false);
            console.log("Transcription complete, setting isTranscribing to false");
          });
        });
      }, 1000);
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
                  <div className="absolute mb-10 -bottom-0 left-1/2 transform -translate-x-1/2"> 
                    <StatusIndicator 
                      isListening={isListening} 
                      isRecording={isSpeaking} 
                      isTranscribing={isTranscribing} 
                    /> 
                  </div>
                </motion.div>
              </AnimatePresence>
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
