import React, { useState, useCallback } from 'react';
import AudioProcessor from './components/AudioProcessor';
import StatusIndicator from './components/StatusIndicator';
import MicrophoneButton from './components/MicrophoneButton';
import TranscriptionDisplay from './components/TranscriptionDisplay';

const App = () => {
  const [isListening, setIsListening] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [transcript, setTranscript] = useState('');
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Toggle listening state
  const toggleListening = useCallback(() => {
    setIsListening(prev => {
      const nextState = !prev;
      if (nextState) {
        // Reset all states when starting to listen
        setTranscript('');
        setIsSpeaking(false);
        setIsTranscribing(false);
      } else {
        // Reset recording states when stopping
        setIsSpeaking(false);
        setIsTranscribing(false);
      }
      return nextState;
    });
  }, []);

  // Audio Processor Callbacks
  const handleAudioLevelChange = useCallback((level: number) => {
    setAudioLevel(level);
  }, []);

  const handleSpeechStart = useCallback(() => {
    console.log('Speech detected, starting recording');
    setIsSpeaking(true);
  }, []);

  const handleSpeechEnd = useCallback(async (audioBlob: Blob) => {
    console.log('Speech ended, processing audio');
    setIsSpeaking(false);
    setIsTranscribing(true);
    
    try {
      setTimeout(() => {
        setTranscript(prev => {
          const newTranscript = prev + (prev ? '\n' : '') + "This is a simulated response";
          console.log('Transcript updated');
          return newTranscript;
        });
        setIsTranscribing(false);
      }, 1000);
    } catch (error) {
      console.error('Error processing speech:', error);
      setIsTranscribing(false);
    }
  }, []);

  return (
    <div>
      <h1 style={{ textAlign: 'center' }}>EchoSight MVP</h1>
      
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <MicrophoneButton 
          isListening={isListening} 
          toggleListening={toggleListening} 
        />
        
        <StatusIndicator 
          isListening={isListening} 
          isRecording={isSpeaking} 
          isTranscribing={isTranscribing} 
        />
      </div>
      
      <div>
        <h2>Transcript:</h2>
        <div style={{ border: '1px solid black', padding: '10px' }}>
          <TranscriptionDisplay transcript={transcript} />
        </div>
      </div>

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
