import React from 'react';

interface StatusIndicatorProps {
  isListening: boolean;
  isRecording: boolean;
  isTranscribing: boolean;
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ 
  isListening, 
  isRecording, 
  isTranscribing 
}) => {
  // Determine status text based on current state
  const getStatusText = () => {
    if (!isListening) return "Ready";
    if (isRecording) return "Recording...";
    if (isTranscribing) return "Transcribing...";
    return "Listening...";
  };

  return (
    <div>
      <p>{getStatusText()}</p>
    </div>
  );
};

export default StatusIndicator;