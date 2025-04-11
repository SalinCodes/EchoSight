import React from 'react';

interface TranscriptionDisplayProps {
  transcript: string;
  fontSize?: string;
}

const TranscriptionDisplay: React.FC<TranscriptionDisplayProps> = ({ 
  transcript, 
  fontSize = 'medium'
}) => {
  const textSizeClass = fontSize === 'small' ? 'text-base' : 
                      fontSize === 'large' ? 'text-xl' : 'text-lg';
  
  // Split the transcript by newline characters and map each line to a paragraph
  const formatTranscript = (text: string) => {
    if (!text) return 'Waiting for speech...';
    
    return text.split('\n').map((line, index) => (
      <p key={index} className="mb-2">{line}</p>
    ));
  };
  
  return (
    <div className={`$'text-gray-100' : 'text-gray-900'} ${textSizeClass}`}>
      {formatTranscript(transcript)}
    </div>
  );
};

export default TranscriptionDisplay;
