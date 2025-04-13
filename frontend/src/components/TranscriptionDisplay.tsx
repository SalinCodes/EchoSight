import React, { useRef, useEffect } from 'react';

interface TranscriptionDisplayProps {
  transcript: string;
  fontSize?: string;
}

const TranscriptionDisplay: React.FC<TranscriptionDisplayProps> = ({
  transcript,
  fontSize = 'medium'
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const textSizeClass = fontSize === 'small' ? 'text-base' :
                      fontSize === 'large' ? 'text-xl' : 'text-lg';

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [transcript]);

  const formatTranscript = (text: string) => {
    if (!text) return 'Waiting for speech...';
    
    return text.split('\n').map((line, index) => (
      <p key={index} className="mb-2">{line}</p>
    ));
  };

  return (
    <div
      ref={containerRef}
      className={`'text-gray-100' : 'text-gray-900'} ${textSizeClass} w-full h-full overflow-y-auto`}
      style={{
        scrollbarWidth: 'thin',
        scrollbarColor: `'rgba(71, 85, 105, 0.5) rgba(30, 41, 59, 0.8)' : '#E5E7EB #F9FAFB'}`,
      }}
    >
      {formatTranscript(transcript)}
      <style>
        {`
          div::-webkit-scrollbar {
            width: 8px;
          }
          div::-webkit-scrollbar-thumb {
            background-color: 'rgba(71, 85, 105, 0.5)' : '#D1D5DB'};
            border-radius: 4px;
          }
          div::-webkit-scrollbar-track {
            background-color: 'rgba(30, 41, 59, 0.8)' : '#F3F4F6'};
          }
        `}
      </style>
    </div>
  );
};

export default TranscriptionDisplay;