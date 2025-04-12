import React from 'react';
import { motion } from 'framer-motion';

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

  // Determine status color based on current state
  const getStatusColor = () => {
    if (!isListening) return "text-indigo-300";
    if (isRecording) return "text-red-300";
    if (isTranscribing) return "text-yellow-300";
    return "text-green-300";
  };

  // Animation for the status text
  const textVariants = {
    initial: { opacity: 0, y: 10 },
    animate: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.3 }
    },
    exit: { 
      opacity: 0, 
      y: -10,
      transition: { duration: 0.2 }
    }
  };

  return (
    <motion.div 
      className="flex items-center justify-center space-x-2 py-1 px-3 rounded-full bg-black/20 backdrop-blur-sm"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2, duration: 0.3 }}
    >
      {/* Status dot */}
      <div className="relative">
        <div className={`h-2 w-2 rounded-full ${
          !isListening ? "bg-indigo-400" : 
          isRecording ? "bg-red-400" : 
          isTranscribing ? "bg-yellow-400" : "bg-green-400"
        }`}>
          {/* Pulsing animation for active states */}
          {isListening && (
            <div className="absolute inset-0 rounded-full animate-ping opacity-75 
              bg-indigo-400 duration-1000" 
              style={{ 
                backgroundColor: isRecording ? '#f87171' : 
                                isTranscribing ? '#facc15' : '#4ade80',
                animationDuration: isRecording ? '0.8s' : '1.2s'
              }}
            />
          )}
        </div>
      </div>
      
      {/* Status text */}
      <motion.p 
        className={`text-xs font-medium ${getStatusColor()}`}
        key={getStatusText()} // Force re-render on text change
        variants={textVariants}
        initial="initial"
        animate="animate"
        exit="exit"
      >
        {getStatusText()}
      </motion.p>
    </motion.div>
  );
};

export default StatusIndicator;