import React, { useRef, useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface MicrophoneButtonProps {
  isListening: boolean;
  toggleListening: () => void;
  audioLevel: number;
  isRecording?: boolean;
  size?: 'normal' | 'large';
}

const MicrophoneButton: React.FC<MicrophoneButtonProps> = ({ 
  isListening, 
  toggleListening,
  audioLevel,
  isRecording = false,
  size = 'normal'
}) => {
  // Add audio reference
  const clickSoundRef = useRef<HTMLAudioElement>(null);
  const [isAnimating, setIsAnimating] = useState(false);
  
  // Function to play click sound
  const playClickSound = () => {
    if (clickSoundRef.current) {
      clickSoundRef.current.currentTime = 0;
      clickSoundRef.current.play().catch(e => console.error("Error playing sound:", e));
    }
  };
  
  // Determine button size based on prop
  const buttonSize = size === 'large' ? 'w-36 h-36' : 'w-24 h-24';
  const iconSize = size === 'large' ? 'h-16 w-16' : 'h-12 w-12';
  
  // Get button colors and effects based on state
  const getButtonColors = () => {
    if (isListening) {
      if (isRecording) {
        return {
          bg: 'bg-red-500',
          shadow: 'shadow-red-500/30',
          hoverBg: 'hover:bg-gradient-to-r hover:from-red-600 hover:to-rose-500',
          hoverText: 'group-hover:text-white',
          glowEffect: 'glow-red',
          innerGlow: 'rgba(248, 113, 113, 0.9)',
          outerGlow: 'rgba(220, 38, 38, 0.7)'
        };
      } else {
        return {
          bg: 'bg-green-500',
          shadow: 'shadow-green-500/30',
          hoverBg: 'hover:bg-gradient-to-r hover:from-green-600 hover:to-emerald-500',
          hoverText: 'group-hover:text-white',
          glowEffect: 'glow-green',
          innerGlow: 'rgba(74, 222, 128, 0.9)',
          outerGlow: 'rgba(22, 163, 74, 0.7)'
        };
      }
    } else {
      return {
        bg: 'bg-purple-600',
        shadow: 'shadow-purple-600/30',
        hoverBg: 'hover:bg-gradient-to-r hover:from-purple-700 hover:to-violet-600',
        hoverText: 'group-hover:text-purple-100',
        glowEffect: 'glow',
        innerGlow: 'rgba(147, 51, 234, 0.8)',
        outerGlow: 'rgba(126, 34, 206, 0.5)'
      };
    }
  };

  const { bg, shadow, hoverBg, hoverText, glowEffect, innerGlow, outerGlow } = getButtonColors();
  
  // Animation variants for button click
  const buttonVariants = {
    idle: { scale: 1, rotateX: 0, rotateY: 0 },
    hover: { 
      scale: 1.05,
      y: [0, -3, 0],
      transition: {
        y: {
          repeat: Infinity,
          duration: 2.5,
          ease: "easeInOut"
        },
        scale: {
          duration: 0.3,
          ease: "easeOut"
        }
      }
    },
    tap: { 
      scale: 0.95,
      rotateX: 0,
      rotateY: 0,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 15
      }
    }
  };

  // Animation variants for the icon with improved reset behavior
  const iconVariants = {
    idle: { scale: 1, rotate: 0 },
    listening: {
      scale: [1, 1.1, 1],
      rotate: 0,
      transition: {
        repeat: Infinity,
        duration: 1.5
      }
    },
    clicked: {
      rotate: [0, -10, 10, -5, 5, 0],
      scale: [1, 1.2, 0.9, 1.1, 1],
      transition: {
        duration: 0.4,
        onComplete: () => setIsAnimating(false)
      }
    }
  };
  
  // Handle button click with sound and animation state
  const handleClick = () => {
    playClickSound();
    setIsAnimating(true);
    toggleListening();
  };
  
  // Effect to handle mouse move for 3D tilt effect
  const buttonRef = useRef<HTMLButtonElement>(null);
  
  useEffect(() => {
    const button = buttonRef.current;
    if (!button) return;
    
    const handleMouseMove = (e: MouseEvent) => {
      if (!button) return;
      
      const rect = button.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      // Calculate the tilt based on mouse position
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      
      // Reduce tilt amount for more subtle effect
      const tiltX = (y - centerY) / 15;
      const tiltY = (centerX - x) / 15;
      
      // Apply the tilt effect with smoother transition
      button.style.transition = 'transform 0.1s ease-out';
      button.style.transform = `perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg)`;
    };
    
    const handleMouseLeave = () => {
      button.style.transition = 'transform 0.3s ease-out';
      button.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg)';
    };
    
    button.addEventListener('mousemove', handleMouseMove);
    button.addEventListener('mouseleave', handleMouseLeave);
    
    return () => {
      button.removeEventListener('mousemove', handleMouseMove);
      button.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, []);
  
  return (
    <>
      {/* Hidden audio element for click sound */}
      <audio ref={clickSoundRef} preload="auto" src="/audio/click.mp3" />
      
      <motion.button
        ref={buttonRef}
        className={`group ${buttonSize} rounded-full flex items-center justify-center shadow-lg 
          ${bg} ${shadow} ${hoverBg} ${glowEffect} three-d-element
          transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)]
          hover:shadow-xl outline-none`}
        style={{ 
          transform: 'perspective(1000px)',
          boxShadow: `0 0 15px ${innerGlow}, 0 0 30px ${outerGlow}, 0 10px 15px -3px rgba(0, 0, 0, 0.2)`,
          backgroundImage: isListening 
            ? isRecording 
              ? 'linear-gradient(145deg, #f87171, #ef4444)' 
              : 'linear-gradient(145deg, #4ade80, #22c55e)'
            : 'linear-gradient(145deg, #9333ea, #7e22ce)',
        }}
        variants={buttonVariants}
        initial="idle"
        whileHover="hover"
        whileTap="tap"
        onClick={handleClick}
        aria-label={isListening ? "Stop listening" : "Start listening"}
      >
        {/* Add inner highlight for 3D effect */}
        <div className="absolute inset-0 rounded-full bg-white opacity-20 
          bg-gradient-to-b from-white to-transparent" 
          style={{clipPath: 'ellipse(50% 40% at 50% 0%)'}}></div>
          
        <motion.div
          variants={iconVariants}
          initial="idle"
          animate={isAnimating ? "clicked" : isListening ? "listening" : "idle"}
          className="transition-transform duration-300 relative z-10"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className={`${iconSize} text-white ${hoverText} transition-colors duration-300`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
        </motion.div>
      </motion.button>
    </>
  );
};

export default MicrophoneButton;
