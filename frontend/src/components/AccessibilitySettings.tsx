import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaCog, FaFont, FaTimes } from 'react-icons/fa';

interface AccessibilitySettingsProps {
  fontSize: string;
  onFontSizeChange: (size: string) => void;
}

const AccessibilitySettings: React.FC<AccessibilitySettingsProps> = ({
  fontSize,
  onFontSizeChange,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleSettings = () => {
    setIsOpen(!isOpen);
  };

  // Animation variants remain the same
  const panelVariants = {
    hidden: { opacity: 0, x: 50, scale: 0.95 },
    visible: { 
      opacity: 1, 
      x: 0, 
      scale: 1,
      transition: { 
        type: "spring", 
        stiffness: 300, 
        damping: 30,
        staggerChildren: 0.07,
        delayChildren: 0.1
      }
    },
    exit: { 
      opacity: 0, 
      x: 50, 
      scale: 0.95,
      transition: { 
        duration: 0.2,
        ease: "easeOut" 
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: { opacity: 1, y: 0 }
  };

  const buttonVariants = {
    hover: { scale: 1.05 },
    tap: { scale: 0.95 }
  };

  return (
    <div className="relative z-50" style={{ marginRight: '10px', marginTop: '15px' }}>
      {/* Settings toggle button */}
      <motion.button
        aria-label="Accessibility settings"
        onClick={toggleSettings}
        className="p-3 rounded-full bg-purple-600 text-white shadow-lg hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-400"
        whileHover={{ scale: 1.1, rotate: isOpen ? 0 : 15 }}
        whileTap={{ scale: 0.9 }}
        transition={{ type: "spring", stiffness: 400, damping: 17 }}
      >
        {isOpen ? FaTimes({ size: 20 }) : FaCog({ size: 20 })}
      </motion.button>

      {/* Settings panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="absolute right-0 mt-3 p-4 bg-slate-800/90 backdrop-blur-md border border-purple-500/30 rounded-lg shadow-xl w-72"
            variants={panelVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
          >
            <motion.h3 
              className="text-lg font-semibold mb-4 text-purple-200 border-b border-purple-500/30 pb-2"
              variants={itemVariants}
            >
              Accessibility
            </motion.h3>

            {/* Font size settings */}
            <motion.div className="mb-4" variants={itemVariants}>
              <div className="flex items-center mb-2">
                {FaFont({ className: "mr-2 text-purple-300" })}
                <span className="text-purple-200">Font Size</span>
              </div>
              <div className="flex justify-between gap-2">
                {['small', 'medium', 'large'].map((size) => (
                  <motion.button
                    key={size}
                    onClick={() => onFontSizeChange(size)}
                    className={`py-2 rounded-md transition-all ${
                      fontSize === size
                        ? 'bg-purple-600 text-white font-medium'
                        : 'bg-slate-700/60 text-slate-300 hover:bg-slate-700'
                    }`}
                    style={{
                      width: size === 'medium' ? '38%' : '31%',
                    }}
                    variants={buttonVariants}
                    whileHover="hover"
                    whileTap="tap"
                  >
                    {size.charAt(0).toUpperCase() + size.slice(1)}
                  </motion.button>
                ))}
              </div>
            </motion.div>

            {/* Close button for mobile */}
            <motion.button
              onClick={toggleSettings}
              className="mt-2 w-full py-2 px-4 bg-purple-600 hover:bg-purple-700 text-white rounded-md transition-colors"
              variants={itemVariants}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              Close
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default AccessibilitySettings;
