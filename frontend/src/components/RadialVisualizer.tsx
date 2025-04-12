import React, { useRef, useEffect } from 'react';

interface RadialVisualizerProps {
  audioLevel: number;
  isListening: boolean;
  isRecording?: boolean;
}

const RadialVisualizer: React.FC<RadialVisualizerProps> = ({
  audioLevel,
  isListening,
  isRecording = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Normalize audio level from dB (-60 to 0) to a 0-1 range
  const normalizeDbLevel = (dbLevel: number) => {
    // Convert dB to normalized value between 0 and 1
    // Typical dB range: -60dB (very quiet) to 0dB (max without clipping)
    const clampedDb = Math.max(-60, Math.min(0, dbLevel));
    return (clampedDb + 60) / 60;
  };

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    const animate = () => {
      if (!ctx || !canvas) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const maxRadius = Math.min(centerX, centerY) * 0.8;

      let baseColor = '#6366F1';
      let gradientStartColor = 'rgba(165, 99, 241, 0.8)';
      let gradientEndColor = 'rgb(126, 70, 229)';

      if (isRecording) {
        baseColor = '#F87171';
        gradientStartColor = 'rgba(248, 113, 113, 0.8)';
        gradientEndColor = 'rgba(220, 38, 38, 0.2)';
      } else if (isListening) {
        baseColor = '#4ADE80';
        gradientStartColor = 'rgba(74, 222, 128, 0.8)';
        gradientEndColor = 'rgba(22, 163, 74, 0.2)';
      }

      if (!isListening && !isRecording) {
        ctx.beginPath();
        ctx.arc(centerX, centerY, maxRadius * 0.5, 0, Math.PI * 2);
        ctx.strokeStyle = baseColor;
        ctx.lineWidth = 2;
        ctx.stroke();
        animationFrameRef.current = requestAnimationFrame(animate);
        return;
      }

      const numPoints = 180;
      const angleStep = (Math.PI * 2) / numPoints;

      const normalizedLevel = normalizeDbLevel(audioLevel);

      ctx.beginPath();

      // Create pulsing effect with 8-point star pattern
      for (let i = 0; i < numPoints; i++) {
        const angle = i * angleStep;

        const frequencyFactor = 8;
        const variation = Math.sin(angle * frequencyFactor) * 0.2;

        const baseSize = 0.5 + (normalizedLevel * 0.5);

        const radiusMultiplier = baseSize + (variation * normalizedLevel);
        const radius = maxRadius * Math.max(0.4, radiusMultiplier);

        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.closePath();

      const gradient = ctx.createRadialGradient(
        centerX, centerY, maxRadius * 0.2,
        centerX, centerY, maxRadius
      );

      gradient.addColorStop(0, gradientStartColor);
      gradient.addColorStop(1, gradientEndColor);

      ctx.fillStyle = gradient;
      ctx.fill();

      ctx.strokeStyle = baseColor;
      ctx.lineWidth = 2;
      ctx.stroke();

      const pulseTime = Date.now() * 0.002;
      const pulseAmount = (Math.sin(pulseTime) * 0.1 + 0.9) * normalizedLevel;

      // Add glow effect that scales with audio level
      if (normalizedLevel > 0.1) {
        ctx.save();
        ctx.filter = `blur(${4 + pulseAmount * 8}px)`;
        ctx.globalAlpha = normalizedLevel * 0.5;
        ctx.strokeStyle = baseColor;
        ctx.lineWidth = 4 + pulseAmount * 6;
        ctx.stroke();
        ctx.restore();
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [audioLevel, isListening, isRecording]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full"
      aria-label="Audio radial visualization"
    />
  );
};

export default RadialVisualizer;