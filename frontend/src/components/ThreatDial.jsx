import React from 'react';

const ThreatDial = ({ threatLevel = 0 }) => {
  // Map threat level (0-10) to an angle (-90 to 90 degrees)
  const angle = (threatLevel / 10) * 180 - 90;
  
  // Determine color based on threat level
  let color = 'var(--color-accent-emerald)';
  let shadowColor = 'rgba(16, 185, 129, 0.5)';
  
  if (threatLevel >= 7) {
    color = 'var(--color-accent-red)';
    shadowColor = 'rgba(239, 68, 68, 0.7)';
  } else if (threatLevel >= 4) {
    color = 'var(--color-accent-amber)';
    shadowColor = 'rgba(245, 158, 11, 0.5)';
  }

  // Generate SVG path for the arc
  const createArc = (radius, startAngle, endAngle) => {
    const start = polarToCartesian(100, 100, radius, endAngle);
    const end = polarToCartesian(100, 100, radius, startAngle);
    const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
    return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`;
  };

  const polarToCartesian = (centerX, centerY, radius, angleInDegrees) => {
    const angleInRadians = (angleInDegrees - 90) * Math.PI / 180.0;
    return {
      x: centerX + (radius * Math.cos(angleInRadians)),
      y: centerY + (radius * Math.sin(angleInRadians))
    };
  };

  return (
    <div className="glass-panel p-6 rounded-2xl flex flex-col items-center relative overflow-hidden transition-all duration-500"
         style={{
           boxShadow: threatLevel >= 7 ? `0 0 40px -10px ${shadowColor}` : 'none',
           borderColor: threatLevel >= 7 ? 'rgba(239, 68, 68, 0.3)' : 'rgba(255,255,255,0.06)'
         }}>
      
      <div className="text-sm text-text-muted font-medium mb-4 tracking-wider uppercase">
        Hazard Proximity Level
      </div>

      <div className="relative w-48 h-28 flex items-end justify-center">
        {/* Background Arc */}
        <svg className="absolute top-0 left-0 w-full h-full" viewBox="0 0 200 120">
          <path
            d={createArc(80, -90, 90)}
            fill="none"
            stroke="var(--color-bg-surface)"
            strokeWidth="16"
            strokeLinecap="round"
          />
          {/* Active Arc */}
          <path
            d={createArc(80, -90, (threatLevel / 10) * 180 - 90)}
            fill="none"
            stroke={color}
            strokeWidth="16"
            strokeLinecap="round"
            style={{
              transition: 'stroke-dasharray 0.5s ease-out, stroke 0.5s ease',
              filter: `drop-shadow(0 0 8px ${shadowColor})`
            }}
          />
          
          {/* Needle */}
          <g style={{ transform: `rotate(${angle}deg)`, transformOrigin: '100px 100px', transition: 'transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)' }}>
            <circle cx="100" cy="100" r="6" fill={color} />
            <path d="M 96 100 L 100 25 L 104 100 Z" fill={color} />
          </g>
        </svg>
        
        {/* Number Display */}
        <div className="absolute bottom-0 w-full flex flex-col items-center pb-2">
          <span className="text-6xl font-black tabular-nums transition-colors duration-500" style={{ color: color, textShadow: `0 0 20px ${shadowColor}` }}>
            {threatLevel}
          </span>
          <span className="text-xs text-text-muted font-bold tracking-widest uppercase mt-1">
            / 10
          </span>
        </div>
      </div>
    </div>
  );
};

export default ThreatDial;
