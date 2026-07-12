"use client";

interface LevelMeterProps {
  levels: number[];
  isRecording: boolean;
}

export function LevelMeter({ levels, isRecording }: LevelMeterProps) {
  return (
    <div className="flex items-end justify-center gap-0.5 h-16 w-full">
      {levels.map((level, i) => {
        const height = Math.max(4, Math.round(level * 64));
        const opacity = isRecording ? 1 : 0.2;
        const color = level > 0.8 ? "#f87171" : level > 0.5 ? "#a78bfa" : "#7c3aed";
        return (
          <div
            key={i}
            className="w-1.5 rounded-full transition-all duration-75"
            style={{ height: `${height}px`, backgroundColor: color, opacity }}
          />
        );
      })}
    </div>
  );
}
