import { memo } from 'react';
import { CheckIcon } from '../icons.jsx';

export const ClassifierHud = memo(function ClassifierHud({ liveScore, passStreak, requiredPasses, scanning, lastLabel, confirmed, passThreshold }) {
  if (confirmed) {
    return (
      <div className="flex items-center gap-2">
        <div className="w-5 h-5 rounded-full bg-apple-green flex items-center justify-center flex-shrink-0 shadow-glow-green">
          <CheckIcon />
        </div>
        <p className="text-white text-sm font-semibold">Verified!</p>
      </div>
    );
  }

  const meterTone = liveScore >= passThreshold * 100 ? 'from-apple-green to-glow-cyan' : liveScore >= 15 ? 'from-amber-400 to-amber-300' : 'from-red-500 to-red-400';

  return (
    <>
      <div className="flex items-center justify-between mb-1.5">
        <p className="text-white/80 text-xs font-medium">{scanning ? 'Scanning…' : `${liveScore}% confidence`}</p>
        <div className="flex items-center gap-1.5">
          {Array.from({ length: requiredPasses }).map((_, i) => (
            <div key={i} className={`w-1.5 h-1.5 rounded-full transition-all duration-300 ease-spring ${i < passStreak ? 'bg-glow-green scale-125' : 'bg-white/30'}`} />
          ))}
        </div>
      </div>

      <div className="w-full h-1.5 bg-white/15 rounded-full overflow-hidden">
        <div className={`h-full bg-gradient-to-r ${meterTone} transition-all duration-700 ease-out rounded-full`} style={{ width: `${liveScore}%` }} />
      </div>
      {lastLabel && <p className="text-white/40 text-[10px] mt-1 truncate">{lastLabel}</p>}
    </>
  );
});
