import { memo } from 'react';
import { CheckIcon } from '../icons.jsx';

export const PoseTrackingBadge = memo(function PoseTrackingBadge({ bodyParts, visible }) {
  return (
    <div className="absolute inset-0 pointer-events-none border-[3px] border-dashed border-cyan-500/30 m-4 rounded-xl">
      <div className="absolute top-3 left-3 bg-black/70 backdrop-blur-md px-2.5 py-1.5 rounded-lg border border-cyan-500/40 text-[10px] font-mono tracking-wider text-cyan-400">
        <div className="flex items-center gap-1.5 mb-1 text-white uppercase font-bold text-[11px]">
          <span className={`w-2 h-2 rounded-full inline-block ${visible ? 'bg-cyan-400 animate-pulse' : 'bg-amber-400'}`} />
          {visible ? 'Pose Engine Locked' : 'Searching…'}
        </div>
        <div className="text-white/60 text-[9px] mb-0.5">Tracking Matrix Focus Points:</div>
        <div className="flex flex-wrap gap-1 max-w-[180px] mt-1">
          {bodyParts.map((part) => (
            <span key={part} className="bg-cyan-950 text-cyan-300 px-1.5 py-0.5 rounded border border-cyan-800/60 font-semibold">
              {part}
            </span>
          ))}
        </div>
      </div>

      <div className="absolute top-0 left-0 w-4 h-4 border-t-4 border-l-4 border-cyan-400" />
      <div className="absolute top-0 right-0 w-4 h-4 border-t-4 border-r-4 border-cyan-400" />
      <div className="absolute bottom-0 left-0 w-4 h-4 border-b-4 border-l-4 border-cyan-400" />
      <div className="absolute bottom-0 right-0 w-4 h-4 border-b-4 border-r-4 border-cyan-400" />
    </div>
  );
});

export const PoseRepHud = memo(function PoseRepHud({ repsDone, targetReps, phase, visible, cue, confirmed, justCounted }) {
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

  return (
    <>
      <div className="flex items-center justify-between mb-1.5">
        <p className="text-white/80 text-xs font-medium">
          {visible ? (phase === 'seek-target' ? 'Ready' : 'Locked in') : 'Move into frame'}
        </p>
        <p
          className={`text-white text-xs font-bold tracking-wide px-2 py-0.5 rounded-md bg-apple-green/80 transition-transform duration-200 ease-spring ${
            justCounted ? 'scale-125' : 'scale-100'
          }`}
        >
          {repsDone} / {targetReps} REPS
        </p>
      </div>

      <div className="w-full h-1.5 bg-white/15 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-apple-green to-glow-cyan transition-all duration-300 ease-out rounded-full"
          style={{ width: `${Math.min(100, (repsDone / targetReps) * 100)}%` }}
        />
      </div>

      <p className="text-white/80 font-medium text-[11px] mt-2 bg-black/40 p-1.5 rounded border border-white/10">
        {!visible ? 'Step back so your whole body is visible.' : phase === 'seek-target' ? `Target: ${cue}` : 'Position locked! Return to complete this rep.'}
      </p>
    </>
  );
});
