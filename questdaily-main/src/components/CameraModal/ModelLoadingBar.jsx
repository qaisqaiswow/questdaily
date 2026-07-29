import { memo } from 'react';

export const ModelLoadingBar = memo(function ModelLoadingBar({ label, percent }) {
  const known = typeof percent === 'number' && percent > 0;
  return (
    <div className="absolute inset-x-0 bottom-0 px-4 pb-3 pt-8" style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.75) 0%, transparent 100%)' }}>
      <p className="text-white/70 text-xs mb-1.5">{label}{known ? ` — ${percent}%` : '…'}</p>
      <div className="w-full h-1.5 bg-white/15 rounded-full overflow-hidden relative">
        <div
          className="h-full bg-gradient-to-r from-apple-blue to-glow-cyan rounded-full transition-all duration-300 ease-smooth"
          style={{ width: known ? `${percent}%` : '35%' }}
        >
          {!known && <div className="absolute inset-0 animate-shimmer bg-gradient-to-r from-transparent via-white/40 to-transparent w-1/3" />}
        </div>
      </div>
    </div>
  );
});
