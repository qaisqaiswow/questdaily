import { memo } from 'react';

// Fixed, pointer-events-none, purely transform/opacity-animated — never
// participates in layout and never causes reflow on the scrollable content
// above it. `dark` only swaps colors via className, no re-mount.
export const BackgroundFX = memo(function BackgroundFX({ dark }) {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none -z-10" aria-hidden="true">
      <div
        className={`bg-blob animate-blob w-[60vw] h-[60vw] max-w-[520px] max-h-[520px] -top-1/4 -left-1/4 ${
          dark ? 'bg-glow-blue/20' : 'bg-glow-blue/25'
        }`}
      />
      <div
        className={`bg-blob animate-blob-slow w-[50vw] h-[50vw] max-w-[440px] max-h-[440px] top-1/3 -right-1/4 ${
          dark ? 'bg-glow-violet/15' : 'bg-glow-violet/20'
        }`}
      />
      <div
        className={`bg-blob animate-blob w-[45vw] h-[45vw] max-w-[400px] max-h-[400px] -bottom-1/4 left-1/4 ${
          dark ? 'bg-glow-green/10' : 'bg-glow-green/15'
        }`}
        style={{ animationDelay: '4s' }}
      />
    </div>
  );
});
