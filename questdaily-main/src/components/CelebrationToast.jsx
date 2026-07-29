import { memo } from 'react';

export const CelebrationToast = memo(function CelebrationToast({ toast }) {
  if (!toast) return null;

  const tone = toast.tone === 'level' ? 'from-apple-blue to-glow-violet' : 'from-apple-green to-glow-cyan';

  return (
    <div className="fixed bottom-6 left-1/2 z-[60] pointer-events-none animate-rise-in" style={{ transform: 'translateX(-50%)' }}>
      <div className={`bg-gradient-to-r ${tone} text-white text-sm font-semibold px-5 py-3 rounded-full shadow-glow-blue flex items-center gap-2 whitespace-nowrap`}>
        <span>{toast.emoji}</span>
        <span>{toast.message}</span>
      </div>
    </div>
  );
});
