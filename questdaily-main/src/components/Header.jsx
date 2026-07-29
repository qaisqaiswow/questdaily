import { memo } from 'react';
import { LogoIcon, SunIcon, MoonIcon } from './icons.jsx';

export const Header = memo(function Header({ dark, onToggleDark, timeLeft, level, xp, xpRequired, xpPct, justLeveledUp }) {
  const cardBg = dark ? 'bg-zinc-900/80' : 'bg-white/80';
  const txt = dark ? 'text-white' : 'text-gray-900';
  const sub = dark ? 'text-zinc-400' : 'text-gray-500';
  const sep = dark ? 'border-white/5' : 'border-black/5';

  return (
    <div className={`${cardBg} backdrop-blur-xl safe-top px-4 pb-3 border-b ${sep} transition-colors duration-300`}>
      <div className="flex items-center justify-between pt-2">
        <div className="flex items-center gap-2.5">
          <LogoIcon size={34} dark={dark} />
          <h1 className={`text-[22px] font-bold tracking-tight ${txt}`}>Side Quests</h1>
        </div>

        <div className="flex items-center gap-2">
          <div className={`flex items-center gap-1 px-2.5 py-1 rounded-full ${dark ? 'bg-white/5' : 'bg-black/5'}`}>
            <span className="text-[10px]">⏱</span>
            <span className={`text-[11px] font-mono font-medium tabular-nums ${dark ? 'text-zinc-300' : 'text-gray-600'}`}>{timeLeft}</span>
          </div>
          <button
            onClick={onToggleDark}
            className={`w-8 h-8 rounded-full flex items-center justify-center transition-all duration-300 ease-spring active:scale-90 ${
              dark ? 'bg-white/5 text-zinc-300 hover:bg-white/10' : 'bg-black/5 text-gray-600 hover:bg-black/10'
            }`}
          >
            {dark ? <SunIcon /> : <MoonIcon />}
          </button>
        </div>
      </div>

      <div className="mt-3">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-1.5">
            <span
              className={`bg-gradient-to-br from-apple-blue to-glow-violet text-white text-[10px] font-bold px-2 py-0.5 rounded-full shadow-glow-blue transition-transform duration-500 ease-spring ${
                justLeveledUp ? 'scale-125' : 'scale-100'
              }`}
            >
              Lv {level}
            </span>
            <span className={`text-[11px] ${sub}`}>Novice Adventurer</span>
          </div>
          <span className={`text-[11px] tabular-nums ${dark ? 'text-zinc-500' : 'text-gray-400'}`}>{xp} / {xpRequired} XP</span>
        </div>
        <div className={`relative w-full h-1.5 rounded-full overflow-hidden ${dark ? 'bg-white/5' : 'bg-black/5'}`}>
          <div
            className="h-full bg-gradient-to-r from-apple-blue to-glow-cyan rounded-full transition-all duration-700 ease-smooth"
            style={{ width: `${xpPct}%` }}
          />
        </div>
      </div>
    </div>
  );
});
