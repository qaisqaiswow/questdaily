import { memo } from 'react';
import { CheckIcon, ChevronIcon } from './icons.jsx';

// Memoized so toggling one quest never re-renders the rest of the list —
// `quest` keeps the same object reference for every row untouched by the
// update in useQuestGame's setQuests(prev => prev.map(...)).
export const QuestRow = memo(function QuestRow({ quest, dark, proofImage, onToggle, onViewProof }) {
  const txt = dark ? 'text-white' : 'text-gray-900';

  return (
    <button
      onClick={() => onToggle(quest)}
      className="w-full flex items-center gap-3 px-3.5 py-3 text-left transition-all duration-150 ease-smooth active:scale-[0.985] active:bg-black/[0.03] dark:active:bg-white/[0.03]"
    >
      <div
        className={`w-[26px] h-[26px] rounded-full flex-shrink-0 flex items-center justify-center border-2 transition-all duration-300 ease-spring ${
          quest.completed
            ? 'bg-apple-green border-apple-green shadow-glow-green'
            : dark ? 'border-zinc-600' : 'border-gray-300'
        }`}
      >
        {quest.completed && (
          <span className="text-white animate-pop-in">
            <CheckIcon size={13} strokeWidth={3.5} />
          </span>
        )}
      </div>

      <span
        className={`flex-1 text-[15px] font-normal leading-snug transition-colors duration-300 ${
          quest.completed ? (dark ? 'text-zinc-600 line-through' : 'text-gray-400 line-through') : txt
        }`}
      >
        {quest.text}
      </span>

      <div className="flex items-center gap-2 flex-shrink-0">
        {quest.completed && proofImage && (
          <button
            onClick={(e) => { e.stopPropagation(); onViewProof(proofImage); }}
            className="w-7 h-7 rounded-md overflow-hidden flex-shrink-0 ring-1 ring-black/10 transition-transform duration-200 ease-spring hover:scale-110 active:scale-95"
          >
            <img src={proofImage} alt="proof" className="w-full h-full object-cover" />
          </button>
        )}

        <span className={`text-xs font-semibold ${quest.completed ? 'text-apple-green' : dark ? 'text-zinc-500' : 'text-gray-400'}`}>
          +{quest.xp}
        </span>

        {quest.completed ? (
          <span className={`text-[10px] ${dark ? 'text-zinc-600' : 'text-gray-300'}`}>XP</span>
        ) : (
          <span className={`text-[11px] ${dark ? 'text-zinc-600' : 'text-gray-300'}`}>
            <ChevronIcon />
          </span>
        )}
      </div>
    </button>
  );
});
