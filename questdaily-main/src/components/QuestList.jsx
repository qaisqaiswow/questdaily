import { memo } from 'react';
import { QuestRow } from './QuestRow.jsx';

export const QuestList = memo(function QuestList({ quests, dark, proofImages, onToggle, onViewProof, allDone }) {
  const sep = dark ? 'border-white/5' : 'border-gray-100';
  const secLabel = dark ? 'text-zinc-500' : 'text-gray-400';
  const cardBg = dark ? 'bg-zinc-900/80' : 'bg-white/80';

  return (
    <div className="flex-1 scroll-ios px-4 pt-4 pb-6 space-y-5">
      <div>
        <p className={`text-[11px] font-semibold uppercase tracking-widest ${secLabel} mb-1.5 px-1`}>
          Today's Objectives
        </p>

        <div className={`${dark ? 'bg-zinc-900/60' : 'bg-white/60'} backdrop-blur-xl rounded-[16px] overflow-hidden shadow-card-light dark:shadow-card-dark`}>
          {quests.map((quest, i) => (
            <div key={quest.id}>
              {i > 0 && <div className={`border-t ${sep} ml-14`} />}
              <QuestRow
                quest={quest}
                dark={dark}
                proofImage={proofImages[quest.id]}
                onToggle={onToggle}
                onViewProof={onViewProof}
              />
            </div>
          ))}
        </div>
      </div>

      {allDone && (
        <div className={`${cardBg} backdrop-blur-xl rounded-[16px] p-5 text-center shadow-card-light dark:shadow-card-dark animate-pop-in`}>
          <p className="text-2xl mb-2">🏆</p>
          <p className={`font-semibold text-[15px] ${dark ? 'text-white' : 'text-gray-900'}`}>All Quests Complete</p>
          <p className={`text-sm mt-1 ${dark ? 'text-zinc-400' : 'text-gray-500'}`}>Rest up. New quests when the timer hits zero.</p>
        </div>
      )}

      {quests.length > 0 && (
        <p className={`text-xs text-center ${secLabel} px-4`}>
          All quests verified by on-device AI · no data leaves your phone
        </p>
      )}
    </div>
  );
});
