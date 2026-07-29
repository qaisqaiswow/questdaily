import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { useDarkMode } from './hooks/useDarkMode.js';
import { useQuestGame } from './hooks/useQuestGame.js';
import { useToast } from './hooks/useToast.js';
import { BackgroundFX } from './components/BackgroundFX.jsx';
import { Header } from './components/Header.jsx';
import { QuestList } from './components/QuestList.jsx';
import { CelebrationToast } from './components/CelebrationToast.jsx';

// Deferred so the heavy AI/camera code path (and the worker bundles it
// pulls in) only ever loads once a quest is actually tapped.
const CameraModal = lazy(() => import('./components/CameraModal/CameraModal.jsx'));

export default function SideQuestsApp() {
  const [dark, setDark] = useDarkMode();
  const game = useQuestGame();
  const [toast, showToast] = useToast();
  const [proofModal, setProofModal] = useState(null);
  const [viewingProof, setViewingProof] = useState(null);
  const [justLeveledUp, setJustLeveledUp] = useState(false);
  const prevLevelRef = useRef(game.level);

  useEffect(() => {
    if (game.level > prevLevelRef.current) {
      showToast({ tone: 'level', emoji: '⭐', message: `Level ${game.level} reached!` });
      setJustLeveledUp(true);
      const id = setTimeout(() => setJustLeveledUp(false), 900);
      prevLevelRef.current = game.level;
      return () => clearTimeout(id);
    }
    prevLevelRef.current = game.level;
    return undefined;
  }, [game.level, showToast]);

  // Depend on the specific handler functions (stable via useCallback inside
  // useQuestGame), not the `game` object itself — useQuestGame returns a new
  // object literal every render, and depending on it here would recreate
  // these callbacks every render and defeat QuestRow's React.memo.
  const { handleQuestClick, handleProofConfirm } = game;

  const handleToggleDark = useCallback(() => setDark((d) => !d), [setDark]);
  const handleToggleQuest = useCallback((quest) => handleQuestClick(quest, setProofModal), [handleQuestClick]);
  const handleViewProof = useCallback((img) => setViewingProof(img), []);
  const handleCancelProof = useCallback(() => setProofModal(null), []);

  const handleConfirmProof = useCallback((img) => {
    if (!proofModal) return;
    handleProofConfirm(proofModal.id, img, proofModal.xp);
    showToast({ tone: 'quest', emoji: '✅', message: `+${proofModal.xp} XP · Quest complete!` });
    setProofModal(null);
  }, [proofModal, handleProofConfirm, showToast]);

  const bg = dark ? 'bg-black' : 'bg-[#F2F2F7]';

  return (
    <div className={`${bg} min-h-screen flex flex-col items-center transition-colors duration-300 relative`}>
      <BackgroundFX dark={dark} />
      <div className="w-full max-w-[430px] flex flex-col min-h-screen">

        {proofModal && (
          <Suspense fallback={null}>
            <CameraModal quest={proofModal} dark={dark} onConfirm={handleConfirmProof} onCancel={handleCancelProof} />
          </Suspense>
        )}

        {viewingProof && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-pop-in"
            style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)' }}
            onClick={() => setViewingProof(null)}
          >
            <img src={viewingProof} alt="proof" className="max-w-full max-h-full rounded-2xl shadow-2xl" />
          </div>
        )}

        <CelebrationToast toast={toast} />

        <Header
          dark={dark}
          onToggleDark={handleToggleDark}
          timeLeft={game.timeLeft}
          level={game.level}
          xp={game.xp}
          xpRequired={game.xpRequired}
          xpPct={game.xpPct}
          justLeveledUp={justLeveledUp}
        />

        <QuestList
          quests={game.quests}
          dark={dark}
          proofImages={game.proofImages}
          onToggle={handleToggleQuest}
          onViewProof={handleViewProof}
          allDone={game.allDone}
        />

        <div className="safe-bottom" />
      </div>
    </div>
  );
}
