import { useCallback, useEffect, useMemo, useState } from 'react';
import { ONE_DAY_MS, pickDailyQuests } from '../data/quests.js';
import { readStorage } from '../lib/storage.js';
import { usePersistentState, usePersistedValue } from './usePersistentState.js';

function readInitialQuests() {
  const saved = readStorage('sq_quests', []);
  const anyDone = saved.some(q => q.completed);
  const lastReset = readStorage('sq_lastReset', 0);
  const expired = Date.now() - lastReset >= ONE_DAY_MS;
  if (!expired && saved.length >= 5) return saved;
  if (!expired && anyDone) return saved;
  return pickDailyQuests();
}

function readInitialProgress() {
  return {
    xp: readStorage('sq_xp', 0),
    level: readStorage('sq_level', 1),
  };
}

export function useQuestGame() {
  const [progress, setProgress] = useState(readInitialProgress);
  usePersistedValue('sq_xp', progress.xp);
  usePersistedValue('sq_level', progress.level);

  // Not routed through usePersistentState: the initial value depends on the
  // daily-reset/expiry check in readInitialQuests, not a plain stored-or-default
  // read, so we compute it once via useState and persist writes separately.
  const [quests, setQuests] = useState(readInitialQuests);
  usePersistedValue('sq_quests', quests);

  const [lastReset, setLastReset] = usePersistentState('sq_lastReset', 0);
  const [proofImages, setProofImages] = usePersistentState('sq_proofs', {});
  const [timeLeft, setTimeLeft] = useState('--:--:--');

  const xpRequired = progress.level * 100;

  const generateNewQuests = useCallback((ts) => {
    setQuests(pickDailyQuests());
    setLastReset(ts);
    setProofImages({});
  }, [setQuests, setLastReset, setProofImages]);

  useEffect(() => {
    const id = setInterval(() => {
      const now = Date.now();
      const remaining = ONE_DAY_MS - (now - lastReset);
      if (remaining <= 0 || quests.length === 0) {
        generateNewQuests(now);
        return;
      }
      const h = Math.floor((remaining / 3_600_000) % 24);
      const m = Math.floor((remaining / 60_000) % 60);
      const s = Math.floor((remaining / 1_000) % 60);
      setTimeLeft(`${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`);
    }, 1000);
    return () => clearInterval(id);
  }, [lastReset, quests.length, generateNewQuests]);

  const applyXpChange = useCallback((amount) => {
    setProgress(prev => {
      let newXp = prev.xp + amount;
      let newLevel = prev.level;
      while (newXp >= newLevel * 100) { newXp -= newLevel * 100; newLevel += 1; }
      while (newXp < 0 && newLevel > 1) { newLevel -= 1; newXp += newLevel * 100; }
      if (newLevel === 1 && newXp < 0) newXp = 0;
      return { xp: newXp, level: newLevel };
    });
  }, []);

  const handleQuestClick = useCallback((quest, onOpenProof) => {
    if (quest.completed) {
      setQuests(prev => prev.map(q => (q.id === quest.id ? { ...q, completed: false } : q)));
      applyXpChange(-quest.xp);
      setProofImages(prev => {
        const next = { ...prev };
        delete next[quest.id];
        return next;
      });
    } else {
      onOpenProof(quest);
    }
  }, [applyXpChange, setQuests, setProofImages]);

  const handleProofConfirm = useCallback((questId, img, xpReward) => {
    setProofImages(prev => ({ ...prev, [questId]: img }));
    setQuests(prev => prev.map(q => (q.id === questId ? { ...q, completed: true } : q)));
    applyXpChange(xpReward ?? 0);
  }, [applyXpChange, setQuests, setProofImages]);

  const xpPct = useMemo(() => Math.min(100, Math.max(0, (progress.xp / xpRequired) * 100)), [progress.xp, xpRequired]);
  const allDone = quests.length > 0 && quests.every(q => q.completed);

  return {
    level: progress.level,
    xp: progress.xp,
    xpRequired,
    xpPct,
    quests,
    proofImages,
    timeLeft,
    allDone,
    handleQuestClick,
    handleProofConfirm,
  };
}
