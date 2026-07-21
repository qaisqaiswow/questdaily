import React, { useState, useEffect, useCallback } from 'react';

// --- QUEST POOL ---
// The master list of side quests. We pick 3-5 randomly every 24 hours.
const QUEST_POOL = [
  { id: 'q1', text: 'Do 20 pushups', xp: 50 },
  { id: 'q2', text: 'Read 10 pages of a book', xp: 40 },
  { id: 'q3', text: 'Go for a 15-minute run', xp: 75 },
  { id: 'q4', text: 'Take a cold shower', xp: 60 },
  { id: 'q5', text: 'Go outside for a walk', xp: 30 },
  { id: 'q6', text: 'Drink 2 liters of water', xp: 25 },
  { id: 'q7', text: 'Meditate for 10 minutes', xp: 45 },
  { id: 'q8', text: 'Stretch for 10 minutes', xp: 35 },
  { id: 'q9', text: 'Write down 3 things you are grateful for', xp: 20 },
  { id: 'q10', text: 'Tidy up your room for 10 minutes', xp: 40 },
];

const ONE_DAY_MS = 24 * 60 * 60 * 1000;

export default function SideQuestsApp() {
  // --- STATE MANAGEMENT ---
  // Initialize state from localStorage or use defaults
  const [level, setLevel] = useState(() => parseInt(localStorage.getItem('sq_level')) || 1);
  const [xp, setXp] = useState(() => parseInt(localStorage.getItem('sq_xp')) || 0);
  const [quests, setQuests] = useState(() => JSON.parse(localStorage.getItem('sq_quests')) || []);
  const [lastReset, setLastReset] = useState(() => parseInt(localStorage.getItem('sq_lastReset')) || 0);
  
  const [timeLeft, setTimeLeft] = useState('24:00:00');

  // Calculate XP required for the current level (Scales linearly for the prototype: L1=100, L2=200, etc.)
  const xpRequiredForNextLevel = level * 100;

  // --- LOGIC: QUEST GENERATION ---
  const generateNewQuests = useCallback((timestamp) => {
    // Randomize between 3 and 5 quests
    const numQuests = Math.floor(Math.random() * 3) + 3; 
    
    // Shuffle the pool and pick the first 'numQuests'
    const shuffled = [...QUEST_POOL].sort(() => 0.5 - Math.random());
    const selectedQuests = shuffled.slice(0, numQuests).map(q => ({
      ...q,
      completed: false // All quests start incomplete
    }));

    setQuests(selectedQuests);
    setLastReset(timestamp);
  }, []);

  // --- LOGIC: TIMER & AUTO-RESET ---
  useEffect(() => {
    const timerInterval = setInterval(() => {
      const now = Date.now();
      const timeSinceReset = now - lastReset;
      const timeRemaining = ONE_DAY_MS - timeSinceReset;

      if (timeRemaining <= 0 || quests.length === 0) {
        // 24 hours have passed, or no quests exist (first load)
        generateNewQuests(now);
      } else {
        // Update countdown timer string
        const h = Math.floor((timeRemaining / (1000 * 60 * 60)) % 24);
        const m = Math.floor((timeRemaining / 1000 / 60) % 60);
        const s = Math.floor((timeRemaining / 1000) % 60);
        
        setTimeLeft(
          `${h.toString().padStart(2, '0')}h ${m.toString().padStart(2, '0')}m ${s.toString().padStart(2, '0')}s`
        );
      }
    }, 1000);

    return () => clearInterval(timerInterval);
  }, [lastReset, quests.length, generateNewQuests]);

  // --- LOGIC: PERSISTENCE ---
  useEffect(() => {
    localStorage.setItem('sq_level', level);
    localStorage.setItem('sq_xp', xp);
    localStorage.setItem('sq_quests', JSON.stringify(quests));
    localStorage.setItem('sq_lastReset', lastReset);
  }, [level, xp, quests, lastReset]);

  // --- LOGIC: XP & LEVELING ---
  const toggleQuest = (questId) => {
    let xpChange = 0;
    
    const updatedQuests = quests.map(quest => {
      if (quest.id === questId) {
        const isNowCompleted = !quest.completed;
        xpChange = isNowCompleted ? quest.xp : -quest.xp; // Add XP if checking, subtract if unchecking
        return { ...quest, completed: isNowCompleted };
      }
      return quest;
    });

    setQuests(updatedQuests);
    handleXpChange(xpChange);
  };

  const handleXpChange = (amount) => {
    let newXp = xp + amount;
    let newLevel = level;

    // Handle Level Ups
    while (newXp >= newLevel * 100) {
      newXp -= newLevel * 100;
      newLevel++;
    }

    // Handle Level Downs (if user unchecks a quest)
    while (newXp < 0 && newLevel > 1) {
      newLevel--;
      newXp += newLevel * 100;
    }

    // Hard floor at Level 1, 0 XP
    if (newLevel === 1 && newXp < 0) newXp = 0;

    setXp(newXp);
    setLevel(newLevel);
  };

  // --- RENDER ---
  const xpPercentage = Math.min(100, Math.max(0, (xp / xpRequiredForNextLevel) * 100));
  const allQuestsCompleted = quests.length > 0 && quests.every(q => q.completed);

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-4 font-sans text-gray-200">
      {/* Main RPG Dashboard Container */}
      <div className="w-full max-w-md bg-gray-900 border-2 border-indigo-900/50 rounded-2xl shadow-[0_0_40px_rgba(49,46,129,0.3)] overflow-hidden relative">
        
        {/* Header / Stats Section */}
        <div className="bg-gradient-to-b from-indigo-900 to-gray-900 p-6 border-b border-indigo-800/50">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h1 className="text-2xl font-bold text-indigo-300 flex items-center gap-2">
                <span>⚔️</span> Side Quests
              </h1>
              <p className="text-indigo-400/70 text-sm mt-1">Daily Habit Tracker</p>
            </div>
            
            {/* Countdown Timer */}
            <div className="text-right flex flex-col items-end">
              <span className="text-xs uppercase tracking-wider text-gray-500 font-semibold mb-1">Time Remaining</span>
              <div className="bg-gray-950 px-3 py-1.5 rounded-lg border border-gray-800 font-mono text-amber-400 shadow-inner">
                ⏳ {timeLeft}
              </div>
            </div>
          </div>

          {/* Level & XP Bar */}
          <div className="mt-6">
            <div className="flex justify-between items-end mb-2">
              <div className="flex items-center gap-2">
                <span className="bg-indigo-600 text-white font-bold text-sm px-2.5 py-1 rounded-md shadow-md">
                  Lv. {level}
                </span>
                <span className="text-gray-400 text-sm font-medium">Novice Adventurer</span>
              </div>
              <span className="text-xs font-bold text-indigo-300">
                {xp} / {xpRequiredForNextLevel} XP
              </span>
            </div>
            {/* Progress Bar Track */}
            <div className="w-full h-3 bg-gray-950 rounded-full overflow-hidden border border-gray-800 shadow-inner">
              {/* Progress Bar Fill */}
              <div 
                className="h-full bg-gradient-to-r from-amber-500 to-yellow-300 transition-all duration-500 ease-out"
                style={{ width: `${xpPercentage}%` }}
              />
            </div>
          </div>
        </div>

        {/* Quests List Section */}
        <div className="p-6">
          <h2 className="text-xs uppercase tracking-widest text-gray-500 font-bold mb-4">
            Today's Objectives
          </h2>
          
          <div className="space-y-3">
            {quests.map((quest) => (
              <label 
                key={quest.id} 
                className={`flex items-center p-4 rounded-xl border cursor-pointer transition-all duration-200 ${
                  quest.completed 
                    ? 'bg-emerald-900/20 border-emerald-800/50 hover:bg-emerald-900/30' 
                    : 'bg-gray-800/50 border-gray-700 hover:bg-gray-800 hover:border-indigo-500/50'
                }`}
              >
                {/* Custom Checkbox */}
                <div className="relative flex items-center justify-center w-6 h-6 mr-4 flex-shrink-0">
                  <input
                    type="checkbox"
                    className="opacity-0 absolute w-full h-full cursor-pointer z-10"
                    checked={quest.completed}
                    onChange={() => toggleQuest(quest.id)}
                  />
                  <div className={`w-full h-full rounded border-2 flex items-center justify-center transition-colors ${
                    quest.completed ? 'bg-emerald-500 border-emerald-500' : 'border-gray-500 bg-gray-900'
                  }`}>
                    {quest.completed && <span className="text-gray-950 text-sm font-bold">✓</span>}
                  </div>
                </div>

                {/* Quest Text */}
                <div className="flex-1">
                  <p className={`text-sm font-medium transition-colors ${
                    quest.completed ? 'text-emerald-400 line-through opacity-70' : 'text-gray-200'
                  }`}>
                    {quest.text}
                  </p>
                </div>

                {/* XP Reward */}
                <div className="flex-shrink-0 ml-4">
                  <span className={`text-xs font-bold px-2 py-1 rounded ${
                    quest.completed ? 'text-emerald-300 bg-emerald-900/40' : 'text-amber-400 bg-amber-900/20'
                  }`}>
                    +{quest.xp} XP
                  </span>
                </div>
              </label>
            ))}
          </div>

          {/* Completion Message */}
          {allQuestsCompleted && (
            <div className="mt-6 p-4 bg-amber-900/20 border border-amber-700/50 rounded-xl text-center animate-pulse">
              <p className="text-amber-400 font-bold text-sm">
                🎉 All Daily Quests Cleared! 
              </p>
              <p className="text-amber-500/70 text-xs mt-1">
                Rest up. New quests arrive when the timer hits zero.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
