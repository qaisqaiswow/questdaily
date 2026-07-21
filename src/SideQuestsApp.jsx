import React, { useState, useEffect, useCallback, useRef } from 'react';

// --- QUEST POOL ---
const QUEST_POOL = [
  { id: 'q1',  text: 'Do 20 pushups',                          xp: 50,  requiresProof: true  },
  { id: 'q2',  text: 'Read 10 pages of a book',                xp: 40,  requiresProof: false },
  { id: 'q3',  text: 'Go for a 15-minute run',                 xp: 75,  requiresProof: true  },
  { id: 'q4',  text: 'Take a cold shower',                     xp: 60,  requiresProof: true  },
  { id: 'q5',  text: 'Go outside for a walk',                  xp: 30,  requiresProof: true  },
  { id: 'q6',  text: 'Drink 2 liters of water',                xp: 25,  requiresProof: false },
  { id: 'q7',  text: 'Meditate for 10 minutes',                xp: 45,  requiresProof: true  },
  { id: 'q8',  text: 'Stretch for 10 minutes',                 xp: 35,  requiresProof: true  },
  { id: 'q9',  text: 'Write down 3 things you are grateful for', xp: 20, requiresProof: false },
  { id: 'q10', text: 'Tidy up your room for 10 minutes',       xp: 40,  requiresProof: true  },
];

const ONE_DAY_MS = 24 * 60 * 60 * 1000;

// --- CAMERA MODAL ---
function CameraModal({ quest, onConfirm, onCancel }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [phase, setPhase] = useState('camera'); // 'camera' | 'preview' | 'upload'
  const [capturedImage, setCapturedImage] = useState(null);
  const [camError, setCamError] = useState(null);
  const [facingMode, setFacingMode] = useState('environment');

  const startCamera = useCallback(async (mode) => {
    // Stop any existing stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: mode, width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCamError(null);
    } catch (err) {
      setCamError(err.name === 'NotAllowedError'
        ? 'Camera access denied. Please allow camera access and try again.'
        : 'Could not access camera. You can upload a photo instead.');
    }
  }, []);

  useEffect(() => {
    startCamera(facingMode);
    return () => {
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    };
  }, [facingMode, startCamera]);

  const snap = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    setCapturedImage(canvas.toDataURL('image/jpeg', 0.8));
    // Stop stream
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    setPhase('preview');
  };

  const retake = () => {
    setCapturedImage(null);
    setPhase('camera');
    startCamera(facingMode);
  };

  const flipCamera = () => {
    setFacingMode(m => m === 'environment' ? 'user' : 'environment');
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      setCapturedImage(ev.target.result);
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
      setPhase('preview');
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4">
      <div className="w-full max-w-md bg-gray-900 border border-indigo-800/50 rounded-2xl overflow-hidden shadow-2xl">

        {/* Modal Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-500 font-semibold mb-0.5">Photo Proof Required</p>
            <p className="text-sm font-bold text-indigo-300">{quest.text}</p>
          </div>
          <button onClick={onCancel} className="text-gray-500 hover:text-gray-300 text-xl leading-none">✕</button>
        </div>

        {/* Camera / Preview area */}
        <div className="relative bg-black aspect-video flex items-center justify-center">
          {phase === 'camera' && (
            <>
              {camError ? (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center">
                  <span className="text-3xl">📷</span>
                  <p className="text-gray-400 text-sm">{camError}</p>
                  <label className="cursor-pointer bg-indigo-700 hover:bg-indigo-600 text-white text-sm font-semibold px-4 py-2 rounded-lg transition-colors">
                    Upload Photo
                    <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
                  </label>
                </div>
              ) : (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
              )}
            </>
          )}
          {phase === 'preview' && capturedImage && (
            <img src={capturedImage} alt="proof" className="w-full h-full object-cover" />
          )}
          <canvas ref={canvasRef} className="hidden" />
        </div>

        {/* Controls */}
        <div className="p-4 border-t border-gray-800">
          {phase === 'camera' && !camError && (
            <div className="flex items-center gap-3">
              <button
                onClick={flipCamera}
                className="flex-shrink-0 w-10 h-10 rounded-full bg-gray-800 hover:bg-gray-700 flex items-center justify-center text-lg transition-colors"
                title="Flip camera"
              >
                🔄
              </button>
              <button
                onClick={snap}
                className="flex-1 bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-2.5 rounded-xl transition-colors text-sm"
              >
                📸 Take Photo
              </button>
              <label className="flex-shrink-0 cursor-pointer w-10 h-10 rounded-full bg-gray-800 hover:bg-gray-700 flex items-center justify-center text-lg transition-colors" title="Upload instead">
                🖼
                <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
              </label>
            </div>
          )}

          {phase === 'preview' && (
            <div className="flex gap-3">
              <button
                onClick={retake}
                className="flex-1 bg-gray-800 hover:bg-gray-700 text-gray-300 font-semibold py-2.5 rounded-xl transition-colors text-sm"
              >
                Retake
              </button>
              <button
                onClick={() => onConfirm(capturedImage)}
                className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-2.5 rounded-xl transition-colors text-sm"
              >
                ✓ Confirm &amp; Complete
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// --- MAIN APP ---
export default function SideQuestsApp() {
  const [level, setLevel]       = useState(() => parseInt(localStorage.getItem('sq_level'))    || 1);
  const [xp, setXp]             = useState(() => parseInt(localStorage.getItem('sq_xp'))       || 0);
  const [quests, setQuests]     = useState(() => JSON.parse(localStorage.getItem('sq_quests')) || []);
  const [lastReset, setLastReset] = useState(() => parseInt(localStorage.getItem('sq_lastReset')) || 0);
  const [timeLeft, setTimeLeft] = useState('24:00:00');

  // Camera modal state
  const [proofModal, setProofModal] = useState(null); // quest object | null

  // Proof images: { [questId]: dataURL }
  const [proofImages, setProofImages] = useState(
    () => JSON.parse(localStorage.getItem('sq_proofs')) || {}
  );
  // Which proof image is being viewed
  const [viewingProof, setViewingProof] = useState(null);

  const xpRequiredForNextLevel = level * 100;

  // --- QUEST GENERATION ---
  const generateNewQuests = useCallback((timestamp) => {
    const numQuests = Math.floor(Math.random() * 3) + 3;
    const shuffled = [...QUEST_POOL].sort(() => 0.5 - Math.random());
    const selected = shuffled.slice(0, numQuests).map(q => ({ ...q, completed: false }));
    setQuests(selected);
    setLastReset(timestamp);
    setProofImages({});
  }, []);

  // --- TIMER & AUTO-RESET ---
  useEffect(() => {
    const id = setInterval(() => {
      const now = Date.now();
      const remaining = ONE_DAY_MS - (now - lastReset);
      if (remaining <= 0 || quests.length === 0) {
        generateNewQuests(now);
      } else {
        const h = Math.floor((remaining / (1000 * 60 * 60)) % 24);
        const m = Math.floor((remaining / 1000 / 60) % 60);
        const s = Math.floor((remaining / 1000) % 60);
        setTimeLeft(`${String(h).padStart(2,'0')}h ${String(m).padStart(2,'0')}m ${String(s).padStart(2,'0')}s`);
      }
    }, 1000);
    return () => clearInterval(id);
  }, [lastReset, quests.length, generateNewQuests]);

  // --- PERSISTENCE ---
  useEffect(() => {
    localStorage.setItem('sq_level',    level);
    localStorage.setItem('sq_xp',       xp);
    localStorage.setItem('sq_quests',   JSON.stringify(quests));
    localStorage.setItem('sq_lastReset', lastReset);
  }, [level, xp, quests, lastReset]);

  useEffect(() => {
    localStorage.setItem('sq_proofs', JSON.stringify(proofImages));
  }, [proofImages]);

  // --- XP & LEVELING ---
  const applyXpChange = (amount) => {
    let newXp = xp + amount, newLevel = level;
    while (newXp >= newLevel * 100)      { newXp -= newLevel * 100; newLevel++; }
    while (newXp < 0 && newLevel > 1)   { newLevel--; newXp += newLevel * 100; }
    if (newLevel === 1 && newXp < 0)     newXp = 0;
    setXp(newXp);
    setLevel(newLevel);
  };

  // --- QUEST TOGGLING ---
  const handleQuestClick = (quest) => {
    if (quest.completed) {
      // Uncomplete — no proof needed
      completeToggle(quest.id, false);
      return;
    }
    if (quest.requiresProof) {
      setProofModal(quest);
    } else {
      completeToggle(quest.id, true);
    }
  };

  const completeToggle = (questId, nowComplete) => {
    let xpChange = 0;
    setQuests(prev => prev.map(q => {
      if (q.id !== questId) return q;
      xpChange = nowComplete ? q.xp : -q.xp;
      return { ...q, completed: nowComplete };
    }));
    applyXpChange(xpChange);
    if (!nowComplete) {
      setProofImages(prev => { const n = { ...prev }; delete n[questId]; return n; });
    }
  };

  const handleProofConfirm = (questId, imageDataUrl) => {
    setProofImages(prev => ({ ...prev, [questId]: imageDataUrl }));
    completeToggle(questId, true);
    setProofModal(null);
  };

  const xpPct = Math.min(100, Math.max(0, (xp / xpRequiredForNextLevel) * 100));
  const allDone = quests.length > 0 && quests.every(q => q.completed);

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-4 font-sans text-gray-200">

      {/* Camera Modal */}
      {proofModal && (
        <CameraModal
          quest={proofModal}
          onConfirm={(img) => handleProofConfirm(proofModal.id, img)}
          onCancel={() => setProofModal(null)}
        />
      )}

      {/* Proof Lightbox */}
      {viewingProof && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4"
          onClick={() => setViewingProof(null)}
        >
          <img src={viewingProof} alt="proof" className="max-w-full max-h-full rounded-xl shadow-2xl" />
          <button className="absolute top-4 right-4 text-white text-2xl">✕</button>
        </div>
      )}

      {/* Main Card */}
      <div className="w-full max-w-md bg-gray-900 border-2 border-indigo-900/50 rounded-2xl shadow-[0_0_40px_rgba(49,46,129,0.3)] overflow-hidden">

        {/* Header */}
        <div className="bg-gradient-to-b from-indigo-900 to-gray-900 p-6 border-b border-indigo-800/50">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h1 className="text-2xl font-bold text-indigo-300 flex items-center gap-2">
                <span>⚔️</span> Side Quests
              </h1>
              <p className="text-indigo-400/70 text-sm mt-1">Daily Habit Tracker</p>
            </div>
            <div className="text-right flex flex-col items-end">
              <span className="text-xs uppercase tracking-wider text-gray-500 font-semibold mb-1">Time Remaining</span>
              <div className="bg-gray-950 px-3 py-1.5 rounded-lg border border-gray-800 font-mono text-amber-400 shadow-inner">
                ⏳ {timeLeft}
              </div>
            </div>
          </div>

          <div className="mt-6">
            <div className="flex justify-between items-end mb-2">
              <div className="flex items-center gap-2">
                <span className="bg-indigo-600 text-white font-bold text-sm px-2.5 py-1 rounded-md shadow-md">
                  Lv. {level}
                </span>
                <span className="text-gray-400 text-sm font-medium">Novice Adventurer</span>
              </div>
              <span className="text-xs font-bold text-indigo-300">{xp} / {xpRequiredForNextLevel} XP</span>
            </div>
            <div className="w-full h-3 bg-gray-950 rounded-full overflow-hidden border border-gray-800 shadow-inner">
              <div
                className="h-full bg-gradient-to-r from-amber-500 to-yellow-300 transition-all duration-500 ease-out"
                style={{ width: `${xpPct}%` }}
              />
            </div>
          </div>
        </div>

        {/* Quest List */}
        <div className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-xs uppercase tracking-widest text-gray-500 font-bold">Today's Objectives</h2>
            <span className="text-xs text-gray-600">— 📷 = photo proof required</span>
          </div>

          <div className="space-y-3">
            {quests.map((quest) => (
              <div
                key={quest.id}
                className={`flex items-center p-4 rounded-xl border transition-all duration-200 ${
                  quest.completed
                    ? 'bg-emerald-900/20 border-emerald-800/50'
                    : 'bg-gray-800/50 border-gray-700 hover:bg-gray-800 hover:border-indigo-500/50'
                }`}
              >
                {/* Checkbox */}
                <button
                  onClick={() => handleQuestClick(quest)}
                  className={`relative flex-shrink-0 w-6 h-6 mr-4 rounded border-2 flex items-center justify-center transition-colors cursor-pointer ${
                    quest.completed ? 'bg-emerald-500 border-emerald-500' : 'border-gray-500 bg-gray-900 hover:border-indigo-400'
                  }`}
                >
                  {quest.completed && <span className="text-gray-950 text-sm font-bold leading-none">✓</span>}
                </button>

                {/* Quest info */}
                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-medium transition-colors ${
                    quest.completed ? 'text-emerald-400 line-through opacity-70' : 'text-gray-200'
                  }`}>
                    {quest.requiresProof && (
                      <span className="mr-1 not-italic no-underline" style={{ textDecoration: 'none' }}>📷</span>
                    )}
                    {quest.text}
                  </p>
                </div>

                {/* Right side: proof thumb + XP badge */}
                <div className="flex items-center gap-2 ml-3 flex-shrink-0">
                  {quest.completed && proofImages[quest.id] && (
                    <button
                      onClick={() => setViewingProof(proofImages[quest.id])}
                      className="w-8 h-8 rounded-md overflow-hidden border border-emerald-700/60 hover:border-emerald-400 transition-colors flex-shrink-0"
                      title="View proof photo"
                    >
                      <img
                        src={proofImages[quest.id]}
                        alt="proof"
                        className="w-full h-full object-cover"
                      />
                    </button>
                  )}
                  <span className={`text-xs font-bold px-2 py-1 rounded ${
                    quest.completed ? 'text-emerald-300 bg-emerald-900/40' : 'text-amber-400 bg-amber-900/20'
                  }`}>
                    +{quest.xp} XP
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* All done banner */}
          {allDone && (
            <div className="mt-6 p-4 bg-amber-900/20 border border-amber-700/50 rounded-xl text-center animate-pulse">
              <p className="text-amber-400 font-bold text-sm">🎉 All Daily Quests Cleared!</p>
              <p className="text-amber-500/70 text-xs mt-1">Rest up. New quests arrive when the timer hits zero.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
