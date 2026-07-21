import React, { useState, useEffect, useCallback, useRef } from 'react';
import { pipeline, env } from '@huggingface/transformers';

// ─── AI SETUP ────────────────────────────────────────────────────────────────
env.allowLocalModels = false;
let classifierPromise = null;
function getClassifier() {
  if (!classifierPromise) {
    classifierPromise = pipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32');
  }
  return classifierPromise;
}

// Labels per quest. All quests now need proof.
const QUEST_LABELS = {
  q1:  { activity: ['person doing pushups on floor', 'pushup exercise', 'floor exercise plank position'],              label: 'doing pushups' },
  q2:  { activity: ['person reading a book', 'open book with pages', 'reading pages of a book'],                      label: 'reading a book' },
  q3:  { activity: ['person running outdoors', 'jogging on road or trail', 'running exercise outside'],               label: 'running' },
  q4:  { activity: ['shower running water', 'bathroom shower head with water', 'wet shower tiles'],                   label: 'in the shower' },
  q5:  { activity: ['person walking outside', 'outdoor street or park', 'person outside in nature or city'],          label: 'outside' },
  q6:  { activity: ['person drinking water', 'water bottle being drunk', 'drinking from glass or bottle'],            label: 'drinking water' },
  q7:  { activity: ['person meditating cross-legged', 'meditation sitting pose eyes closed', 'mindfulness exercise'], label: 'meditating' },
  q8:  { activity: ['person stretching muscles', 'yoga stretch pose', 'flexibility exercise stretching'],             label: 'stretching' },
  q9:  { activity: ['handwriting in notebook or journal', 'pen writing on paper', 'gratitude journal writing'],       label: 'writing in a journal' },
  q10: { activity: ['clean tidy organized room', 'neatly arranged furniture bedroom', 'organized clean living space'],label: 'tidying up' },
};
const NEGATIVE_LABELS = ['person sitting doing nothing', 'phone or computer screen', 'random everyday object'];

// ─── QUEST POOL ───────────────────────────────────────────────────────────────
const QUEST_POOL = [
  { id: 'q1',  text: 'Do 20 pushups',                              xp: 50 },
  { id: 'q2',  text: 'Read 10 pages of a book',                    xp: 40 },
  { id: 'q3',  text: 'Go for a 15-minute run',                     xp: 75 },
  { id: 'q4',  text: 'Take a cold shower',                         xp: 60 },
  { id: 'q5',  text: 'Go outside for a walk',                      xp: 30 },
  { id: 'q6',  text: 'Drink 2 liters of water',                    xp: 25 },
  { id: 'q7',  text: 'Meditate for 10 minutes',                    xp: 45 },
  { id: 'q8',  text: 'Stretch for 10 minutes',                     xp: 35 },
  { id: 'q9',  text: 'Write down 3 things you are grateful for',   xp: 20 },
  { id: 'q10', text: 'Tidy up your room for 10 minutes',           xp: 40 },
];

const ONE_DAY_MS = 24 * 60 * 60 * 1000;
// How many consecutive passing scans before we unlock the Complete button
const REQUIRED_PASSES = 2;
// Confidence threshold to count as a pass
const PASS_THRESHOLD = 0.28;

// ─── CAMERA MODAL ─────────────────────────────────────────────────────────────
function CameraModal({ quest, onConfirm, onCancel }) {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const streamRef   = useRef(null);
  const scanTimerRef = useRef(null);

  // phase: 'starting' | 'live' | 'error'
  const [phase,         setPhase]         = useState('starting');
  const [camError,      setCamError]      = useState(null);
  const [facingMode,    setFacingMode]    = useState('environment');

  // AI state
  const [modelReady,    setModelReady]    = useState(false);
  const [modelProgress, setModelProgress] = useState(null);
  const [scanning,      setScanning]      = useState(false);
  const [liveScore,     setLiveScore]     = useState(0);      // 0-100
  const [passStreak,    setPassStreak]    = useState(0);      // consecutive passes
  const [confirmed,     setConfirmed]     = useState(false);  // locked in
  const [lastLabel,     setLastLabel]     = useState('');

  const labels     = QUEST_LABELS[quest.id];
  const allLabels  = labels ? [...labels.activity, ...NEGATIVE_LABELS] : [];

  // ── Camera startup ──────────────────────────────────────────────────────────
  const startCamera = useCallback(async (mode) => {
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    const attempts = [
      { video: { facingMode: { ideal: mode }, width: { ideal: 1280 }, height: { ideal: 720 } } },
      { video: { facingMode: { ideal: mode } } },
      { video: true },
    ];
    for (const constraints of attempts) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;
        await new Promise(r => setTimeout(r, 60));
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }
        setPhase('live');
        setCamError(null);
        return;
      } catch (err) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          setCamError('permission'); setPhase('error'); return;
        }
      }
    }
    setCamError('unavailable'); setPhase('error');
  }, []);

  useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCamError('unsupported'); setPhase('error'); return;
    }
    startCamera(facingMode);
    return () => {
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [facingMode]);

  // ── Preload model as soon as modal opens ────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await getClassifier((evt) => {
          if (cancelled) return;
          if (evt.status === 'progress' && evt.total) {
            setModelProgress(Math.round((evt.loaded / evt.total) * 100));
          }
        });
        if (!cancelled) { setModelReady(true); setModelProgress(null); }
      } catch {
        if (!cancelled) { setModelReady(true); } // fail open so camera still works
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // ── Live scanning loop — every 2.5 s once camera + model are ready ──────────
  const passStreakRef = useRef(0); // keep ref in sync for the interval closure

  useEffect(() => {
    if (phase !== 'live' || !modelReady || confirmed || !labels) return;

    const grabAndScan = async () => {
      const video  = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;

      setScanning(true);
      canvas.width  = video.videoWidth  || 640;
      canvas.height = video.videoHeight || 480;
      canvas.getContext('2d').drawImage(video, 0, 0);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.7);

      try {
        const classifier = await getClassifier();
        const results = await classifier(dataUrl, allLabels);

        const actScore = results
          .filter(r => labels.activity.includes(r.label))
          .reduce((s, r) => s + r.score, 0);
        const negScore = results
          .filter(r => NEGATIVE_LABELS.includes(r.label))
          .reduce((s, r) => s + r.score, 0);

        const pct    = Math.round(actScore * 100);
        const passed = actScore > negScore && actScore >= PASS_THRESHOLD;
        const topLabel = results[0]?.label ?? '';

        setLiveScore(pct);
        setLastLabel(topLabel);

        if (passed) {
          passStreakRef.current += 1;
        } else {
          passStreakRef.current = 0;
        }
        setPassStreak(passStreakRef.current);

        if (passStreakRef.current >= REQUIRED_PASSES) {
          setConfirmed(true);
        }
      } catch {
        // ignore individual scan errors
      } finally {
        setScanning(false);
      }
    };

    grabAndScan(); // fire immediately
    scanTimerRef.current = setInterval(grabAndScan, 2500);
    return () => clearInterval(scanTimerRef.current);
  }, [phase, modelReady, confirmed, labels, allLabels]);

  // ── Helpers ─────────────────────────────────────────────────────────────────
  const flipCamera = () => {
    clearInterval(scanTimerRef.current);
    setPhase('starting');
    setLiveScore(0);
    setPassStreak(0);
    passStreakRef.current = 0;
    setConfirmed(false);
    setFacingMode(m => m === 'environment' ? 'user' : 'environment');
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (ev) => {
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
      clearInterval(scanTimerRef.current);
      const dataUrl = ev.target.result;

      if (!labels) { onConfirm(dataUrl); return; }

      setPhase('live'); // temporarily set to show analyzing overlay
      setScanning(true);
      try {
        const classifier = await getClassifier();
        const results = await classifier(dataUrl, allLabels);
        const actScore = results.filter(r => labels.activity.includes(r.label)).reduce((s, r) => s + r.score, 0);
        const negScore = results.filter(r => NEGATIVE_LABELS.includes(r.label)).reduce((s, r) => s + r.score, 0);
        const pct    = Math.round(actScore * 100);
        const passed = actScore > negScore && actScore >= PASS_THRESHOLD;
        setLiveScore(pct);
        setLastLabel(results[0]?.label ?? '');
        if (passed) {
          passStreakRef.current = REQUIRED_PASSES;
          setPassStreak(REQUIRED_PASSES);
          setConfirmed(true);
        } else {
          passStreakRef.current = 0;
          setPassStreak(0);
        }
      } catch { /* ignore */ }
      finally { setScanning(false); }
    };
    reader.readAsDataURL(file);
  };

  const captureAndConfirm = () => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);
    onConfirm(canvas.toDataURL('image/jpeg', 0.82));
  };

  // ── Meter colour ─────────────────────────────────────────────────────────────
  const meterColor =
    liveScore >= PASS_THRESHOLD * 100 ? 'bg-emerald-400' :
    liveScore >= PASS_THRESHOLD * 50  ? 'bg-amber-400'   : 'bg-red-500';

  const errorMessages = {
    permission:  { icon: '🚫', title: 'Camera access blocked',   body: 'Open the app in a new tab, then allow camera access when prompted.' },
    unavailable: { icon: '📷', title: 'No camera found',          body: 'Could not access a camera. Upload a photo instead.' },
    unsupported: { icon: '⚠️', title: 'Camera not supported',    body: 'Try Chrome or Safari, or upload a photo.' },
  };
  const errInfo = errorMessages[camError] || errorMessages.unavailable;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4">
      <div className="w-full max-w-md bg-gray-900 border border-indigo-800/50 rounded-2xl overflow-hidden shadow-2xl">

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-500 font-semibold mb-0.5">AI Verification</p>
            <p className="text-sm font-bold text-indigo-300">{quest.text}</p>
          </div>
          <button onClick={onCancel} className="text-gray-500 hover:text-gray-300 text-xl leading-none">✕</button>
        </div>

        {/* Viewfinder */}
        <div className="relative bg-black aspect-video flex items-center justify-center overflow-hidden">
          {/* Live video — always mounted so ref is stable */}
          <video
            ref={videoRef}
            autoPlay playsInline muted
            className={`w-full h-full object-cover ${phase === 'live' ? 'block' : 'hidden'}`}
          />

          {/* Camera starting spinner */}
          {phase === 'starting' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
              <div className="w-8 h-8 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
              <p className="text-gray-400 text-xs">Starting camera…</p>
            </div>
          )}

          {/* Error state */}
          {phase === 'error' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center">
              <span className="text-4xl">{errInfo.icon}</span>
              <p className="text-white font-semibold text-sm">{errInfo.title}</p>
              <p className="text-gray-400 text-xs leading-relaxed">{errInfo.body}</p>
              {camError === 'permission' && (
                <a href={window.location.href} target="_blank" rel="noreferrer"
                  className="mt-1 bg-indigo-700 hover:bg-indigo-600 text-white text-xs font-semibold px-4 py-2 rounded-lg transition-colors">
                  Open in new tab ↗
                </a>
              )}
            </div>
          )}

          {/* Live AI overlay — shown when camera is live */}
          {phase === 'live' && (
            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-3">
              {/* Model loading */}
              {!modelReady && modelProgress !== null && (
                <div className="mb-2">
                  <p className="text-gray-300 text-xs mb-1">Loading AI model… {modelProgress}%</p>
                  <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
                    <div className="h-full bg-indigo-400 transition-all duration-300" style={{ width: `${modelProgress}%` }} />
                  </div>
                </div>
              )}

              {/* Confidence meter */}
              {modelReady && (
                <>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-300 font-semibold">
                      {confirmed ? '✅ Verified!' : scanning ? '🔍 Scanning…' : `Confidence: ${liveScore}%`}
                    </span>
                    <span className="text-xs text-gray-500 truncate max-w-[55%] text-right">{lastLabel}</span>
                  </div>
                  <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-700 ease-out ${confirmed ? 'bg-emerald-400' : meterColor}`}
                      style={{ width: `${confirmed ? 100 : liveScore}%` }}
                    />
                  </div>
                  {/* Pass streak dots */}
                  <div className="flex gap-1.5 mt-1.5 justify-center">
                    {Array.from({ length: REQUIRED_PASSES }).map((_, i) => (
                      <div
                        key={i}
                        className={`w-2 h-2 rounded-full transition-colors duration-300 ${
                          i < passStreak || confirmed ? 'bg-emerald-400' : 'bg-gray-600'
                        }`}
                      />
                    ))}
                  </div>
                </>
              )}

              {/* Not ready hint */}
              {!modelReady && modelProgress === null && (
                <p className="text-gray-400 text-xs text-center">Loading AI…</p>
              )}
            </div>
          )}

          <canvas ref={canvasRef} className="hidden" />
        </div>

        {/* Controls */}
        <div className="p-4 border-t border-gray-800 space-y-3">

          {phase === 'live' && (
            <>
              <div className="flex items-center gap-3">
                <button onClick={flipCamera}
                  className="flex-shrink-0 w-10 h-10 rounded-full bg-gray-800 hover:bg-gray-700 flex items-center justify-center text-lg transition-colors"
                  title="Flip camera">🔄</button>

                <button
                  onClick={captureAndConfirm}
                  disabled={!confirmed}
                  className={`flex-1 font-bold py-2.5 rounded-xl transition-all duration-300 text-sm ${
                    confirmed
                      ? 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-[0_0_16px_rgba(52,211,153,0.4)]'
                      : 'bg-gray-800 text-gray-600 cursor-not-allowed'
                  }`}
                >
                  {confirmed ? '✓ Complete Quest' : 'Hold still — AI is watching…'}
                </button>

                <label className="flex-shrink-0 cursor-pointer w-10 h-10 rounded-full bg-gray-800 hover:bg-gray-700 flex items-center justify-center text-lg transition-colors" title="Upload photo">
                  🖼
                  <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
                </label>
              </div>

              {!confirmed && (
                <p className="text-center text-gray-600 text-xs">
                  {!modelReady ? 'AI model loading…' : `Show the camera you're ${labels?.label ?? 'doing the activity'}. The button unlocks when the AI is sure.`}
                </p>
              )}
            </>
          )}

          {(phase === 'error' || phase === 'starting') && (
            <label className="flex items-center justify-center gap-2 cursor-pointer w-full bg-gray-800 hover:bg-gray-700 text-gray-300 font-semibold py-2.5 rounded-xl transition-colors text-sm">
              🖼 Upload a Photo Instead
              <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
            </label>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function SideQuestsApp() {
  const [level,     setLevel]     = useState(() => parseInt(localStorage.getItem('sq_level'))    || 1);
  const [xp,        setXp]        = useState(() => parseInt(localStorage.getItem('sq_xp'))       || 0);
  const [quests,    setQuests]    = useState(() => JSON.parse(localStorage.getItem('sq_quests')) || []);
  const [lastReset, setLastReset] = useState(() => parseInt(localStorage.getItem('sq_lastReset')) || 0);
  const [timeLeft,  setTimeLeft]  = useState('--:--:--');

  const [proofModal,   setProofModal]   = useState(null);
  const [proofImages,  setProofImages]  = useState(() => JSON.parse(localStorage.getItem('sq_proofs')) || {});
  const [viewingProof, setViewingProof] = useState(null);

  // ── XP fix: keep refs so callbacks always see latest values ─────────────────
  const xpRef    = useRef(xp);
  const levelRef = useRef(level);
  useEffect(() => { xpRef.current    = xp;    }, [xp]);
  useEffect(() => { levelRef.current = level; }, [level]);

  const xpRequiredForNextLevel = level * 100;

  // ── Quest generation ────────────────────────────────────────────────────────
  const generateNewQuests = useCallback((timestamp) => {
    const numQuests = Math.floor(Math.random() * 3) + 3;
    const shuffled  = [...QUEST_POOL].sort(() => 0.5 - Math.random());
    const selected  = shuffled.slice(0, numQuests).map(q => ({ ...q, completed: false }));
    setQuests(selected);
    setLastReset(timestamp);
    setProofImages({});
  }, []);

  // ── Timer & auto-reset ──────────────────────────────────────────────────────
  useEffect(() => {
    const id = setInterval(() => {
      const now       = Date.now();
      const remaining = ONE_DAY_MS - (now - lastReset);
      if (remaining <= 0 || quests.length === 0) {
        generateNewQuests(now);
      } else {
        const h = Math.floor((remaining / 3_600_000) % 24);
        const m = Math.floor((remaining /    60_000) % 60);
        const s = Math.floor((remaining /     1_000) % 60);
        setTimeLeft(`${String(h).padStart(2,'0')}h ${String(m).padStart(2,'0')}m ${String(s).padStart(2,'0')}s`);
      }
    }, 1000);
    return () => clearInterval(id);
  }, [lastReset, quests.length, generateNewQuests]);

  // ── Persistence ─────────────────────────────────────────────────────────────
  useEffect(() => {
    localStorage.setItem('sq_level',     level);
    localStorage.setItem('sq_xp',        xp);
    localStorage.setItem('sq_quests',    JSON.stringify(quests));
    localStorage.setItem('sq_lastReset', lastReset);
  }, [level, xp, quests, lastReset]);

  useEffect(() => {
    localStorage.setItem('sq_proofs', JSON.stringify(proofImages));
  }, [proofImages]);

  // ── XP & leveling — uses refs to avoid stale closure ───────────────────────
  const applyXpChange = useCallback((amount) => {
    let newXp    = xpRef.current + amount;
    let newLevel = levelRef.current;
    while (newXp >= newLevel * 100)    { newXp -= newLevel * 100; newLevel++; }
    while (newXp < 0 && newLevel > 1) { newLevel--; newXp += newLevel * 100; }
    if (newLevel === 1 && newXp < 0)   newXp = 0;
    setXp(newXp);
    setLevel(newLevel);
  }, []);

  // ── Quest completion ────────────────────────────────────────────────────────
  const handleQuestClick = (quest) => {
    if (quest.completed) {
      // Un-complete: get xp from quest data directly (not from state callback)
      const xpChange = -quest.xp;
      setQuests(prev => prev.map(q => q.id === quest.id ? { ...q, completed: false } : q));
      applyXpChange(xpChange);
      setProofImages(prev => { const n = { ...prev }; delete n[quest.id]; return n; });
    } else {
      setProofModal(quest);
    }
  };

  const handleProofConfirm = (questId, imageDataUrl) => {
    const quest    = quests.find(q => q.id === questId);
    const xpChange = quest ? quest.xp : 0;
    setProofImages(prev => ({ ...prev, [questId]: imageDataUrl }));
    setQuests(prev => prev.map(q => q.id === questId ? { ...q, completed: true } : q));
    applyXpChange(xpChange);
    setProofModal(null);
  };

  const xpPct  = Math.min(100, Math.max(0, (xp / xpRequiredForNextLevel) * 100));
  const allDone = quests.length > 0 && quests.every(q => q.completed);

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-4 font-sans text-gray-200">

      {/* Camera modal */}
      {proofModal && (
        <CameraModal
          quest={proofModal}
          onConfirm={(img) => handleProofConfirm(proofModal.id, img)}
          onCancel={() => setProofModal(null)}
        />
      )}

      {/* Proof lightbox */}
      {viewingProof && (
        <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4" onClick={() => setViewingProof(null)}>
          <img src={viewingProof} alt="proof" className="max-w-full max-h-full rounded-xl shadow-2xl" />
          <button className="absolute top-4 right-4 text-white text-2xl">✕</button>
        </div>
      )}

      {/* Main card */}
      <div className="w-full max-w-md bg-gray-900 border-2 border-indigo-900/50 rounded-2xl shadow-[0_0_40px_rgba(49,46,129,0.3)] overflow-hidden">

        {/* Header */}
        <div className="bg-gradient-to-b from-indigo-900 to-gray-900 p-6 border-b border-indigo-800/50">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h1 className="text-2xl font-bold text-indigo-300 flex items-center gap-2">⚔️ Side Quests</h1>
              <p className="text-indigo-400/70 text-sm mt-1">Daily Habit Tracker</p>
            </div>
            <div className="text-right flex flex-col items-end">
              <span className="text-xs uppercase tracking-wider text-gray-500 font-semibold mb-1">Time Remaining</span>
              <div className="bg-gray-950 px-3 py-1.5 rounded-lg border border-gray-800 font-mono text-amber-400 shadow-inner">
                ⏳ {timeLeft}
              </div>
            </div>
          </div>

          {/* Level & XP bar */}
          <div className="mt-6">
            <div className="flex justify-between items-end mb-2">
              <div className="flex items-center gap-2">
                <span className="bg-indigo-600 text-white font-bold text-sm px-2.5 py-1 rounded-md shadow-md">Lv. {level}</span>
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

        {/* Quest list */}
        <div className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-xs uppercase tracking-widest text-gray-500 font-bold">Today's Objectives</h2>
            <span className="text-xs text-gray-600">— all verified by AI 🤖</span>
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
                {/* Checkbox button */}
                <button
                  onClick={() => handleQuestClick(quest)}
                  className={`flex-shrink-0 w-6 h-6 mr-4 rounded border-2 flex items-center justify-center transition-colors cursor-pointer ${
                    quest.completed ? 'bg-emerald-500 border-emerald-500' : 'border-gray-500 bg-gray-900 hover:border-indigo-400'
                  }`}
                >
                  {quest.completed && <span className="text-gray-950 text-sm font-bold leading-none">✓</span>}
                </button>

                {/* Quest text */}
                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-medium transition-colors ${
                    quest.completed ? 'text-emerald-400 line-through opacity-70' : 'text-gray-200'
                  }`}>
                    {quest.text}
                  </p>
                </div>

                {/* Proof thumb + XP */}
                <div className="flex items-center gap-2 ml-3 flex-shrink-0">
                  {quest.completed && proofImages[quest.id] && (
                    <button
                      onClick={() => setViewingProof(proofImages[quest.id])}
                      className="w-8 h-8 rounded-md overflow-hidden border border-emerald-700/60 hover:border-emerald-400 transition-colors flex-shrink-0"
                    >
                      <img src={proofImages[quest.id]} alt="proof" className="w-full h-full object-cover" />
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
