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

const QUEST_LABELS = {
  q1:  { activity: ['person doing pushups on floor', 'pushup exercise', 'floor exercise plank position'],                      label: 'doing pushups' },
  q2:  { activity: ['person doing squats exercise', 'squat workout legs bent', 'lower body exercise squat'],                   label: 'doing squats' },
  q3:  { activity: ['person running outdoors', 'jogging on road or trail', 'running exercise outside'],                        label: 'running' },
  q4:  { activity: ['person doing pullups on bar', 'pullup bar exercise arms raised', 'hanging from bar doing pullups'],        label: 'doing pullups' },
  q5:  { activity: ['person walking outside', 'outdoor street or park', 'person outside in nature or city'],                   label: 'outside walking' },
  q6:  { activity: ['person drinking water', 'water bottle being drunk', 'drinking from glass or bottle'],                     label: 'drinking water' },
  q7:  { activity: ['person meditating cross-legged', 'meditation sitting pose eyes closed', 'mindfulness exercise'],          label: 'meditating' },
  q8:  { activity: ['person stretching muscles', 'yoga stretch pose', 'flexibility exercise stretching'],                      label: 'stretching' },
  q9:  { activity: ['healthy food meal salad vegetables fruits', 'nutritious meal on plate', 'fresh vegetables or fruit bowl'],label: 'eating healthy' },
  q10: { activity: ['person sleeping in bed', 'person resting in bed eyes closed', 'bedroom with person lying down'],          label: 'getting good sleep' },
  q11: { activity: ['person doing jumping jacks or burpees', 'cardio exercise arms raised', 'full body workout jumping'],      label: 'doing cardio' },
  q12: { activity: ['person doing situps or crunches', 'abdominal exercise on floor', 'core workout crunches'],                label: 'doing situps' },
  q13: { activity: ['handwriting in notebook or journal', 'pen writing on paper', 'gratitude journal writing'],                label: 'journaling' },
  q14: { activity: ['person drinking green smoothie or juice', 'blender with green smoothie', 'healthy green drink'],         label: 'drinking a smoothie' },
  q15: { activity: ['person doing plank exercise', 'plank position core exercise', 'forearm plank on floor'],                  label: 'holding a plank' },
  q16: { activity: ['person cycling on bike', 'riding bicycle outdoors or stationary bike', 'cycling exercise'],               label: 'cycling' },
  q17: { activity: ['person jumping rope', 'skipping rope exercise', 'jump rope workout'],                                     label: 'jumping rope' },
  q18: { activity: ['shower running water', 'bathroom shower head with water', 'wet shower tiles'],                            label: 'in the shower' },
  q19: { activity: ['healthy food no sugar', 'fruits vegetables whole foods', 'clean healthy meal on plate'],                  label: 'eating clean' },
  q20: { activity: ['person cooking in kitchen', 'food being cooked on stove', 'homemade meal preparation'],                  label: 'cooking' },
  q21: { activity: ['person breathing deeply eyes closed', 'relaxed breathing exercise', 'calm seated breathing practice'],   label: 'deep breathing' },
  q22: { activity: ['person walking outdoors on path', 'person walking in park or street', 'outdoor walking trail'],          label: 'walking' },
  q23: { activity: ['person doing lunges exercise', 'lunge workout legs split stance', 'leg exercise lunge position'],        label: 'doing lunges' },
  q24: { activity: ['person sleeping in bed at night', 'person in bed eyes closed lights off', 'sleeping in dark bedroom'],   label: 'sleeping early' },
  q25: { activity: ['person in ice bath tub', 'cold plunge tub with ice', 'ice water bath cold immersion'],                   label: 'in a cold plunge' },
};
const NEGATIVE_LABELS = ['person sitting doing nothing', 'phone or computer screen', 'random everyday object'];

// ─── QUEST POOL ───────────────────────────────────────────────────────────────
const QUEST_POOL = [
  { id: 'q1',  text: 'Do 20 pushups',                            xp: 50 },
  { id: 'q2',  text: 'Do 30 squats',                             xp: 45 },
  { id: 'q3',  text: 'Go for a 15-minute run',                   xp: 75 },
  { id: 'q4',  text: 'Do 10 pullups',                            xp: 60 },
  { id: 'q16', text: 'Do 15 minutes of cycling',                xp: 55 },
  { id: 'q17', text: 'Do 50 jumping rope reps',                 xp: 40 },
  { id: 'q18', text: 'Take a cold shower',                      xp: 50 },
  { id: 'q19', text: 'Eat no sugar today',                      xp: 65 },
  { id: 'q20', text: 'Cook a meal from scratch',                xp: 45 },
  { id: 'q21', text: 'Do 10 minutes of deep breathing',         xp: 30 },
  { id: 'q22', text: 'Take 10,000 steps',                       xp: 70 },
  { id: 'q23', text: 'Do 3 sets of lunges',                     xp: 40 },
  { id: 'q24', text: 'Go to bed before 11pm',                   xp: 35 },
  { id: 'q25', text: 'Do a 5-minute ice bath or cold plunge',   xp: 80 },
  { id: 'q5',  text: 'Walk outside for 20 minutes',              xp: 35 },
  { id: 'q6',  text: 'Drink 2 liters of water today',            xp: 25 },
  { id: 'q7',  text: 'Meditate for 10 minutes',                  xp: 45 },
  { id: 'q8',  text: 'Stretch for 10 minutes',                   xp: 35 },
  { id: 'q9',  text: 'Eat a healthy meal',                       xp: 40 },
  { id: 'q10', text: 'Get 8 hours of sleep',                     xp: 55 },
  { id: 'q11', text: 'Do 3 minutes of jumping jacks',            xp: 30 },
  { id: 'q12', text: 'Do 20 situps',                             xp: 40 },
  { id: 'q13', text: 'Write in your journal',                    xp: 20 },
  { id: 'q14', text: 'Drink a green smoothie',                   xp: 30 },
  { id: 'q15', text: 'Hold a plank for 60 seconds',              xp: 50 },
];

const ONE_DAY_MS = 24 * 60 * 60 * 1000;
const REQUIRED_PASSES = 2;
const PASS_THRESHOLD  = 0.28;

// ─── ICONS (inline SVG, no deps) ─────────────────────────────────────────────
const LogoIcon = ({ size = 34, dark }) => (
  <img src="/logo-transparent.png" alt="Side Quests" width={size} height={size}
    style={{ filter: dark ? 'invert(0)' : 'invert(1)', opacity: 0.9 }} />
);

const SunIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>
  </svg>
);
const MoonIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
  </svg>
);
const CheckIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
);
const ChevronIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="9 18 15 12 9 6"/>
  </svg>
);

// ─── CAMERA / AI MODAL ────────────────────────────────────────────────────────
function CameraModal({ quest, onConfirm, onCancel, dark }) {
  const videoRef     = useRef(null);
  const canvasRef    = useRef(null);
  const streamRef    = useRef(null);
  const scanTimerRef = useRef(null);

  const [phase,         setPhase]         = useState('starting');
  const [camError,      setCamError]      = useState(null);
  const [facingMode,    setFacingMode]    = useState('environment');
  const [modelReady,    setModelReady]    = useState(false);
  const [modelProgress, setModelProgress] = useState(null);
  const [scanning,      setScanning]      = useState(false);
  const [liveScore,     setLiveScore]     = useState(0);
  const [passStreak,    setPassStreak]    = useState(0);
  const [confirmed,     setConfirmed]     = useState(false);
  const [lastLabel,     setLastLabel]     = useState('');

  const labels    = QUEST_LABELS[quest.id];
  const allLabels = labels ? [...labels.activity, ...NEGATIVE_LABELS] : [];

  const startCamera = useCallback(async (mode) => {
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    const attempts = [
      { video: { facingMode: { ideal: mode }, width: { ideal: 1280 }, height: { ideal: 720 } } },
      { video: { facingMode: { ideal: mode } } },
      { video: true },
    ];
    for (const c of attempts) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(c);
        streamRef.current = stream;
        await new Promise(r => setTimeout(r, 60));
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }
        setPhase('live'); setCamError(null); return;
      } catch (err) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          setCamError('permission'); setPhase('error'); return;
        }
      }
    }
    setCamError('unavailable'); setPhase('error');
  }, []);

  useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia) { setCamError('unsupported'); setPhase('error'); return; }
    startCamera(facingMode);
    return () => { if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop()); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [facingMode]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await getClassifier(evt => {
          if (cancelled) return;
          if (evt.status === 'progress' && evt.total)
            setModelProgress(Math.round((evt.loaded / evt.total) * 100));
        });
        if (!cancelled) { setModelReady(true); setModelProgress(null); }
      } catch { if (!cancelled) setModelReady(true); }
    })();
    return () => { cancelled = true; };
  }, []);

  const passStreakRef = useRef(0);

  useEffect(() => {
    if (phase !== 'live' || !modelReady || confirmed || !labels) return;
    const grabAndScan = async () => {
      const video = videoRef.current, canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;
      setScanning(true);
      canvas.width = video.videoWidth || 640; canvas.height = video.videoHeight || 480;
      canvas.getContext('2d').drawImage(video, 0, 0);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
      try {
        const classifier = await getClassifier();
        const results = await classifier(dataUrl, allLabels);
        const actScore = results.filter(r => labels.activity.includes(r.label)).reduce((s, r) => s + r.score, 0);
        const negScore = results.filter(r => NEGATIVE_LABELS.includes(r.label)).reduce((s, r) => s + r.score, 0);
        const pct = Math.round(actScore * 100);
        const passed = actScore > negScore && actScore >= PASS_THRESHOLD;
        setLiveScore(pct); setLastLabel(results[0]?.label ?? '');
        if (passed) { passStreakRef.current += 1; } else { passStreakRef.current = 0; }
        setPassStreak(passStreakRef.current);
        if (passStreakRef.current >= REQUIRED_PASSES) setConfirmed(true);
      } catch { /* ignore */ } finally { setScanning(false); }
    };
    grabAndScan();
    scanTimerRef.current = setInterval(grabAndScan, 2500);
    return () => clearInterval(scanTimerRef.current);
  }, [phase, modelReady, confirmed, labels, allLabels]);

  const flipCamera = () => {
    clearInterval(scanTimerRef.current);
    setPhase('starting'); setLiveScore(0); setPassStreak(0); passStreakRef.current = 0; setConfirmed(false);
    setFacingMode(m => m === 'environment' ? 'user' : 'environment');
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = async (ev) => {
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
      clearInterval(scanTimerRef.current);
      const dataUrl = ev.target.result;
      if (!labels) { onConfirm(dataUrl); return; }
      setPhase('live'); setScanning(true);
      try {
        const classifier = await getClassifier();
        const results = await classifier(dataUrl, allLabels);
        const actScore = results.filter(r => labels.activity.includes(r.label)).reduce((s, r) => s + r.score, 0);
        const negScore = results.filter(r => NEGATIVE_LABELS.includes(r.label)).reduce((s, r) => s + r.score, 0);
        setLiveScore(Math.round(actScore * 100)); setLastLabel(results[0]?.label ?? '');
        if (actScore > negScore && actScore >= PASS_THRESHOLD) {
          passStreakRef.current = REQUIRED_PASSES; setPassStreak(REQUIRED_PASSES); setConfirmed(true);
        } else { passStreakRef.current = 0; setPassStreak(0); }
      } catch { /* ignore */ } finally { setScanning(false); }
    };
    reader.readAsDataURL(file);
  };

  const captureAndConfirm = () => {
    const video = videoRef.current, canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth || 640; canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);
    onConfirm(canvas.toDataURL('image/jpeg', 0.82));
  };

  const meterColor = liveScore >= PASS_THRESHOLD * 100 ? 'bg-green-500' : liveScore >= PASS_THRESHOLD * 50 ? 'bg-yellow-400' : 'bg-red-500';

  const bg     = dark ? 'bg-zinc-900'  : 'bg-white';
  const border = dark ? 'border-zinc-700' : 'border-gray-200';
  const txt    = dark ? 'text-white'   : 'text-gray-900';
  const sub    = dark ? 'text-zinc-400' : 'text-gray-500';
  const pill   = dark ? 'bg-zinc-800'  : 'bg-gray-100';

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center" style={{ background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}>
      <div className={`${bg} w-full max-w-lg rounded-t-[28px] overflow-hidden shadow-2xl`} style={{ paddingBottom: 'max(env(safe-area-inset-bottom), 16px)' }}>

        {/* Drag handle */}
        <div className="flex justify-center pt-3 pb-1">
          <div className={`w-10 h-1 rounded-full ${dark ? 'bg-zinc-600' : 'bg-gray-300'}`} />
        </div>

        {/* Header */}
        <div className={`flex items-center justify-between px-5 py-3 border-b ${border}`}>
          <button onClick={onCancel} className="text-[#007AFF] text-sm font-medium">Cancel</button>
          <div className="text-center">
            <p className={`text-sm font-semibold ${txt}`}>AI Verification</p>
            <p className={`text-xs ${sub} mt-0.5 max-w-[200px] truncate`}>{quest.text}</p>
          </div>
          <div className="w-14" />
        </div>

        {/* Viewfinder */}
        <div className="relative bg-black" style={{ aspectRatio: '4/3' }}>
          <video ref={videoRef} autoPlay playsInline muted
            className={`w-full h-full object-cover ${phase === 'live' ? 'opacity-100' : 'opacity-0'}`} />

          {phase === 'starting' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
              <div className="w-8 h-8 border-2 border-white/40 border-t-white rounded-full animate-spin" />
              <p className="text-white/60 text-xs">Starting camera…</p>
            </div>
          )}

          {phase === 'error' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-8 text-center">
              <p className="text-white font-semibold text-sm">
                {camError === 'permission' ? 'Camera Access Blocked' : 'No Camera Found'}
              </p>
              <p className="text-white/60 text-xs leading-relaxed">
                {camError === 'permission'
                  ? 'Open this app in a new tab and allow camera access when prompted.'
                  : 'Upload a photo instead.'}
              </p>
              {camError === 'permission' && (
                <a href={window.location.href} target="_blank" rel="noreferrer"
                  className="mt-1 bg-white/20 text-white text-xs font-medium px-4 py-2 rounded-full">
                  Open in New Tab ↗
                </a>
              )}
            </div>
          )}

          {/* AI overlay */}
          {phase === 'live' && modelReady && (
            <div className="absolute inset-x-0 bottom-0 px-4 pb-3 pt-8"
              style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.7) 0%, transparent 100%)' }}>

              {confirmed ? (
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center flex-shrink-0">
                    <CheckIcon />
                  </div>
                  <p className="text-white text-sm font-semibold">Verified!</p>
                </div>
              ) : (
                <>
                  <div className="flex items-center justify-between mb-1.5">
                    <p className="text-white/80 text-xs">
                      {scanning ? 'Scanning…' : `${liveScore}% confidence`}
                    </p>
                    <div className="flex gap-1">
                      {Array.from({ length: REQUIRED_PASSES }).map((_, i) => (
                        <div key={i} className={`w-1.5 h-1.5 rounded-full transition-colors duration-300 ${i < passStreak ? 'bg-green-400' : 'bg-white/30'}`} />
                      ))}
                    </div>
                  </div>
                  <div className="w-full h-1 bg-white/20 rounded-full overflow-hidden">
                    <div className={`h-full ${meterColor} transition-all duration-700 ease-out rounded-full`}
                      style={{ width: `${liveScore}%` }} />
                  </div>
                  {lastLabel && <p className="text-white/40 text-[10px] mt-1 truncate">{lastLabel}</p>}
                </>
              )}
            </div>
          )}

          {phase === 'live' && !modelReady && modelProgress !== null && (
            <div className="absolute inset-x-0 bottom-0 px-4 pb-3 pt-8"
              style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.7) 0%, transparent 100%)' }}>
              <p className="text-white/60 text-xs mb-1">Loading AI model… {modelProgress}%</p>
              <div className="w-full h-1 bg-white/20 rounded-full overflow-hidden">
                <div className="h-full bg-[#007AFF] transition-all duration-300 rounded-full"
                  style={{ width: `${modelProgress}%` }} />
              </div>
            </div>
          )}

          <canvas ref={canvasRef} className="hidden" />
        </div>

        {/* Controls */}
        <div className="px-4 pt-4 pb-2 space-y-3">
          {phase === 'live' && (
            <>
              <button
                onClick={captureAndConfirm}
                disabled={!confirmed}
                className={`w-full py-4 rounded-[14px] text-sm font-semibold transition-all duration-300 ${
                  confirmed
                    ? 'bg-[#007AFF] text-white shadow-lg active:scale-[0.97]'
                    : `${pill} ${sub} cursor-not-allowed`
                }`}
              >
                {confirmed ? 'Complete Quest' : `Show the camera you're ${labels?.label ?? 'doing it'}…`}
              </button>

              <div className="flex gap-2">
                <button onClick={flipCamera}
                  className={`flex-1 py-3 rounded-[14px] text-sm font-medium ${pill} ${sub} active:opacity-70`}>
                  Flip Camera
                </button>
                <label className={`flex-1 py-3 rounded-[14px] text-sm font-medium ${pill} ${sub} text-center cursor-pointer active:opacity-70`}>
                  Upload Photo
                  <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
                </label>
              </div>
            </>
          )}

          {(phase === 'error' || phase === 'starting') && (
            <label className={`flex items-center justify-center w-full py-4 rounded-[14px] text-sm font-medium ${pill} ${sub} cursor-pointer`}>
              Upload a Photo
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
  // Theme
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem('sq_dark');
    if (saved !== null) return saved === 'true';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('sq_dark', dark);
  }, [dark]);

  // State
  const [level,     setLevel]     = useState(() => parseInt(localStorage.getItem('sq_level'))    || 1);
  const [xp,        setXp]        = useState(() => parseInt(localStorage.getItem('sq_xp'))       || 0);
  const [quests,    setQuests]    = useState(() => {
    const saved     = JSON.parse(localStorage.getItem('sq_quests')) || [];
    const anyDone   = saved.some(q => q.completed);
    const lastReset = parseInt(localStorage.getItem('sq_lastReset')) || 0;
    const expired   = Date.now() - lastReset >= ONE_DAY_MS;
    // use saved if valid, otherwise generate fresh immediately
    if (!expired && saved.length >= 5) return saved;
    if (!expired && anyDone)            return saved;
    const n = Math.floor(Math.random() * 3) + 5;
    return [...QUEST_POOL].sort(() => 0.5 - Math.random()).slice(0, n)
      .map(q => ({ ...q, completed: false }));
  });
  const [lastReset, setLastReset] = useState(() => parseInt(localStorage.getItem('sq_lastReset')) || 0);
  const [timeLeft,  setTimeLeft]  = useState('--:--:--');
  const [proofModal,   setProofModal]   = useState(null);
  const [proofImages,  setProofImages]  = useState(() => JSON.parse(localStorage.getItem('sq_proofs')) || {});
  const [viewingProof, setViewingProof] = useState(null);

  const xpRef    = useRef(xp);
  const levelRef = useRef(level);
  useEffect(() => { xpRef.current    = xp;    }, [xp]);
  useEffect(() => { levelRef.current = level; }, [level]);

  const xpRequired = level * 100;

  // Quest generation
  const generateNewQuests = useCallback((ts) => {
    const n = Math.floor(Math.random() * 3) + 5;
    const selected = [...QUEST_POOL].sort(() => 0.5 - Math.random()).slice(0, n).map(q => ({ ...q, completed: false }));
    setQuests(selected); setLastReset(ts); setProofImages({});
  }, []);

  // Timer
  useEffect(() => {
    const id = setInterval(() => {
      const now = Date.now(), remaining = ONE_DAY_MS - (now - lastReset);
      if (remaining <= 0 || quests.length === 0) { generateNewQuests(now); return; }
      const h = Math.floor((remaining / 3_600_000) % 24);
      const m = Math.floor((remaining /    60_000) % 60);
      const s = Math.floor((remaining /     1_000) % 60);
      setTimeLeft(`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`);
    }, 1000);
    return () => clearInterval(id);
  }, [lastReset, quests.length, generateNewQuests]);

  // Persistence
  useEffect(() => {
    localStorage.setItem('sq_level',     level);
    localStorage.setItem('sq_xp',        xp);
    localStorage.setItem('sq_quests',    JSON.stringify(quests));
    localStorage.setItem('sq_lastReset', lastReset);
  }, [level, xp, quests, lastReset]);
  useEffect(() => { localStorage.setItem('sq_proofs', JSON.stringify(proofImages)); }, [proofImages]);

  // XP
  const applyXpChange = useCallback((amount) => {
    let newXp = xpRef.current + amount, newLevel = levelRef.current;
    while (newXp >= newLevel * 100)    { newXp -= newLevel * 100; newLevel++; }
    while (newXp < 0 && newLevel > 1) { newLevel--; newXp += newLevel * 100; }
    if (newLevel === 1 && newXp < 0)   newXp = 0;
    setXp(newXp); setLevel(newLevel);
  }, []);

  const handleQuestClick = (quest) => {
    if (quest.completed) {
      setQuests(prev => prev.map(q => q.id === quest.id ? { ...q, completed: false } : q));
      applyXpChange(-quest.xp);
      setProofImages(prev => { const n = { ...prev }; delete n[quest.id]; return n; });
    } else {
      setProofModal(quest);
    }
  };

  const handleProofConfirm = (questId, img) => {
    const quest = quests.find(q => q.id === questId);
    setProofImages(prev => ({ ...prev, [questId]: img }));
    setQuests(prev => prev.map(q => q.id === questId ? { ...q, completed: true } : q));
    applyXpChange(quest?.xp ?? 0);
    setProofModal(null);
  };

  const xpPct  = Math.min(100, Math.max(0, (xp / xpRequired) * 100));
  const allDone = quests.length > 0 && quests.every(q => q.completed);

  // Theme tokens
  const bg       = dark ? 'bg-black'      : 'bg-[#F2F2F7]';
  const cardBg   = dark ? 'bg-zinc-900'   : 'bg-white';
  const txt      = dark ? 'text-white'    : 'text-gray-900';
  const sub      = dark ? 'text-zinc-400' : 'text-gray-500';
  const sep      = dark ? 'border-zinc-800' : 'border-gray-100';
  const secLabel = dark ? 'text-zinc-500' : 'text-gray-400';

  return (
    <div className={`${bg} min-h-screen flex flex-col items-center transition-colors duration-200`}>
      <div className="w-full max-w-[430px] flex flex-col min-h-screen">

      {/* Camera modal */}
      {proofModal && (
        <CameraModal quest={proofModal} dark={dark}
          onConfirm={img => handleProofConfirm(proofModal.id, img)}
          onCancel={() => setProofModal(null)} />
      )}

      {/* Proof lightbox */}
      {viewingProof && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)' }}
          onClick={() => setViewingProof(null)}>
          <img src={viewingProof} alt="proof" className="max-w-full max-h-full rounded-2xl shadow-2xl" />
        </div>
      )}

      {/* Header */}
      <div className={`${cardBg} safe-top px-4 pb-3 border-b ${sep} transition-colors duration-200`}>
        <div className="flex items-center justify-between pt-2">
          {/* Logo + Title */}
          <div className="flex items-center gap-2.5">
            <LogoIcon size={34} dark={dark} />
            <h1 className={`text-[22px] font-bold tracking-tight ${txt}`}>Side Quests</h1>
          </div>

          <div className="flex items-center gap-2">
            {/* Timer pill */}
            <div className={`flex items-center gap-1 px-2.5 py-1 rounded-full ${dark ? 'bg-zinc-800' : 'bg-gray-100'}`}>
              <span className="text-[10px]">⏱</span>
              <span className={`text-[11px] font-mono font-medium ${dark ? 'text-zinc-300' : 'text-gray-600'}`}>{timeLeft}</span>
            </div>
            {/* Theme toggle */}
            <button onClick={() => setDark(d => !d)}
              className={`w-8 h-8 rounded-full flex items-center justify-center ${dark ? 'bg-zinc-800 text-zinc-300' : 'bg-gray-100 text-gray-600'} active:opacity-70 transition-colors`}>
              {dark ? <SunIcon /> : <MoonIcon />}
            </button>
          </div>
        </div>

        {/* XP row */}
        <div className="mt-3">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <span className="bg-[#007AFF] text-white text-[10px] font-bold px-2 py-0.5 rounded-full">Lv {level}</span>
              <span className={`text-[11px] ${sub}`}>Novice Adventurer</span>
            </div>
            <span className={`text-[11px] ${dark ? 'text-zinc-500' : 'text-gray-400'}`}>{xp} / {xpRequired} XP</span>
          </div>
          <div className={`w-full h-1 rounded-full overflow-hidden ${dark ? 'bg-zinc-800' : 'bg-gray-200'}`}>
            <div className="h-full bg-[#007AFF] rounded-full transition-all duration-700 ease-out" style={{ width: `${xpPct}%` }} />
          </div>
        </div>
      </div>

      {/* Quest List */}
      <div className="flex-1 scroll-ios px-4 pt-4 pb-6 space-y-5">

        {/* Section */}
        <div>
          <p className={`text-[11px] font-semibold uppercase tracking-widest ${secLabel} mb-1.5 px-1`}>
            Today's Objectives
          </p>

          <div className={`${dark ? 'bg-zinc-900' : 'bg-[#F2F2F7]'} rounded-[14px] overflow-hidden`}>
            {quests.map((quest, i) => (
              <div key={quest.id}>
                {i > 0 && <div className={`border-t ${sep} ml-14`} />}
                <button
                  onClick={() => handleQuestClick(quest)}
                  className={`w-full flex items-center gap-3 px-3.5 py-3 text-left active:opacity-60 transition-opacity`}
                >
                  {/* Check circle */}
                  <div className={`w-[26px] h-[26px] rounded-full flex-shrink-0 flex items-center justify-center border-2 transition-all duration-200 ${
                    quest.completed
                      ? 'bg-[#34C759] border-[#34C759]'
                      : dark ? 'border-zinc-600' : 'border-gray-300'
                  }`}>
                    {quest.completed && (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"/>
                      </svg>
                    )}
                  </div>

                  {/* Text */}
                  <span className={`flex-1 text-[15px] font-normal leading-snug transition-colors ${
                    quest.completed ? (dark ? 'text-zinc-600 line-through' : 'text-gray-400 line-through') : txt
                  }`}>
                    {quest.text}
                  </span>

                  {/* Right side */}
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {/* Proof thumb */}
                    {quest.completed && proofImages[quest.id] && (
                      <button
                        onClick={e => { e.stopPropagation(); setViewingProof(proofImages[quest.id]); }}
                        className="w-7 h-7 rounded-md overflow-hidden flex-shrink-0 ring-1 ring-black/10">
                        <img src={proofImages[quest.id]} alt="proof" className="w-full h-full object-cover" />
                      </button>
                    )}

                    {/* XP */}
                    <span className={`text-xs font-semibold ${
                      quest.completed
                        ? 'text-[#34C759]'
                        : dark ? 'text-zinc-500' : 'text-gray-400'
                    }`}>
                      +{quest.xp}
                    </span>

                    {/* Chevron or camera */}
                    {quest.completed ? (
                      <span className={`text-[10px] ${dark ? 'text-zinc-600' : 'text-gray-300'}`}>XP</span>
                    ) : (
                      <span className={`text-[11px] ${dark ? 'text-zinc-600' : 'text-gray-300'}`}>
                        <svg width="8" height="14" viewBox="0 0 8 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="1 1 7 7 1 13"/>
                        </svg>
                      </span>
                    )}
                  </div>
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* All done */}
        {allDone && (
          <div className={`${cardBg} rounded-[16px] p-5 text-center`}>
            <p className="text-2xl mb-2">🏆</p>
            <p className={`font-semibold text-[15px] ${txt}`}>All Quests Complete</p>
            <p className={`text-sm mt-1 ${sub}`}>Rest up. New quests when the timer hits zero.</p>
          </div>
        )}

        {/* AI note */}
        {quests.length > 0 && (
          <p className={`text-xs text-center ${secLabel} px-4`}>
            All quests verified by on-device AI · no data leaves your phone
          </p>
        )}
      </div>

      {/* Safe area bottom spacer */}
      <div className="safe-bottom" />
      </div>
    </div>
  );
}
