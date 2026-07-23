import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { pipeline, env } from '@huggingface/transformers';

// ─── AI SETUP ────────────────────────────────────────────────────────────────
env.allowLocalModels = false;
let classifierPromise = null;
function getClassifier(onProgress) {
  if (!classifierPromise) {
    classifierPromise = pipeline(
      'zero-shot-image-classification',
      'Xenova/clip-vit-base-patch32',
      onProgress ? { progress_callback: onProgress } : undefined
    ).catch(error => {
      // Do not cache a rejected model load forever. A later modal can retry
      // after a transient network or browser-cache failure.
      classifierPromise = null;
      throw error;
    });
  }
  return classifierPromise;
}

// Added 'type' and highly tuned strict visual labels for the AI
const QUEST_LABELS = {
  // REPS: Kept for live camera counting
  q1:  { type: 'reps', activity: ['person doing pushups on floor', 'pushup exercise'], label: 'doing pushups' },
  q2:  { type: 'reps', activity: ['person doing squats exercise', 'squat workout legs bent'], label: 'doing squats' },
  q4:  { type: 'reps', activity: ['person doing pullups on bar', 'pullup bar exercise'], label: 'doing pullups' },
  q12: { type: 'reps', activity: ['person doing situps or crunches', 'abdominal exercise on floor'], label: 'doing situps' },
  q17: { type: 'reps', activity: ['person jumping rope', 'skipping rope exercise'], label: 'jumping rope' },
  q23: { type: 'reps', activity: ['person doing lunges exercise', 'lunge workout legs split stance'], label: 'doing lunges' },

  // MAPS: Strict rules to look for GPS app screenshots (Strava, Apple Fitness, etc.)
  q3:  { type: 'map', activity: ['gps tracking map route screenshot', 'fitness tracker map running route'], label: 'running map screenshot' },
  q16: { type: 'map', activity: ['gps tracking map route screenshot', 'cycling route map on phone screen'], label: 'cycling map screenshot' },
  q5:  { type: 'map', activity: ['gps tracking map route screenshot', 'walking route map tracker'], label: 'walking map screenshot' },
  q22: { type: 'map', activity: ['gps tracking map route screenshot', 'step counter fitness app screenshot'], label: 'step tracking map' },

  // FOOD: Strict rules to look for actual food on plates/bowls
  q9:  { type: 'food', activity: ['healthy food meal salad vegetables on a plate', 'nutritious meal in a bowl'], label: 'plate of healthy food' },
  q19: { type: 'food', activity: ['clean healthy meal on plate', 'plate of vegetables and whole foods'], label: 'plate of clean food' },
  q20: { type: 'food', activity: ['cooked food on a plate', 'homemade meal in a bowl or plate'], label: 'cooked meal' },
  q14: { type: 'food', activity: ['glass of green smoothie', 'blended green juice drink'], label: 'green smoothie' },
  q6:  { type: 'food', activity: ['glass of water', 'reusable water bottle filled'], label: 'water bottle' },

  // STANDARD ACTIONS
  q7:  { type: 'action', activity: ['person meditating cross-legged', 'mindfulness exercise'], label: 'meditating' },
  q8:  { type: 'action', activity: ['person stretching muscles', 'yoga stretch pose'], label: 'stretching' },
  q10: { type: 'action', activity: ['person sleeping in bed', 'person resting in bed eyes closed'], label: 'getting good sleep' },
  q11: { type: 'action', activity: ['person doing jumping jacks or burpees'], label: 'doing cardio' },
  q13: { type: 'action', activity: ['handwriting in notebook or journal'], label: 'journaling' },
  q15: { type: 'action', activity: ['person doing plank exercise', 'plank position core exercise'], label: 'holding a plank' },
  q18: { type: 'action', activity: ['shower running water', 'bathroom shower head with water'], label: 'in the shower' },
  q21: { type: 'action', activity: ['person breathing deeply eyes closed'], label: 'deep breathing' },
  q24: { type: 'action', activity: ['person sleeping in bed at night', 'sleeping in dark bedroom'], label: 'sleeping early' },
  q25: { type: 'action', activity: ['person in ice bath tub', 'cold plunge tub with ice'], label: 'in a cold plunge' },
};

// Dynamically generate negative labels so the AI knows exactly what to reject
const getNegativeLabels = (type) => {
  // Added "person standing still straight" so the AI actively rejects just standing around
  const base = ['person sitting doing nothing', 'person standing still straight', 'random everyday object'];
  
  if (type === 'map') return [...base, 'sweaty selfie face', 'picture of running shoes', 'treadmill machine indoors', 'person running outside'];
  if (type === 'food') return [...base, 'empty plate or bowl', 'restaurant paper menu', 'store product barcode', 'person eating face'];
  return [...base, 'phone or computer screen'];
};

// ─── QUEST POOL ───────────────────────────────────────────────────────────────
const QUEST_POOL = [
  { id: 'q1',  text: 'Do 20 pushups',                            xp: 50, reps: 20 },
  { id: 'q2',  text: 'Do 30 squats',                             xp: 45, reps: 30 },
  { id: 'q3',  text: 'Go for a 15-minute run',                   xp: 75 },
  { id: 'q4',  text: 'Do 10 pullups',                            xp: 60, reps: 10 },
  { id: 'q16', text: 'Do 15 minutes of cycling',                 xp: 55 },
  { id: 'q17', text: 'Do 50 jumping rope reps',                  xp: 40, reps: 50 },
  { id: 'q18', text: 'Take a cold shower',                       xp: 50 },
  { id: 'q19', text: 'Eat no sugar today',                       xp: 65 },
  { id: 'q20', text: 'Cook a meal from scratch',                 xp: 45 },
  { id: 'q21', text: 'Do 10 minutes of deep breathing',          xp: 30 },
  { id: 'q22', text: 'Take 10,000 steps',                        xp: 70 },
  { id: 'q23', text: 'Do 3 sets of lunges',                      xp: 40, reps: 30 },
  { id: 'q24', text: 'Go to bed before 11pm',                    xp: 35 },
  { id: 'q25', text: 'Do a 5-minute ice bath or cold plunge',    xp: 80 },
  { id: 'q5',  text: 'Walk outside for 20 minutes',              xp: 35 },
  { id: 'q6',  text: 'Drink 2 liters of water today',            xp: 25 },
  { id: 'q7',  text: 'Meditate for 10 minutes',                  xp: 45 },
  { id: 'q8',  text: 'Stretch for 10 minutes',                   xp: 35 },
  { id: 'q9',  text: 'Eat a healthy meal',                       xp: 40 },
  { id: 'q10', text: 'Get 8 hours of sleep',                     xp: 55 },
  { id: 'q11', text: 'Do 3 minutes of jumping jacks',            xp: 30 },
  { id: 'q12', text: 'Do 20 situps',                             xp: 40, reps: 20 },
  { id: 'q13', text: 'Write in your journal',                    xp: 20 },
  { id: 'q14', text: 'Drink a green smoothie',                   xp: 30 },
  { id: 'q15', text: 'Hold a plank for 60 seconds',              xp: 50 },
];

const ONE_DAY_MS = 24 * 60 * 60 * 1000;
const REQUIRED_PASSES = 2;

// Increased these thresholds so it requires a clearer visual match
const PASS_THRESHOLD  = 0.35; 
const REP_SCAN_INTERVAL_MS = 280;
const REP_PHASE_HOLD_MS = 180;
const REP_MIN_DURATION_MS = 650;
const REP_RETURN_TIMEOUT_MS = 7000;
const REP_FORM_CONFIDENCE = 0.20;
const REP_FORM_MARGIN = 0.035;
const REP_ACTIVITY_MARGIN = 0.01;

// CLIP is an image classifier, not a temporal pose tracker. Counting a rep
// therefore needs two visually distinct positions and a completed trip between
// them. Each profile defines the contracted position first, followed by the
// fully returned position that is required before a rep is credited.
const REP_PROFILES = {
  q1: {
    target: ['person at the bottom of a pushup with elbows bent', 'person lowering chest close to floor in a pushup'],
    reset: ['person at the top of a pushup with arms straight', 'person holding a straight-arm plank after a pushup'],
    cue: 'Lower into the pushup, then return to straight arms.',
  },
  q2: {
    target: ['person at the bottom of a deep squat with knees bent', 'person squatting with thighs near parallel to floor'],
    reset: ['person standing tall after a squat with legs straight', 'person at the top of a squat standing upright'],
    cue: 'Reach squat depth, then stand fully upright.',
  },
  q4: {
    target: ['person at the top of a pullup with chin near bar', 'person pulling body up on a pullup bar'],
    reset: ['person hanging from a pullup bar with arms straight', 'person at the bottom of a pullup with arms extended'],
    cue: 'Pull up to the bar, then lower to straight arms.',
  },
  q12: {
    target: ['person at the top of a situp with torso raised', 'person crunching with shoulders lifted from floor'],
    reset: ['person lying back down after a situp', 'person on back with torso extended on floor'],
    cue: 'Raise your torso, then return to the floor.',
  },
  q17: {
    target: ['person jumping in the air while skipping rope', 'person airborne during a jump rope exercise'],
    reset: ['person landing with both feet while jumping rope', 'person standing on the ground with a jump rope'],
    cue: 'Jump, land, and let the camera see both positions.',
  },
  q23: {
    target: ['person at the bottom of a lunge with knees bent', 'person in a deep split-stance lunge'],
    reset: ['person standing upright after a lunge', 'person at the top of a lunge with legs straight'],
    cue: 'Lower into the lunge, then return to standing.',
  },
};

function getMaxLabelScore(results, candidates) {
  if (!candidates?.length) return 0;
  return results.reduce(
    (best, result) => (candidates.includes(result.label) ? Math.max(best, result.score) : best),
    0
  );
}

function createRepTracker() {
  return {
    phase: 'seek-target',
    reps: 0,
    smoothedTarget: 0,
    smoothedReset: 0,
    lastSampleAt: 0,
    phaseStartedAt: 0,
    targetHeldAt: 0,
    resetHeldAt: 0,
    lastRepAt: 0,
  };
}

function advanceRepTracker(tracker, { targetScore, resetScore, negativeScore, now }) {
  const alpha = tracker.lastSampleAt ? 0.45 : 1;
  tracker.lastSampleAt = now;
  tracker.smoothedTarget = tracker.smoothedTarget * (1 - alpha) + targetScore * alpha;
  tracker.smoothedReset = tracker.smoothedReset * (1 - alpha) + resetScore * alpha;

  const formScore = Math.max(tracker.smoothedTarget, tracker.smoothedReset);
  const activityVisible =
    formScore >= REP_FORM_CONFIDENCE &&
    formScore >= negativeScore + REP_ACTIVITY_MARGIN;
  const atTarget =
    activityVisible &&
    tracker.smoothedTarget >= REP_FORM_CONFIDENCE &&
    tracker.smoothedTarget >= tracker.smoothedReset + REP_FORM_MARGIN;
  const atReset =
    tracker.smoothedReset >= REP_FORM_CONFIDENCE &&
    tracker.smoothedReset >= tracker.smoothedTarget + REP_FORM_MARGIN;

  let counted = false;
  let phaseChanged = false;

  if (tracker.phase === 'seek-target') {
    tracker.resetHeldAt = 0;
    if (atTarget) {
      tracker.targetHeldAt ||= now;
      if (now - tracker.targetHeldAt >= REP_PHASE_HOLD_MS) {
        tracker.phase = 'seek-reset';
        tracker.phaseStartedAt = now;
        tracker.targetHeldAt = 0;
        phaseChanged = true;
      }
    } else {
      tracker.targetHeldAt = 0;
    }
  } else {
    tracker.targetHeldAt = 0;
    if (now - tracker.phaseStartedAt > REP_RETURN_TIMEOUT_MS) {
      tracker.phase = 'seek-target';
      tracker.phaseStartedAt = 0;
      tracker.resetHeldAt = 0;
      phaseChanged = true;
    } else if (atReset) {
      tracker.resetHeldAt ||= now;
      const hasHeldReturn = now - tracker.resetHeldAt >= REP_PHASE_HOLD_MS;
      const isNewRep = now - tracker.lastRepAt >= REP_MIN_DURATION_MS;
      const hasValidDuration = now - tracker.phaseStartedAt >= REP_MIN_DURATION_MS;
      if (hasHeldReturn && isNewRep && hasValidDuration) {
        tracker.reps += 1;
        tracker.lastRepAt = now;
        tracker.phase = 'seek-target';
        tracker.phaseStartedAt = 0;
        tracker.resetHeldAt = 0;
        counted = true;
        phaseChanged = true;
      }
    } else {
      tracker.resetHeldAt = 0;
    }
  }

  return {
    counted,
    phaseChanged,
    phase: tracker.phase,
    confidence: formScore,
    activityVisible,
  };
}

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

// ─── CAMERA / AI MODAL ────────────────────────────────────────────────────────
function CameraModal({ quest, onConfirm, onCancel, dark }) {
  const videoRef     = useRef(null);
  const canvasRef    = useRef(null);
  const streamRef    = useRef(null);
  const scanTimerRef = useRef(null);
  const isScanningRef = useRef(false);
  const scanRunRef = useRef(0);
  const cameraSessionRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const confirmedRef = useRef(false);
  const passStreakRef = useRef(0);
  const repTrackerRef = useRef(createRepTracker());

  const [phase,         setPhase]         = useState('starting');
  const [camError,      setCamError]      = useState(null);
  const [facingMode,    setFacingMode]    = useState('environment');
  const [cameraVersion, setCameraVersion] = useState(0);
  const [modelReady,    setModelReady]    = useState(false);
  const [modelProgress, setModelProgress] = useState(null);
  const [modelError,    setModelError]    = useState(null);
  const [modelVersion,  setModelVersion]  = useState(0);
  const [notice,        setNotice]        = useState(null);
  const [scanning,      setScanning]      = useState(false);
  const [uploading,     setUploading]     = useState(false);
  const [liveScore,     setLiveScore]     = useState(0);
  const [passStreak,    setPassStreak]    = useState(0);
  const [repsDone,      setRepsDone]      = useState(0);
  const [repPhase,      setRepPhase]      = useState('seek-target');
  const [confirmed,     setConfirmed]     = useState(false);
  const [lastLabel,     setLastLabel]     = useState('');
  const [uploadedProof, setUploadedProof] = useState(null);

  const labels = QUEST_LABELS[quest.id];
  const questType = labels?.type || 'action';
  const repProfile = questType === 'reps' ? REP_PROFILES[quest.id] : null;
  // Memoized so identity stays stable across re-renders of the same quest —
  // otherwise every setState in the scan loop (setLiveScore, setScanning, etc.)
  // creates new array refs here, which re-triggers the scan effect below and
  // bypasses its setTimeout throttle, hammering the classifier with rapid-fire
  // frames and making a single noisy read (e.g. while standing still) far more
  // likely to produce a noisy, one-frame classification result.
  const activeNegatives = useMemo(() => getNegativeLabels(questType), [questType]);
  const classifierLabels = useMemo(
    () => [...new Set([
      ...(labels?.activity ?? []),
      ...(repProfile?.target ?? []),
      ...(repProfile?.reset ?? []),
      ...activeNegatives,
    ])],
    [labels, repProfile, activeNegatives]
  );

  // Dynamic UI Instructions based on Quest Type
  let instructionText = `Show the camera you're ${labels?.label ?? 'doing it'}…`;
  let uiSubtext = quest.text;

  if (questType === 'map') {
      instructionText = "Please upload a screenshot of your GPS/map route.";
      uiSubtext = "Requires a screenshot from a tracking app (e.g. Strava) showing time & distance.";
  } else if (questType === 'food') {
      instructionText = "Take a clear picture of the food on your plate.";
      uiSubtext = "Must be a real photo of a prepared meal or plate.";
  } else if (questType === 'reps') {
      instructionText = `Show the camera your full body to count ${quest.reps} reps.`;
      uiSubtext = "Keep your entire body in the frame while moving.";
  }

  const stopCamera = useCallback(() => {
    const stream = streamRef.current;
    streamRef.current = null;
    stream?.getTracks().forEach(track => track.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const startCamera = useCallback(async (mode) => {
    const session = ++cameraSessionRef.current;
    stopCamera();
    setCamError(null);
    setNotice(null);
    setUploadedProof(null);
    setPhase('starting');

    const attempts = [
      { video: { facingMode: { ideal: mode }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
      { video: { facingMode: { ideal: mode } }, audio: false },
      { video: true, audio: false },
    ];

    for (const constraints of attempts) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (session !== cameraSessionRef.current) {
          stream.getTracks().forEach(track => track.stop());
          return;
        }

        streamRef.current = stream;
        const video = videoRef.current;
        if (!video) {
          stopCamera();
          return;
        }

        video.srcObject = stream;
        await video.play();
        if (session !== cameraSessionRef.current) {
          stream.getTracks().forEach(track => track.stop());
          return;
        }

        stream.getVideoTracks().forEach(track => {
          track.addEventListener('ended', () => {
            if (session === cameraSessionRef.current) {
              setCamError('unavailable');
              setPhase('error');
            }
          });
        });

        setPhase('live');
        return;
      } catch (err) {
        if (session !== cameraSessionRef.current) return;
        stopCamera();
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError' || err.name === 'SecurityError') {
          setCamError('permission');
          setPhase('error');
          return;
        }
      }
    }

    if (session === cameraSessionRef.current) {
      setCamError('unavailable');
      setPhase('error');
    }
  }, [stopCamera]);

  useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCamError('unsupported');
      setPhase('error');
      return undefined;
    }

    startCamera(facingMode);
    return () => {
      cameraSessionRef.current += 1;
      stopCamera();
    };
  }, [facingMode, cameraVersion, startCamera, stopCamera]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setModelError(null);
        await getClassifier(evt => {
          if (cancelled) return;
          if (evt.status === 'progress' && evt.total)
            setModelProgress(Math.round((evt.loaded / evt.total) * 100));
        });
        if (!cancelled) { setModelReady(true); setModelProgress(null); }
      } catch (err) { 
        console.error("Model load error:", err);
        if (!cancelled) {
          setModelError('The AI model could not be loaded. Check your connection and try again.');
          setModelProgress(null);
        }
      }
    })();
    return () => { cancelled = true; };
  }, [modelVersion]);

  const resetRepTracking = useCallback(() => {
    repTrackerRef.current = createRepTracker();
    setRepsDone(0);
    setRepPhase('seek-target');
  }, []);

  useEffect(() => {
    if (phase !== 'live' || !modelReady || confirmed || uploading || !labels || !classifierLabels.length) return undefined;

    const runId = ++scanRunRef.current;
    let disposed = false;
    let consecutiveErrors = 0;
    const isCurrentRun = () => !disposed && runId === scanRunRef.current && !confirmedRef.current;
    const scheduleNext = (delay) => {
      if (!isCurrentRun()) return;
      clearTimeout(scanTimerRef.current);
      scanTimerRef.current = window.setTimeout(scanLoop, delay);
    };

    const scanLoop = async () => {
      const video = videoRef.current, canvas = canvasRef.current;
      
      if (!isCurrentRun()) return;
      if (document.visibilityState === 'hidden') {
        scheduleNext(1000);
        return;
      }
      if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) {
        scheduleNext(250);
        return;
      }
      if (video.currentTime === lastVideoTimeRef.current || isScanningRef.current) {
        scheduleNext(80);
        return;
      }
      lastVideoTimeRef.current = video.currentTime;
      isScanningRef.current = true;
      setScanning(true);

      try {
        // Letterbox rather than centre-cropping: portrait video otherwise loses
        // the user's head or feet, making full-range form impossible to detect.
        if (canvas.width !== 224 || canvas.height !== 224) {
          canvas.width = 224;
          canvas.height = 224;
        }
        const context = canvas.getContext('2d');
        if (!context) throw new Error('Canvas 2D context is unavailable');
        const scale = Math.min(224 / video.videoWidth, 224 / video.videoHeight);
        const width = video.videoWidth * scale;
        const height = video.videoHeight * scale;
        context.fillStyle = '#000';
        context.fillRect(0, 0, 224, 224);
        context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, (224 - width) / 2, (224 - height) / 2, width, height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.55);

        const classifier = await getClassifier();
        const results = await classifier(dataUrl, classifierLabels);
        if (!isCurrentRun()) return;
        consecutiveErrors = 0;
        // Use the single best-matching phrase per group rather than summing all
        // phrase scores together. Summing biased the comparison against activity
        // whenever a quest had fewer activity phrasings than negative phrasings
        // (e.g. 2 activity phrases vs 4 negative phrases) — the negative side
        // would out-score genuine reps just by having more candidate strings
        // sharing the same softmax distribution, so reps never registered.
        const actScore = getMaxLabelScore(results, labels.activity);
        const negScore = getMaxLabelScore(results, activeNegatives);

        setLastLabel(results[0]?.label ?? '');

        if (quest.reps && repProfile) {
          const update = advanceRepTracker(repTrackerRef.current, {
            targetScore: getMaxLabelScore(results, repProfile.target),
            resetScore: getMaxLabelScore(results, repProfile.reset),
            negativeScore: negScore,
            now: performance.now(),
          });
          setLiveScore(Math.round(update.confidence * 100));
          if (update.phaseChanged) setRepPhase(update.phase);
          if (update.counted) {
            setRepsDone(repTrackerRef.current.reps);
            if (repTrackerRef.current.reps >= quest.reps) {
              confirmedRef.current = true;
              setConfirmed(true);
            }
          }
        } else {
          setLiveScore(Math.round(actScore * 100));
          const passed = actScore > negScore && actScore >= PASS_THRESHOLD;
          passStreakRef.current = passed ? passStreakRef.current + 1 : 0;
          setPassStreak(passStreakRef.current);
          if (passStreakRef.current >= REQUIRED_PASSES) {
            confirmedRef.current = true;
            setConfirmed(true);
          }
        }
      } catch (err) { 
        console.error("AI Inference Error:", err);
        consecutiveErrors += 1;
        if (consecutiveErrors >= 3 && isCurrentRun()) {
          setModelError('AI analysis is temporarily unavailable. Try closing and reopening the camera.');
        }
      } finally { 
        isScanningRef.current = false;
        if (isCurrentRun()) {
          setScanning(false);
          scheduleNext(consecutiveErrors ? 1000 : (quest.reps ? REP_SCAN_INTERVAL_MS : 1200));
        }
      }
    };

    scanLoop();
    return () => {
      disposed = true;
      scanRunRef.current += 1;
      clearTimeout(scanTimerRef.current);
    };
  }, [phase, modelReady, confirmed, uploading, labels, classifierLabels, quest.reps, repProfile, activeNegatives]);

  const retryCamera = () => {
    scanRunRef.current += 1;
    lastVideoTimeRef.current = -1;
    setCamError(null);
    setPhase('starting');
    setCameraVersion(version => version + 1);
  };

  const retryModel = () => {
    setModelReady(false);
    setModelProgress(null);
    setModelError(null);
    setModelVersion(version => version + 1);
  };

  const flipCamera = () => {
    clearTimeout(scanTimerRef.current);
    scanRunRef.current += 1;
    setPhase('starting'); setLiveScore(0); setPassStreak(0); setRepsDone(0);
    passStreakRef.current = 0;
    confirmedRef.current = false;
    lastVideoTimeRef.current = -1;
    resetRepTracking();
    setConfirmed(false);
    setFacingMode(m => m === 'environment' ? 'user' : 'environment');
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0]; if (!file) return;
    // A still photo cannot prove a completed sequence of repetitions.
    if (quest.reps) {
      setNotice('Rep quests need live camera tracking; a single photo cannot verify a full set.');
      e.target.value = '';
      return;
    }
    const reader = new FileReader();
    reader.onload = async (ev) => {
      let accepted = false;
      scanRunRef.current += 1;
      setUploading(true);
      stopCamera();
      clearTimeout(scanTimerRef.current);
      const dataUrl = ev.target.result;
      setUploadedProof(dataUrl);
      if (!labels) { onConfirm(dataUrl); return; }
      setPhase('live'); setScanning(true);
      try {
        const classifier = await getClassifier();
        const results = await classifier(dataUrl, classifierLabels);
        const actScore = getMaxLabelScore(results, labels.activity);
        const negScore = getMaxLabelScore(results, activeNegatives);
        setLiveScore(Math.round(actScore * 100)); setLastLabel(results[0]?.label ?? '');
        
        // Slightly lower threshold for uploaded screenshots to ensure maps/photos pass easily if valid
        if (actScore > negScore && actScore >= (PASS_THRESHOLD - 0.05)) {
          setPassStreak(REQUIRED_PASSES);
          confirmedRef.current = true;
          accepted = true;
          setConfirmed(true);
        } else { 
          setPassStreak(0); 
        }
      } catch (err) { 
        console.error("File upload AI error:", err);
      } finally { 
        setScanning(false); 
        setUploading(false);
        if (!accepted) {
          setUploadedProof(null);
          retryCamera();
        }
      }
    };
    reader.readAsDataURL(file);
  };

  const captureAndConfirm = () => {
    if (uploadedProof) {
      onConfirm(uploadedProof);
      return;
    }
    const video = videoRef.current, canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth || 640; canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);
    onConfirm(canvas.toDataURL('image/jpeg', 0.82));
  };

  const meterColor = liveScore >= (quest.reps ? REP_FORM_CONFIDENCE : PASS_THRESHOLD) * 100 ? 'bg-green-500' : liveScore >= 15 ? 'bg-yellow-400' : 'bg-red-500';

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
            <p className={`text-xs ${sub} mt-0.5 max-w-[240px] truncate`}>{uiSubtext}</p>
          </div>
          <div className="w-14" />
        </div>

        {/* Viewfinder */}
       <div className="relative bg-black" style={{ aspectRatio: '4/3' }}>
         <video ref={videoRef} autoPlay playsInline muted
            className={`w-full h-full object-cover ${phase === 'live' && !uploadedProof ? 'opacity-100' : 'opacity-0'}`} />
          {uploadedProof && (
            <img src={uploadedProof} alt="Uploaded verification" className="absolute inset-0 w-full h-full object-cover" />
          )}

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
                  : quest.reps
                    ? 'This quest needs a live camera. Check your camera and try again.'
                    : 'Upload a photo instead.'}
              </p>
              <button onClick={retryCamera}
                className="mt-1 bg-white/20 text-white text-xs font-medium px-4 py-2 rounded-full">
                Try Camera Again
              </button>
              {camError === 'permission' && (
                <a href={window.location.href} target="_blank" rel="noreferrer"
                  className="mt-1 bg-white/20 text-white text-xs font-medium px-4 py-2 rounded-full">
                  Open in New Tab ↗
                </a>
              )}
            </div>
          )}

          {(modelError || notice) && (
            <div className="absolute inset-x-3 top-3 rounded-xl bg-red-950/85 px-3 py-2 text-center shadow-lg">
              <p className="text-xs leading-relaxed text-white">{modelError || notice}</p>
              {modelError && (
                <button onClick={retryModel} className="mt-1 text-xs font-semibold text-blue-300">
                  Retry AI model
                </button>
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
                    <p className="text-white/80 text-xs font-medium">
                      {scanning ? 'Scanning…' : `${liveScore}% confidence`}
                    </p>
                    
                    {/* Progress tracking: Reps vs Standard */}
                    <div className="flex items-center gap-1.5">
                      {quest.reps ? (
                        <p className="text-white text-xs font-bold tracking-wide">
                            {repsDone} / {quest.reps} REPS
                        </p>
                      ) : (
                        Array.from({ length: REQUIRED_PASSES }).map((_, i) => (
                            <div key={i} className={`w-1.5 h-1.5 rounded-full transition-colors duration-300 ${i < passStreak ? 'bg-green-400' : 'bg-white/30'}`} />
                        ))
                      )}
                    </div>
                  </div>
                  
                  {/* Progress Bar */}
                  <div className="w-full h-1 bg-white/20 rounded-full overflow-hidden">
                    {quest.reps ? (
                        <div className={`h-full bg-green-400 transition-all duration-300 ease-out rounded-full`}
                          style={{ width: `${Math.min(100, (repsDone / quest.reps) * 100)}%` }} />
                    ) : (
                        <div className={`h-full ${meterColor} transition-all duration-700 ease-out rounded-full`}
                          style={{ width: `${liveScore}%` }} />
                    )}
                  </div>
                  {quest.reps && !confirmed && (
                    <p className="text-white/70 text-[11px] mt-2">
                      {repPhase === 'seek-target'
                        ? repProfile?.cue
                        : 'Good position. Return to the fully extended position to count the rep.'}
                    </p>
                  )}
                  {lastLabel && <p className="text-white/40 text-[10px] mt-1 truncate">{lastLabel}</p>}
                </>
              )}
            </div>
          )}

          {phase === 'live' && !modelReady && !modelError && (
            <div className="absolute inset-x-0 bottom-0 px-4 pb-3 pt-8"
              style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.7) 0%, transparent 100%)' }}>
              <p className="text-white/60 text-xs mb-1">Loading AI model… {modelProgress ?? 0}%</p>
              <div className="w-full h-1 bg-white/20 rounded-full overflow-hidden">
                <div className="h-full bg-[#007AFF] transition-all duration-300 rounded-full"
                  style={{ width: `${modelProgress ?? 0}%` }} />
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
                {confirmed ? 'Complete Quest' : instructionText}
              </button>

              <div className="flex gap-2">
                <button onClick={flipCamera}
                  className={`${quest.reps ? 'w-full' : 'flex-1'} py-3 rounded-[14px] text-sm font-medium ${pill} ${sub} active:opacity-70`}>
                  Flip Camera
                </button>
                {!quest.reps && (
                <label className={`flex-1 py-3 rounded-[14px] text-sm font-medium ${pill} ${sub} text-center cursor-pointer active:opacity-70 ${questType === 'map' ? 'ring-2 ring-[#007AFF]' : ''}`}>
                  Upload Photo
                  <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
                </label>
                )}
              </div>
            </>
          )}

          {(phase === 'error' || phase === 'starting') && !quest.reps && (
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

  const generateNewQuests = useCallback((ts) => {
    const n = Math.floor(Math.random() * 3) + 5;
    const selected = [...QUEST_POOL].sort(() => 0.5 - Math.random()).slice(0, n).map(q => ({ ...q, completed: false }));
    setQuests(selected); setLastReset(ts); setProofImages({});
  }, []);

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

  useEffect(() => {
    localStorage.setItem('sq_level',     level);
    localStorage.setItem('sq_xp',        xp);
    localStorage.setItem('sq_quests',    JSON.stringify(quests));
    localStorage.setItem('sq_lastReset', lastReset);
  }, [level, xp, quests, lastReset]);
  useEffect(() => { localStorage.setItem('sq_proofs', JSON.stringify(proofImages)); }, [proofImages]);

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
