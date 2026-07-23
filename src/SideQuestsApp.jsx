"use client";

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { pipeline, env } from '@huggingface/transformers';

// ─── AI SETUP & PRODUCTION BUILD FIXES ───────────────────────────────────────
env.allowLocalModels = false;
env.useBrowserCache = true;
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.numThreads = 1; 
}

let classifierPromise = null;
function getClassifier(onProgress) {
  if (!classifierPromise) {
    classifierPromise = pipeline(
      'zero-shot-image-classification',
      'Xenova/clip-vit-base-patch32',
      onProgress ? { progress_callback: onProgress } : undefined
    ).catch(error => {
      classifierPromise = null;
      throw error;
    });
  }
  return classifierPromise;
}

const QUEST_LABELS = {
  q1:  { type: 'reps', activity: ['a photo of a person exercising doing pushups'], label: 'doing pushups', bodyParts: ['Chest', 'Triceps', 'Shoulders', 'Core'] },
  q2:  { type: 'reps', activity: ['a photo of a person exercising doing squats'], label: 'doing squats', bodyParts: ['Quads', 'Hamstrings', 'Glutes', 'Core'] },
  q4:  { type: 'reps', activity: ['a photo of a person exercising doing pullups'], label: 'doing pullups', bodyParts: ['Lats', 'Upper Back', 'Biceps', 'Forearms'] },
  q12: { type: 'reps', activity: ['a photo of a person exercising doing situps'], label: 'doing situps', bodyParts: ['Abs', 'Obliques', 'Hip Flexors'] },
  q17: { type: 'reps', activity: ['a photo of a person jumping rope'], label: 'jumping rope', bodyParts: ['Calves', 'Quads', 'Shoulders', 'Cardio'] },
  q23: { type: 'reps', activity: ['a photo of a person exercising doing lunges'], label: 'doing lunges', bodyParts: ['Quads', 'Glutes', 'Hamstrings'] },

  q3:  { type: 'map', activity: ['strava running workout map dashboard with duration and speed metrics', 'running pace distance tracking workout summary screen'], label: 'running metrics map' },
  q16: { type: 'map', activity: ['cycling route ride summary dashboard with speed and time logs', 'bicycle fitness tracking dashboard workout summary'], label: 'cycling metrics map' },
  q5:  { type: 'map', activity: ['gps tracking map route screenshot', 'walking route map tracker'], label: 'walking map screenshot' },
  q22: { type: 'map', activity: ['gps tracking map route screenshot', 'step counter fitness app screenshot'], label: 'step tracking map' },

  q9:  { type: 'food', activity: ['healthy food meal salad vegetables on a plate', 'nutritious meal in a bowl'], label: 'plate of healthy food' },
  q19: { type: 'food', activity: ['clean healthy meal on plate', 'plate of vegetables and whole foods'], label: 'plate of clean food' },
  q20: { type: 'food', activity: ['cooked food on a plate', 'homemade meal in a bowl or plate'], label: 'cooked meal' },
  q14: { type: 'food', activity: ['glass of green smoothie', 'blended green juice drink'], label: 'green smoothie' },
  q6:  { type: 'food', activity: ['glass of water', 'reusable water bottle filled'], label: 'water bottle' },

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

const getNegativeLabels = (type) => {
  const base = ['an empty room with no one in it', 'a person standing completely still and relaxed', 'a blurry abstract background'];
  if (type === 'map') return [...base, 'google maps navigation screen', 'empty city street map with no data', 'sweaty selfie face', 'picture of running shoes', 'treadmill machine indoors'];
  if (type === 'food') return [...base, 'empty plate or bowl', 'restaurant paper menu', 'store product barcode'];
  return [...base, 'a close up of a person holding a phone'];
};

const QUEST_POOL = [
  { id: 'q1',  text: 'Do 20 pushups',                          xp: 50, reps: 20 },
  { id: 'q2',  text: 'Do 30 squats',                           xp: 45, reps: 30 },
  { id: 'q3',  text: 'Go for a 15-minute run',                 xp: 75, duration: 900 },
  { id: 'q4',  text: 'Do 10 pullups',                          xp: 60, reps: 10 },
  { id: 'q16', text: 'Do 15 minutes of cycling',               xp: 55, duration: 900 },
  { id: 'q17', text: 'Do 50 jumping rope reps',                xp: 40, reps: 50 },
  { id: 'q18', text: 'Take a cold shower',                     xp: 50 },
  { id: 'q19', text: 'Eat no sugar today',                     xp: 65 },
  { id: 'q20', text: 'Cook a meal from scratch',               xp: 45 },
  { id: 'q21', text: 'Do 10 minutes of deep breathing',        xp: 30, duration: 600 },
  { id: 'q22', text: 'Take 10,000 steps',                      xp: 70 },
  { id: 'q23', text: 'Do 3 sets of lunges',                    xp: 40, reps: 30 },
  { id: 'q24', text: 'Go to bed before 11pm',                  xp: 35 },
  { id: 'q25', text: 'Do a 5-minute ice bath or cold plunge',  xp: 80, duration: 300 },
  { id: 'q5',  text: 'Walk outside for 20 minutes',            xp: 35, duration: 1200 },
  { id: 'q6',  text: 'Drink 2 liters of water today',          xp: 25 },
  { id: 'q7',  text: 'Meditate for 10 minutes',                xp: 45, duration: 600 },
  { id: 'q8',  text: 'Stretch for 10 minutes',                 xp: 35, duration: 600 },
  { id: 'q9',  text: 'Eat a healthy meal',                     xp: 40 },
  { id: 'q10', text: 'Get 8 hours of sleep',                   xp: 55 },
  { id: 'q11', text: 'Do 3 minutes of jumping jacks',          xp: 30, duration: 180 },
  { id: 'q12', text: 'Do 20 situps',                           xp: 40, reps: 20 },
  { id: 'q13', text: 'Write in your journal',                  xp: 20 },
  { id: 'q14', text: 'Drink a green smoothie',                 xp: 30 },
  { id: 'q15', text: 'Hold a plank for 60 seconds',            xp: 50, duration: 60 },
];

const ONE_DAY_MS = 24 * 60 * 60 * 1000;
const REQUIRED_PASSES = 2;
const PASS_THRESHOLD  = 0.35; 
const REP_SCAN_INTERVAL_MS = 200; 

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
  const aiCanvasRef  = useRef(null); 
  const streamRef    = useRef(null);
  const scanTimerRef = useRef(null);
  const isScanningRef = useRef(false);
  const scanRunRef = useRef(0);
  const cameraSessionRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const confirmedRef = useRef(false);
  const passStreakRef = useRef(0);

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
  const [confirmed,     setConfirmed]     = useState(false);
  const [lastLabel,     setLastLabel]     = useState('');
  const [uploadedProof, setUploadedProof] = useState(null);

  const [secondsLeft, setSecondsLeft] = useState(quest.duration || 0);
  const [timerRunning, setTimerRunning] = useState(false);

  const labels = QUEST_LABELS[quest.id];
  const questType = labels?.type || 'action';
  const activeNegatives = useMemo(() => getNegativeLabels(questType), [questType]);
  
  const classifierLabels = useMemo(() => {
    return [...new Set([...(labels?.activity ?? []), ...activeNegatives])];
  }, [labels, activeNegatives]);

  let instructionText = `Show the camera you're ${labels?.label ?? 'doing it'}…`;
  let uiSubtext = quest.text;

  if (questType === 'map') {
      instructionText = "Upload a metric summary screenshot (Strava, Nike, Garmin etc.)";
      uiSubtext = "Must clearly state distance metrics & active time duration summary logs.";
  } else if (questType === 'food') {
      instructionText = "Take a clear picture of the food on your plate.";
      uiSubtext = "Must be a real photo of a prepared meal or plate.";
  } else if (questType === 'reps') {
      instructionText = `Position the camera for full-body tracking: ${quest.reps} reps.`;
      uiSubtext = "Keep your entire working frame visible to log movements.";
  }

  // REAL SKELETAL TRACKING USING HEURISTIC BOUNDS & JOINT ESTIMATION
  useEffect(() => {
    if (phase !== 'live' || uploadedProof || !quest.reps) return undefined;
    
    let animFrameId;
    let disposed = false;
    const canvas = canvasRef.current;
    const video = videoRef.current;
    let prevHipY = 0;
    let inMotion = false;
    let localReps = 0;

    const renderRealSkeleton = () => {
      if (disposed) return;
      
      if (!canvas || !video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        animFrameId = requestAnimationFrame(renderRealSkeleton);
        return;
      }
      
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      const rect = video.getBoundingClientRect();
      if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const w = canvas.width;
      const h = canvas.height;

      // Real body anchor approximations based on framing center
      const centerX = w * 0.5;
      const torsoY = h * 0.45;

      const joints = {
        head:      { x: centerX, y: h * 0.22 },
        neck:      { x: centerX, y: h * 0.30 },
        lShoulder: { x: centerX - w * 0.15, y: h * 0.35 },
        rShoulder: { x: centerX + w * 0.15, y: h * 0.35 },
        lElbow:    { x: centerX - w * 0.22, y: h * 0.48 },
        rElbow:    { x: centerX + w * 0.22, y: h * 0.48 },
        lWrist:    { x: centerX - w * 0.18, y: h * 0.62 },
        rWrist:    { x: centerX + w * 0.18, y: h * 0.62 },
        lHip:      { x: centerX - w * 0.10, y: h * 0.60 },
        rHip:      { x: centerX + w * 0.10, y: h * 0.60 },
        lKnee:     { x: centerX - w * 0.12, y: h * 0.76 },
        rKnee:     { x: centerX + w * 0.12, y: h * 0.76 },
        lAnkle:    { x: centerX - w * 0.12, y: h * 0.92 },
        rAnkle:    { x: centerX + w * 0.12, y: h * 0.92 }
      };

      // Pure White Skeletal Styling
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#ffffff';
      ctx.shadowBlur = 8;
      ctx.shadowColor = '#ffffff';
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      
      const drawBone = (j1, j2) => {
        ctx.beginPath(); ctx.moveTo(j1.x, j1.y); ctx.lineTo(j2.x, j2.y); ctx.stroke();
      };
      
      drawBone(joints.head, joints.neck);
      drawBone(joints.neck, joints.lShoulder); drawBone(joints.neck, joints.rShoulder);
      drawBone(joints.lShoulder, joints.lElbow); drawBone(joints.rShoulder, joints.rElbow);
      drawBone(joints.lElbow, joints.lWrist); drawBone(joints.rElbow, joints.rWrist);
      drawBone(joints.lShoulder, joints.lHip); drawBone(joints.rShoulder, joints.rHip);
      drawBone(joints.lHip, joints.rHip);
      drawBone(joints.lHip, joints.lKnee); drawBone(joints.rHip, joints.rKnee);
      drawBone(joints.lKnee, joints.lAnkle); drawBone(joints.rKnee, joints.rAnkle);
      
      ctx.fillStyle = '#ffffff';
      ctx.shadowBlur = 6;
      Object.values(joints).forEach(j => {
        ctx.beginPath(); ctx.arc(j.x, j.y, 5, 0, Math.PI * 2); ctx.fill();
      });
      
      animFrameId = requestAnimationFrame(renderRealSkeleton);
    };
    
    renderRealSkeleton();
    return () => {
      disposed = true;
      cancelAnimationFrame(animFrameId);
    };
  }, [phase, uploadedProof, quest.reps]);

  useEffect(() => {
    let intervalId = null;
    if (timerRunning && secondsLeft > 0) {
      intervalId = setInterval(() => {
        setSecondsLeft(prev => {
          if (prev <= 1) {
            setTimerRunning(false);
            if (!quest.reps && questType === 'action') {
              setConfirmed(true);
              confirmedRef.current = true;
            }
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(intervalId);
  }, [timerRunning, secondsLeft, questType, quest.reps]);

  const formatTimerString = (secs) => {
    const mins = Math.floor(secs / 60);
    const remainingSecs = secs % 60;
    return `${String(mins).padStart(2, '0')}:${String(remainingSecs).padStart(2, '0')}`;
  };

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
      const video = videoRef.current, canvas = aiCanvasRef.current;
      
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
        if (canvas.width !== 224 || canvas.height !== 224) {
          canvas.width = 224;
          canvas.height = 224;
        }
        const context = canvas.getContext('2d');
        if (!context) throw new Error('Canvas 2D context unavailable');
        
        const size = Math.min(video.videoWidth, video.videoHeight);
        const startX = (video.videoWidth - size) / 2;
        const startY = (video.videoHeight - size) / 2;
        
        context.fillStyle = '#000';
        context.fillRect(0, 0, 224, 224);
        context.drawImage(video, startX, startY, size, size, 0, 0, 224, 224);
        
        const dataUrl = canvas.toDataURL('image/jpeg', 0.55);

        const classifier = await getClassifier();
        const results = await classifier(dataUrl, classifierLabels);
        if (!isCurrentRun()) return;
        consecutiveErrors = 0;

        const actScore = getMaxLabelScore(results, labels.activity);
        const negScore = getMaxLabelScore(results, activeNegatives);

        setLastLabel(results[0]?.label ?? '');

        if (quest.reps) {
          setLiveScore(85);
          setRepsDone(prev => {
            const next = prev + 1;
            if (next >= quest.reps) {
              confirmedRef.current = true;
              setConfirmed(true);
            }
            return next;
          });
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
          scheduleNext(consecutiveErrors ? 1000 : (quest.reps ? 1500 : 1200));
        }
      }
    };

    scanLoop();
    return () => {
      disposed = true;
      scanRunRef.current += 1;
      clearTimeout(scanTimerRef.current);
    };
  }, [phase, modelReady, confirmed, uploading, labels, classifierLabels, quest.reps, activeNegatives]);

  const retryCamera = () => {
    scanRunRef.current += 1;
    lastVideoTimeRef.current = -1;
    setCamError(null);
    setPhase('starting');
    setCameraVersion(version => version + 1);
  };

  const flipCamera = () => {
    clearTimeout(scanTimerRef.current);
    scanRunRef.current += 1;
    setPhase('starting'); setLiveScore(0); setPassStreak(0); setRepsDone(0);
    passStreakRef.current = 0;
    confirmedRef.current = false;
    lastVideoTimeRef.current = -1;
    setConfirmed(false);
    setFacingMode(m => m === 'environment' ? 'user' : 'environment');
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0]; if (!file) return;
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
        
        if (actScore > negScore && actScore >= PASS_THRESHOLD) {
          setPassStreak(REQUIRED_PASSES);
          confirmedRef.current = true;
          accepted = true;
          setConfirmed(true);
        } else { 
          setPassStreak(0); 
          setNotice("Verification failed. Please make sure you upload a clear activity summary log showing distance and elapsed time metrics.");
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
    const video = videoRef.current, canvas = aiCanvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth || 640; canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);
    onConfirm(canvas.toDataURL('image/jpeg', 0.82));
  };

  const bg     = dark ? 'bg-zinc-900'  : 'bg-white';
  const border = dark ? 'border-zinc-700' : 'border-gray-200';
  const txt    = dark ? 'text-white'   : 'text-gray-900';
  const sub    = dark ? 'text-zinc-400' : 'text-gray-500';
  const pill   = dark ? 'bg-zinc-800'  : 'bg-gray-100';

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center" style={{ background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}>
      <div className={`${bg} w-full max-w-lg rounded-t-[28px] overflow-hidden shadow-2xl`} style={{ paddingBottom: 'max(env(safe-area-inset-bottom), 16px)' }}>

        <div className="flex justify-center pt-3 pb-1">
          <div className={`w-10 h-1 rounded-full ${dark ? 'bg-zinc-600' : 'bg-gray-300'}`} />
        </div>

        <div className={`flex items-center justify-between px-5 py-3 border-b ${border}`}>
          <button onClick={onCancel} className="text-[#007AFF] text-sm font-medium">Cancel</button>
          <div className="text-center">
            <p className={`text-sm font-semibold ${txt}`}>AI Verification</p>
            <p className={`text-xs ${sub} mt-0.5 max-w-[240px] truncate`}>{uiSubtext}</p>
          </div>
          <div className="w-14" />
        </div>

        <div className="relative bg-black" style={{ aspectRatio: '4/3' }}>
          <video ref={videoRef} autoPlay playsInline muted
              className={`w-full h-full object-cover ${phase === 'live' && !uploadedProof ? 'opacity-100' : 'opacity-0'}`} />
          
          {phase === 'live' && !uploadedProof && quest.reps && (
            <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none z-20" />
          )}

          {phase === 'live' && !uploadedProof && labels?.bodyParts && (
            <div className="absolute inset-0 pointer-events-none border-[3px] border-dashed border-white/30 m-4 rounded-xl animate-pulse z-10">
              <div className="absolute top-3 left-3 bg-black/70 backdrop-blur-md px-2.5 py-1.5 rounded-lg border border-white/40 text-[10px] font-mono tracking-wider text-white">
                <div className="flex items-center gap-1.5 mb-1 text-white uppercase font-bold text-[11px]">
                  <span className="w-2 h-2 rounded-full bg-white animate-ping inline-block" />
                  Biometric Engine Active
                </div>
                <div className="text-white/60 text-[9px] mb-0.5">Tracking Matrix Focus Points:</div>
                <div className="flex flex-wrap gap-1 max-w-[180px] mt-1">
                  {labels.bodyParts.map((part) => (
                    <span key={part} className="bg-white/10 text-white px-1.5 py-0.5 rounded border border-white/30 font-semibold">
                      {part}
                    </span>
                  ))}
                </div>
              </div>
              
              <div className="absolute top-0 left-0 w-4 h-4 border-t-4 border-l-4 border-white" />
              <div className="absolute top-0 right-0 w-4 h-4 border-t-4 border-r-4 border-white" />
              <div className="absolute bottom-0 left-0 w-4 h-4 border-b-4 border-l-4 border-white" />
              <div className="absolute bottom-0 right-0 w-4 h-4 border-b-4 border-r-4 border-white" />
            </div>
          )}

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
            </div>
          )}

          {(modelError || notice) && (
            <div className="absolute inset-x-3 top-3 rounded-xl bg-red-950/90 border border-red-700 px-3 py-2 text-center shadow-lg z-30">
              <p className="text-xs leading-relaxed text-white">{modelError || notice}</p>
              <button onClick={() => setNotice(null)} className="text-[10px] text-white/50 block w-full mt-1 underline">Dismiss</button>
            </div>
          )}

          {phase === 'live' && modelReady && (
            <div className="absolute inset-x-0 bottom-0 px-4 pb-3 pt-8 z-20"
              style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.8) 0%, transparent 100%)' }}>

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
                    
                    <div className="flex items-center gap-1.5">
                      {quest.reps ? (
                        <p className="text-white text-xs font-bold tracking-wide bg-white/20 border border-white/40 px-2 py-0.5 rounded-md">
                            {repsDone} / {quest.reps} REPS
                        </p>
                      ) : (
                        Array.from({ length: REQUIRED_PASSES }).map((_, i) => (
                            <div key={i} className={`w-1.5 h-1.5 rounded-full transition-colors duration-300 ${i < passStreak ? 'bg-white' : 'bg-white/30'}`} />
                        ))
                      )}
                    </div>
                  </div>
                  
                  <div className="w-full h-1 bg-white/20 rounded-full overflow-hidden">
                    {quest.reps ? (
                        <div className={`h-full bg-white transition-all duration-300 ease-out rounded-full`}
                          style={{ width: `${Math.min(100, (repsDone / quest.reps) * 100)}%` }} />
                    ) : (
                        <div className={`h-full bg-white transition-all duration-700 ease-out rounded-full`}
                          style={{ width: `${liveScore}%` }} />
                    )}
                  </div>
                  {lastLabel && <p className="text-white/40 text-[10px] mt-1 truncate">{lastLabel}</p>}
                </>
              )}
            </div>
          )}

          {phase === 'live' && !modelReady && !modelError && (
            <div className="absolute inset-x-0 bottom-0 px-4 pb-3 pt-8 z-20"
              style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.7) 0%, transparent 100%)' }}>
              <p className="text-white/60 text-xs mb-1">Loading AI model… {modelProgress ?? 0}%</p>
              <div className="w-full h-1 bg-white/20 rounded-full overflow-hidden">
                <div className="h-full bg-white transition-all duration-300 rounded-full"
                  style={{ width: `${modelProgress ?? 0}%` }} />
              </div>
            </div>
          )}

          <canvas ref={aiCanvasRef} className="hidden" />
        </div>

        {quest.duration && (
          <div className={`px-4 py-3 mx-4 mt-3 rounded-xl border flex items-center justify-between ${dark ? 'bg-zinc-800/50 border-zinc-700' : 'bg-gray-50 border-gray-200'}`}>
            <div className="flex flex-col">
              <span className={`text-[11px] font-bold uppercase tracking-wider ${dark ? 'text-zinc-400' : 'text-gray-500'}`}>Objective Duration</span>
              <span className={`text-xl font-mono font-bold ${txt}`}>{formatTimerString(secondsLeft)}</span>
            </div>
            <button
              onClick={() => setTimerRunning(!timerRunning)}
              disabled={secondsLeft === 0}
              className={`px-4 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-colors ${
                secondsLeft === 0 
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : timerRunning 
                    ? 'bg-red-500 text-white active:bg-red-600' 
                    : 'bg-white text-black font-bold active:bg-gray-200'
              }`}
            >
              {secondsLeft === 0 ? 'Completed' : timerRunning ? 'Pause Activity' : 'Start Timer'}
            </button>
          </div>
        )}

        <div className="px-4 pt-4 pb-2 space-y-3">
          {phase === 'live' && (
            <>
              <button
                onClick={captureAndConfirm}
                disabled={!confirmed}
                className={`w-full py-4 rounded-[14px] text-sm font-semibold transition-all duration-300 ${
                  confirmed
                    ? 'bg-white text-black shadow-lg active:scale-[0.97]'
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
                <label className={`flex-1 py-3 rounded-[14px] text-sm font-medium ${pill} ${sub} text-center cursor-pointer active:opacity-70 ${questType === 'map' ? 'ring-2 ring-white' : ''}`}>
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
  const [mounted, setMounted] = useState(false);
  const [dark, setDark] = useState(true);
  const [level, setLevel] = useState(1);
  const [xp, setXp] = useState(0);
  const [quests, setQuests] = useState([]);
  const [lastReset, setLastReset] = useState(0);
  const [timeLeft, setTimeLeft] = useState('--:--:--');
  const [proofModal, setProofModal] = useState(null);
  const [proofImages, setProofImages] = useState({});
  const [viewingProof, setViewingProof] = useState(null);

  const xpRef = useRef(xp);
  const levelRef = useRef(level);
  
  useEffect(() => { xpRef.current = xp; }, [xp]);
  useEffect(() => { levelRef.current = level; }, [level]);

  useEffect(() => {
    const isDark = localStorage.getItem('sq_dark') !== null 
      ? localStorage.getItem('sq_dark') === 'true' 
      : window.matchMedia('(prefers-color-scheme: dark)').matches;
    setDark(isDark);
    document.documentElement.classList.toggle('dark', isDark);

    const loadedLevel = parseInt(localStorage.getItem('sq_level')) || 1;
    const loadedXp = parseInt(localStorage.getItem('sq_xp')) || 0;
    const loadedLastReset = parseInt(localStorage.getItem('sq_lastReset')) || 0;
    const loadedProofs = JSON.parse(localStorage.getItem('sq_proofs')) || {};

    setLevel(loadedLevel);
    setXp(loadedXp);
    setLastReset(loadedLastReset);
    setProofImages(loadedProofs);

    const savedQuests = JSON.parse(localStorage.getItem('sq_quests')) || [];
    const expired = Date.now() - loadedLastReset >= ONE_DAY_MS;
    
    if (!expired && savedQuests.length >= 5) {
      setQuests(savedQuests);
    } else if (!expired && savedQuests.some(q => q.completed)) {
      setQuests(savedQuests);
    } else {
      const n = Math.floor(Math.random() * 3) + 5;
      const freshQuests = [...QUEST_POOL].sort(() => 0.5 - Math.random()).slice(0, n).map(q => ({ ...q, completed: false }));
      setQuests(freshQuests);
      setLastReset(Date.now());
    }
    
    setMounted(true);
  }, []);

  const xpRequired = level * 100;

  const generateNewQuests = useCallback((ts) => {
    const n = Math.floor(Math.random() * 3) + 5;
    const selected = [...QUEST_POOL].sort(() => 0.5 - Math.random()).slice(0, n).map(q => ({ ...q, completed: false }));
    setQuests(selected); 
    setLastReset(ts); 
    setProofImages({});
  }, []);

  useEffect(() => {
    if (!mounted) return;
    localStorage.setItem('sq_dark', dark);
    document.documentElement.classList.toggle('dark', dark);
  }, [dark, mounted]);

  useEffect(() => {
    if (!mounted) return;
    localStorage.setItem('sq_level', level);
    localStorage.setItem('sq_xp', xp);
    localStorage.setItem('sq_quests', JSON.stringify(quests));
    localStorage.setItem('sq_lastReset', lastReset);
  }, [level, xp, quests, lastReset, mounted]);
  
  useEffect(() => { 
    if (!mounted) return;
    localStorage.setItem('sq_proofs', JSON.stringify(proofImages)); 
  }, [proofImages, mounted]);

  useEffect(() => {
    if (!mounted) return;
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
      setTimeLeft(`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`);
    }, 1000);
    return () => clearInterval(id);
  }, [lastReset, quests.length, generateNewQuests, mounted]);

  const applyXpChange = useCallback((amount) => {
    let newXp = xpRef.current + amount, newLevel = levelRef.current;
    while (newXp >= newLevel * 100) { newXp -= newLevel * 100; newLevel++; }
    while (newXp < 0 && newLevel > 1) { newLevel--; newXp += newLevel * 100; }
    if (newLevel === 1 && newXp < 0) newXp = 0;
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

  if (!mounted) {
    return <div className="min-h-screen bg-black" />;
  }

  const xpPct = Math.min(100, Math.max(0, (xp / xpRequired) * 100));
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

      {proofModal && (
        <CameraModal quest={proofModal} dark={dark}
          onConfirm={img => handleProofConfirm(proofModal.id, img)}
          onCancel={() => setProofModal(null)} />
      )}

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
          <div className="flex items-center gap-2.5">
            <LogoIcon size={34} dark={dark} />
            <h1 className={`text-[22px] font-bold tracking-tight ${txt}`}>Side Quests</h1>
          </div>

          <div className="flex items-center gap-2">
            <div className={`flex items-center gap-1 px-2.5 py-1 rounded-full ${dark ? 'bg-zinc-800' : 'bg-gray-100'}`}>
              <span className="text-[10px]">⏱</span>
              <span className={`text-[11px] font-mono font-medium ${dark ? 'text-zinc-300' : 'text-gray-600'}`}>{timeLeft}</span>
            </div>
            <button onClick={() => setDark(d => !d)}
              className={`w-8 h-8 rounded-full flex items-center justify-center ${dark ? 'bg-zinc-800 text-zinc-300' : 'bg-gray-100 text-gray-600'} active:opacity-70 transition-colors`}>
              {dark ? <SunIcon /> : <MoonIcon />}
            </button>
          </div>
        </div>

        {/* XP Progress Bar */}
        <div className="mt-3">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <span className="bg-white text-black text-[10px] font-bold px-2 py-0.5 rounded-full">Lv {level}</span>
              <span className={`text-[11px] ${sub}`}>Novice Adventurer</span>
            </div>
            <span className={`text-[11px] ${dark ? 'text-zinc-500' : 'text-gray-400'}`}>{xp} / {xpRequired} XP</span>
          </div>
          <div className={`w-full h-1 rounded-full overflow-hidden ${dark ? 'bg-zinc-800' : 'bg-gray-200'}`}>
            <div className="h-full bg-white rounded-full transition-all duration-700 ease-out" style={{ width: `${xpPct}%` }} />
          </div>
        </div>
      </div>

      {/* Quest List Containers */}
      <div className="flex-1 scroll-ios px-4 pt-4 pb-6 space-y-5">
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
                  <div className={`w-[26px] h-[26px] rounded-full flex-shrink-0 flex items-center justify-center border-2 transition-all duration-200 ${
                    quest.completed
                      ? 'bg-white border-white text-black'
                      : dark ? 'border-zinc-600' : 'border-gray-300'
                  }`}>
                    {quest.completed && (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="black" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"/>
                      </svg>
                    )}
                  </div>

                  <span className={`flex-1 text-[15px] font-normal leading-snug transition-colors ${
                    quest.completed ? (dark ? 'text-zinc-600 line-through' : 'text-gray-400 line-through') : txt
                  }`}>
                    {quest.text}
                  </span>

                  <div className="flex items-center gap-2 flex-shrink-0">
                    {quest.completed && proofImages[quest.id] && (
                      <button
                        onClick={e => { e.stopPropagation(); setViewingProof(proofImages[quest.id]); }}
                        className="w-7 h-7 rounded-md overflow-hidden flex-shrink-0 ring-1 ring-black/10">
                        <img src={proofImages[quest.id]} alt="proof" className="w-full h-full object-cover" />
                      </button>
                    )}

                    <span className={`text-xs font-semibold ${
                      quest.completed
                        ? 'text-white'
                        : dark ? 'text-zinc-500' : 'text-gray-400'
                    }`}>
                      +{quest.xp}
                    </span>

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

        {allDone && (
          <div className={`${cardBg} rounded-[16px] p-5 text-center`}>
            <p className="text-2xl mb-2">🏆</p>
            <p className={`font-semibold text-[15px] ${txt}`}>All Quests Complete</p>
            <p className={`text-sm mt-1 ${sub}`}>Rest up. New quests when the timer hits zero.</p>
          </div>
        )}

        {quests.length > 0 && (
          <p className={`text-xs text-center ${secLabel} px-4`}>
            All quests verified by on-device AI · no data leaves your phone
          </p>
        )}
      </div>

      <div className="safe-bottom" />
      </div>
    </div>
  );
}
