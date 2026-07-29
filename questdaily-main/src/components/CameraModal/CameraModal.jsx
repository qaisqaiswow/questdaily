import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { QUEST_LABELS, getNegativeLabels, REQUIRED_PASSES, PASS_THRESHOLD, CLASSIFIER_SCAN_INTERVAL_MS } from '../../data/quests.js';
import { REP_PROFILES } from '../../data/repProfiles.js';
import { REP_SCAN_INTERVAL_MS } from '../../lib/poseMath.js';
import { usePoseWorker } from '../../hooks/usePoseWorker.js';
import { useClassifierWorker } from '../../hooks/useClassifierWorker.js';
import { useFrameLoop } from '../../hooks/useFrameLoop.js';
import { usePendingRequests } from '../../hooks/usePendingRequests.js';
import { ModelLoadingBar } from './ModelLoadingBar.jsx';
import { PoseRepHud, PoseTrackingBadge } from './PoseRepHud.jsx';
import { ClassifierHud } from './ClassifierHud.jsx';

function formatTimerString(secs) {
  const mins = Math.floor(secs / 60);
  const remainingSecs = secs % 60;
  return `${String(mins).padStart(2, '0')}:${String(remainingSecs).padStart(2, '0')}`;
}

export default function CameraModal({ quest, onConfirm, onCancel, dark }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const cameraSessionRef = useRef(0);
  const confirmedRef = useRef(false);
  const repsSeenRef = useRef(0);
  const justCountedTimerRef = useRef(null);
  const frameRequestIdRef = useRef(0);

  const [phase, setPhase] = useState('starting');
  const [camError, setCamError] = useState(null);
  const [facingMode, setFacingMode] = useState('environment');
  const [cameraVersion, setCameraVersion] = useState(0);
  const [modelRetryKey, setModelRetryKey] = useState(0);
  const [modelError, setModelError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [uploadedProof, setUploadedProof] = useState(null);

  // Pose (rep quests) display state
  const [repsDone, setRepsDone] = useState(0);
  const [repPhase, setRepPhase] = useState('seek-target');
  const [repVisible, setRepVisible] = useState(false);
  const [justCounted, setJustCounted] = useState(false);

  // Classifier (map/food/action quests) display state
  const [liveScore, setLiveScore] = useState(0);
  const [passStreak, setPassStreak] = useState(0);
  const [lastLabel, setLastLabel] = useState('');
  const passStreakRef = useRef(0);

  const [secondsLeft, setSecondsLeft] = useState(quest.duration || 0);
  const [timerRunning, setTimerRunning] = useState(false);

  const labels = QUEST_LABELS[quest.id];
  const questType = labels?.type || 'action';
  const isReps = questType === 'reps';
  const repProfile = isReps ? REP_PROFILES[quest.id] : null;

  const activeNegatives = useMemo(() => (isReps ? [] : getNegativeLabels(questType)), [isReps, questType]);
  const classifierLabels = useMemo(
    () => (isReps ? [] : [...new Set([...(labels?.activity ?? []), ...activeNegatives])]),
    [isReps, labels, activeNegatives]
  );

  let instructionText = `Show the camera you're ${labels?.label ?? 'doing it'}…`;
  let uiSubtext = quest.text;
  if (questType === 'map') {
    instructionText = 'Upload a metric summary screenshot (Strava, Nike, Garmin etc.)';
    uiSubtext = 'Must clearly state distance metrics & active time duration summary logs.';
  } else if (questType === 'food') {
    instructionText = 'Take a clear picture of the food on your plate.';
    uiSubtext = 'Must be a real photo of a prepared meal or plate.';
  } else if (isReps) {
    instructionText = `Position the camera for full-body tracking: ${quest.reps} reps.`;
    uiSubtext = 'Keep your entire working frame visible to log movements.';
  }

  useEffect(() => () => clearTimeout(justCountedTimerRef.current), []);

  // ── Duration timer ──────────────────────────────────────────────────────
  useEffect(() => {
    let intervalId = null;
    if (timerRunning && secondsLeft > 0) {
      intervalId = setInterval(() => {
        setSecondsLeft((prev) => {
          if (prev <= 1) {
            setTimerRunning(false);
            if (!quest.reps && questType === 'action') {
              confirmedRef.current = true;
              setConfirmed(true);
            }
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(intervalId);
  }, [timerRunning, secondsLeft, questType, quest.reps]);

  // ── Camera lifecycle ────────────────────────────────────────────────────
  const stopCamera = useCallback(() => {
    const stream = streamRef.current;
    streamRef.current = null;
    stream?.getTracks().forEach((track) => track.stop());
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
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        const video = videoRef.current;
        if (!video) { stopCamera(); return; }

        video.srcObject = stream;
        await video.play();
        if (session !== cameraSessionRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        stream.getVideoTracks().forEach((track) => {
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

  // ── AI workers ───────────────────────────────────────────────────────────
  const posePending = usePendingRequests();
  const classifierPending = usePendingRequests();

  const onPoseResult = useCallback((msg) => {
    posePending.resolve(msg.requestId, msg);
    if (confirmedRef.current) return;
    setRepVisible(msg.visible);
    setRepPhase(msg.phase);
    if (msg.reps !== repsSeenRef.current) {
      repsSeenRef.current = msg.reps;
      setJustCounted(true);
      clearTimeout(justCountedTimerRef.current);
      justCountedTimerRef.current = setTimeout(() => setJustCounted(false), 400);
    }
    setRepsDone(msg.reps);
    if (msg.done) {
      confirmedRef.current = true;
      setConfirmed(true);
    }
  }, [posePending]);

  const onClassifierMessage = useCallback((msg) => {
    classifierPending.resolve(msg.requestId, msg);
  }, [classifierPending]);

  const { ready: poseReady, sendFrame: sendPoseFrame, reset: resetPoseTracker } = usePoseWorker({
    active: isReps && !confirmed,
    questId: quest.id,
    targetReps: quest.reps,
    onResult: onPoseResult,
    onError: setModelError,
    retryKey: modelRetryKey,
  });

  const { ready: classifierReady, progress: classifierProgress, classify } = useClassifierWorker({
    active: !isReps && !confirmed,
    onResult: onClassifierMessage,
    onError: (message, requestId) => {
      if (requestId) classifierPending.resolve(requestId, { results: [] });
      else setModelError(message);
    },
    retryKey: modelRetryKey,
  });

  const modelReady = isReps ? poseReady : classifierReady;

  // ── Pose frame loop ──────────────────────────────────────────────────────
  useFrameLoop({
    active: isReps && phase === 'live' && poseReady && !confirmed && !uploading,
    videoRef,
    minDelayMs: REP_SCAN_INTERVAL_MS,
    onFrame: async (video) => {
      const bitmap = await createImageBitmap(video);
      const requestId = ++frameRequestIdRef.current;
      const wait = posePending.wait(requestId);
      sendPoseFrame(bitmap, Math.round(performance.now()), requestId);
      await wait;
    },
  });

  // ── Classifier frame loop ────────────────────────────────────────────────
  useFrameLoop({
    active: !isReps && phase === 'live' && classifierReady && !confirmed && !uploading && classifierLabels.length > 0,
    videoRef,
    minDelayMs: CLASSIFIER_SCAN_INTERVAL_MS,
    onFrame: async (video) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      setScanning(true);
      try {
        canvas.width = 224;
        canvas.height = 224;
        const context = canvas.getContext('2d');
        const scale = Math.min(224 / video.videoWidth, 224 / video.videoHeight);
        const width = video.videoWidth * scale;
        const height = video.videoHeight * scale;
        context.fillStyle = '#000';
        context.fillRect(0, 0, 224, 224);
        context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, (224 - width) / 2, (224 - height) / 2, width, height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.55);

        const requestId = ++frameRequestIdRef.current;
        const wait = classifierPending.wait(requestId);
        classify(dataUrl, classifierLabels, requestId);
        const msg = await wait;
        if (confirmedRef.current) return;

        const results = msg.results ?? [];
        const actScore = results.reduce((best, r) => (labels.activity.includes(r.label) ? Math.max(best, r.score) : best), 0);
        const negScore = results.reduce((best, r) => (activeNegatives.includes(r.label) ? Math.max(best, r.score) : best), 0);
        setLastLabel(results[0]?.label ?? '');
        setLiveScore(Math.round(actScore * 100));

        const passed = actScore > negScore && actScore >= PASS_THRESHOLD;
        passStreakRef.current = passed ? passStreakRef.current + 1 : 0;
        setPassStreak(passStreakRef.current);
        if (passStreakRef.current >= REQUIRED_PASSES) {
          confirmedRef.current = true;
          setConfirmed(true);
        }
      } finally {
        setScanning(false);
      }
    },
  });

  const resetRepDisplay = useCallback(() => {
    repsSeenRef.current = 0;
    setRepsDone(0);
    setRepPhase('seek-target');
    setRepVisible(false);
    resetPoseTracker();
  }, [resetPoseTracker]);

  const retryCamera = () => {
    setCamError(null);
    setPhase('starting');
    setCameraVersion((v) => v + 1);
  };

  const retryModel = () => {
    setModelError(null);
    setModelRetryKey((v) => v + 1);
  };

  const flipCamera = () => {
    setLiveScore(0);
    setPassStreak(0);
    passStreakRef.current = 0;
    confirmedRef.current = false;
    resetRepDisplay();
    setConfirmed(false);
    setFacingMode((m) => (m === 'environment' ? 'user' : 'environment'));
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (quest.reps) {
      setNotice('Rep quests need live camera tracking; a single photo cannot verify a full set.');
      e.target.value = '';
      return;
    }
    const reader = new FileReader();
    reader.onload = async (ev) => {
      let accepted = false;
      setUploading(true);
      stopCamera();
      const dataUrl = ev.target.result;
      setUploadedProof(dataUrl);
      if (!labels) { onConfirm(dataUrl); return; }
      setPhase('live');
      setScanning(true);
      try {
        const requestId = ++frameRequestIdRef.current;
        const wait = classifierPending.wait(requestId);
        classify(dataUrl, classifierLabels, requestId);
        const msg = await wait;
        const results = msg.results ?? [];
        const actScore = results.reduce((best, r) => (labels.activity.includes(r.label) ? Math.max(best, r.score) : best), 0);
        const negScore = results.reduce((best, r) => (activeNegatives.includes(r.label) ? Math.max(best, r.score) : best), 0);
        setLiveScore(Math.round(actScore * 100));
        setLastLabel(results[0]?.label ?? '');

        if (actScore > negScore && actScore >= PASS_THRESHOLD) {
          setPassStreak(REQUIRED_PASSES);
          confirmedRef.current = true;
          accepted = true;
          setConfirmed(true);
        } else {
          setPassStreak(0);
          setNotice('Verification failed. Please make sure you upload a clear activity summary log showing distance and elapsed time metrics.');
        }
      } catch (err) {
        console.error('File upload AI error:', err);
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
    if (uploadedProof) { onConfirm(uploadedProof); return; }
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);
    onConfirm(canvas.toDataURL('image/jpeg', 0.82));
  };

  const bg = dark ? 'bg-zinc-900' : 'bg-white';
  const border = dark ? 'border-zinc-700' : 'border-gray-200';
  const txt = dark ? 'text-white' : 'text-gray-900';
  const sub = dark ? 'text-zinc-400' : 'text-gray-500';
  const pill = dark ? 'bg-zinc-800' : 'bg-gray-100';

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center" style={{ background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}>
      <div className={`${bg} w-full max-w-lg rounded-t-[28px] overflow-hidden shadow-2xl transition-transform duration-300 ease-spring`} style={{ paddingBottom: 'max(env(safe-area-inset-bottom), 16px)' }}>

        <div className="flex justify-center pt-3 pb-1">
          <div className={`w-10 h-1 rounded-full ${dark ? 'bg-zinc-600' : 'bg-gray-300'}`} />
        </div>

        <div className={`flex items-center justify-between px-5 py-3 border-b ${border}`}>
          <button onClick={onCancel} className="text-apple-blue text-sm font-medium active:opacity-60 transition-opacity">Cancel</button>
          <div className="text-center">
            <p className={`text-sm font-semibold ${txt}`}>AI Verification</p>
            <p className={`text-xs ${sub} mt-0.5 max-w-[240px] truncate`}>{uiSubtext}</p>
          </div>
          <div className="w-14" />
        </div>

        <div className="relative bg-black" style={{ aspectRatio: '4/3' }}>
          <video ref={videoRef} autoPlay playsInline muted
            className={`w-full h-full object-cover transition-opacity duration-300 ${phase === 'live' && !uploadedProof ? 'opacity-100' : 'opacity-0'}`} />

          {phase === 'live' && !uploadedProof && labels?.bodyParts && (
            <PoseTrackingBadge bodyParts={labels.bodyParts} visible={isReps ? repVisible : true} />
          )}

          {uploadedProof && (
            <img src={uploadedProof} alt="Uploaded verification" className="absolute inset-0 w-full h-full object-cover" />
          )}

          {phase === 'starting' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
              <div className="w-8 h-8 border-2 border-white/40 border-t-apple-blue rounded-full animate-spin" />
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
              <button onClick={retryCamera} className="mt-1 bg-white/20 text-white text-xs font-medium px-4 py-2 rounded-full active:scale-95 transition-transform">
                Try Camera Again
              </button>
            </div>
          )}

          {(modelError || notice) && (
            <div className="absolute inset-x-3 top-3 rounded-xl bg-red-950/90 border border-red-700 px-3 py-2 text-center shadow-lg z-10 animate-rise-in" style={{ transform: 'none' }}>
              <p className="text-xs leading-relaxed text-white">{modelError || notice}</p>
              <div className="flex items-center justify-center gap-3 mt-1">
                {modelError && <button onClick={retryModel} className="text-[10px] text-white/70 underline">Retry</button>}
                <button onClick={() => { setNotice(null); setModelError(null); }} className="text-[10px] text-white/50 underline">Dismiss</button>
              </div>
            </div>
          )}

          {phase === 'live' && modelReady && (
            <div className="absolute inset-x-0 bottom-0 px-4 pb-3 pt-8" style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.8) 0%, transparent 100%)' }}>
              {isReps ? (
                <PoseRepHud
                  repsDone={repsDone}
                  targetReps={quest.reps}
                  phase={repPhase}
                  visible={repVisible}
                  cue={repProfile?.cue}
                  confirmed={confirmed}
                  justCounted={justCounted}
                />
              ) : (
                <ClassifierHud
                  liveScore={liveScore}
                  passStreak={passStreak}
                  requiredPasses={REQUIRED_PASSES}
                  scanning={scanning}
                  lastLabel={lastLabel}
                  confirmed={confirmed}
                  passThreshold={PASS_THRESHOLD}
                />
              )}
            </div>
          )}

          {phase === 'live' && !modelReady && !modelError && (
            <ModelLoadingBar label={isReps ? 'Loading pose model' : 'Loading AI model'} percent={isReps ? null : classifierProgress} />
          )}

          <canvas ref={canvasRef} className="hidden" />
        </div>

        {quest.duration && (
          <div className={`px-4 py-3 mx-4 mt-3 rounded-xl border flex items-center justify-between ${dark ? 'bg-zinc-800/50 border-zinc-700' : 'bg-gray-50 border-gray-200'}`}>
            <div className="flex flex-col">
              <span className={`text-[11px] font-bold uppercase tracking-wider ${dark ? 'text-zinc-400' : 'text-gray-500'}`}>Objective Duration</span>
              <span className={`text-xl font-mono font-bold tabular-nums ${txt}`}>{formatTimerString(secondsLeft)}</span>
            </div>
            <button
              onClick={() => setTimerRunning(!timerRunning)}
              disabled={secondsLeft === 0}
              className={`px-4 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-all duration-200 ease-spring active:scale-95 ${
                secondsLeft === 0
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : timerRunning
                    ? 'bg-red-500 text-white'
                    : 'bg-apple-green text-white shadow-glow-green'
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
                className={`w-full py-4 rounded-[14px] text-sm font-semibold transition-all duration-300 ease-spring ${
                  confirmed
                    ? 'bg-apple-blue text-white shadow-glow-blue active:scale-[0.97]'
                    : `${pill} ${sub} cursor-not-allowed`
                }`}
              >
                {confirmed ? 'Complete Quest' : instructionText}
              </button>

              <div className="flex gap-2">
                <button onClick={flipCamera}
                  className={`${quest.reps ? 'w-full' : 'flex-1'} py-3 rounded-[14px] text-sm font-medium ${pill} ${sub} active:opacity-70 transition-opacity`}>
                  Flip Camera
                </button>
                {!quest.reps && (
                  <label className={`flex-1 py-3 rounded-[14px] text-sm font-medium ${pill} ${sub} text-center cursor-pointer active:opacity-70 transition-opacity ${questType === 'map' ? 'ring-2 ring-apple-blue' : ''}`}>
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
