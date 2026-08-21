"use client";

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { pipeline, env } from '@huggingface/transformers';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import * as exifr from 'exifr';
import { initializeApp, getApps } from 'firebase/app';
import { getAuth, signInAnonymously, onAuthStateChanged } from 'firebase/auth';
import { getFirestore, doc, setDoc, onSnapshot, collection, query, orderBy, limit } from 'firebase/firestore';
import {
  Sun, Moon, CheckSquare, History as HistoryIcon,
  Dumbbell, PersonStanding, Bike, Footprints, CircleDot, Timer, ChevronsUp,
  Droplet, Leaf, Utensils, CookingPot, Flower2, Move, Zap, Flame,
  Pencil, Snowflake, Wind, Waves, Target, GlassWater, CupSoda,
} from 'lucide-react';

// firebase / leaderboard stuff
// swap in your own project config (Firebase console > project settings > your apps)
// this is fine to ship client-side btw, it's not secret — the real access control
// is in the firestore security rules
const firebaseConfig = {
  apiKey: "AIzaSyDWaoCQjCc8mpF9jE8FIZvSKDKkxgTUELA",
  authDomain: "quest-daily-8debb.firebaseapp.com",
  projectId: "quest-daily-8debb",
  storageBucket: "quest-daily-8debb.firebasestorage.app",
  messagingSenderId: "628879821292",
  appId: "1:628879821292:web:997e8155a24b806eb2d8c9",
  measurementId: "G-C6S86XMFRZ",
};

const firebaseApp = getApps().length ? getApps()[0] : initializeApp(firebaseConfig);
const auth = getAuth(firebaseApp);
const db = getFirestore(firebaseApp);
const LEADERBOARD_COLLECTION = 'leaderboard';
const LEADERBOARD_SIZE = 100;

// pushes the player's stats up to their leaderboard doc. ranked by lifetime
// xp earned (not current xp) so leveling up never makes your rank go down
async function syncLeaderboardEntry(uid, { username, level, xp, totalXpEarned }) {
  if (!uid || !username) return { ok: false, error: 'Not signed in yet — try again in a moment.' };
  try {
    await setDoc(doc(db, LEADERBOARD_COLLECTION, uid), {
      username: username.slice(0, 20),
      level,
      xp,
      totalXpEarned,
      updatedAt: Date.now(),
    }, { merge: true });
    return { ok: true };
  } catch (err) {
    console.error('Leaderboard sync failed:', err);
    return { ok: false, error: `${err.code || 'error'}: ${err.message}` };
  }
}

// anonymous auth so the device has a stable uid, no account/password needed
function useAnonymousAuth() {
  const [uid, setUid] = useState(null);
  useEffect(() => {
    const unsub = onAuthStateChanged(auth, user => {
      if (user) setUid(user.uid);
      else signInAnonymously(auth).catch(err => console.error('Anonymous auth failed:', err));
    });
    return unsub;
  }, []);
  return uid;
}

// live top-100, onSnapshot keeps it updated in realtime so no polling needed
function useLeaderboard() {
  const [entries, setEntries] = useState([]);
  const [status, setStatus] = useState('loading'); // 'loading' | 'ready' | 'error'
  const [errorDetail, setErrorDetail] = useState(null);
  useEffect(() => {
    const q = query(collection(db, LEADERBOARD_COLLECTION), orderBy('totalXpEarned', 'desc'), limit(LEADERBOARD_SIZE));
    const timeout = setTimeout(() => {
      setStatus(prev => prev === 'loading' ? 'error' : prev);
      setErrorDetail(prev => prev || 'Timed out waiting for a response. Firestore Database and Anonymous Auth may not be enabled yet in the Firebase console.');
    }, 8000);
    const unsub = onSnapshot(
        q,
        snap => { clearTimeout(timeout); setEntries(snap.docs.map(d => ({ id: d.id, ...d.data() }))); setStatus('ready'); setErrorDetail(null); },
        err => { clearTimeout(timeout); console.error('Leaderboard listener error:', err); setStatus('error'); setErrorDetail(`${err.code || 'error'}: ${err.message}`); }
    );
    return () => { clearTimeout(timeout); unsub(); };
  }, []);
  return { entries, status, errorDetail };
}

// haptics
function haptic(pattern = 10) {
  try { if (typeof navigator !== 'undefined' && navigator.vibrate) navigator.vibrate(pattern); } catch { /* unsupported */ }
}

function getLevelTitle(level) {
  if (level < 5)  return 'Novice';
  if (level < 10) return 'Apprentice';
  if (level < 20) return 'Skilled';
  if (level < 35) return 'Veteran';
  return 'Legendary';
}

// ai model setup
env.allowLocalModels = false;

// fast model, used by the live scan loop while the camera is open (~every 1.2s).
// quantized so it's fast enough to run on-device — we make up for the accuracy
// hit with multi-frame smoothing + requiring a streak of passes (see below)
let classifierPromise = null;
function getClassifier(onProgress) {
  if (!classifierPromise) {
    classifierPromise = pipeline(
        'zero-shot-image-classification',
        'Xenova/siglip-base-patch16-224',
        { dtype: 'q8', ...(onProgress ? { progress_callback: onProgress } : {}) }
    ).catch(error => {
      classifierPromise = null;
      throw error;
    });
  }
  return classifierPromise;
}

// bigger/slower model, only used for one-shot checks (uploaded photos, the final
// confirm frame) — basically a second opinion before we actually mark it verified
let staticClassifierPromise = null;
function getStaticClassifier(onProgress) {
  if (!staticClassifierPromise) {
    staticClassifierPromise = pipeline(
        'zero-shot-image-classification',
        'Xenova/siglip-large-patch16-256',
        onProgress ? { progress_callback: onProgress } : undefined
    ).catch(error => {
      staticClassifierPromise = null;
      throw error;
    });
  }
  return staticClassifierPromise;
}

// photo source labels (for the "is this actually your photo" check)
const SOURCE_CAMERA_LABEL = 'a real unedited photo taken with a phone camera';
const SOURCE_DOWNLOADED_LABELS = [
  'a professional stock photography image',
  'a screenshot or image saved from a website',
  'an image downloaded from a search engine or social media',
];
const SOURCE_LABELS = [SOURCE_CAMERA_LABEL, ...SOURCE_DOWNLOADED_LABELS];

// pose model — this is what actually counts reps
let landmarkerPromise = null;
function getPoseLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      );
      return PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
      });
    })().catch(error => {
      landmarkerPromise = null;
      throw error;
    });
  }
  return landmarkerPromise;
}

const LM = {
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_ELBOW: 13, R_ELBOW: 14,
  L_WRIST: 15, R_WRIST: 16,
  L_HIP: 23, R_HIP: 24,
  L_KNEE: 25, R_KNEE: 26,
  L_ANKLE: 27, R_ANKLE: 28,
};

const POSE_CONNECTIONS = [
  [11, 12], [11, 23], [12, 24], [23, 24], // Torso
  [11, 13], [13, 15],                     // Left Arm
  [12, 14], [14, 16],                     // Right Arm
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31], // Left Leg
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32], // Right Leg
  [0, 1], [1, 2], [2, 3], [3, 7],         // Left Face
  [0, 4], [4, 5], [5, 6], [6, 8],         // Right Face
  [9, 10]                                 // Mouth
];

function angleBetween(a, b, c) {
  if (!a || !b || !c) return null;
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.hypot(v1.x, v1.y);
  const mag2 = Math.hypot(v2.x, v2.y);
  if (mag1 === 0 || mag2 === 0) return null;
  const cos = Math.min(1, Math.max(-1, dot / (mag1 * mag2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

function visiblePt(pt, minVis = 0.5) {
  return pt && (pt.visibility === undefined || pt.visibility >= minVis);
}

function pickSide(lms, leftKeys, rightKeys) {
  const left = leftKeys.map(k => lms[k]);
  const right = rightKeys.map(k => lms[k]);
  const leftOk = left.every(p => visiblePt(p));
  const rightOk = right.every(p => visiblePt(p));
  if (leftOk && rightOk) {
    return left.map((p, i) => ({
      x: (p.x + right[i].x) / 2,
      y: (p.y + right[i].y) / 2,
      visibility: Math.min(p.visibility ?? 1, right[i].visibility ?? 1),
    }));
  }
  if (leftOk) return left;
  if (rightOk) return right;
  return null;
}

const REP_METRICS = {
  q1: (lms) => { const p = pickSide(lms, [LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST], [LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST]); return p ? angleBetween(p[0], p[1], p[2]) : null; },
  q2: (lms) => { const p = pickSide(lms, [LM.L_HIP, LM.L_KNEE, LM.L_ANKLE], [LM.R_HIP, LM.R_KNEE, LM.R_ANKLE]); return p ? angleBetween(p[0], p[1], p[2]) : null; },
  q4: (lms) => { const p = pickSide(lms, [LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST], [LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST]); return p ? angleBetween(p[0], p[1], p[2]) : null; },
  q12:(lms) => { const p = pickSide(lms, [LM.L_SHOULDER, LM.L_HIP, LM.L_KNEE], [LM.R_SHOULDER, LM.R_HIP, LM.R_KNEE]); return p ? angleBetween(p[0], p[1], p[2]) : null; },
  q17:(lms) => { const hips = [lms[LM.L_HIP], lms[LM.R_HIP]].filter(p => visiblePt(p)); if (!hips.length) return null; return hips.reduce((s, p) => s + p.y, 0) / hips.length; },
  q23:(lms) => { const p = pickSide(lms, [LM.L_HIP, LM.L_KNEE, LM.L_ANKLE], [LM.R_HIP, LM.R_KNEE, LM.R_ANKLE]); return p ? angleBetween(p[0], p[1], p[2]) : null; },
};

const REP_CONFIG = {
  q1:  { mode: 'angle',  downThreshold: 100,  upThreshold: 155,   cueDown: 'Lower into the pushup', cueUp: 'Push back up to full extension' },
  q2:  { mode: 'angle',  downThreshold: 110,  upThreshold: 160,   cueDown: 'Squat down',             cueUp: 'Stand back up' },
  q4:  { mode: 'angle',  downThreshold: 80,   upThreshold: 150,   cueDown: 'Pull up to the bar',     cueUp: 'Lower to a full hang' },
  q12: { mode: 'angle',  downThreshold: 110,  upThreshold: 150,   cueDown: 'Crunch up',              cueUp: 'Lower back down' },
  q17: { mode: 'height', downThreshold: 0.02, upThreshold: 0.005, cueDown: 'Jump!',                  cueUp: 'Land' },
  q23: { mode: 'angle',  downThreshold: 110,  upThreshold: 160,   cueDown: 'Lower into the lunge',   cueUp: 'Return to standing' },
};

const MIN_REP_MS = 500;

function createPoseRepState(initialReps = 0) {
  return { phase: 'up', reps: initialReps, baselineY: null, lastRepAt: 0, smoothed: null };
}

function updatePoseRepState(state, questId, landmarks, now) {
  const config = REP_CONFIG[questId];
  const metricFn = REP_METRICS[questId];
  if (!config || !metricFn || !landmarks) {
    return { reps: state.reps, phase: state.phase, cue: 'Move into frame', counted: false };
  }
  const raw = metricFn(landmarks);
  if (raw === null) {
    return { reps: state.reps, phase: state.phase, cue: 'Move fully into frame', counted: false };
  }
  state.smoothed = state.smoothed === null ? raw : state.smoothed * 0.6 + raw * 0.4;
  const value = state.smoothed;
  let counted = false;

  if (config.mode === 'height') {
    if (state.baselineY === null) state.baselineY = value;
    else state.baselineY = state.baselineY * 0.98 + value * 0.02;
    const jumpHeight = state.baselineY - value;
    if (state.phase === 'up' && jumpHeight >= config.downThreshold) {
      state.phase = 'down';
    } else if (state.phase === 'down' && jumpHeight <= config.upThreshold) {
      if (now - state.lastRepAt >= MIN_REP_MS) { state.reps += 1; state.lastRepAt = now; counted = true; }
      state.phase = 'up';
    }
  } else {
    if (state.phase === 'up' && value <= config.downThreshold) {
      state.phase = 'down';
    } else if (state.phase === 'down' && value >= config.upThreshold) {
      if (now - state.lastRepAt >= MIN_REP_MS) { state.reps += 1; state.lastRepAt = now; counted = true; }
      state.phase = 'up';
    }
  }

  const cue = state.phase === 'up' ? config.cueDown : config.cueUp;
  return { reps: state.reps, phase: state.phase, cue, counted };
}

// quest labels
const QUEST_LABELS = {
  q1:  { type: 'reps', activity: ['person doing pushups on floor', 'pushup exercise'], label: 'doing pushups', bodyParts: ['Chest', 'Triceps', 'Shoulders', 'Core'] },
  q2:  { type: 'reps', activity: ['person doing squats exercise', 'squat workout legs bent'], label: 'doing squats', bodyParts: ['Quads', 'Hamstrings', 'Glutes', 'Core'] },
  q4:  { type: 'reps', activity: ['person doing pullups on bar', 'pullup bar exercise'], label: 'doing pullups', bodyParts: ['Lats', 'Upper Back', 'Biceps', 'Forearms'] },
  q12: { type: 'reps', activity: ['person doing situps or crunches', 'abdominal exercise on floor'], label: 'doing situps', bodyParts: ['Abs', 'Obliques', 'Hip Flexors'] },
  q17: { type: 'reps', activity: ['person jumping rope', 'skipping rope exercise'], label: 'jumping rope', bodyParts: ['Calves', 'Quads', 'Shoulders', 'Cardio'] },
  q23: { type: 'reps', activity: ['person doing lunges exercise', 'lunge workout legs split stance'], label: 'doing lunges', bodyParts: ['Quads', 'Glutes', 'Hamstrings'] },
  q3:  { type: 'map', activity: ['gps tracking map route screenshot', 'fitness tracker map running route'], label: 'running map screenshot' },
  q16: { type: 'map', activity: ['gps tracking map route screenshot', 'cycling route map on phone screen'], label: 'cycling map screenshot' },
  q5:  { type: 'map', activity: ['gps tracking map route screenshot', 'walking route map tracker'], label: 'walking map screenshot' },
  q22: { type: 'map', activity: ['gps tracking map route screenshot', 'step counter fitness app screenshot'], label: 'step tracking map' },
  q9:  { type: 'food', activity: ['healthy food meal salad vegetables on a plate', 'nutritious meal in a bowl'], label: 'plate of healthy food' },
  q19: { type: 'food', activity: ['clean healthy meal on plate', 'plate of vegetables and whole foods'], label: 'plate of clean food' },
  q20: { type: 'food', activity: ['cooked food on a plate', 'homemade meal in a bowl or plate'], label: 'cooked meal' },
  q14: { type: 'food', activity: ['glass of green smoothie', 'blended green juice drink'], label: 'green smoothie' },
  q6:  { type: 'food', activity: ['glass of water', 'reusable water bottle filled'], label: 'water bottle' },
  q7:  { type: 'action', activity: ['person meditating cross-legged', 'mindfulness exercise'], label: 'meditating' },
  q8:  { type: 'action', activity: ['person stretching muscles', 'yoga stretch pose'], label: 'stretching', bodyParts: ['Full Body', 'Flexibility'] },
  q10: { type: 'action', activity: ['person sleeping in bed', 'person resting in bed eyes closed'], label: 'getting good sleep' },
  q11: { type: 'action', activity: ['person doing jumping jacks or burpees'], label: 'doing cardio', bodyParts: ['Cardio', 'Full Body'] },
  q13: { type: 'action', activity: ['handwriting in notebook or journal'], label: 'journaling' },
  q15: { type: 'action', activity: ['person doing plank exercise', 'plank position core exercise'], label: 'holding a plank', bodyParts: ['Core', 'Shoulders'] },
  q18: { type: 'action', activity: ['shower running water', 'bathroom shower head with water'], label: 'in the shower' },
  q21: { type: 'action', activity: ['person breathing deeply eyes closed'], label: 'deep breathing' },
  q24: { type: 'action', activity: ['person sleeping in bed at night', 'sleeping in dark bedroom'], label: 'sleeping early' },
  q25: { type: 'action', activity: ['person in ice bath tub', 'cold plunge tub with ice'], label: 'in a cold plunge' },
  q26: { type: 'action', activity: ['person doing a wall sit exercise against a wall'], label: 'holding a wall sit', bodyParts: ['Quads', 'Core'] },
  q27: { type: 'action', activity: ['person doing a glute bridge exercise on floor'], label: 'holding a glute bridge', bodyParts: ['Glutes', 'Core'] },
  q28: { type: 'action', activity: ['person doing high knees exercise'], label: 'doing high knees', bodyParts: ['Cardio', 'Quads'] },
  q29: { type: 'action', activity: ['person doing mountain climbers exercise'], label: 'doing mountain climbers', bodyParts: ['Core', 'Cardio'] },
  q30: { type: 'action', activity: ['person doing a superman back exercise lying face down'], label: 'holding a superman pose', bodyParts: ['Lower Back', 'Glutes'] },
  q31: { type: 'action', activity: ['person doing burpees exercise'], label: 'doing burpees', bodyParts: ['Full Body', 'Cardio'] },
};

const MOVEMENT_ACTION_IDS = new Set(['q8', 'q11', 'q15', 'q26', 'q27', 'q28', 'q29', 'q30', 'q31']);

// labels that catch the most common ways people try to fake a check-in:
// holding a printed photo up to the lens, or pointing the camera at a
// second screen (phone/tablet/laptop) that's playing a photo or video of
// someone else doing the activity. These are added to every negative set
// so the classifier is actively working against replay attempts, not just
// scoring "does this look vaguely like the activity".
const ANTI_SPOOF_LABELS = [
  'a photo or video playing on a phone or tablet screen',
  'a laptop or computer monitor screen',
  'a printed photograph being held up to the camera',
  'a picture of a picture',
];
const SPOOF_BLOCK_THRESHOLD = 0.30;
const PASS_MARGIN = 0.12; // how much the activity score must clear the negative/spoof score by

const getNegativeLabels = (type) => {
  const base = ['person sitting doing nothing', 'person standing still straight', 'random everyday object', ...ANTI_SPOOF_LABELS];
  if (type === 'map') return [...base, 'sweaty selfie face', 'picture of running shoes', 'treadmill machine indoors', 'person running outside'];
  if (type === 'food') return [...base, 'empty plate or bowl', 'restaurant paper menu', 'store product barcode', 'person eating face'];
  return [...base, 'phone or computer screen'];
};

const QUEST_POOL = [
  { id: 'q1',  textTemplate: 'Do {n} pushups',                       xp: 50, reps: 20 },
  { id: 'q2',  textTemplate: 'Do {n} squats',                        xp: 45, reps: 30 },
  { id: 'q3',  textTemplate: 'Go for a {n}-minute run',              xp: 75, duration: 180 },
  { id: 'q4',  textTemplate: 'Do {n} pullups',                       xp: 60, reps: 10 },
  { id: 'q16', textTemplate: 'Do {n} minutes of cycling',            xp: 55, duration: 180 },
  { id: 'q17', textTemplate: 'Do {n} jumping rope reps',             xp: 40, reps: 50 },
  { id: 'q21', textTemplate: 'Do {n} minutes of deep breathing',     xp: 30, duration: 180 },
  { id: 'q23', textTemplate: 'Do {n} lunges',                        xp: 40, reps: 30 },
  { id: 'q25', textTemplate: 'Do a {n}-minute ice bath or cold plunge', xp: 80, duration: 180 },
  { id: 'q5',  textTemplate: 'Walk outside for {n} minutes',         xp: 35, duration: 180 },
  { id: 'q7',  textTemplate: 'Meditate for {n} minutes',             xp: 45, duration: 180 },
  { id: 'q8',  textTemplate: 'Stretch for {n} minutes',              xp: 35, duration: 180 },
  { id: 'q11', textTemplate: 'Do {n} minutes of jumping jacks',      xp: 30, duration: 180 },
  { id: 'q12', textTemplate: 'Do {n} situps',                        xp: 40, reps: 20 },
  { id: 'q15', textTemplate: 'Hold a plank for {n} minutes',         xp: 50, duration: 180 },
  { id: 'q26', textTemplate: 'Hold a wall sit for {n} minutes',      xp: 40, duration: 180 },
  { id: 'q27', textTemplate: 'Hold a glute bridge for {n} minutes',  xp: 35, duration: 180 },
  { id: 'q28', textTemplate: 'Do {n} minutes of high knees',         xp: 35, duration: 180 },
  { id: 'q29', textTemplate: 'Do {n} minutes of mountain climbers',  xp: 40, duration: 180 },
  { id: 'q30', textTemplate: 'Hold a superman pose for {n} minutes', xp: 35, duration: 180 },
  { id: 'q31', textTemplate: 'Do {n} minutes of burpees',            xp: 45, duration: 180 },
];

function randomizeQuest(pool) {
  if (pool.reps) {
    const n = Math.floor(Math.random() * 55) + 1; // 1-55 reps
    return { ...pool, reps: n, text: pool.textTemplate.replace('{n}', n), completed: false, progress: 0 };
  }
  if (pool.duration) {
    const minutes = Math.floor(Math.random() * 5) + 1; // 1-5 minutes
    return { ...pool, duration: minutes * 60, text: pool.textTemplate.replace('{n}', minutes), completed: false, progress: 0 };
  }
  return { ...pool, text: pool.textTemplate, completed: false, progress: 0 };
}

const ONE_DAY_MS = 24 * 60 * 60 * 1000;
const REQUIRED_PASSES = 4;
const PASS_THRESHOLD  = 0.42;
const SCORE_SMOOTHING = 0.38; // weight given to each new frame when smoothing confidence — lower = slower to react, harder to fool with a quick flash

function getMaxLabelScore(results, candidates) {
  if (!candidates?.length) return 0;
  return results.reduce((best, result) => (candidates.includes(result.label) ? Math.max(best, result.score) : best), 0);
}

async function checkPhotoAuthenticity(file, dataUrl, classifierLabels) {
  let hasCameraMetadata = false;
  try {
    const tags = await exifr.parse(file, ['Make', 'Model']);
    hasCameraMetadata = Boolean(tags && (tags.Make || tags.Model));
  } catch { /* no/unreadable EXIF — treated as absent, not fatal */ }

  let cameraScore = 0, downloadedScore = 0;
  try {
    const classifier = await getStaticClassifier();
    const results = await classifier(dataUrl, classifierLabels);
    cameraScore = getMaxLabelScore(results, [SOURCE_CAMERA_LABEL]);
    downloadedScore = getMaxLabelScore(results, SOURCE_DOWNLOADED_LABELS);
  } catch { /* if the model call fails, fall back to EXIF alone */ }

  const margin = downloadedScore - cameraScore;
  const modelThinksDownloaded = margin > 0.05;
  const suspicious = !hasCameraMetadata && modelThinksDownloaded;
  const confidence = Math.round(Math.max(cameraScore, downloadedScore) * 100);
  return { hasCameraMetadata, cameraScore, downloadedScore, suspicious, confidence };
}

// colors
// just the standard ios system colors, light + dark. using plain objects instead
// of tailwind's dark: classes since we're already passing `dark` around as state
const C = {
  light: {
    bg: '#F2F2F7', bgElevated: '#FFFFFF', bgSecondary: '#F2F2F7', bgTertiary: '#E5E5EA',
    label: '#000000', labelSecondary: 'rgba(60,60,67,0.60)', labelTertiary: 'rgba(60,60,67,0.30)',
    separator: 'rgba(60,60,67,0.29)', fill: 'rgba(120,120,128,0.12)',
    blue: '#007AFF', green: '#34C759', orange: '#FF9500', red: '#FF3B30',
    indigo: '#5856D6', teal: '#30B0C7', purple: '#AF52DE', yellow: '#FFCC00', pink: '#FF2D55', gray: '#8E8E93',
    glass: 'rgba(255,255,255,0.55)', glassStrong: 'rgba(255,255,255,0.72)',
    glassBorder: 'rgba(255,255,255,0.6)', glassHighlight: 'rgba(255,255,255,0.9)',
    glassShadow: '0 1px 1px rgba(0,0,0,0.03), 0 8px 24px -12px rgba(0,0,0,0.14)',
  },
  dark: {
    bg: '#000000', bgElevated: '#1C1C1E', bgSecondary: '#1C1C1E', bgTertiary: '#2C2C2E',
    label: '#FFFFFF', labelSecondary: 'rgba(235,235,245,0.60)', labelTertiary: 'rgba(235,235,245,0.30)',
    separator: 'rgba(84,84,88,0.65)', fill: 'rgba(120,120,128,0.24)',
    blue: '#0A84FF', green: '#30D158', orange: '#FF9F0A', red: '#FF453A',
    indigo: '#5E5CE6', teal: '#40C8E0', purple: '#BF5AF2', yellow: '#FFD60A', pink: '#FF375F', gray: '#98989D',
    glass: 'rgba(28,28,30,0.55)', glassStrong: 'rgba(28,28,30,0.72)',
    glassBorder: 'rgba(255,255,255,0.1)', glassHighlight: 'rgba(255,255,255,0.14)',
    glassShadow: '0 1px 1px rgba(0,0,0,0.2), 0 8px 24px -12px rgba(0,0,0,0.5)',
  },
};

// one helper so every glass surface looks the same — real "liquid glass":
// heavy blur + saturation boost, a soft glossy wash top-to-bottom, and a
// thin specular highlight traced along the inner top edge like light
// catching the rim of actual glass.
const glassStyle = (c, strong = false) => ({
  background: `linear-gradient(165deg, ${c.glassHighlight} 0%, ${strong ? c.glassStrong : c.glass} 22%, ${strong ? c.glassStrong : c.glass} 100%)`,
  backdropFilter: 'blur(22px) saturate(165%)',
  WebkitBackdropFilter: 'blur(22px) saturate(165%)',
  border: `0.5px solid ${c.glassBorder}`,
  boxShadow: `inset 0 1px 0 ${c.glassHighlight}, inset 0 0 0 0.5px rgba(255,255,255,0.04), ${c.glassShadow}`,
  transform: 'translateZ(0)',
});

// system font stack, no google fonts import — applied to every descendant so
// nothing inside the app can accidentally fall back to a non-system typeface
const SystemType = () => (
    <style>{`
    .sq-root, .sq-root * { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif; }
    .sq-mono, .sq-mono * { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-variant-numeric: tabular-nums; }
    .sq-large-title { letter-spacing: -0.6px; }
    .sq-title { letter-spacing: -0.2px; }
    * { -webkit-tap-highlight-color: transparent; }
    /* Theme crossfade — kept to cheap, compositor-friendly properties only.
       box-shadow/backdrop-filter are deliberately excluded here: animating
       those on many overlapping glass surfaces at once is what causes janky,
       "unpolished" frame drops, so those stay instant while color/background
       still crossfade smoothly. */
    .sq-root, .sq-root :where(div, span, p, h1, h2, h3, button, a, input) {
      transition: background-color 0.3s ease, border-color 0.3s ease, color 0.25s ease;
    }
    svg { transition: stroke 0.2s ease, fill 0.2s ease; }
    .sq-scroll { -webkit-overflow-scrolling: touch; }
    @keyframes sq-fade-up { from { opacity: 0; transform: translate3d(0,7px,0); } to { opacity: 1; transform: translate3d(0,0,0); } }
    @keyframes sq-check-in { 0% { opacity: 0; transform: scale(0.5); } 70% { opacity: 1; transform: scale(1.06); } 100% { opacity: 1; transform: scale(1); } }
    .sq-anim-in { animation: sq-fade-up 0.32s cubic-bezier(0.22,1,0.36,1) both; will-change: transform, opacity; }
    .sq-anim-check { animation: sq-check-in 0.4s cubic-bezier(0.22,1,0.36,1) both; }
    button, a { transition: opacity 0.2s ease-out, transform 0.2s cubic-bezier(0.22,1,0.36,1), background-color 0.25s ease; }
    button:active { transform: scale(0.96); }
    @keyframes sq-icon-pop { 0% { opacity: 0; transform: scale(0.6) rotate(-8deg); } 60% { opacity: 1; transform: scale(1.12) rotate(2deg); } 100% { opacity: 1; transform: scale(1) rotate(0deg); } }
    .sq-icon-tap { transition: transform 0.2s cubic-bezier(0.34,1.56,0.64,1); display: inline-flex; will-change: transform; }
    button:active .sq-icon-tap { transform: scale(1.2) rotate(-6deg); }
    .sq-icon-pop-in { animation: sq-icon-pop 0.4s cubic-bezier(0.22,1,0.36,1) both; }
    .sq-tab-icon { transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1); will-change: transform; }
    .sq-tab-icon-active { transform: scale(1.14) translateY(-1px); }
    @keyframes sq-drift-a { 0%,100% { transform: translate3d(-6%,-4%,0) scale(1); } 50% { transform: translate3d(8%,10%,0) scale(1.15); } }
    @keyframes sq-drift-b { 0%,100% { transform: translate3d(10%,6%,0) scale(1.1); } 50% { transform: translate3d(-8%,-8%,0) scale(0.95); } }
    @keyframes sq-drift-c { 0%,100% { transform: translate3d(-4%,8%,0) scale(0.95); } 50% { transform: translate3d(6%,-10%,0) scale(1.1); } }
    .sq-orb-a { animation: sq-drift-a 26s ease-in-out infinite; will-change: transform; }
    .sq-orb-b { animation: sq-drift-b 32s ease-in-out infinite; will-change: transform; }
    .sq-orb-c { animation: sq-drift-c 22s ease-in-out infinite; will-change: transform; }
    @media (prefers-reduced-motion: reduce) {
      .sq-orb-a, .sq-orb-b, .sq-orb-c { animation: none; }
    }
  `}</style>
);

// background blobs
// 3 slow-drifting blurred color fields, fixed behind everything. this is what
// the glass cards are actually blurring/tinting — without it the "glass" is
// just a translucent gray box, which looks flat
const AmbientBackground = ({ c }) => (
    <div className="fixed inset-0 pointer-events-none z-0" aria-hidden="true" style={{ overflow: 'hidden' }}>
      <div className="sq-orb-a" style={{ position: 'absolute', top: '-10%', left: '-15%', width: '75%', height: '42%', borderRadius: '50%', background: c.blue, opacity: 0.16, filter: 'blur(70px)' }} />
      <div className="sq-orb-b" style={{ position: 'absolute', top: '30%', right: '-20%', width: '70%', height: '46%', borderRadius: '50%', background: c.purple, opacity: 0.13, filter: 'blur(80px)' }} />
      <div className="sq-orb-c" style={{ position: 'absolute', bottom: '-14%', left: '5%', width: '65%', height: '40%', borderRadius: '50%', background: c.teal, opacity: 0.13, filter: 'blur(75px)' }} />
    </div>
);

// icons
// SF-Symbols-style icons via lucide-react — clean, rounded, consistent 1.5–2px strokes
const CheckIcon = ({ size = 12 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
);
const ChevronIcon = ({ color }) => (
    <svg width="8" height="14" viewBox="0 0 8 14" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="1 1 7 7 1 13"/></svg>
);
const BackChevron = ({ color }) => (
    <svg width="11" height="18" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
);
const IOSShareIcon = () => (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/>
    </svg>
);
const IOSAddIcon = () => (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="4" ry="4"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/>
    </svg>
);
const AndroidMenuIcon = () => (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="5" r="2.2"/><circle cx="12" cy="12" r="2.2"/><circle cx="12" cy="19" r="2.2"/></svg>
);
const AndroidAddIcon = () => (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="5" y="2" width="14" height="20" rx="2" ry="2"/><line x1="12" y1="18" x2="12.01" y2="18"/><line x1="9" y1="11" x2="15" y2="11"/><line x1="12" y1="8" x2="12" y2="14"/>
    </svg>
);
const TrophyIcon = ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 4h8v5a4 4 0 0 1-8 0V4Z"/><path d="M8 5H5a3 3 0 0 0 3 4"/><path d="M16 5h3a3 3 0 0 1-3 4"/><path d="M12 13v3"/><path d="M9 20h6"/><path d="M10 16h4l.5 4h-5l.5-4Z"/>
    </svg>
);
const WarningIcon = ({ size = 15 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.3 3.6 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.6a2 2 0 0 0-3.4 0Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
);
const SignalOffIcon = ({ size = 24 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12.5a11 11 0 0 1 4-2.5"/><path d="M9.5 8.8A11 11 0 0 1 19 10.5"/><path d="M12.5 15a4 4 0 0 1 3 1.8"/><circle cx="8" cy="19" r="1"/><line x1="2" y1="2" x2="22" y2="22"/>
    </svg>
);
const FlagIcon = ({ size = 24 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 21V4"/><path d="M5 4h13l-3 4 3 4H5"/>
    </svg>
);

// quest icons — mapped straight onto Lucide (SF-Symbols-style) components.
// Each is a drop-in React component, so existing call sites (which pass
// size/color/style props) work unchanged.
const QuestSvg = {
  cup: GlassWater,
  smoothie: CupSoda,
  dumbbell: Dumbbell,
  run: PersonStanding,
  bike: Bike,
  footsteps: Footprints,
  legs: Footprints,
  core: CircleDot,
  stopwatch: Timer,
  bar: ChevronsUp,
  droplet: Droplet,
  leaf: Leaf,
  bowl: Utensils,
  pot: CookingPot,
  moon: Moon,
  lotus: Flower2,
  stretch: Move,
  bolt: Zap,
  flame: Flame,
  pencil: Pencil,
  snowflake: Snowflake,
  wind: Wind,
  rope: Waves,
  target: Target,
};

// Quest categories map to system colors, not a rotating decorative palette —
// each color means the same category everywhere in the app.
const CAT = { strength: 'orange', cardio: 'red', mind: 'indigo', recover: 'purple', fuel: 'green', cold: 'teal', track: 'blue' };
const QUEST_THEME = {
  q1:  { icon: 'dumbbell', cat: CAT.strength }, q2:  { icon: 'legs',      cat: CAT.strength },
  q3:  { icon: 'run',      cat: CAT.cardio },    q4:  { icon: 'bar',       cat: CAT.strength },
  q5:  { icon: 'footsteps',cat: CAT.track },      q6:  { icon: 'cup',       cat: CAT.fuel },
  q7:  { icon: 'lotus',    cat: CAT.mind },       q8:  { icon: 'stretch',   cat: CAT.recover },
  q9:  { icon: 'bowl',     cat: CAT.fuel },        q10: { icon: 'moon',      cat: CAT.recover },
  q11: { icon: 'bolt',     cat: CAT.cardio },      q12: { icon: 'core',      cat: CAT.strength },
  q13: { icon: 'pencil',   cat: CAT.mind },        q14: { icon: 'smoothie',  cat: CAT.fuel },
  q15: { icon: 'stopwatch',cat: CAT.recover },     q16: { icon: 'bike',      cat: CAT.cardio },
  q17: { icon: 'rope',     cat: CAT.cardio },      q18: { icon: 'droplet',   cat: CAT.recover },
  q19: { icon: 'leaf',     cat: CAT.fuel },        q20: { icon: 'pot',       cat: CAT.fuel },
  q21: { icon: 'wind',     cat: CAT.mind },        q22: { icon: 'footsteps', cat: CAT.track },
  q23: { icon: 'legs',     cat: CAT.strength },    q24: { icon: 'moon',      cat: CAT.recover },
  q25: { icon: 'snowflake',cat: CAT.cold },
  q26: { icon: 'legs',     cat: CAT.strength },    q27: { icon: 'core',      cat: CAT.strength },
  q28: { icon: 'bolt',     cat: CAT.cardio },      q29: { icon: 'core',      cat: CAT.cardio },
  q30: { icon: 'stretch',  cat: CAT.recover },     q31: { icon: 'flame',     cat: CAT.cardio },
};

const QUEST_ABOUT = {
  q1: "Pushups build raw upper-body strength and core stability in one clean movement.", q2: "Squats fire up your biggest muscle groups and reinforce the mechanics behind athletic movement.",
  q3: "A steady run gets your heart rate up, clears your head, and builds endurance.", q4: "Pullups are one of the purest tests of back and grip strength.",
  q5: "A brisk walk outside boosts circulation, mood, and gives your eyes a break.", q6: "Staying hydrated helps your body perform better and keeps your mind sharp.",
  q7: "A short meditation resets your focus and lowers stress before it builds up.", q8: "Stretching keeps your muscles loose and your joints moving.",
  q9: "A balanced, whole-food meal fuels recovery and keeps your energy steady.", q10: "Consistent, sufficient sleep is the single biggest lever for recovery.",
  q11: "A quick cardio burst spikes your heart rate and wakes up your whole body fast.", q12: "Situps target your core and build the stability everything else is built on.",
  q13: "Journaling for a few minutes helps you process the day and plan the next one.", q14: "A green smoothie is an easy way to pack in nutrients when short on time.",
  q15: "Holding a plank builds isometric core strength that carries over to everything else.", q16: "Cycling is easy on the joints while still building serious cardio endurance.",
  q17: "Jump rope sharpens coordination and torches calories in a small amount of time.", q18: "A cold shower is a quick way to train discipline and wake up your nervous system.",
  q19: "Cutting added sugar for a day gives your energy levels a noticeably steadier baseline.", q20: "Cooking from scratch puts you in control of what goes into your body.",
  q21: "A few minutes of deep breathing calms your nervous system and sharpens focus.", q22: "Hitting your step count keeps your body moving steadily throughout the day.",
  q23: "Lunges build single-leg strength and balance that squats alone don't cover.", q24: "An earlier bedtime compounds — better sleep tonight means a better day tomorrow.",
  q25: "Cold exposure trains resilience and gives your recovery a real boost.",
  q26: "Wall sits build isometric leg strength without any equipment needed.",
  q27: "Glute bridges wake up and strengthen the muscles that keep your hips stable.",
  q28: "High knees spike your heart rate fast and sharpen coordination.",
  q29: "Mountain climbers combine cardio and core work in one fluid movement.",
  q30: "The superman hold strengthens your lower back and posterior chain.",
  q31: "Burpees are a full-body blast that builds strength and conditioning together.",
};

const QUEST_QUOTES = ["Small steps every day lead to big changes.", "Discipline is choosing between what you want now and what you want most.", "Progress, not perfection.", "The body achieves what the mind believes.", "One quest at a time.", "Consistency beats intensity.", "You didn't come this far to only come this far."];
const QUEST_QUOTE = { q1: "Strength grows one rep at a time.", q2: "Every squat builds a stronger foundation.", q3: "Miles don't lie — you earned this one.", q4: "Small steps every day lead to big changes.", q5: "One step at a time is still progress.", q6: "Small steps every day lead to big changes.", q7: "A quiet mind carries the loudest strength.", q8: "Flexibility today, resilience tomorrow.", q9: "You fueled the body that carries you.", q10: "Rest is where the real gains happen.", q11: "Energy in motion stays in motion.", q12: "A strong core holds everything else together.", q13: "The pen remembers what the mind forgets.", q14: "Good fuel, good day.", q15: "Stillness can be the hardest work of all.", q16: "Every mile ridden is a mile earned.", q17: "Rhythm builds more than just your legs.", q18: "Discomfort today, discipline for life.", q19: "Progress, not perfection.", q20: "What you cook is what you become.", q21: "Breathe in control, breathe out doubt.", q22: "One step at a time is still progress.", q23: "Balance is built one side at a time.", q24: "Tonight's rest is tomorrow's edge.", q25: "You didn't come this far to only come this far." };

// flat icon tile, one color one glyph, no gradient/gloss nonsense
const QuestIconBadge = ({ questId, size = 88, c }) => {
  const theme = QUEST_THEME[questId] || { icon: 'target', cat: 'blue' };
  const Icon = QuestSvg[theme.icon] || QuestSvg.target;
  const color = c[theme.cat];
  return (
      <div
          style={{
            width: size, height: size, borderRadius: size * 0.36,
            background: color, display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
          }}
      >
        <Icon className="sq-icon-pop-in sq-icon-tap" width={Math.round(size * 0.44)} height={Math.round(size * 0.44)} strokeWidth={1.75} color="#FFFFFF" style={{ color: '#FFFFFF' }} />
      </div>
  );
};

// smaller version for list rows
const QuestRowIcon = ({ questId, c, muted }) => {
  const theme = QUEST_THEME[questId] || { icon: 'target', cat: 'blue' };
  const Icon = QuestSvg[theme.icon] || QuestSvg.target;
  const color = c[theme.cat];
  return (
      <div style={{ width: 30, height: 30, borderRadius: 12, background: muted ? c.fill : color, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
        <Icon className="sq-icon-pop-in sq-icon-tap" width={16} height={16} strokeWidth={1.75} color={muted ? c.labelTertiary : '#FFFFFF'} style={{ color: muted ? c.labelTertiary : '#FFFFFF' }} />
      </div>
  );
};

// like the activity rings but just one flat stroke, no glow
const ProgressRing = ({ pct, size = 46, stroke = 5, c }) => {
  const r = (size - stroke) / 2;
  const circumference = 2 * Math.PI * r;
  const offset = circumference - (pct / 100) * circumference;
  return (
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={c.fill} strokeWidth={stroke} />
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={c.blue} strokeWidth={stroke} strokeLinecap="round"
                  strokeDasharray={circumference} style={{ strokeDashoffset: offset, transition: 'stroke-dashoffset 0.6s cubic-bezier(0.22,1,0.36,1)' }} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="sq-mono" style={{ fontSize: 11, fontWeight: 600, color: c.label }}>{Math.round(pct)}%</span>
        </div>
      </div>
  );
};

// little stat tile
const StatChip = ({ label, value, c, accentColor }) => (
    <div style={{ ...glassStyle(c), borderRadius: 20, padding: '14px 8px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <span className="sq-mono" style={{ fontSize: 20, fontWeight: 600, color: accentColor || c.label, lineHeight: 1 }}>{value}</span>
      <span style={{ fontSize: 11, fontWeight: 500, color: c.labelSecondary }}>{label}</span>
    </div>
);

// confetti
const Confetti = ({ count = 22, big = false, c }) => {
  const colors = [c.blue, c.green, c.orange, c.purple, c.teal];
  const pieces = useMemo(() => {
    const total = big ? Math.round(count * 1.6) : count;
    return Array.from({ length: total }).map((_, i) => ({
      id: i,
      left: Math.random() * 100,
      delay: Math.random() * 0.3,
      duration: 1.3 + Math.random() * 1.1,
      color: colors[i % colors.length],
      w: 5 + Math.random() * 5,
      h: 3 + Math.random() * 4,
      rot: Math.round(Math.random() * 360),
      drift: Math.round((Math.random() - 0.5) * 140),
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [count, big]);

  return (
      <div className="pointer-events-none absolute inset-0 overflow-hidden z-20" aria-hidden="true">
        {pieces.map(p => (
            <span key={p.id} className="sq-confetti-piece" style={{
              left: `${p.left}%`, '--sq-drift': `${p.drift}px`, '--sq-rot': `${p.rot}deg`,
              animationDelay: `${p.delay}s`, animationDuration: `${p.duration}s`,
              background: p.color, width: p.w, height: p.h,
            }} />
        ))}
        <style>{`
        .sq-confetti-piece { position: absolute; top: -6%; border-radius: 2px; opacity: 0.9; animation-name: sq-confetti-fall; animation-timing-function: cubic-bezier(0.35,0,0.65,1); animation-fill-mode: forwards; }
        @keyframes sq-confetti-fall { 0% { transform: translate(0,0) rotate(0deg); opacity: 1; } 85% { opacity: 1; } 100% { transform: translate(var(--sq-drift), 115vh) rotate(var(--sq-rot)); opacity: 0; } }
      `}</style>
      </div>
  );
};

// pick a username
function UsernameModal({ onSubmit, c }) {
  const [name, setName] = useState('');
  const [error, setError] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (trimmed.length < 3) { setError('At least 3 characters.'); return; }
    if (trimmed.length > 20) { setError('20 characters max.'); return; }
    if (!/^[a-zA-Z0-9_ ]+$/.test(trimmed)) { setError('Letters, numbers, and underscores only.'); return; }
    haptic(15);
    onSubmit(trimmed);
  };

  return (
      <div className="fixed inset-0 z-[100] flex items-center justify-center p-5 sq-anim-in" style={{ background: 'rgba(0,0,0,0.4)', backdropFilter: 'blur(6px)', WebkitBackdropFilter: 'blur(6px)' }}>
        <div style={{ ...glassStyle(c, true), borderRadius: 26, width: '100%', maxWidth: 340, padding: '28px 24px 20px', textAlign: 'center' }}>
          <div style={{ width: 52, height: 52, borderRadius: 26, background: c.blue, display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
            <TrophyIcon size={24} />
            <style>{`div > svg { color: #fff; }`}</style>
          </div>
          <h2 className="sq-title" style={{ fontSize: 17, fontWeight: 600, color: c.label }}>Pick a name</h2>
          <p style={{ fontSize: 13, lineHeight: 1.4, color: c.labelSecondary, marginTop: 6 }}>
            This is how you'll appear on the leaderboard. You can only set it once.
          </p>
          <form onSubmit={handleSubmit} style={{ marginTop: 18 }}>
            <input
                type="text" value={name} autoFocus maxLength={20}
                onChange={e => { setName(e.target.value); setError(null); }}
                placeholder="QuestMaster99"
                style={{ width: '100%', borderRadius: 14, border: `1px solid ${c.separator}`, background: c.bgSecondary, color: c.label, padding: '11px 14px', fontSize: 15, fontWeight: 500, textAlign: 'center', outline: 'none' }}
            />
            {error && <p style={{ color: c.red, fontSize: 12, fontWeight: 500, marginTop: 8 }}>{error}</p>}
            <button type="submit" style={{ width: '100%', marginTop: 14, padding: '13px', borderRadius: 999, fontSize: 15, fontWeight: 600, color: '#FFFFFF', background: c.blue }}>
              Continue
            </button>
          </form>
          <p style={{ fontSize: 11, color: c.labelTertiary, marginTop: 12 }}>Only your name, level, and XP are shared.</p>
        </div>
      </div>
  );
}

// leaderboard
function LeaderboardScreen({ dark, onBack, myUid, myUsername, myLevel, myXp, syncError, onRetrySync, c }) {
  const { entries, status, errorDetail } = useLeaderboard();

  // don't surface a name until they've actually earned XP — someone who's
  // just picked a username but hasn't completed anything yet stays
  // anonymous. Checked against level too (not just totalXpEarned) so an
  // older or partially-synced doc missing that field doesn't wrongly hide
  // someone who's clearly made progress.
  const rankedEntries = useMemo(
      () => entries.filter(e => (e.totalXpEarned ?? 0) > 0 || (e.level ?? 1) > 1),
      [entries]
  );

  const myRank = useMemo(() => {
    const idx = rankedEntries.findIndex(e => e.id === myUid);
    return idx === -1 ? null : idx + 1;
  }, [rankedEntries, myUid]);

  const RANK_COLOR = { 1: c.yellow, 2: c.gray, 3: c.orange };

  return (
      <div className="sq-anim-in relative z-10 min-h-screen flex flex-col">
        <div className="flex items-center px-4 pb-3" style={{ paddingTop: 'max(env(safe-area-inset-top), 16px)' }}>
          <button onClick={onBack} className="flex items-center gap-1" style={{ color: c.blue }}>
            <BackChevron color={c.blue} />
            <span style={{ fontSize: 17 }}>Quests</span>
          </button>
        </div>
        <div className="px-4 pb-2">
          <h1 className="sq-large-title" style={{ fontSize: 30, fontWeight: 700, color: c.label }}>Leaderboard</h1>
        </div>

        {syncError && (
            <div className="px-4 mb-3">
              <div style={{ borderRadius: 16, padding: '10px 14px', display: 'flex', gap: 10, background: dark ? 'rgba(255,69,58,0.14)' : 'rgba(255,59,48,0.08)' }}>
                <WarningIcon size={15} />
                <div style={{ flex: 1, color: c.red }}>
                  <p style={{ fontSize: 12, fontWeight: 500, lineHeight: 1.4 }}>Your last score didn't save: {syncError}</p>
                  {onRetrySync && <button onClick={onRetrySync} style={{ fontSize: 11, fontWeight: 600, textDecoration: 'underline', marginTop: 4 }}>Try again</button>}
                </div>
              </div>
            </div>
        )}

        {myRank && (
            <div className="px-4 mb-3">
              <div style={{ borderRadius: 20, padding: '12px 14px', display: 'flex', alignItems: 'center', gap: 12, background: c.blue }}>
                <div style={{ width: 34, height: 34, borderRadius: 999, background: 'rgba(255,255,255,0.22)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontWeight: 700, fontSize: 13 }} className="sq-mono">
                  #{myRank}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{ color: '#fff', fontWeight: 600, fontSize: 14, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{myUsername} (You)</p>
                  <p className="sq-mono" style={{ color: 'rgba(255,255,255,0.75)', fontSize: 11, fontWeight: 500 }}>Level {myLevel} · {myXp} XP this level</p>
                </div>
              </div>
            </div>
        )}

        <div className="flex-1 sq-scroll overflow-y-auto px-4 pb-10">
          {status === 'loading' && (
              <div className="flex flex-col items-center justify-center py-16 gap-3">
                <div style={{ width: 26, height: 26, border: `2.5px solid ${c.fill}`, borderTopColor: c.blue, borderRadius: '50%' }} className="animate-spin" />
                <p style={{ fontSize: 13, color: c.labelSecondary }}>Loading rankings…</p>
              </div>
          )}

          {status === 'error' && (
              <div className="flex flex-col items-center justify-center py-16 gap-2 text-center px-6" style={{ color: c.labelTertiary }}>
                <SignalOffIcon size={24} />
                <p style={{ fontSize: 14, fontWeight: 600, color: c.label }}>Can't reach the leaderboard</p>
                <p style={{ fontSize: 12, color: c.labelSecondary }}>{errorDetail || 'Check your connection, or the backend may not be configured yet.'}</p>
              </div>
          )}

          {status === 'ready' && rankedEntries.length === 0 && (
              <div className="flex flex-col items-center justify-center py-16 gap-2 text-center px-6" style={{ color: c.labelTertiary }}>
                <FlagIcon size={24} />
                <p style={{ fontSize: 14, fontWeight: 600, color: c.label }}>No rankings yet</p>
                <p style={{ fontSize: 12, color: c.labelSecondary }}>
                  {myLevel > 1 || myXp > 0 ? "Your progress hasn't reached the server yet. Complete another quest and check back." : 'Complete a quest to be the first on the board.'}
                </p>
              </div>
          )}

          {status === 'ready' && rankedEntries.length > 0 && (
              <div style={{ ...glassStyle(c), borderRadius: 20, overflow: 'hidden' }}>
                {rankedEntries.map((entry, i) => {
                  const rank = i + 1;
                  const isMe = entry.id === myUid;
                  return (
                      <div key={entry.id}>
                        {i > 0 && <div style={{ marginLeft: 58, borderTop: `1px solid ${c.separator}` }} />}
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 14px', background: isMe ? c.fill : 'transparent' }}>
                          <div className="sq-mono" style={{ width: 30, height: 30, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, fontWeight: 700, background: RANK_COLOR[rank] || c.fill, color: RANK_COLOR[rank] ? '#fff' : c.labelSecondary }}>
                            {rank}
                          </div>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <p style={{ fontSize: 14, fontWeight: 600, color: c.label, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              {entry.username || 'Adventurer'}{isMe ? ' (You)' : ''}
                            </p>
                            <p style={{ fontSize: 11, color: c.labelSecondary }}>Level {entry.level ?? 1}</p>
                          </div>
                          <span className="sq-mono" style={{ fontSize: 12, fontWeight: 600, color: c.labelSecondary, flexShrink: 0 }}>
                      {(entry.totalXpEarned ?? 0).toLocaleString()} XP
                    </span>
                        </div>
                      </div>
                  );
                })}
              </div>
          )}
        </div>
      </div>
  );
}

// history tab
function groupLabelForDate(ts) {
  const d = new Date(ts);
  const startOf = (date) => { const x = new Date(date); x.setHours(0, 0, 0, 0); return x.getTime(); };
  const today = startOf(Date.now());
  const yesterday = today - ONE_DAY_MS;
  const day = startOf(ts);
  if (day === today) return 'Today';
  if (day === yesterday) return 'Yesterday';
  const sameYear = d.getFullYear() === new Date().getFullYear();
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: sameYear ? undefined : 'numeric' });
}

function HistoryScreen({ history, dark, onToggleTheme, c }) {
  const groups = useMemo(() => {
    const sorted = [...history].sort((a, b) => b.ts - a.ts);
    const out = [];
    let current = null;
    for (const entry of sorted) {
      const label = groupLabelForDate(entry.ts);
      if (!current || current.label !== label) { current = { label, items: [] }; out.push(current); }
      current.items.push(entry);
    }
    return out;
  }, [history]);

  const totalXp = useMemo(() => history.reduce((sum, e) => sum + (e.xp || 0), 0), [history]);

  return (
      <div className="relative z-10 flex flex-col flex-1 sq-anim-in">
        <div className="px-4 pb-2" style={{ paddingTop: 'max(env(safe-area-inset-top), 16px)' }}>
          <div className="flex items-center justify-between">
            <h1 className="sq-large-title" style={{ fontSize: 30, fontWeight: 700, color: c.label }}>History</h1>
            <button onClick={onToggleTheme} aria-label="Toggle theme"
                    style={{ ...glassStyle(c), width: 32, height: 32, borderRadius: 999, display: 'flex', alignItems: 'center', justifyContent: 'center', color: c.label }}>
              <span className="sq-icon-tap">{dark ? <Sun key="sun" size={16} strokeWidth={1.75} className="sq-icon-pop-in" /> : <Moon key="moon" size={16} strokeWidth={1.75} className="sq-icon-pop-in" />}</span>
            </button>
          </div>
          <p style={{ fontSize: 13, color: c.labelSecondary, marginTop: 2 }}>Every quest you've ever completed</p>
        </div>

        <div className="flex-1 sq-scroll overflow-y-auto px-4 pt-2 space-y-5" style={{ paddingBottom: 'calc(100px + env(safe-area-inset-bottom))' }}>
          <div className="grid grid-cols-3 gap-2.5 sq-anim-in">
            <StatChip label="Completed" value={history.length} c={c} />
            <StatChip label="Total XP" value={totalXp.toLocaleString()} c={c} />
            <StatChip label="Days logged" value={groups.length} c={c} />
          </div>

          {groups.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 gap-2 text-center px-6" style={{ color: c.labelTertiary }}>
                <FlagIcon size={24} />
                <p style={{ fontSize: 14, fontWeight: 600, color: c.label }}>No history yet</p>
                <p style={{ fontSize: 12, color: c.labelSecondary }}>Quests you complete will show up here, forever.</p>
              </div>
          ) : (
              groups.map(group => (
                  <div key={group.label}>
                    <p style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, color: c.labelSecondary, marginBottom: 6, paddingLeft: 2 }}>
                      {group.label}
                    </p>
                    <div style={{ ...glassStyle(c), borderRadius: 20, overflow: 'hidden' }}>
                      {group.items.map((entry, i) => (
                          <div key={entry.id}>
                            {i > 0 && <div style={{ marginLeft: 58, borderTop: `1px solid ${c.separator}` }} />}
                            <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '14px 14px' }}>
                              <QuestRowIcon questId={entry.questId} c={c} />
                              <div style={{ flex: 1, minWidth: 0 }}>
                                <p style={{ fontSize: 15, fontWeight: 500, color: c.label, lineHeight: 1.3 }}>{entry.text}</p>
                                <p className="sq-mono" style={{ fontSize: 11, color: c.labelTertiary, marginTop: 2 }}>
                                  {new Date(entry.ts).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
                                  {entry.leveledUp ? ' · Level up' : ''}
                                </p>
                              </div>
                              <span className="sq-mono" style={{ fontSize: 12, fontWeight: 700, color: c.green, flexShrink: 0 }}>+{entry.xp}</span>
                            </div>
                          </div>
                      ))}
                    </div>
                  </div>
              ))
          )}
        </div>
      </div>
  );
}

// bottom tab bar — native iOS floating dock: heavy frosted glass, no sliding
// pill highlight, iOS system blue for the active tab and iOS system gray
// for inactive tabs.
function TabBar({ activeTab, onChange, c }) {
  const items = [
    { key: 'quests', label: 'Quests', Icon: CheckSquare },
    { key: 'history', label: 'History', Icon: HistoryIcon },
  ];
  return (
      <div style={{ position: 'fixed', left: '50%', bottom: 'max(env(safe-area-inset-bottom), 16px)', transform: 'translateX(-50%)', zIndex: 50 }}>
        <div
            className="backdrop-blur-2xl"
            style={{
              ...glassStyle(c, true),
              borderRadius: 999,
              padding: '6px 6px',
              display: 'flex',
              width: 216,
            }}
        >
          {items.map(({ key, label, Icon }) => {
            const active = activeTab === key;
            const color = active ? c.blue : c.gray;
            return (
                <button
                    key={key}
                    onClick={() => onChange(key)}
                    style={{
                      flex: 1,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: 3,
                      padding: '8px 0 7px',
                      borderRadius: 999,
                      color,
                      background: active ? c.fill : 'transparent',
                      transition: 'background-color 0.25s ease',
                    }}
                >
                  <span className={`sq-tab-icon ${active ? 'sq-tab-icon-active' : ''}`}>
                    <Icon size={21} strokeWidth={active ? 2.1 : 1.75} color={color} />
                  </span>
                  <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: -0.1, color }}>{label}</span>
                </button>
            );
          })}
        </div>
      </div>
  );
}

// add to home screen prompt
function DeviceInstallPrompt({ onDismiss, c }) {
  const [step, setStep] = useState(0);
  const [os, setOs] = useState(null);

  const nextStep = () => { if (step === 1) setStep(2); else onDismiss(); };

  const rowStyle = { width: '100%', padding: '13px 16px', borderRadius: 18, fontSize: 15, fontWeight: 500, textAlign: 'left', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: c.bgSecondary, color: c.label };

  return (
      <div className="fixed inset-0 z-[100] flex items-end sm:items-center justify-center p-4 sq-anim-in" style={{ background: 'rgba(0,0,0,0.4)', backdropFilter: 'blur(6px)', WebkitBackdropFilter: 'blur(6px)' }}>
        <div style={{ ...glassStyle(c, true), borderRadius: 26, width: '100%', maxWidth: 340, padding: 20, textAlign: 'center' }}>
          {step === 0 && (
              <>
                <h3 className="sq-title" style={{ fontSize: 17, fontWeight: 600, color: c.label, marginBottom: 2 }}>Add to Home Screen</h3>
                <p style={{ fontSize: 13, color: c.labelSecondary, marginBottom: 16 }}>Which device are you using?</p>
                <div className="flex flex-col gap-2">
                  <button onClick={() => { setOs('ios'); setStep(1); }} style={rowStyle}>
                    <span>iPhone or iPad</span><ChevronIcon color={c.labelTertiary} />
                  </button>
                  <button onClick={() => { setOs('android'); setStep(1); }} style={rowStyle}>
                    <span>Android</span><ChevronIcon color={c.labelTertiary} />
                  </button>
                  <button onClick={onDismiss} style={{ width: '100%', padding: '13px', borderRadius: 999, fontSize: 15, fontWeight: 600, color: c.blue, marginTop: 4 }}>
                    Not Now
                  </button>
                </div>
              </>
          )}

          {step > 0 && os === 'ios' && (
              <>
                <h3 className="sq-title" style={{ fontSize: 17, fontWeight: 600, color: c.label, marginBottom: 2 }}>Install on iOS</h3>
                <p style={{ fontSize: 13, color: c.labelSecondary, marginBottom: 16 }}>Step {step} of 2</p>
                <div style={{ background: c.bgSecondary, borderRadius: 20, padding: 20, marginBottom: 16 }}>
                  <div style={{ width: 44, height: 44, borderRadius: 999, background: c.blue, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px' }}>
                    {step === 1 ? <IOSShareIcon /> : <IOSAddIcon />}
                  </div>
                  <p style={{ fontSize: 14, fontWeight: 600, color: c.label, marginBottom: 4 }}>
                    {step === 1 ? 'Tap the Share button' : 'Tap Add to Home Screen'}
                  </p>
                  <p style={{ fontSize: 12, color: c.labelSecondary }}>
                    {step === 1 ? 'Find it in the Safari toolbar.' : 'Scroll down the share sheet to find it.'}
                  </p>
                </div>
                <div className="flex gap-2">
                  {step === 2 && <button onClick={() => setStep(1)} style={{ flex: 1, padding: '13px', borderRadius: 999, fontSize: 15, fontWeight: 600, background: c.bgSecondary, color: c.label }}>Back</button>}
                  <button onClick={nextStep} style={{ flex: 2, padding: '13px', borderRadius: 999, fontSize: 15, fontWeight: 600, background: c.blue, color: '#fff' }}>
                    {step === 1 ? 'Next' : 'Done'}
                  </button>
                </div>
              </>
          )}

          {step > 0 && os === 'android' && (
              <>
                <h3 className="sq-title" style={{ fontSize: 17, fontWeight: 600, color: c.label, marginBottom: 2 }}>Install on Android</h3>
                <p style={{ fontSize: 13, color: c.labelSecondary, marginBottom: 16 }}>Step {step} of 2</p>
                <div style={{ background: c.bgSecondary, borderRadius: 20, padding: 20, marginBottom: 16 }}>
                  <div style={{ width: 44, height: 44, borderRadius: 999, background: c.green, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px' }}>
                    {step === 1 ? <AndroidMenuIcon /> : <AndroidAddIcon />}
                  </div>
                  <p style={{ fontSize: 14, fontWeight: 600, color: c.label, marginBottom: 4 }}>
                    {step === 1 ? 'Tap the menu icon' : 'Tap Add to Home Screen'}
                  </p>
                  <p style={{ fontSize: 12, color: c.labelSecondary }}>
                    {step === 1 ? 'Three dots, usually top right in Chrome.' : 'Select it from the menu, or "Install app".'}
                  </p>
                </div>
                <div className="flex gap-2">
                  {step === 2 && <button onClick={() => setStep(1)} style={{ flex: 1, padding: '13px', borderRadius: 999, fontSize: 15, fontWeight: 600, background: c.bgSecondary, color: c.label }}>Back</button>}
                  <button onClick={nextStep} style={{ flex: 2, padding: '13px', borderRadius: 999, fontSize: 15, fontWeight: 600, background: c.green, color: '#fff' }}>
                    {step === 1 ? 'Next' : 'Done'}
                  </button>
                </div>
              </>
          )}
        </div>
      </div>
  );
}

// quest detail
function QuestDetailScreen({ quest, dark, onToggleTheme, timeLeft, onBack, onMarkComplete, c }) {
  const about = QUEST_ABOUT[quest.id] || 'Stay consistent — every quest you complete adds up to real progress.';
  const theme = QUEST_THEME[quest.id] || { cat: 'blue' };
  const accent = c[theme.cat];

  return (
      <div className="sq-anim-in relative z-10 min-h-screen flex flex-col">
        <div className="flex items-center justify-between px-4 pb-2" style={{ paddingTop: 'max(env(safe-area-inset-top), 16px)' }}>
          <button onClick={onBack} className="flex items-center gap-1" style={{ color: c.blue }}>
            <BackChevron color={c.blue} /><span style={{ fontSize: 17 }}>Quests</span>
          </button>
          <div className="flex items-center gap-3">
            <span className="sq-mono" style={{ fontSize: 13, fontWeight: 500, color: c.labelSecondary }}>{timeLeft}</span>
            <button onClick={onToggleTheme} style={{ ...glassStyle(c), width: 30, height: 30, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: c.label }}>
              <span className="sq-icon-tap">{dark ? <Sun key="sun" size={16} strokeWidth={1.75} className="sq-icon-pop-in" /> : <Moon key="moon" size={16} strokeWidth={1.75} className="sq-icon-pop-in" />}</span>
            </button>
          </div>
        </div>

        <div className="px-4 mt-3 flex flex-col items-center text-center">
          <QuestIconBadge questId={quest.id} size={84} c={c} />
          <h2 style={{ marginTop: 16, fontSize: 22, fontWeight: 700, color: c.label, maxWidth: 260 }}>{quest.text}</h2>
          <span className="sq-mono" style={{ marginTop: 8, fontSize: 13, fontWeight: 700, color: accent }}>+{quest.xp} XP</span>
        </div>

        <div className="px-4 mt-8 flex-1">
          <p style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, color: c.labelSecondary, marginBottom: 6, paddingLeft: 2 }}>About</p>
          <div style={{ ...glassStyle(c), borderRadius: 20, padding: 16 }}>
            <p style={{ fontSize: 14, lineHeight: 1.5, color: c.labelSecondary }}>{about}</p>
          </div>
        </div>

        <div className="px-4 pb-8 mt-6">
          <button onClick={onMarkComplete}
                  style={{ width: '100%', padding: '15px', borderRadius: 999, fontSize: 16, fontWeight: 600, color: '#fff', background: c.blue }}>
            Start
          </button>
        </div>
      </div>
  );
}

// completion screen
function CompletionScreen({ quest, dark, onToggleTheme, timeLeft, onBack, c }) {
  const quote = useMemo(() => QUEST_QUOTE[quest?.id] || QUEST_QUOTES[Math.floor(Math.random() * QUEST_QUOTES.length)], [quest?.id]);
  const leveledUp = Boolean(quest?.leveledUp);
  const accent = leveledUp ? c.orange : c.green;

  useEffect(() => { haptic(leveledUp ? [25, 40, 25, 40, 70] : [15, 30, 15]); }, [leveledUp]);

  return (
      <div className="sq-anim-in relative z-10 min-h-screen flex flex-col">
        <div className="flex items-center justify-between px-4 pb-2" style={{ paddingTop: 'max(env(safe-area-inset-top), 16px)' }}>
          <button onClick={onBack} className="flex items-center gap-1" style={{ color: c.blue }}>
            <BackChevron color={c.blue} /><span style={{ fontSize: 17 }}>Quests</span>
          </button>
          <div className="flex items-center gap-3">
            <span className="sq-mono" style={{ fontSize: 13, fontWeight: 500, color: c.labelSecondary }}>{timeLeft}</span>
            <button onClick={onToggleTheme} style={{ ...glassStyle(c), width: 30, height: 30, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: c.label }}>
              <span className="sq-icon-tap">{dark ? <Sun key="sun" size={16} strokeWidth={1.75} className="sq-icon-pop-in" /> : <Moon key="moon" size={16} strokeWidth={1.75} className="sq-icon-pop-in" />}</span>
            </button>
          </div>
        </div>

        <div className="relative px-4 mt-6 flex flex-col items-center text-center">
          {leveledUp && <Confetti big c={c} />}
          {leveledUp && (
              <span className="sq-mono" style={{ marginBottom: 12, fontSize: 11, fontWeight: 700, letterSpacing: 0.6, textTransform: 'uppercase', color: '#fff', background: c.orange, padding: '5px 14px', borderRadius: 999 }}>
            Level up
          </span>
          )}
          <div className="sq-anim-check" style={{ width: 88, height: 88, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: accent }}>
            <CheckIcon size={38} />
            <style>{`div > svg { color: #fff; }`}</style>
          </div>
          <h2 style={{ marginTop: 18, fontSize: 22, fontWeight: 700, color: c.label }}>Quest complete</h2>
          <span className="sq-mono" style={{ marginTop: 8, fontSize: 13, fontWeight: 700, color: '#fff', background: accent, padding: '4px 12px', borderRadius: 999 }}>
          +{quest?.xp ?? 0} XP
        </span>
        </div>

        <div className="px-4 mt-8 flex-1">
          <div style={{ ...glassStyle(c), borderRadius: 20, padding: 20, textAlign: 'center' }}>
            <p style={{ fontSize: 15, fontWeight: 500, lineHeight: 1.5, color: c.label }}>{quote}</p>
          </div>
        </div>

        <div className="px-4 pb-8 mt-6">
          <button onClick={onBack} style={{ width: '100%', padding: '15px', borderRadius: 999, fontSize: 16, fontWeight: 600, color: c.label, background: c.fill }}>
            Back to Quests
          </button>
        </div>
      </div>
  );
}

// camera modal (the big one)
function CameraModal({ quest, onConfirm, onCancel, c }) {
  const videoRef          = useRef(null);
  const aiCanvasRef       = useRef(null);
  const skeletonCanvasRef = useRef(null);
  const streamRef         = useRef(null);
  const scanTimerRef      = useRef(null);
  const isScanningRef     = useRef(false);
  const scanRunRef        = useRef(0);
  const cameraSessionRef  = useRef(0);
  const lastVideoTimeRef  = useRef(-1);
  const confirmedRef      = useRef(false);
  const passStreakRef     = useRef(0);
  const poseStateRef      = useRef(createPoseRepState(quest.reps ? (quest.progress || 0) : 0));
  const poseRafRef        = useRef(null);
  const poseLastVideoTimeRef = useRef(-1);
  const smoothedActRef    = useRef(null);
  const smoothedNegRef    = useRef(null);
  const smoothedSpoofRef  = useRef(null);
  const spoofNoticeActiveRef = useRef(false);
  const verifyingRef      = useRef(false);
  const repSpoofSuspectedRef = useRef(false);
  const smoothedRepSpoofRef  = useRef(null);
  const repSpoofNoticeActiveRef = useRef(false);

  const [phase,         setPhase]         = useState('starting');
  const [camError,      setCamError]      = useState(null);
  const [facingMode,    setFacingMode]    = useState('environment');
  const [cameraVersion, setCameraVersion] = useState(0);
  const [modelReady,    setModelReady]    = useState(false);
  const [modelProgress, setModelProgress] = useState(null);
  const [modelError,    setModelError]    = useState(null);
  const [modelVersion,  setModelVersion]  = useState(0);
  const [poseReady,     setPoseReady]     = useState(false);
  const [poseError,     setPoseError]     = useState(null);
  const [repCue,        setRepCue]        = useState('Get in frame');
  const [notice,        setNotice]        = useState(() => {
    if (quest.reps && quest.progress > 0) return `Resuming — you already logged ${quest.progress} of ${quest.reps} reps.`;
    if (quest.duration && quest.progress != null && quest.progress < quest.duration) {
      const m = Math.floor(quest.progress / 60), s = quest.progress % 60;
      return `Resuming — ${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')} left on the clock.`;
    }
    return null;
  });
  const [scanning,      setScanning]      = useState(false);
  const [uploading,     setUploading]     = useState(false);
  const [liveScore,     setLiveScore]     = useState(0);
  const [passStreak,    setPassStreak]    = useState(0);
  const [repsDone,      setRepsDone]      = useState(quest.reps ? (quest.progress || 0) : 0);
  const [repPhase,      setRepPhase]      = useState('up');
  const [confirmed,     setConfirmed]     = useState(false);
  const [verifying,     setVerifying]     = useState(false);
  const [lastLabel,     setLastLabel]     = useState('');
  const [uploadedProof, setUploadedProof] = useState(null);

  const [secondsLeft, setSecondsLeft] = useState(quest.duration ? (quest.progress ?? quest.duration) : 0);
  const [timerRunning, setTimerRunning] = useState(false);

  const labels = QUEST_LABELS[quest.id];
  const questType = labels?.type || 'action';
  const hasSkeletonTracking = Boolean(quest.reps) || MOVEMENT_ACTION_IDS.has(quest.id);

  const activeNegatives = useMemo(() => getNegativeLabels(questType), [questType]);
  const classifierLabels = useMemo(
      () => [...new Set([...(labels?.activity ?? []), ...activeNegatives])],
      [labels, activeNegatives]
  );
  const [sourceWarning, setSourceWarning] = useState(null);

  let instructionText = `Show the camera you're ${labels?.label ?? 'doing it'}…`;
  let uiSubtext = quest.text;

  if (questType === 'map') { instructionText = "Upload a metric summary screenshot (Strava, Nike, Garmin etc.)"; uiSubtext = "Must clearly state distance metrics & active time duration summary logs."; }
  else if (questType === 'food') { instructionText = "Take a clear picture of the food on your plate."; uiSubtext = "Must be a real photo of a prepared meal or plate."; }
  else if (questType === 'reps') { instructionText = `Position the camera for full-body tracking: ${quest.reps} reps.`; uiSubtext = "Keep your entire working frame visible to log movements."; }

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

  const formatTimerString = (secs) => `${String(Math.floor(secs / 60)).padStart(2, '0')}:${String(secs % 60).padStart(2, '0')}`;

  const stopCamera = useCallback(() => {
    const stream = streamRef.current;
    streamRef.current = null;
    stream?.getTracks().forEach(track => track.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const startCamera = useCallback(async (mode) => {
    const session = ++cameraSessionRef.current;
    stopCamera();
    setCamError(null); setNotice(null); setUploadedProof(null); setPhase('starting');
    const attempts = [
      { video: { facingMode: { ideal: mode }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
      { video: { facingMode: { ideal: mode } }, audio: false },
      { video: true, audio: false },
    ];
    for (const constraints of attempts) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (session !== cameraSessionRef.current) { stream.getTracks().forEach(track => track.stop()); return; }
        streamRef.current = stream;
        const video = videoRef.current;
        if (!video) { stopCamera(); return; }
        video.srcObject = stream;
        await video.play();
        if (session !== cameraSessionRef.current) { stream.getTracks().forEach(track => track.stop()); return; }
        stream.getVideoTracks().forEach(track => {
          track.addEventListener('ended', () => { if (session === cameraSessionRef.current) { setCamError('unavailable'); setPhase('error'); } });
        });
        setPhase('live');
        return;
      } catch (err) {
        if (session !== cameraSessionRef.current) return;
        stopCamera();
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError' || err.name === 'SecurityError') {
          setCamError('permission'); setPhase('error'); return;
        }
      }
    }
    if (session === cameraSessionRef.current) { setCamError('unavailable'); setPhase('error'); }
  }, [stopCamera]);

  useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia) { setCamError('unsupported'); setPhase('error'); return undefined; }
    startCamera(facingMode);
    return () => { cameraSessionRef.current += 1; stopCamera(); };
  }, [facingMode, cameraVersion, startCamera, stopCamera]);

  useEffect(() => {
    if (quest.reps) { setModelReady(true); return undefined; }
    let cancelled = false;
    (async () => {
      try {
        setModelError(null);
        await getClassifier(evt => {
          if (cancelled) return;
          if (evt.status === 'progress' && evt.total) setModelProgress(Math.round((evt.loaded / evt.total) * 100));
        });
        if (!cancelled) { setModelReady(true); setModelProgress(null); }
      } catch (err) {
        if (!cancelled) { setModelError('The AI model could not be loaded. Check your connection and try again.'); setModelProgress(null); }
      }
    })();
    return () => { cancelled = true; };
  }, [modelVersion, quest.reps]);

  useEffect(() => {
    if (quest.reps || !modelReady) return undefined;
    let cancelled = false;
    getStaticClassifier().catch(() => { /* falls back gracefully in runFinalVerification */ });
    return () => { cancelled = true; };
  }, [modelReady, quest.reps]);

  const resetRepTracking = useCallback(() => {
    const baseline = quest.reps ? (quest.progress || 0) : 0;
    poseStateRef.current = createPoseRepState(baseline);
    setRepsDone(baseline);
    setRepPhase('up');
    setRepCue('Get in frame');
  }, [quest.reps, quest.progress]);

  useEffect(() => {
    if (quest.reps) return undefined;
    if (phase !== 'live' || !modelReady || confirmed || uploading || !labels || !classifierLabels.length) return undefined;
    const runId = ++scanRunRef.current;
    let disposed = false;
    let consecutiveErrors = 0;
    const isCurrentRun = () => !disposed && runId === scanRunRef.current && !confirmedRef.current;
    const scheduleNext = (delay) => { if (!isCurrentRun()) return; clearTimeout(scanTimerRef.current); scanTimerRef.current = window.setTimeout(scanLoop, delay); };

    const scanLoop = async () => {
      const video = videoRef.current, canvas = aiCanvasRef.current;
      if (!isCurrentRun()) return;
      if (document.visibilityState === 'hidden') { scheduleNext(1000); return; }
      if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) { scheduleNext(250); return; }
      if (video.currentTime === lastVideoTimeRef.current || isScanningRef.current) { scheduleNext(80); return; }
      lastVideoTimeRef.current = video.currentTime;
      isScanningRef.current = true;
      setScanning(true);

      try {
        if (canvas.width !== 224 || canvas.height !== 224) { canvas.width = 224; canvas.height = 224; }
        const context = canvas.getContext('2d');
        if (!context) throw new Error('Canvas 2D context unavailable');

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

        const actScore = getMaxLabelScore(results, labels.activity);
        const negScore = getMaxLabelScore(results, activeNegatives);
        const spoofScore = getMaxLabelScore(results, ANTI_SPOOF_LABELS);
        setLastLabel(results[0]?.label ?? '');

        smoothedActRef.current = smoothedActRef.current === null ? actScore : smoothedActRef.current * (1 - SCORE_SMOOTHING) + actScore * SCORE_SMOOTHING;
        smoothedNegRef.current = smoothedNegRef.current === null ? negScore : smoothedNegRef.current * (1 - SCORE_SMOOTHING) + negScore * SCORE_SMOOTHING;
        smoothedSpoofRef.current = smoothedSpoofRef.current === null ? spoofScore : smoothedSpoofRef.current * (1 - SCORE_SMOOTHING) + spoofScore * SCORE_SMOOTHING;
        const smoothedAct = smoothedActRef.current;
        const smoothedNeg = smoothedNegRef.current;
        const smoothedSpoof = smoothedSpoofRef.current;

        setLiveScore(Math.round(smoothedAct * 100));
        const spoofSuspected = smoothedSpoof >= SPOOF_BLOCK_THRESHOLD;
        const passed = !spoofSuspected && (smoothedAct - smoothedNeg) >= PASS_MARGIN && smoothedAct >= PASS_THRESHOLD;
        passStreakRef.current = passed
            ? Math.min(REQUIRED_PASSES, passStreakRef.current + 1)
            : Math.max(0, passStreakRef.current - 1);
        setPassStreak(passStreakRef.current);
        if (spoofSuspected) {
          spoofNoticeActiveRef.current = true;
          setNotice('This looks like it\'s coming from a screen or printed photo rather than you, live. Point the camera at yourself doing it in person.');
        } else if (spoofNoticeActiveRef.current) {
          spoofNoticeActiveRef.current = false;
          setNotice(null);
        }
        if (passStreakRef.current >= REQUIRED_PASSES && !verifyingRef.current && !confirmedRef.current) {
          verifyingRef.current = true;
          setVerifying(true);
          runFinalVerification();
        } else if (passed) {
          haptic(8);
        }
      } catch (err) {
        consecutiveErrors += 1;
        if (consecutiveErrors >= 3 && isCurrentRun()) setModelError('AI analysis is temporarily unavailable. Try closing and reopening the camera.');
      } finally {
        isScanningRef.current = false;
        if (isCurrentRun()) { setScanning(false); scheduleNext(consecutiveErrors ? 1000 : 1200); }
      }
    };

    const runFinalVerification = async () => {
      try {
        const video = videoRef.current;
        if (!video || !video.videoWidth) throw new Error('no live frame available');
        const size = 256;
        const canvas = document.createElement('canvas');
        canvas.width = size; canvas.height = size;
        const ctx = canvas.getContext('2d');
        const scale = Math.min(size / video.videoWidth, size / video.videoHeight);
        const w = video.videoWidth * scale, h = video.videoHeight * scale;
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, size, size);
        ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, (size - w) / 2, (size - h) / 2, w, h);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

        const staticClassifier = await getStaticClassifier();
        const results = await staticClassifier(dataUrl, classifierLabels);
        const actScore = getMaxLabelScore(results, labels.activity);
        const negScore = getMaxLabelScore(results, activeNegatives);
        const spoofScore = getMaxLabelScore(results, ANTI_SPOOF_LABELS);
        const finalPass = spoofScore < SPOOF_BLOCK_THRESHOLD && (actScore - negScore) >= PASS_MARGIN && actScore >= PASS_THRESHOLD;

        if (!isCurrentRun()) return;
        if (finalPass) {
          confirmedRef.current = true;
          setConfirmed(true);
          haptic([15, 30, 15]);
        } else {
          // fail closed: a second, higher-resolution model disagreed with the
          // live estimate, so drop most of the streak instead of confirming
          if (spoofScore >= SPOOF_BLOCK_THRESHOLD) {
            spoofNoticeActiveRef.current = true;
            setNotice('This looks like it\'s coming from a screen or printed photo rather than you, live. Point the camera at yourself doing it in person.');
          }
          passStreakRef.current = Math.max(0, REQUIRED_PASSES - 2);
          setPassStreak(passStreakRef.current);
        }
      } catch {
        // model call failed — fail closed rather than auto-confirming, so a
        // dropped network request or model hiccup can't be used as a free pass
        if (isCurrentRun()) {
          passStreakRef.current = Math.max(0, REQUIRED_PASSES - 2);
          setPassStreak(passStreakRef.current);
        }
      } finally {
        verifyingRef.current = false;
        setVerifying(false);
        if (isCurrentRun()) scheduleNext(400);
      }
    };
    scanLoop();
    return () => { disposed = true; scanRunRef.current += 1; clearTimeout(scanTimerRef.current); };
  }, [phase, modelReady, confirmed, uploading, labels, classifierLabels, quest.reps, activeNegatives]);

  useEffect(() => {
    if (!hasSkeletonTracking || phase !== 'live' || confirmed) return undefined;
    let cancelled = false;
    (async () => {
      try {
        setPoseError(null);
        const landmarker = await getPoseLandmarker();
        if (cancelled) return;
        setPoseReady(true);

        const loop = () => {
          if (cancelled) return;
          const video = videoRef.current;
          const canvas = skeletonCanvasRef.current;
          if (video && video.readyState >= 2 && video.currentTime !== poseLastVideoTimeRef.current) {
            poseLastVideoTimeRef.current = video.currentTime;
            const now = performance.now();
            try {
              const result = landmarker.detectForVideo(video, now);
              const landmarks = result?.landmarks?.[0] ?? null;
              if (landmarks && quest.reps) {
                const update = updatePoseRepState(poseStateRef.current, quest.id, landmarks, now);
                setRepPhase(update.phase);
                setRepCue(update.cue);
                if (update.counted) {
                  setRepsDone(update.reps);
                  if (update.reps >= quest.reps) {
                    if (repSpoofSuspectedRef.current) {
                      // hold one rep short until the screen/photo signal clears —
                      // this stops a looped video of someone else exercising
                      // (which pose angles alone can't distinguish from the real thing)
                      poseStateRef.current.reps = quest.reps - 1;
                      setRepsDone(quest.reps - 1);
                      haptic(10);
                    } else {
                      confirmedRef.current = true;
                      setConfirmed(true);
                      haptic([15, 30, 15]);
                    }
                  } else {
                    haptic(10);
                  }
                }
              } else if (!landmarks && quest.reps) {
                setRepCue('Step back so your full body is visible');
              }

              if (canvas && video) {
                const ctx = canvas.getContext('2d');
                if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
                  canvas.width = canvas.clientWidth || 300;
                  canvas.height = canvas.clientHeight || 300;
                }
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                if (landmarks && !confirmedRef.current) {
                  const Wc = canvas.width;
                  const Hc = canvas.height;
                  const Wv = video.videoWidth || 640;
                  const Hv = video.videoHeight || 480;
                  const scale = Math.max(Wc / Wv, Hc / Hv);
                  const Wr = Wv * scale;
                  const Hr = Hv * scale;
                  const Ox = (Wc - Wr) / 2;
                  const Oy = (Hc - Hr) / 2;

                  const mapPt = (pt) => ({ x: Ox + pt.x * Wr, y: Oy + pt.y * Hr, vis: pt.visibility ?? 1 });
                  const mapped = landmarks.map(mapPt);
                  const currentPhase = poseStateRef.current.phase;
                  const lineColor = currentPhase === 'down' ? '#30D158' : '#0A84FF';

                  ctx.save();
                  ctx.lineWidth = 3;
                  ctx.lineCap = 'round';
                  ctx.lineJoin = 'round';
                  ctx.strokeStyle = lineColor;
                  POSE_CONNECTIONS.forEach(([i, j]) => {
                    const p1 = mapped[i]; const p2 = mapped[j];
                    if (p1 && p2 && p1.vis > 0.4 && p2.vis > 0.4) {
                      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
                    }
                  });
                  ctx.restore();

                  ctx.save();
                  mapped.forEach((pt, idx) => {
                    if (pt.vis > 0.4 && (idx === 0 || (idx >= 11 && idx <= 32))) {
                      const isMajorJoint = [11, 12, 13, 14, 23, 24, 25, 26].includes(idx);
                      ctx.beginPath();
                      ctx.arc(pt.x, pt.y, isMajorJoint ? 6 : 4, 0, 2 * Math.PI);
                      ctx.fillStyle = 'rgba(0,0,0,0.65)';
                      ctx.fill();
                      ctx.lineWidth = isMajorJoint ? 2 : 1.5;
                      ctx.strokeStyle = lineColor;
                      ctx.stroke();
                      ctx.beginPath();
                      ctx.arc(pt.x, pt.y, isMajorJoint ? 2.5 : 1.6, 0, 2 * Math.PI);
                      ctx.fillStyle = '#ffffff';
                      ctx.fill();
                    }
                  });
                  ctx.restore();
                }
              }
            } catch (err) { /* transient frame errors are fine, keep looping */ }
          }
          if (!cancelled) poseRafRef.current = requestAnimationFrame(loop);
        };
        poseRafRef.current = requestAnimationFrame(loop);
      } catch (err) {
        if (!cancelled) setPoseError('Could not load the pose tracking model. Check your connection and try again.');
      }
    })();
    return () => {
      cancelled = true;
      if (poseRafRef.current) cancelAnimationFrame(poseRafRef.current);
      if (skeletonCanvasRef.current) {
        const ctx = skeletonCanvasRef.current.getContext('2d');
        ctx?.clearRect(0, 0, skeletonCanvasRef.current.width, skeletonCanvasRef.current.height);
      }
    };
  }, [hasSkeletonTracking, quest.reps, quest.id, phase, confirmed]);

  // rep quests are verified by joint-angle thresholds, which a static photo
  // can't fake (angles never move) — but a *video* of someone else
  // exercising, played on another screen and pointed at the camera, would
  // still cross those thresholds. This runs alongside pose tracking purely
  // to catch that: it doesn't judge the activity, only whether the feed
  // looks like it's coming from a screen or printed photo.
  useEffect(() => {
    if (!quest.reps || phase !== 'live' || confirmed) return undefined;
    let cancelled = false;
    let timer = null;

    const check = async () => {
      try {
        const video = videoRef.current;
        const canvas = aiCanvasRef.current;
        if (video && canvas && video.videoWidth) {
          if (canvas.width !== 224 || canvas.height !== 224) { canvas.width = 224; canvas.height = 224; }
          const ctx = canvas.getContext('2d');
          const scale = Math.min(224 / video.videoWidth, 224 / video.videoHeight);
          const w = video.videoWidth * scale, h = video.videoHeight * scale;
          ctx.fillStyle = '#000';
          ctx.fillRect(0, 0, 224, 224);
          ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, (224 - w) / 2, (224 - h) / 2, w, h);
          const dataUrl = canvas.toDataURL('image/jpeg', 0.5);
          const classifier = await getClassifier();
          const results = await classifier(dataUrl, [...ANTI_SPOOF_LABELS, 'a real person exercising in a room']);
          if (!cancelled) {
            const spoofScore = getMaxLabelScore(results, ANTI_SPOOF_LABELS);
            smoothedRepSpoofRef.current = smoothedRepSpoofRef.current === null ? spoofScore : smoothedRepSpoofRef.current * 0.5 + spoofScore * 0.5;
            const suspected = smoothedRepSpoofRef.current >= SPOOF_BLOCK_THRESHOLD;
            repSpoofSuspectedRef.current = suspected;
            if (suspected) {
              repSpoofNoticeActiveRef.current = true;
              setNotice('This looks like it\'s coming from a screen or printed photo rather than you, live. Point the camera at yourself doing it in person.');
            } else if (repSpoofNoticeActiveRef.current) {
              repSpoofNoticeActiveRef.current = false;
              setNotice(null);
            }
          }
        }
      } catch { /* transient — try again next tick */ }
      finally { if (!cancelled) timer = window.setTimeout(check, 2500); }
    };
    timer = window.setTimeout(check, 1500);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [quest.reps, phase, confirmed]);

  const retryCamera = () => { scanRunRef.current += 1; lastVideoTimeRef.current = -1; setCamError(null); setPhase('starting'); setCameraVersion(v => v + 1); setSourceWarning(null); };
  const flipCamera = () => {
    clearTimeout(scanTimerRef.current); scanRunRef.current += 1; setPhase('starting'); setLiveScore(0); setPassStreak(0); setRepsDone(quest.reps ? (quest.progress || 0) : 0); passStreakRef.current = 0; confirmedRef.current = false; lastVideoTimeRef.current = -1; smoothedActRef.current = null; smoothedNegRef.current = null; smoothedSpoofRef.current = null; spoofNoticeActiveRef.current = false; repSpoofSuspectedRef.current = false; smoothedRepSpoofRef.current = null; repSpoofNoticeActiveRef.current = false; verifyingRef.current = false; setVerifying(false); resetRepTracking(); setConfirmed(false); setSourceWarning(null); setFacingMode(m => m === 'environment' ? 'user' : 'environment');
    if (skeletonCanvasRef.current) {
      const ctx = skeletonCanvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, skeletonCanvasRef.current.width, skeletonCanvasRef.current.height);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0]; if (!file) return;
    if (quest.reps) { setNotice('Rep quests need live camera tracking; a single photo cannot verify a full set.'); e.target.value = ''; return; }
    const reader = new FileReader();
    reader.onload = async (ev) => {
      let accepted = false; scanRunRef.current += 1; setUploading(true); stopCamera(); clearTimeout(scanTimerRef.current);
      setSourceWarning(null);
      const dataUrl = ev.target.result; setUploadedProof(dataUrl);
      if (!labels) { onConfirm(dataUrl); return; }
      setPhase('live'); setScanning(true);
      try {
        if (questType !== 'map') {
          const authCheck = await checkPhotoAuthenticity(file, dataUrl, SOURCE_LABELS);
          if (authCheck.suspicious) {
            setPassStreak(0);
            setNotice(`This looks like a stock photo, screenshot, or image pulled from the internet rather than one you took yourself (${authCheck.confidence}% confidence). Please upload an original photo.`);
            return;
          }
          if (!authCheck.hasCameraMetadata) {
            setSourceWarning("This photo has no camera metadata, so we can't fully confirm it's an original. Live camera capture is the most reliable option.");
          }
        }
        const classifier = await getStaticClassifier();
        const results = await classifier(dataUrl, classifierLabels);
        const actScore = getMaxLabelScore(results, labels.activity);
        const negScore = getMaxLabelScore(results, activeNegatives);
        const spoofScore = getMaxLabelScore(results, ANTI_SPOOF_LABELS);
        setLiveScore(Math.round(actScore * 100)); setLastLabel(results[0]?.label ?? '');
        if (spoofScore >= SPOOF_BLOCK_THRESHOLD) {
          setPassStreak(0);
          setNotice('This looks like a photo of a screen or a printed photo rather than an original shot. Please upload something you photographed directly.');
        } else if ((actScore - negScore) >= PASS_MARGIN && actScore >= PASS_THRESHOLD) {
          setPassStreak(REQUIRED_PASSES); confirmedRef.current = true; accepted = true; setConfirmed(true); haptic([15, 30, 15]);
        } else { setPassStreak(0); setNotice("Verification failed. Please make sure you upload a clear activity summary log showing distance and elapsed time metrics."); }
      } catch (err) { } finally { setScanning(false); setUploading(false); if (!accepted) { setUploadedProof(null); retryCamera(); } }
    };
    reader.readAsDataURL(file);
  };

  const handleCancel = () => {
    stopCamera();
    if (quest.reps) onCancel(repsDone > 0 ? repsDone : undefined);
    else if (quest.duration) onCancel(secondsLeft < quest.duration ? secondsLeft : undefined);
    else onCancel(undefined);
  };

  const confirmingRef = useRef(false);
  const captureAndConfirm = () => {
    if (confirmingRef.current) return;
    confirmingRef.current = true;
    if (uploadedProof) { onConfirm(uploadedProof); return; }
    const video = videoRef.current, canvas = aiCanvasRef.current;
    if (!video || !canvas) { confirmingRef.current = false; return; }
    canvas.width = video.videoWidth || 640; canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);
    onConfirm(canvas.toDataURL('image/jpeg', 0.82));
  };

  const meterColor = liveScore >= PASS_THRESHOLD * 100 ? c.green : liveScore >= 15 ? c.orange : c.red;

  return (
      <div className="fixed inset-0 z-[60] flex items-end justify-center" style={{ background: 'rgba(0,0,0,0.5)' }}>
        <div style={{ background: c.bg, width: '100%', height: '95vh', borderTopLeftRadius: 24, borderTopRightRadius: 24, overflow: 'hidden', display: 'flex', flexDirection: 'column', paddingBottom: 'max(env(safe-area-inset-bottom), 14px)' }}>
          <div className="flex justify-center pt-3 pb-2">
            <div style={{ width: 36, height: 5, borderRadius: 999, background: c.fill }} />
          </div>

          <div className="flex items-center justify-center px-6 pb-3">
            <div className="text-center">
              <p style={{ fontSize: 15, fontWeight: 600, color: c.label }}>Quest check-in</p>
              <p style={{ fontSize: 11, color: c.labelSecondary, marginTop: 2, maxWidth: 240, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{uiSubtext}</p>
            </div>
          </div>

          <div className="relative flex-1 mx-4 rounded-3xl overflow-hidden" style={{ background: '#000' }}>
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" style={{ opacity: phase === 'live' && !uploadedProof ? 1 : 0, transition: 'opacity 0.3s' }} />
            <canvas ref={skeletonCanvasRef} className="absolute inset-0 w-full h-full pointer-events-none object-cover z-10"
                    style={{ opacity: phase === 'live' && !uploadedProof && hasSkeletonTracking ? 1 : 0, transition: 'opacity 0.3s' }} />

            {phase === 'live' && !uploadedProof && labels?.bodyParts && (
                <div className="absolute top-3 left-3 z-10" style={{ background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(10px)', borderRadius: 12, padding: '9px 11px' }}>
                  <div className="flex items-center gap-1.5 mb-1.5" style={{ color: '#fff', fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4 }}>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: c.blue, display: 'inline-block' }} className="animate-pulse" />
                    Tracking
                  </div>
                  <div className="flex flex-wrap gap-1 max-w-[150px]">
                    {labels.bodyParts.map((part) => (
                        <span key={part} style={{ background: 'rgba(255,255,255,0.14)', color: '#fff', padding: '2px 7px', borderRadius: 8, fontSize: 9, fontWeight: 500 }} className="sq-mono">
                    {part}
                  </span>
                    ))}
                  </div>
                </div>
            )}

            {uploadedProof && (<img src={uploadedProof} alt="Uploaded verification" className="absolute inset-0 w-full h-full object-cover" />)}

            {phase === 'starting' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                  <div style={{ width: 30, height: 30, border: '2.5px solid rgba(255,255,255,0.2)', borderTopColor: '#fff', borderRadius: '50%' }} className="animate-spin" />
                  <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: 12, fontWeight: 500 }}>Opening camera…</p>
                </div>
            )}

            {phase === 'error' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-8 text-center">
                  <p style={{ color: '#fff', fontWeight: 700, fontSize: 15 }}>{camError === 'permission' ? 'Camera access blocked' : 'No camera found'}</p>
                  <p style={{ color: 'rgba(255,255,255,0.65)', fontSize: 13, lineHeight: 1.5, marginBottom: 4 }}>
                    {camError === 'permission' ? 'Open this app in a new tab and allow camera access when prompted.' : quest.reps ? 'This quest needs a live camera. Check your camera and try again.' : 'Upload a photo instead.'}
                  </p>
                  <button onClick={retryCamera} style={{ background: c.blue, color: '#fff', fontSize: 13, fontWeight: 600, padding: '9px 22px', borderRadius: 999 }}>Try again</button>
                </div>
            )}

            {(modelError || notice) && (
                <div className="absolute inset-x-3 top-3 z-30" style={{ borderRadius: 12, background: 'rgba(255,59,48,0.92)', backdropFilter: 'blur(10px)', padding: '10px 14px' }}>
                  <p style={{ fontSize: 13, fontWeight: 500, lineHeight: 1.4, color: '#fff' }}>{modelError || notice}</p>
                  <button onClick={() => setNotice(null)} style={{ fontSize: 11, fontWeight: 700, color: 'rgba(255,255,255,0.75)', marginTop: 6, textTransform: 'uppercase', letterSpacing: 0.3 }}>Dismiss</button>
                </div>
            )}

            {!modelError && !notice && sourceWarning && (
                <div className="absolute inset-x-3 top-3 z-30" style={{ borderRadius: 12, background: 'rgba(255,159,10,0.92)', backdropFilter: 'blur(10px)', padding: '10px 14px' }}>
                  <p style={{ fontSize: 13, fontWeight: 500, lineHeight: 1.4, color: '#fff' }}>{sourceWarning}</p>
                  <button onClick={() => setSourceWarning(null)} style={{ fontSize: 11, fontWeight: 700, color: 'rgba(255,255,255,0.75)', marginTop: 6, textTransform: 'uppercase', letterSpacing: 0.3 }}>Dismiss</button>
                </div>
            )}

            {phase === 'live' && quest.reps && (
                <div className="absolute inset-x-0 bottom-0 px-4 pb-4 pt-10 z-20" style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.75), transparent)' }}>
                  {confirmed ? (
                      <div className="flex items-center gap-2 w-max px-3 py-1.5 rounded-full" style={{ background: 'rgba(48,209,88,0.2)' }}>
                        <div style={{ width: 18, height: 18, borderRadius: '50%', background: c.green, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><CheckIcon size={10} /></div>
                        <p style={{ color: '#DFFBE8', fontSize: 12, fontWeight: 600 }}>Verified</p>
                      </div>
                  ) : poseError ? (
                      <p style={{ color: '#FF9F9A', fontSize: 12, fontWeight: 500 }}>{poseError}</p>
                  ) : !poseReady ? (
                      <p style={{ color: 'rgba(255,255,255,0.8)', fontSize: 11, fontWeight: 600 }}>Getting ready to track your reps…</p>
                  ) : (
                      <>
                        <div className="flex items-end justify-between mb-2">
                          <p style={{ color: 'rgba(255,255,255,0.9)', fontSize: 13, fontWeight: 500 }}>{repCue}</p>
                          <p className="sq-mono" style={{ color: '#fff', fontSize: 12, fontWeight: 700, background: c.blue, padding: '4px 10px', borderRadius: 8 }}>
                            {repsDone} / {quest.reps} reps
                          </p>
                        </div>
                        <div style={{ width: '100%', height: 4, borderRadius: 999, background: 'rgba(255,255,255,0.2)', overflow: 'hidden' }}>
                          <div style={{ height: '100%', background: c.green, width: `${Math.min(100, (repsDone / quest.reps) * 100)}%`, transition: 'width 0.3s ease-out' }} />
                        </div>
                      </>
                  )}
                </div>
            )}

            {phase === 'live' && !quest.reps && modelReady && (
                <div className="absolute inset-x-0 bottom-0 px-4 pb-4 pt-10 z-20" style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.75), transparent)' }}>
                  {confirmed ? (
                      <div className="flex items-center gap-2 w-max px-3 py-1.5 rounded-full" style={{ background: 'rgba(48,209,88,0.2)' }}>
                        <div style={{ width: 18, height: 18, borderRadius: '50%', background: c.green, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><CheckIcon size={10} /></div>
                        <p style={{ color: '#DFFBE8', fontSize: 12, fontWeight: 600 }}>Verified</p>
                      </div>
                  ) : verifying ? (
                      <div className="flex items-center gap-2 w-max px-3 py-1.5 rounded-full" style={{ background: 'rgba(10,132,255,0.2)' }}>
                        <div style={{ width: 14, height: 14, border: '2px solid rgba(255,255,255,0.3)', borderTopColor: '#fff', borderRadius: '50%' }} className="animate-spin" />
                        <p style={{ color: '#fff', fontSize: 12, fontWeight: 600 }}>Confirming…</p>
                      </div>
                  ) : (
                      <>
                        <div className="flex items-end justify-between mb-2">
                          <p style={{ color: 'rgba(255,255,255,0.9)', fontSize: 13, fontWeight: 500 }}>{scanning ? 'Scanning…' : `${liveScore}% confidence`}</p>
                          <div className="flex items-center gap-1">
                            {Array.from({ length: REQUIRED_PASSES }).map((_, i) => (
                                <div key={i} style={{ width: 6, height: 6, borderRadius: '50%', background: i < passStreak ? c.green : 'rgba(255,255,255,0.3)', transition: 'background 0.3s' }} />
                            ))}
                          </div>
                        </div>
                        <div style={{ width: '100%', height: 4, borderRadius: 999, background: 'rgba(255,255,255,0.2)', overflow: 'hidden' }}>
                          <div style={{ height: '100%', background: meterColor, width: `${liveScore}%`, transition: 'width 0.5s ease-out' }} />
                        </div>
                        {lastLabel && <p className="sq-mono" style={{ color: 'rgba(255,255,255,0.4)', fontSize: 9, marginTop: 6, textTransform: 'uppercase', letterSpacing: 0.3, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{lastLabel}</p>}
                      </>
                  )}
                </div>
            )}

            {phase === 'live' && !quest.reps && !modelReady && !modelError && (
                <div className="absolute inset-x-0 bottom-0 px-4 pb-4 pt-10 z-20" style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.7), transparent)' }}>
                  <p style={{ color: 'rgba(255,255,255,0.8)', fontSize: 11, fontWeight: 600, marginBottom: 8 }}>Getting ready… {modelProgress ?? 0}%</p>
                  <div style={{ width: '100%', height: 4, borderRadius: 999, background: 'rgba(255,255,255,0.2)', overflow: 'hidden' }}>
                    <div style={{ height: '100%', background: c.blue, width: `${modelProgress ?? 0}%`, transition: 'width 0.3s' }} />
                  </div>
                </div>
            )}
            <canvas ref={aiCanvasRef} className="hidden" />
          </div>

          {quest.duration && (
              <div className="mx-4 mt-4" style={{ padding: '13px 16px', borderRadius: 20, background: c.bgElevated, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <span style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, color: c.labelSecondary }}>Duration</span>
                  <p className="sq-mono" style={{ fontSize: 20, fontWeight: 700, color: c.label, marginTop: 2 }}>{formatTimerString(secondsLeft)}</p>
                </div>
                <button onClick={() => setTimerRunning(!timerRunning)} disabled={secondsLeft === 0}
                        style={{ padding: '10px 20px', borderRadius: 12, fontSize: 13, fontWeight: 600, color: '#fff', background: secondsLeft === 0 ? c.gray : timerRunning ? c.red : c.green, opacity: secondsLeft === 0 ? 0.5 : 1 }}>
                  {secondsLeft === 0 ? 'Done' : timerRunning ? 'Pause' : 'Start'}
                </button>
              </div>
          )}

          <div className="px-4 pt-4 pb-1">
            {phase === 'live' && (
                <div className="space-y-2.5">
                  <button onClick={captureAndConfirm} disabled={!confirmed}
                          style={{ width: '100%', padding: '15px', borderRadius: 999, fontSize: 16, fontWeight: 600, color: confirmed ? '#fff' : c.labelTertiary, background: confirmed ? c.blue : c.fill }}>
                    {confirmed ? 'Complete quest' : verifying ? 'Confirming…' : instructionText}
                  </button>
                  <div className="flex gap-2.5">
                    <button onClick={flipCamera} style={{ flex: quest.reps ? '1 1 100%' : 1, padding: '13px', borderRadius: 999, fontSize: 14, fontWeight: 600, background: c.fill, color: c.label }}>
                      Flip camera
                    </button>
                    {!quest.reps && (
                        <label style={{ flex: 1, padding: '13px', borderRadius: 999, fontSize: 14, fontWeight: 600, textAlign: 'center', cursor: 'pointer', background: questType === 'map' ? c.bgElevated : c.fill, color: questType === 'map' ? c.blue : c.label, border: questType === 'map' ? `1.5px solid ${c.blue}` : 'none' }}>
                          Upload photo
                          <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
                        </label>
                    )}
                  </div>
                </div>
            )}

            {(phase === 'error' || phase === 'starting') && !quest.reps && (
                <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', padding: '15px', borderRadius: 999, fontSize: 16, fontWeight: 600, background: c.fill, color: c.label, cursor: 'pointer' }}>
                  Upload a photo
                  <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
                </label>
            )}

            <button onClick={handleCancel} style={{ width: '100%', marginTop: 10, padding: '15px', borderRadius: 999, fontSize: 16, fontWeight: 600, background: 'transparent', color: c.red }}>
              Cancel
            </button>
          </div>
        </div>
      </div>
  );
}

// main app
export default function QuestDailyApp() {
  const [isMounted, setIsMounted] = useState(false);
  const uid = useAnonymousAuth();

  const [dark, setDark] = useState(true);
  const [showInstallPrompt, setShowInstallPrompt] = useState(false);
  const [level, setLevel] = useState(1);
  const [xp, setXp] = useState(0);
  const [totalXpEarned, setTotalXpEarned] = useState(0);
  const [streak, setStreak] = useState(0);
  const [username, setUsername] = useState(null);
  const [showLeaderboard, setShowLeaderboard] = useState(false);
  const [leaderboardSyncError, setLeaderboardSyncError] = useState(null);
  const [quests, setQuests] = useState([]);
  const [lastReset, setLastReset] = useState(0);
  const [proofImages, setProofImages] = useState({});
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('quests');

  useEffect(() => {
    setIsMounted(true);
    const savedDark = localStorage.getItem('sq_dark');
    setDark(savedDark !== null ? savedDark === 'true' : window.matchMedia('(prefers-color-scheme: dark)').matches);

    if (!localStorage.getItem('sq_has_seen_install_v2')) {
      setShowInstallPrompt(true);
    }

    setLevel(parseInt(localStorage.getItem('sq_level')) || 1);
    setXp(parseInt(localStorage.getItem('sq_xp')) || 0);
    setTotalXpEarned(parseInt(localStorage.getItem('sq_totalXpEarned')) || 0);
    setStreak(parseInt(localStorage.getItem('sq_streak')) || 0);
    setUsername(localStorage.getItem('sq_username') || null);
    setLastReset(parseInt(localStorage.getItem('sq_lastReset')) || 0);
    setProofImages(JSON.parse(localStorage.getItem('sq_proofs')) || {});
    setHistory(JSON.parse(localStorage.getItem('sq_history')) || []);

    const savedQuests = JSON.parse(localStorage.getItem('sq_quests')) || [];
    const anyDone = savedQuests.some(q => q.completed);
    const lastResetTs = parseInt(localStorage.getItem('sq_lastReset')) || 0;
    const expired = Date.now() - lastResetTs >= ONE_DAY_MS;

    if (!expired && savedQuests.length >= 5) setQuests(savedQuests);
    else if (!expired && anyDone) setQuests(savedQuests);
    else {
      const n = Math.floor(Math.random() * 3) + 5;
      setQuests([...QUEST_POOL].sort(() => 0.5 - Math.random()).slice(0, n).map(randomizeQuest));
    }
  }, []);

  useEffect(() => {
    if (isMounted) {
      document.documentElement.classList.toggle('dark', dark);
      localStorage.setItem('sq_dark', dark);
    }
  }, [dark, isMounted]);

  const handleDismissInstall = () => {
    localStorage.setItem('sq_has_seen_install_v2', 'true');
    setShowInstallPrompt(false);
  };

  const handleSetUsername = (name) => {
    localStorage.setItem('sq_username', name);
    setUsername(name);
  };

  const retryLeaderboardSync = () => {
    if (uid && username) {
      syncLeaderboardEntry(uid, { username, level, xp, totalXpEarned })
          .then(res => setLeaderboardSyncError(res.ok ? null : res.error));
    }
  };

  useEffect(() => {
    if (uid && username) {
      syncLeaderboardEntry(uid, { username, level, xp, totalXpEarned })
          .then(res => setLeaderboardSyncError(res.ok ? null : res.error));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uid, username]);

  const [timeLeft,  setTimeLeft]  = useState('--:--:--');
  const [proofModalId,   setProofModalId]   = useState(null);
  const [viewingProof, setViewingProof] = useState(null);
  const [detailQuestId,     setDetailQuestId]     = useState(null);
  const [completionQuest, setCompletionQuest] = useState(null);

  const proofModal  = quests.find(q => q.id === proofModalId)  || null;
  const detailQuest = quests.find(q => q.id === detailQuestId) || null;

  const xpRef    = useRef(xp);
  const levelRef = useRef(level);
  const totalXpEarnedRef = useRef(totalXpEarned);
  useEffect(() => { xpRef.current    = xp;    }, [xp]);
  useEffect(() => { levelRef.current = level; }, [level]);
  useEffect(() => { totalXpEarnedRef.current = totalXpEarned; }, [totalXpEarned]);

  const xpRequired = level * 100;

  const generateNewQuests = useCallback((ts) => {
    const n = Math.floor(Math.random() * 3) + 5;
    const selected = [...QUEST_POOL].sort(() => 0.5 - Math.random()).slice(0, n).map(randomizeQuest);
    setQuests(selected); setLastReset(ts); setProofImages({});
  }, []);

  useEffect(() => {
    if (!isMounted) return;
    const id = setInterval(() => {
      const now = Date.now(), remaining = ONE_DAY_MS - (now - lastReset);
      if (remaining <= 0 || quests.length === 0) { generateNewQuests(now); return; }
      const h = Math.floor((remaining / 3_600_000) % 24);
      const m = Math.floor((remaining /    60_000) % 60);
      const s = Math.floor((remaining /     1_000) % 60);
      setTimeLeft(`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`);
    }, 1000);
    return () => clearInterval(id);
  }, [lastReset, quests.length, generateNewQuests, isMounted]);

  useEffect(() => {
    if (isMounted) {
      localStorage.setItem('sq_level', level);
      localStorage.setItem('sq_xp', xp);
      localStorage.setItem('sq_totalXpEarned', totalXpEarned);
      localStorage.setItem('sq_streak', streak);
      localStorage.setItem('sq_quests', JSON.stringify(quests));
      localStorage.setItem('sq_lastReset', lastReset);
      localStorage.setItem('sq_proofs', JSON.stringify(proofImages));
      localStorage.setItem('sq_history', JSON.stringify(history));
    }
  }, [level, xp, totalXpEarned, streak, quests, lastReset, proofImages, history, isMounted]);

  const applyXpChange = useCallback((amount) => {
    const startLevel = levelRef.current;
    let newXp = xpRef.current + amount, newLevel = levelRef.current;
    while (newXp >= newLevel * 100) { newXp -= newLevel * 100; newLevel++; }
    while (newXp < 0 && newLevel > 1) { newLevel--; newXp += newLevel * 100; }
    if (newLevel === 1 && newXp < 0) newXp = 0;
    xpRef.current = newXp; levelRef.current = newLevel;
    setXp(newXp); setLevel(newLevel);

    let newTotal = totalXpEarnedRef.current;
    if (amount > 0) {
      newTotal += amount;
      totalXpEarnedRef.current = newTotal;
      setTotalXpEarned(newTotal);
    }

    return { leveledUp: newLevel > startLevel, newXp, newLevel, newTotal };
  }, []);

  const handleQuestClick = (quest) => {
    if (quest.completed) return;
    haptic(6);
    setDetailQuestId(quest.id);
  };

  const handleProofConfirm = (questId, img) => {
    const quest = quests.find(q => q.id === questId);
    if (!quest || quest.completed) { setProofModalId(null); setDetailQuestId(null); return; }
    setProofImages(prev => ({ ...prev, [questId]: img }));
    setQuests(prev => prev.map(q => q.id === questId ? { ...q, completed: true, progress: 0 } : q));
    const { leveledUp, newXp, newLevel, newTotal } = applyXpChange(quest?.xp ?? 0);

    setHistory(prev => {
      const entry = { id: `${questId}-${Date.now()}`, questId, text: quest.text, xp: quest.xp, ts: Date.now(), leveledUp };
      const next = [entry, ...prev];
      return next.length > 500 ? next.slice(0, 500) : next;
    });

    const todayStr = new Date().toDateString();
    const lastStreakDate = localStorage.getItem('sq_streak_date');
    if (lastStreakDate !== todayStr) {
      const yesterdayStr = new Date(Date.now() - ONE_DAY_MS).toDateString();
      setStreak(prev => (lastStreakDate === yesterdayStr ? prev + 1 : 1));
      localStorage.setItem('sq_streak_date', todayStr);
    }

    syncLeaderboardEntry(uid, { username, level: newLevel, xp: newXp, totalXpEarned: newTotal })
        .then(res => setLeaderboardSyncError(res.ok ? null : res.error));

    setProofModalId(null);
    setDetailQuestId(null);
    setCompletionQuest(quest ? { ...quest, leveledUp } : null);
  };

  const handleCancelProof = (progress) => {
    if (proofModalId && progress !== undefined) {
      setQuests(prev => prev.map(q => q.id === proofModalId ? { ...q, progress } : q));
    }
    setProofModalId(null);
  };

  if (!isMounted) return null;

  const c = dark ? C.dark : C.light;
  const xpPct  = Math.min(100, Math.max(0, (xp / xpRequired) * 100));
  const allDone = quests.length > 0 && quests.every(q => q.completed);

  const completedCount = quests.filter(q => q.completed).length;
  const dailyPct = quests.length ? (completedCount / quests.length) * 100 : 0;

  return (
      <div className="sq-root" style={{ background: c.bg, minHeight: '100vh', width: '100vw', display: 'flex', flexDirection: 'column', alignItems: 'center', transition: 'background-color 0.3s ease' }}>
        <SystemType />
        <div className="relative w-full flex flex-col min-h-screen" style={{ background: c.bg }}>
          <AmbientBackground c={c} />

          {isMounted && !username && <UsernameModal onSubmit={handleSetUsername} c={c} />}
          {username && showInstallPrompt && <DeviceInstallPrompt onDismiss={handleDismissInstall} c={c} />}

          {proofModal && (
              <CameraModal quest={proofModal} c={c}
                           onConfirm={img => handleProofConfirm(proofModal.id, img)}
                           onCancel={handleCancelProof} />
          )}

          {viewingProof && (
              <div className="fixed inset-0 z-[70] flex items-center justify-center p-4 sq-anim-in"
                   style={{ background: 'rgba(0,0,0,0.85)' }}
                   onClick={() => setViewingProof(null)}>
                <img src={viewingProof} alt="proof" className="max-w-full max-h-full rounded-2xl" />
              </div>
          )}

          {showLeaderboard ? (
              <LeaderboardScreen dark={dark} c={c} onBack={() => setShowLeaderboard(false)}
                                 myUid={uid} myUsername={username} myLevel={level} myXp={xp}
                                 syncError={leaderboardSyncError} onRetrySync={retryLeaderboardSync} />
          ) : completionQuest ? (
              <CompletionScreen quest={completionQuest} dark={dark} c={c} timeLeft={timeLeft}
                                onToggleTheme={() => setDark(d => !d)} onBack={() => setCompletionQuest(null)} />
          ) : detailQuest ? (
              <QuestDetailScreen quest={detailQuest} dark={dark} c={c} timeLeft={timeLeft}
                                 onToggleTheme={() => setDark(d => !d)} onBack={() => setDetailQuestId(null)}
                                 onMarkComplete={() => setProofModalId(detailQuest.id)} />
          ) : activeTab === 'history' ? (
              <HistoryScreen history={history} dark={dark} c={c} onToggleTheme={() => setDark(d => !d)} />
          ) : (
              <div className="relative z-10 flex flex-col flex-1 sq-anim-in">
                {/* Header */}
                <div className="px-4 pb-2" style={{ paddingTop: 'max(env(safe-area-inset-top), 16px)' }}>
                  <div className="flex items-center justify-between">
                    <h1 className="sq-large-title" style={{ fontSize: 32, fontWeight: 800, color: c.label }}>QuestDaily</h1>
                    <div className="flex items-center gap-2">
                      <button onClick={() => setShowLeaderboard(true)} aria-label="Leaderboard"
                              style={{ ...glassStyle(c), width: 32, height: 32, borderRadius: 999, display: 'flex', alignItems: 'center', justifyContent: 'center', color: c.blue }}>
                        <span className="sq-icon-tap"><TrophyIcon size={15} /></span>
                      </button>
                      <button onClick={() => setDark(d => !d)} aria-label="Toggle theme"
                              style={{ ...glassStyle(c), width: 32, height: 32, borderRadius: 999, display: 'flex', alignItems: 'center', justifyContent: 'center', color: c.label }}>
                        <span className="sq-icon-tap">{dark ? <Sun key="sun" size={16} strokeWidth={1.75} className="sq-icon-pop-in" /> : <Moon key="moon" size={16} strokeWidth={1.75} className="sq-icon-pop-in" />}</span>
                      </button>
                    </div>
                  </div>

                  <div className="mt-3 flex items-center gap-2.5">
                <span className="sq-mono" style={{ fontSize: 12, fontWeight: 700, color: '#fff', background: c.blue, padding: '3px 9px', borderRadius: 8 }}>
                  LV {level}
                </span>
                    <span style={{ fontSize: 12, fontWeight: 500, color: c.labelSecondary }}>{getLevelTitle(level)}</span>
                    <span className="sq-mono ml-auto" style={{ fontSize: 11, fontWeight: 600, color: c.labelTertiary }}>{xp} / {xpRequired} XP</span>
                  </div>
                  <div style={{ marginTop: 8, height: 5, width: '100%', borderRadius: 999, background: c.fill, overflow: 'hidden' }}>
                    <div style={{ height: '100%', borderRadius: 999, width: `${xpPct}%`, background: c.blue, transition: 'width 0.6s cubic-bezier(0.22,1,0.36,1)' }} />
                  </div>
                </div>

                <div className="flex-1 sq-scroll overflow-y-auto px-4 pt-2 space-y-4" style={{ paddingBottom: 'calc(100px + env(safe-area-inset-bottom))' }}>
                  <div style={{ ...glassStyle(c), borderRadius: 20, padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 14 }} className="sq-anim-in">
                    <ProgressRing pct={dailyPct} c={c} size={44} />
                    <div className="flex-1" style={{ minWidth: 0 }}>
                      <p style={{ fontSize: 14, fontWeight: 600, color: c.label }}>Daily progress</p>
                      <p style={{ fontSize: 12, marginTop: 2, color: c.labelSecondary }}>{completedCount} of {quests.length} quests done</p>
                    </div>
                    {streak > 0 && (
                        <div className="flex flex-col items-center flex-shrink-0" style={{ paddingLeft: 12, borderLeft: `1px solid ${c.separator}` }}>
                          <QuestSvg.flame width={15} height={15} strokeWidth={1.75} style={{ color: c.orange }} />
                          <span className="sq-mono" style={{ fontSize: 12, fontWeight: 700, color: c.orange, marginTop: 2 }}>{streak}d</span>
                        </div>
                    )}
                  </div>

                  <div>
                    <p style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, color: c.labelSecondary, marginBottom: 8, paddingLeft: 2 }}>
                      Today
                    </p>
                    <div className="flex flex-col gap-2.5">
                      {quests.map((quest, i) => (
                          <button key={quest.id}
                                  onClick={() => { if (!quest.completed) handleQuestClick(quest); }}
                                  className="sq-anim-in"
                                  style={{
                                    ...glassStyle(c),
                                    width: '100%', display: 'flex', alignItems: 'center', gap: 12,
                                    padding: '14px 14px', textAlign: 'left', borderRadius: 20,
                                    opacity: quest.completed ? 0.7 : 1,
                                    animationDelay: `${Math.min(i, 8) * 0.035}s`,
                                  }}>
                            <div className="relative flex-shrink-0">
                              <QuestRowIcon questId={quest.id} c={c} muted={quest.completed} />
                              {quest.completed && (
                                  <div style={{ position: 'absolute', bottom: -3, right: -3, width: 16, height: 16, borderRadius: '50%', background: c.green, border: `2px solid ${c.bgElevated}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <CheckIcon size={8} />
                                    <style>{`div > svg { color: #fff; }`}</style>
                                  </div>
                              )}
                            </div>
                            <span style={{ flex: 1, fontSize: 15, fontWeight: 500, lineHeight: 1.3, color: quest.completed ? c.labelTertiary : c.label, textDecoration: quest.completed ? 'line-through' : 'none' }}>
                        {quest.text}
                              {!quest.completed && quest.progress > 0 && (
                                  <span className="sq-mono" style={{ marginLeft: 8, fontSize: 10, fontWeight: 700, color: c.blue, background: c.fill, padding: '2px 7px', borderRadius: 999 }}>
                            In progress
                          </span>
                              )}
                      </span>
                            <div className="flex items-center gap-2 flex-shrink-0">
                              {quest.completed && proofImages[quest.id] && (
                                  <button onClick={e => { e.stopPropagation(); setViewingProof(proofImages[quest.id]); }}
                                          style={{ width: 26, height: 26, borderRadius: 10, overflow: 'hidden', flexShrink: 0 }}>
                                    <img src={proofImages[quest.id]} alt="proof" className="w-full h-full object-cover" />
                                  </button>
                              )}
                              <span className="sq-mono" style={{ fontSize: 12, fontWeight: 700, color: quest.completed ? c.green : c.labelTertiary }}>+{quest.xp}</span>
                              {!quest.completed && <span className="sq-icon-tap"><ChevronIcon color={c.labelTertiary} /></span>}
                            </div>
                          </button>
                      ))}
                    </div>
                  </div>

                  {allDone && (
                      <div style={{ ...glassStyle(c), borderRadius: 20, padding: 24, textAlign: 'center' }} className="sq-anim-in">
                        <TrophyIcon size={26} style={{ color: c.orange, margin: '0 auto 10px' }} />
                        <p style={{ fontSize: 15, fontWeight: 600, color: c.label }}>All quests complete</p>
                        <p style={{ fontSize: 13, marginTop: 4, color: c.labelSecondary }}>Rest up. New quests when the timer hits zero.</p>
                      </div>
                  )}

                  {quests.length > 0 && (
                      <p style={{ fontSize: 11, fontWeight: 500, textAlign: 'center', color: c.labelTertiary, padding: '0 16px 8px' }}>
                        Checked on your device — nothing you record ever leaves your phone.
                      </p>
                  )}
                </div>
              </div>
          )}

          {isMounted && username && !showLeaderboard && !completionQuest && !detailQuest && !proofModal && (
              <TabBar activeTab={activeTab} onChange={setActiveTab} c={c} />
          )}
        </div>
      </div>
  );
}
