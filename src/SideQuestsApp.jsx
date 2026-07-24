"use client";

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { pipeline, env } from '@huggingface/transformers';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';

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
      classifierPromise = null;
      throw error;
    });
  }
  return classifierPromise;
}

// ─── POSE MODEL SETUP (real rep counting via joint tracking) ────────────────
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

// Metric extractors: return an angle (degrees) or normalized position we track
// through a down/up state machine per exercise.
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

// ─── QUEST LABELS & PROFILES ─────────────────────────────────────────────────
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
  q8:  { type: 'action', activity: ['person stretching muscles', 'yoga stretch pose'], label: 'stretching' },
  q10: { type: 'action', activity: ['person sleeping in bed', 'person resting in bed eyes closed'], label: 'getting good sleep' },
  q11: { type: 'action', activity: ['person doing jumping jacks or burpees'], label: 'doing cardio' },
  q13: { type: 'action', activity: ['handwriting in notebook or journal'], label: 'journaling' },
  q15: { type: 'action', activity: ['person doing plank exercise', 'plank position core exercise'], label: 'holding a plank' },
  q18: { type: 'action', activity: ['shower running water', 'bathroom shower head with water'], label: 'in the shower' },
  q21: { type: 'action', activity: ['person breathing deeply eyes closed'], label: 'deep breathing' },
  q24: { type: 'action', activity: ['person sleeping in bed at night', 'sleeping in dark bedroom'], label: 'sleeping early' },
  q25: { type: 'action', activity: ['person in ice bath tub', 'cold plunge tub with ice'], label: 'in a cold plunge' },
  q26: { type: 'action', activity: ['person doing a wall sit exercise against a wall'], label: 'holding a wall sit' },
  q27: { type: 'action', activity: ['person doing a glute bridge exercise on floor'], label: 'holding a glute bridge' },
  q28: { type: 'action', activity: ['person doing high knees exercise'], label: 'doing high knees' },
  q29: { type: 'action', activity: ['person doing mountain climbers exercise'], label: 'doing mountain climbers' },
  q30: { type: 'action', activity: ['person doing a superman back exercise lying face down'], label: 'holding a superman pose' },
  q31: { type: 'action', activity: ['person doing burpees exercise'], label: 'doing burpees' },
};

const getNegativeLabels = (type) => {
  const base = ['person sitting doing nothing', 'person standing still straight', 'random everyday object'];
  if (type === 'map') return [...base, 'sweaty selfie face', 'picture of running shoes', 'treadmill machine indoors', 'person running outside'];
  if (type === 'food') return [...base, 'empty plate or bowl', 'restaurant paper menu', 'store product barcode', 'person eating face'];
  return [...base, 'phone or computer screen'];
};

// Every quest is either reps-based or duration-based so it can be timeboxed,
// progress-saved, and given a randomized amount each time it's assigned.
// textTemplate uses "{n}" as a placeholder for the randomized reps/minutes.
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

// Randomizes a fresh copy of a pool quest: reps get a random count 1-55,
// durations get a random length between 1 and 5 minutes, and the display
// text is filled in with that same random number so they always match.
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
const REQUIRED_PASSES = 2;
const PASS_THRESHOLD  = 0.35;

function getMaxLabelScore(results, candidates) {
  if (!candidates?.length) return 0;
  return results.reduce((best, result) => (candidates.includes(result.label) ? Math.max(best, result.score) : best), 0);
}

// ─── ICONS ───────────────────────────────────────────────────────────────────
const LogoIcon = ({ size = 34, dark }) => (
  <img src="/logo-transparent.png" alt="QuestDaily" width={size} height={size}
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

const IOSShareIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/>
    <polyline points="16 6 12 2 8 6"/>
    <line x1="12" y1="2" x2="12" y2="15"/>
  </svg>
);

const IOSAddIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="4" ry="4"/>
    <line x1="12" y1="8" x2="12" y2="16"/>
    <line x1="8" y1="12" x2="16" y2="12"/>
  </svg>
);

// ─── AMBIENT ANIMATED BACKGROUND ───────────────────────────────────────────────
const AnimatedBackground = ({ dark }) => {
  const vars = {
    '--sq-streak-c1': dark ? 'rgba(139,92,246,0.55)' : 'rgba(99,102,241,0.30)',
    '--sq-streak-c2': dark ? 'rgba(99,102,241,0.35)' : 'rgba(168,85,247,0.20)',
    '--sq-glow-1':    dark ? '#7c3aed' : '#c7d2fe',
    '--sq-glow-2':    dark ? '#4f46e5' : '#e0d4fc',
    '--sq-glow-o':    dark ? 0.28 : 0.55,
  };
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden z-0" style={vars}>
      <div className="sq-streak sq-streak-1" />
      <div className="sq-streak sq-streak-2" />
      <div className="sq-streak sq-streak-3" />
      <div className="sq-glow sq-glow-1" />
      <div className="sq-glow sq-glow-2" />
      <style>{`
        .sq-streak { position: absolute; width: 180%; height: 1.5px; left: -40%; background: linear-gradient(90deg, transparent, var(--sq-streak-c1), var(--sq-streak-c2), transparent); transform-origin: center; filter: blur(0.5px); opacity: 0.7; }
        .sq-streak-1 { top: 14%;  transform: rotate(-18deg); animation: sq-drift-a 9s ease-in-out infinite; }
        .sq-streak-2 { top: 42%;  transform: rotate(-12deg); animation: sq-drift-b 13s ease-in-out infinite; opacity: 0.45; }
        .sq-streak-3 { top: 68%;  transform: rotate(-22deg); animation: sq-drift-a 11s ease-in-out infinite reverse; opacity: 0.35; }
        @keyframes sq-drift-a { 0% { transform: translateX(-6%) rotate(-18deg); opacity: 0.35; } 50% { transform: translateX(6%)  rotate(-16deg); opacity: 0.8; } 100% { transform: translateX(-6%) rotate(-18deg); opacity: 0.35; } }
        @keyframes sq-drift-b { 0% { transform: translateX(5%)  rotate(-12deg); opacity: 0.25; } 50% { transform: translateX(-5%) rotate(-10deg); opacity: 0.55; } 100% { transform: translateX(5%)  rotate(-12deg); opacity: 0.25; } }
        .sq-glow { position: absolute; width: 260px; height: 260px; border-radius: 999px; filter: blur(70px); opacity: var(--sq-glow-o); }
        .sq-glow-1 { top: -60px; left: -60px; background: var(--sq-glow-1); animation: sq-float 10s ease-in-out infinite; }
        .sq-glow-2 { bottom: -80px; right: -60px; background: var(--sq-glow-2); animation: sq-float 12s ease-in-out infinite reverse; }
        @keyframes sq-float { 0%, 100% { transform: translate(0,0) scale(1); } 50% { transform: translate(20px, -15px) scale(1.15); } }
        @keyframes sq-pop-in { 0% { opacity: 0; transform: scale(0.9) translateY(8px); } 100% { opacity: 1; transform: scale(1) translateY(0); } }
        @keyframes sq-check-in { 0% { opacity: 0; transform: scale(0.4); } 60% { opacity: 1; transform: scale(1.15); } 100% { opacity: 1; transform: scale(1); } }
        @keyframes sq-icon-float { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-4px); } }
        .sq-anim-pop { animation: sq-pop-in 0.45s cubic-bezier(0.22, 1, 0.36, 1) both; }
        .sq-anim-check { animation: sq-check-in 0.55s cubic-bezier(0.22, 1, 0.36, 1) both; }
        .sq-anim-float { animation: sq-icon-float 3.2s ease-in-out infinite; }
      `}</style>
    </div>
  );
};

// ─── QUEST THEMES & CONFIG ───────────────────────────────────────────────────
const QuestSvg = {
  cup: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M7 3h10l-1 12a4 4 0 0 1-4 4h0a4 4 0 0 1-4-4L7 3Z"/><path d="M9 3v3"/><path d="M15 3v3"/></svg>),
  smoothie: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M8 2h8l-1 3H9L8 2Z"/><path d="M7 5h10l-1.2 14.2A2 2 0 0 1 13.8 21h-3.6a2 2 0 0 1-2-1.8L7 5Z"/><path d="M7.6 11h8.8"/></svg>),
  dumbbell: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M4 9v6"/><path d="M2 10v4"/><path d="M20 9v6"/><path d="M22 10v4"/><path d="M6.5 8v8"/><path d="M17.5 8v8"/><path d="M6.5 12h11"/></svg>),
  run: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="13" cy="4" r="1.6"/><path d="M9.5 21l2-5 2.2 1.8L16 21"/><path d="M6 14l3-3 3 1 3.5-3.5"/><path d="M9 11 7 8"/></svg>),
  bike: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="6" cy="17" r="3.2"/><circle cx="18" cy="17" r="3.2"/><path d="M6 17l4-8h4l3 8"/><path d="M10 9h4"/><path d="M13 5h3l2 4"/></svg>),
  footsteps: (p) => (<svg {...p} viewBox="0 0 24 24" fill="currentColor" stroke="none"><ellipse cx="8.3" cy="15.6" rx="2.3" ry="3.5" transform="rotate(-12 8.3 15.6)"/><circle cx="6.6" cy="11" r="0.95"/><circle cx="8" cy="10.1" r="0.95"/><circle cx="9.4" cy="10.3" r="0.9"/><circle cx="10.6" cy="11" r="0.8"/><ellipse cx="15.9" cy="8.6" rx="2.3" ry="3.5" transform="rotate(10 15.9 8.6)"/><circle cx="14.1" cy="4" r="0.95"/><circle cx="15.5" cy="3.2" r="0.95"/><circle cx="16.9" cy="3.4" r="0.9"/><circle cx="18.1" cy="4.1" r="0.8"/></svg>),
  legs: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="4" r="1.6"/><path d="M12 6v5"/><path d="M12 11 8 14v7"/><path d="M12 11l4 3v7"/><path d="M8 21h2"/><path d="M14 21h2"/></svg>),
  core: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="7" y="3.5" width="10" height="17" rx="4.5"/><path d="M12 3.5v17"/><path d="M7.5 9.5h9"/><path d="M7.5 14.5h9"/></svg>),
  stopwatch: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="13.5" r="7.2"/><path d="M12 13.5V9.2"/><path d="M9.5 2.5h5"/><path d="M18 5.5l1.4-1.4"/></svg>),
  bar: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M4 5h16"/><circle cx="12" cy="9" r="1.6"/><path d="M12 10.6v4"/><path d="M9 6.5l2 3"/><path d="M15 6.5l-2 3"/><path d="M10 14.6l-1.6 4"/><path d="M14 14.6l1.6 4"/></svg>),
  droplet: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3s6 6.5 6 10.5a6 6 0 0 1-12 0C6 9.5 12 3 12 3Z"/></svg>),
  leaf: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M20 5c-9 0-15 6-15 15 9 0 15-6 15-15Z"/><path d="M6 19 18 6"/></svg>),
  bowl: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M4 12h16a8 6 0 0 1-16 0Z"/><path d="M12 12V5"/><path d="M9 7l3-2 3 2"/></svg>),
  pot: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M4 11h16v3a5 5 0 0 1-5 5H9a5 5 0 0 1-5-5v-3Z"/><path d="M2 11h20"/><path d="M6 11V8h12v3"/></svg>),
  moon: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z"/></svg>),
  lotus: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 21c-4-1-6-3.5-6-7 3 0 6 1.5 6 5 0-3.5 3-5 6-5 0 3.5-2 6-6 7Z"/><path d="M12 21V9"/><path d="M8 9c0-3 2-6 4-7 2 1 4 4 4 7"/></svg>),
  stretch: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="4" r="1.6"/><path d="M12 6v5"/><path d="M6 8l6 3 6-3"/><path d="M12 11l-3 9"/><path d="M12 11l3 9"/></svg>),
  bolt: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z"/></svg>),
  flame: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22c4 0 7-2.7 7-6.5 0-3-2-5-3-7-.3 2-1.5 3-2.5 2.2.7-2.3-.2-4.7-2-6.2C11 7 8 9 8 13c-1-.6-1.5-1.8-1.5-3.2C5.3 11.2 5 13 5 15.2 5 19 8 22 12 22Z"/></svg>),
  pencil: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M4 20h4L18.5 9.5a2.1 2.1 0 0 0-3-3L5 17v3Z"/><path d="M14 6l4 4"/></svg>),
  snowflake: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v20M4.5 7l15 10M19.5 7l-15 10"/></svg>),
  wind: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 8h11a2.5 2.5 0 1 0-2.5-2.5"/><path d="M3 13h15a2.5 2.5 0 1 1-2.5 2.5"/><path d="M3 18h9a2 0 1 0-2-2"/></svg>),
  rope: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M4 20c4-8 12-8 16 0"/><path d="M4 4c4 8 12 8 16 0"/></svg>),
  target: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="0.6" fill="currentColor"/></svg>),
};

const QUEST_THEME = {
  q1:  { icon: 'dumbbell', grad: 'from-fuchsia-500 to-purple-600' }, q2:  { icon: 'legs',     grad: 'from-violet-500 to-indigo-600' },
  q3:  { icon: 'run',      grad: 'from-orange-400 to-rose-500' },    q4:  { icon: 'bar',      grad: 'from-purple-500 to-blue-600' },
  q5:  { icon: 'footsteps',grad: 'from-teal-400 to-cyan-600' },      q6:  { icon: 'cup',      grad: 'from-indigo-400 to-violet-600' },
  q7:  { icon: 'lotus',    grad: 'from-emerald-400 to-teal-600' },   q8:  { icon: 'stretch',  grad: 'from-sky-400 to-indigo-600' },
  q9:  { icon: 'bowl',     grad: 'from-lime-400 to-emerald-600' },   q10: { icon: 'moon',     grad: 'from-indigo-500 to-slate-700' },
  q11: { icon: 'bolt',     grad: 'from-yellow-400 to-orange-600' },  q12: { icon: 'core',     grad: 'from-rose-500 to-red-600' },
  q13: { icon: 'pencil',   grad: 'from-amber-400 to-orange-600' },   q14: { icon: 'smoothie', grad: 'from-green-400 to-emerald-600' },
  q15: { icon: 'stopwatch',grad: 'from-cyan-400 to-blue-600' },      q16: { icon: 'bike',     grad: 'from-blue-400 to-indigo-600' },
  q17: { icon: 'rope',     grad: 'from-pink-500 to-fuchsia-600' },   q18: { icon: 'droplet',  grad: 'from-sky-400 to-blue-600' },
  q19: { icon: 'leaf',     grad: 'from-emerald-400 to-green-600' },  q20: { icon: 'pot',      grad: 'from-orange-400 to-amber-600' },
  q21: { icon: 'wind',     grad: 'from-cyan-300 to-teal-600' },      q22: { icon: 'footsteps',grad: 'from-violet-400 to-purple-600' },
  q23: { icon: 'legs',     grad: 'from-fuchsia-500 to-rose-600' },   q24: { icon: 'moon',     grad: 'from-indigo-400 to-blue-700' },
  q25: { icon: 'snowflake',grad: 'from-cyan-300 to-blue-600' },
  q26: { icon: 'legs',     grad: 'from-amber-500 to-orange-600' },  q27: { icon: 'core',     grad: 'from-rose-400 to-pink-600' },
  q28: { icon: 'bolt',     grad: 'from-yellow-400 to-red-500' },    q29: { icon: 'core',     grad: 'from-blue-500 to-cyan-600' },
  q30: { icon: 'stretch',  grad: 'from-indigo-400 to-purple-600' }, q31: { icon: 'flame',    grad: 'from-orange-500 to-red-600' },
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

const QuestIconBadge = ({ questId, size = 96, dark, floating = false }) => {
  const theme = QUEST_THEME[questId] || { icon: 'target', grad: 'from-indigo-500 to-purple-600' };
  const Icon = QuestSvg[theme.icon] || QuestSvg.target;
  return (
    <div className={`relative flex items-center justify-center rounded-full bg-gradient-to-br ${theme.grad} ${floating ? 'sq-anim-float' : ''}`}
      style={{ width: size, height: size, boxShadow: `0 0 0 1px rgba(255,255,255,0.15) inset, 0 8px 30px -8px rgba(139,92,246,0.65)` }}>
      <div className="absolute inset-[3px] rounded-full border border-white/25" />
      <Icon width={Math.round(size * 0.42)} height={Math.round(size * 0.42)} className="text-white relative z-10" />
    </div>
  );
};

const ProgressRing = ({ pct, size = 56, stroke = 5, dark }) => {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const offset = c - (pct / 100) * c;
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={dark ? '#27272a' : '#e5e7eb'} strokeWidth={stroke} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="url(#sq-ring-gradient)" strokeWidth={stroke} strokeLinecap="round" strokeDasharray={c} style={{ strokeDashoffset: offset, transition: 'stroke-dashoffset 0.8s cubic-bezier(0.22,1,0.36,1)' }} />
        <defs>
          <linearGradient id="sq-ring-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#818cf8" />
            <stop offset="100%" stopColor="#c084fc" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className={`text-[11px] font-bold ${dark ? 'text-white' : 'text-gray-900'}`}>{Math.round(pct)}%</span>
      </div>
    </div>
  );
};

// ─── INSTALL PROMPT ONBOARDING MODAL ──────────────────────────────────────────
function IOSInstallPrompt({ onDismiss, dark }) {
  const [step, setStep] = useState(1);
  const cardBg = dark ? 'bg-zinc-900' : 'bg-white';
  const txt = dark ? 'text-white' : 'text-gray-900';
  const sub = dark ? 'text-zinc-400' : 'text-gray-500';
  const pill = dark ? 'bg-zinc-800' : 'bg-gray-100';

  const nextStep = () => {
    if (step === 1) setStep(2);
    else onDismiss();
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sq-anim-pop" style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}>
      <div className={`w-full max-w-sm rounded-[24px] ${cardBg} shadow-2xl p-6 text-center border ${dark ? 'border-zinc-800' : 'border-gray-100'}`}>
        <h3 className={`text-xl font-bold mb-2 ${txt}`}>Install QuestDaily</h3>
        <p className={`text-sm mb-6 ${sub}`}>Add this app to your home screen for the full experience.</p>
        
        <div className={`relative mb-8 ${pill} rounded-2xl p-5 flex flex-col items-center justify-center border ${dark ? 'border-white/5' : 'border-black/5'}`}>
          <div className="flex items-center justify-center w-14 h-14 rounded-xl bg-[#007AFF] text-white shadow-lg mb-3">
            {step === 1 ? <IOSShareIcon /> : <IOSAddIcon />}
          </div>
          <p className={`text-sm font-semibold ${txt} mb-1`}>
            {step === 1 ? 'Step 1: Tap the Share button' : 'Step 2: Add to Home Screen'}
          </p>
          <p className={`text-xs ${sub}`}>
            {step === 1 ? 'Look for the share icon in your Safari menu bar below.' : 'Scroll down the share menu and select this option.'}
          </p>
        </div>

        <div className="flex gap-3">
          {step === 2 && (
            <button onClick={() => setStep(1)} className={`flex-1 py-3.5 rounded-[14px] text-sm font-semibold ${pill} ${txt}`}>
              Back
            </button>
          )}
          <button onClick={nextStep} className="flex-[2] py-3.5 rounded-[14px] text-sm font-semibold bg-[#007AFF] text-white shadow-lg active:scale-95 transition-transform">
            {step === 1 ? 'Next' : 'Got it!'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── QUEST DETAIL SCREEN ────────────────────────────────────────────────────────
function QuestDetailScreen({ quest, dark, onToggleTheme, timeLeft, onBack, onMarkComplete }) {
  const txt = dark ? 'text-white' : 'text-gray-900';
  const sub = dark ? 'text-zinc-400' : 'text-gray-500';
  const pill = dark ? 'bg-zinc-800/80' : 'bg-gray-100';
  const cardBg = dark ? 'bg-zinc-900/70' : 'bg-white';
  const theme = QUEST_THEME[quest.id] || { grad: 'from-indigo-500 to-purple-600' };
  const about = QUEST_ABOUT[quest.id] || 'Stay consistent — every quest you complete adds up to real progress.';

  return (
    <div className="sq-anim-pop relative z-10 min-h-screen flex flex-col">
      <div className="flex items-center justify-between px-4 pb-3" style={{ paddingTop: 'max(env(safe-area-inset-top), 18px)' }}>
        <button onClick={onBack}
          className={`w-9 h-9 rounded-full flex items-center justify-center ${dark ? 'bg-zinc-800/80 text-zinc-300' : 'bg-gray-100 text-gray-600'} active:opacity-70 transition-colors`}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
        </button>
        <div className={`flex items-center gap-1 px-2.5 py-1 rounded-full ${pill}`}>
          <span className="text-[10px]">⏱</span>
          <span className={`text-[11px] font-mono font-medium ${dark ? 'text-zinc-300' : 'text-gray-600'}`}>{timeLeft}</span>
        </div>
        <button onClick={onToggleTheme}
          className={`w-9 h-9 rounded-full flex items-center justify-center ${dark ? 'bg-zinc-800/80 text-zinc-300' : 'bg-gray-100 text-gray-600'} active:opacity-70 transition-colors`}>
          {dark ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>

      <div className="px-4">
        <div className={`relative overflow-hidden rounded-[26px] px-6 pt-10 pb-8 flex flex-col items-center text-center bg-gradient-to-b ${dark ? 'from-[#1c1530] to-[#0d0a17]' : 'from-indigo-50 to-white'} border ${dark ? 'border-white/10' : 'border-gray-100'}`}>
          <AnimatedBackground dark={dark} />
          <div className="relative z-10 flex flex-col items-center">
            <QuestIconBadge questId={quest.id} size={92} dark={dark} floating />
            <h2 className={`mt-5 text-[22px] font-bold leading-tight max-w-[240px] ${txt}`}>{quest.text}</h2>
            <span className={`mt-3 px-3 py-1 rounded-full text-xs font-semibold bg-gradient-to-r ${theme.grad} text-white`}>
              +{quest.xp} XP
            </span>
          </div>
        </div>
      </div>

      <div className="px-4 mt-5 flex-1">
        <p className={`text-[11px] font-semibold uppercase tracking-widest mb-1.5 px-1 ${dark ? 'text-zinc-500' : 'text-gray-400'}`}>About this quest</p>
        <div className={`${cardBg} rounded-[16px] p-4 border ${dark ? 'border-white/5' : 'border-gray-100'}`}>
          <p className={`text-[14px] leading-relaxed ${sub}`}>{about}</p>
        </div>
      </div>

      <div className="px-4 pb-8 mt-6">
        <button onClick={onMarkComplete}
          className={`w-full py-4 rounded-[16px] text-[15px] font-semibold text-white bg-gradient-to-r ${theme.grad} shadow-lg active:scale-[0.97] transition-transform`}
          style={{ boxShadow: '0 10px 30px -10px rgba(139,92,246,0.6)' }}>
          Start Challenge
        </button>
      </div>
    </div>
  );
}

// ─── COMPLETION SCREEN ───────────────────────────────────────────────────────────
function CompletionScreen({ quest, dark, onToggleTheme, timeLeft, onBack }) {
  const txt = dark ? 'text-white' : 'text-gray-900';
  const sub = dark ? 'text-zinc-400' : 'text-gray-500';
  const pill = dark ? 'bg-zinc-800/80' : 'bg-gray-100';
  const cardBg = dark ? 'bg-zinc-900/70' : 'bg-white';
  const quote = useMemo(() => QUEST_QUOTE[quest?.id] || QUEST_QUOTES[Math.floor(Math.random() * QUEST_QUOTES.length)], [quest?.id]);

  return (
    <div className="sq-anim-pop relative z-10 min-h-screen flex flex-col">
      <div className="flex items-center justify-between px-4 pb-3" style={{ paddingTop: 'max(env(safe-area-inset-top), 18px)' }}>
        <button onClick={onBack}
          className={`w-9 h-9 rounded-full flex items-center justify-center ${dark ? 'bg-zinc-800/80 text-zinc-300' : 'bg-gray-100 text-gray-600'} active:opacity-70 transition-colors`}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
        </button>
        <div className={`flex items-center gap-1 px-2.5 py-1 rounded-full ${pill}`}>
          <span className="text-[10px]">⏱</span>
          <span className={`text-[11px] font-mono font-medium ${dark ? 'text-zinc-300' : 'text-gray-600'}`}>{timeLeft}</span>
        </div>
        <button onClick={onToggleTheme}
          className={`w-9 h-9 rounded-full flex items-center justify-center ${dark ? 'bg-zinc-800/80 text-zinc-300' : 'bg-gray-100 text-gray-600'} active:opacity-70 transition-colors`}>
          {dark ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>

      <div className="px-4">
        <div className={`relative overflow-hidden rounded-[26px] px-6 pt-12 pb-10 flex flex-col items-center text-center bg-gradient-to-b ${dark ? 'from-[#0e2318] to-[#081712]' : 'from-emerald-50 to-white'} border ${dark ? 'border-emerald-500/20' : 'border-emerald-100'}`}>
          <div className="pointer-events-none absolute inset-0 overflow-hidden">
            <div className="absolute -top-10 -left-10 w-40 h-40 rounded-full bg-emerald-500/25 blur-3xl" />
            <div className="absolute -bottom-14 -right-10 w-48 h-48 rounded-full bg-green-400/20 blur-3xl" />
          </div>
          <div className="relative z-10 flex flex-col items-center">
            <div className="sq-anim-check w-24 h-24 rounded-full flex items-center justify-center border-2 border-emerald-400"
              style={{ boxShadow: '0 0 40px -6px rgba(52,211,153,0.55)' }}>
              <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12"/>
              </svg>
            </div>
            <h2 className={`mt-5 text-[22px] font-bold ${txt}`}>Quest Completed!</h2>
            <span className="mt-3 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/90 text-white">
              +{quest?.xp ?? 0} XP
            </span>
          </div>
        </div>
      </div>

      <div className="px-4 mt-5 flex-1">
        <div className={`${cardBg} rounded-[16px] p-5 border ${dark ? 'border-white/5' : 'border-gray-100'} relative`}>
          <span className={`absolute top-2 left-3 text-3xl leading-none ${dark ? 'text-zinc-700' : 'text-gray-200'}`}>&ldquo;</span>
          <p className={`text-[15px] font-medium text-center leading-relaxed px-3 ${txt}`}>{quote}</p>
          <span className={`absolute bottom-1 right-3 text-3xl leading-none ${dark ? 'text-zinc-700' : 'text-gray-200'}`}>&rdquo;</span>
        </div>
      </div>

      <div className="px-4 pb-8 mt-6">
        <button onClick={onBack}
          className={`w-full py-4 rounded-[16px] text-[15px] font-semibold ${dark ? 'bg-zinc-800 text-white' : 'bg-gray-100 text-gray-800'} active:opacity-70`}>
          Back to Quests
        </button>
      </div>
    </div>
  );
}

// ─── CAMERA / AI MODAL ────────────────────────────────────────────────────────
function CameraModal({ quest, onConfirm, onCancel, dark }) {
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
  const [lastLabel,     setLastLabel]     = useState('');
  const [uploadedProof, setUploadedProof] = useState(null);

  const [secondsLeft, setSecondsLeft] = useState(quest.duration ? (quest.progress ?? quest.duration) : 0);
  const [timerRunning, setTimerRunning] = useState(false);

  const labels = QUEST_LABELS[quest.id];
  const questType = labels?.type || 'action';

  const activeNegatives = useMemo(() => getNegativeLabels(questType), [questType]);
  const classifierLabels = useMemo(
    () => [...new Set([...(labels?.activity ?? []), ...activeNegatives])],
    [labels, activeNegatives]
  );

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

  // CLIP model — only needed for food / map / action verification, not reps.
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

  const resetRepTracking = useCallback(() => {
    const baseline = quest.reps ? (quest.progress || 0) : 0;
    poseStateRef.current = createPoseRepState(baseline);
    setRepsDone(baseline);
    setRepPhase('up');
    setRepCue('Get in frame');
  }, [quest.reps, quest.progress]);

  // CLIP scanning loop — food / map / action quests only. Reps use pose tracking below.
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
        
        // Strict letterbox cropping for full body frame capturing
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
        setLastLabel(results[0]?.label ?? '');

        setLiveScore(Math.round(actScore * 100));
        const passed = actScore > negScore && actScore >= PASS_THRESHOLD;
        passStreakRef.current = passed ? passStreakRef.current + 1 : 0;
        setPassStreak(passStreakRef.current);
        if (passStreakRef.current >= REQUIRED_PASSES) { confirmedRef.current = true; setConfirmed(true); }
      } catch (err) { 
        consecutiveErrors += 1;
        if (consecutiveErrors >= 3 && isCurrentRun()) setModelError('AI analysis is temporarily unavailable. Try closing and reopening the camera.');
      } finally { 
        isScanningRef.current = false;
        if (isCurrentRun()) { setScanning(false); scheduleNext(consecutiveErrors ? 1000 : 1200); }
      }
    };
    scanLoop();
    return () => { disposed = true; scanRunRef.current += 1; clearTimeout(scanTimerRef.current); };
  }, [phase, modelReady, confirmed, uploading, labels, classifierLabels, quest.reps, activeNegatives]);

  // Pose tracking & Skeleton HUD loop — reps challenges only. Uses joint-angle detection and HUD drawing.
  useEffect(() => {
    if (!quest.reps || phase !== 'live' || confirmed) return undefined;
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
              if (landmarks) {
                const update = updatePoseRepState(poseStateRef.current, quest.id, landmarks, now);
                setRepPhase(update.phase);
                setRepCue(update.cue);
                if (update.counted) {
                  setRepsDone(update.reps);
                  if (update.reps >= quest.reps) {
                    confirmedRef.current = true;
                    setConfirmed(true);
                  }
                }
              } else {
                setRepCue('Step back so your full body is visible');
              }

              // ── SKELETON HUD OVERLAY ──
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

                  const mapPt = (pt) => ({
                    x: Ox + pt.x * Wr,
                    y: Oy + pt.y * Hr,
                    vis: pt.visibility ?? 1
                  });

                  const mapped = landmarks.map(mapPt);
                  const currentPhase = poseStateRef.current.phase;
                  const lineColor = currentPhase === 'down' ? '#4ade80' : '#38bdf8'; // Neon green when lowered into pushup/squat, cyan when up
                  const shadowColor = currentPhase === 'down' ? 'rgba(74, 222, 128, 0.8)' : 'rgba(56, 189, 248, 0.8)';

                  // Draw connecting skeleton rods
                  ctx.save();
                  ctx.lineWidth = 4;
                  ctx.lineCap = 'round';
                  ctx.lineJoin = 'round';
                  ctx.shadowColor = shadowColor;
                  ctx.shadowBlur = 10;
                  ctx.strokeStyle = lineColor;

                  POSE_CONNECTIONS.forEach(([i, j]) => {
                    const p1 = mapped[i];
                    const p2 = mapped[j];
                    if (p1 && p2 && p1.vis > 0.4 && p2.vis > 0.4) {
                      ctx.beginPath();
                      ctx.moveTo(p1.x, p1.y);
                      ctx.lineTo(p2.x, p2.y);
                      ctx.stroke();
                    }
                  });
                  ctx.restore();

                  // Draw joint nodes and HUD accent rings
                  ctx.save();
                  mapped.forEach((pt, idx) => {
                    if (pt.vis > 0.4 && (idx === 0 || (idx >= 11 && idx <= 32))) {
                      const isMajorJoint = [11, 12, 13, 14, 23, 24, 25, 26].includes(idx);
                      
                      ctx.beginPath();
                      ctx.arc(pt.x, pt.y, isMajorJoint ? 7 : 5, 0, 2 * Math.PI);
                      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
                      ctx.fill();
                      ctx.lineWidth = isMajorJoint ? 2.5 : 2;
                      ctx.strokeStyle = lineColor;
                      ctx.stroke();

                      // Inner glowing dot
                      ctx.beginPath();
                      ctx.arc(pt.x, pt.y, isMajorJoint ? 3 : 2, 0, 2 * Math.PI);
                      ctx.fillStyle = '#ffffff';
                      ctx.fill();

                      // Extra HUD crosshair ring on shoulders & hips
                      if ([11, 12, 23, 24].includes(idx)) {
                        ctx.beginPath();
                        ctx.arc(pt.x, pt.y, 12, 0, 2 * Math.PI);
                        ctx.lineWidth = 1;
                        ctx.strokeStyle = shadowColor;
                        ctx.stroke();
                      }
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
  }, [quest.reps, quest.id, phase, confirmed]);

  const retryCamera = () => { scanRunRef.current += 1; lastVideoTimeRef.current = -1; setCamError(null); setPhase('starting'); setCameraVersion(v => v + 1); };
  const retryModel = () => { setModelReady(false); setModelProgress(null); setModelError(null); setModelVersion(v => v + 1); };
  const flipCamera = () => { 
    clearTimeout(scanTimerRef.current); scanRunRef.current += 1; setPhase('starting'); setLiveScore(0); setPassStreak(0); setRepsDone(quest.reps ? (quest.progress || 0) : 0); passStreakRef.current = 0; confirmedRef.current = false; lastVideoTimeRef.current = -1; resetRepTracking(); setConfirmed(false); setFacingMode(m => m === 'environment' ? 'user' : 'environment');
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
      const dataUrl = ev.target.result; setUploadedProof(dataUrl);
      if (!labels) { onConfirm(dataUrl); return; }
      setPhase('live'); setScanning(true);
      try {
        const classifier = await getClassifier();
        const results = await classifier(dataUrl, classifierLabels);
        const actScore = getMaxLabelScore(results, labels.activity);
        const negScore = getMaxLabelScore(results, activeNegatives);
        setLiveScore(Math.round(actScore * 100)); setLastLabel(results[0]?.label ?? '');
        if (actScore > negScore && actScore >= (PASS_THRESHOLD - 0.05)) {
          setPassStreak(REQUIRED_PASSES); confirmedRef.current = true; accepted = true; setConfirmed(true);
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

  const captureAndConfirm = () => {
    if (uploadedProof) { onConfirm(uploadedProof); return; }
    const video = videoRef.current, canvas = aiCanvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth || 640; canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);
    onConfirm(canvas.toDataURL('image/jpeg', 0.82));
  };

  const meterColor = liveScore >= PASS_THRESHOLD * 100 ? 'bg-emerald-500' : liveScore >= 15 ? 'bg-amber-400' : 'bg-rose-500';
  const bg = dark ? 'bg-zinc-950' : 'bg-white';
  const border = dark ? 'border-white/10' : 'border-gray-200';
  const txt = dark ? 'text-white' : 'text-gray-900';
  const sub = dark ? 'text-zinc-400' : 'text-gray-500';
  const pill = dark ? 'bg-zinc-800/80' : 'bg-gray-100';

  return (
    <div className="fixed inset-0 z-[60] flex items-end justify-center" style={{ background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}>
      <div className={`relative ${bg} w-full h-[95vh] max-w-lg rounded-t-[32px] overflow-hidden shadow-2xl flex flex-col`} style={{ paddingBottom: 'max(env(safe-area-inset-bottom), 16px)' }}>
        <div className="pointer-events-none absolute inset-x-0 top-0 h-40 overflow-hidden opacity-60 z-0">
          <AnimatedBackground dark={dark} />
        </div>

        <div className="relative z-10 flex justify-center pt-4 pb-2">
          <div className={`w-12 h-1.5 rounded-full ${dark ? 'bg-zinc-700' : 'bg-gray-300'}`} />
        </div>

        <div className={`relative z-10 flex items-center justify-between px-6 py-2 pb-4`}>
          <button onClick={handleCancel} className="text-[#007AFF] text-[15px] font-semibold active:opacity-70 transition-opacity">Cancel</button>
          <div className="text-center">
            <p className={`text-[15px] font-bold ${txt}`}>AI Verification</p>
            <p className={`text-[11px] font-medium ${sub} mt-0.5 max-w-[220px] truncate`}>{uiSubtext}</p>
          </div>
          <div className="w-[52px]" />
        </div>

        <div className="relative flex-1 bg-black overflow-hidden mx-4 rounded-3xl shadow-inner border border-white/10">
          <video ref={videoRef} autoPlay playsInline muted className={`w-full h-full object-cover transition-opacity duration-500 ${phase === 'live' && !uploadedProof ? 'opacity-100' : 'opacity-0'}`} />

          {/* Skeleton HUD Overlay for reps/challenges */}
          <canvas
            ref={skeletonCanvasRef}
            className={`absolute inset-0 w-full h-full pointer-events-none object-cover z-10 transition-opacity duration-500 ${phase === 'live' && !uploadedProof && quest.reps ? 'opacity-100' : 'opacity-0'}`}
          />

          {phase === 'live' && !uploadedProof && labels?.bodyParts && (
            <div className="absolute inset-0 pointer-events-none border-[2px] border-dashed border-violet-500/40 m-5 rounded-2xl animate-pulse z-10">
              <div className="absolute top-4 left-4 bg-black/75 backdrop-blur-md px-3 py-2 rounded-xl border border-violet-500/50 shadow-2xl">
                <div className="flex items-center gap-2 mb-1.5 text-white uppercase font-bold text-[10px] tracking-wider">
                  <span className="w-2 h-2 rounded-full bg-violet-400 animate-ping inline-block" />
                  Biometric Engine
                </div>
                <div className="text-white/60 text-[9px] mb-1 font-mono">Tracking Nodes Active:</div>
                <div className="flex flex-wrap gap-1.5 max-w-[160px]">
                  {labels.bodyParts.map((part) => (
                    <span key={part} className="bg-violet-950/80 text-violet-300 px-1.5 py-0.5 rounded text-[9px] border border-violet-800/60 font-semibold tracking-wide">
                      {part}
                    </span>
                  ))}
                </div>
              </div>
              <div className="absolute top-0 left-0 w-6 h-6 border-t-4 border-l-4 border-violet-500/80 rounded-tl-xl" />
              <div className="absolute top-0 right-0 w-6 h-6 border-t-4 border-r-4 border-violet-500/80 rounded-tr-xl" />
              <div className="absolute bottom-0 left-0 w-6 h-6 border-b-4 border-l-4 border-violet-500/80 rounded-bl-xl" />
              <div className="absolute bottom-0 right-0 w-6 h-6 border-b-4 border-r-4 border-violet-500/80 rounded-br-xl" />
            </div>
          )}

          {uploadedProof && (
            <img src={uploadedProof} alt="Uploaded verification" className="absolute inset-0 w-full h-full object-cover" />
          )}

          {phase === 'starting' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-zinc-900">
              <div className="w-10 h-10 border-4 border-white/20 border-t-[#007AFF] rounded-full animate-spin" />
              <p className="text-white/70 text-xs font-medium tracking-wide">Initializing Camera Engine…</p>
            </div>
          )}

          {phase === 'error' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-8 text-center bg-zinc-900">
              <p className="text-white font-bold text-[15px]">{camError === 'permission' ? 'Camera Access Blocked' : 'No Camera Found'}</p>
              <p className="text-white/60 text-[13px] leading-relaxed mb-2">
                {camError === 'permission' ? 'Open this app in a new tab and allow camera access when prompted.' : quest.reps ? 'This quest needs a live camera. Check your camera and try again.' : 'Upload a photo instead.'}
              </p>
              <button onClick={retryCamera} className="bg-[#007AFF] text-white text-[13px] font-semibold px-6 py-2.5 rounded-full active:scale-95 transition-transform">
                Try Camera Again
              </button>
            </div>
          )}

          {(modelError || notice) && (
            <div className="absolute inset-x-4 top-4 rounded-xl bg-red-500/90 backdrop-blur border border-red-400 px-4 py-3 text-center shadow-2xl z-30">
              <p className="text-[13px] font-medium leading-relaxed text-white">{modelError || notice}</p>
              <button onClick={() => setNotice(null)} className="text-[11px] font-bold text-white/70 block w-full mt-2 uppercase tracking-wider">Dismiss</button>
            </div>
          )}

          {/* ── REP QUESTS: pose-tracking overlay ── */}
          {phase === 'live' && quest.reps && (
            <div className="absolute inset-x-0 bottom-0 px-5 pb-5 pt-12 z-20" style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.85) 0%, transparent 100%)' }}>
              {confirmed ? (
                <div className="flex items-center gap-2.5 bg-green-500/20 w-max px-4 py-2 rounded-full border border-green-500/30 backdrop-blur-md">
                  <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center flex-shrink-0">
                    <CheckIcon />
                  </div>
                  <p className="text-green-50 text-[13px] font-bold tracking-wide">Verified & Locked!</p>
                </div>
              ) : poseError ? (
                <p className="text-rose-300 text-[12px] font-semibold">{poseError}</p>
              ) : !poseReady ? (
                <p className="text-white/80 text-[11px] font-bold tracking-widest uppercase mb-2">Loading Pose Tracking Model…</p>
              ) : (
                <>
                  <div className="flex items-end justify-between mb-2">
                    <p className="text-white/90 text-[13px] font-semibold tracking-wide drop-shadow-md">{repCue}</p>
                    <p className="text-white text-[13px] font-black tracking-widest bg-[#007AFF] px-3 py-1 rounded-lg shadow-lg">
                      {repsDone} / {quest.reps} REPS
                    </p>
                  </div>
                  <div className="w-full h-1.5 bg-white/20 rounded-full overflow-hidden shadow-inner">
                    <div className="h-full bg-green-400 transition-all duration-300 ease-out rounded-full" style={{ width: `${Math.min(100, (repsDone / quest.reps) * 100)}%` }} />
                  </div>
                  <p className="text-white/40 text-[9px] font-mono mt-1.5 truncate uppercase tracking-widest">{repPhase === 'up' ? 'Ready position' : 'Mid-rep'}</p>
                </>
              )}
            </div>
          )}

          {/* ── FOOD / MAP / ACTION QUESTS: CLIP overlay ── */}
          {phase === 'live' && !quest.reps && modelReady && (
            <div className="absolute inset-x-0 bottom-0 px-5 pb-5 pt-12 z-20" style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.85) 0%, transparent 100%)' }}>
              {confirmed ? (
                <div className="flex items-center gap-2.5 bg-green-500/20 w-max px-4 py-2 rounded-full border border-green-500/30 backdrop-blur-md">
                  <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center flex-shrink-0">
                    <CheckIcon />
                  </div>
                  <p className="text-green-50 text-[13px] font-bold tracking-wide">Verified & Locked!</p>
                </div>
              ) : (
                <>
                  <div className="flex items-end justify-between mb-2">
                    <p className="text-white/90 text-[13px] font-semibold tracking-wide drop-shadow-md">
                      {scanning ? 'Scanning environment…' : `${liveScore}% confidence`}
                    </p>
                    <div className="flex items-center gap-1.5">
                      {Array.from({ length: REQUIRED_PASSES }).map((_, i) => (
                          <div key={i} className={`w-2 h-2 rounded-full transition-all duration-300 ${i < passStreak ? 'bg-green-400 shadow-[0_0_8px_rgba(74,222,128,0.8)]' : 'bg-white/30'}`} />
                      ))}
                    </div>
                  </div>
                  
                  <div className="w-full h-1.5 bg-white/20 rounded-full overflow-hidden shadow-inner">
                    <div className={`h-full ${meterColor} transition-all duration-700 ease-out rounded-full`} style={{ width: `${liveScore}%` }} />
                  </div>
                  {lastLabel && <p className="text-white/40 text-[9px] font-mono mt-1.5 truncate uppercase tracking-widest">{lastLabel}</p>}
                </>
              )}
            </div>
          )}

          {phase === 'live' && !quest.reps && !modelReady && !modelError && (
            <div className="absolute inset-x-0 bottom-0 px-5 pb-5 pt-12 z-20" style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.8) 0%, transparent 100%)' }}>
              <p className="text-white/80 text-[11px] font-bold tracking-widest uppercase mb-2">Loading Vision Model… {modelProgress ?? 0}%</p>
              <div className="w-full h-1.5 bg-white/20 rounded-full overflow-hidden">
                <div className="h-full bg-[#007AFF] transition-all duration-300 rounded-full shadow-[0_0_10px_rgba(0,122,255,0.8)]" style={{ width: `${modelProgress ?? 0}%` }} />
              </div>
            </div>
          )}
          <canvas ref={aiCanvasRef} className="hidden" />
        </div>

        {quest.duration && (
          <div className={`px-5 py-3.5 mx-4 mt-4 rounded-[18px] border flex items-center justify-between shadow-sm ${dark ? 'bg-zinc-900 border-zinc-800' : 'bg-gray-50 border-gray-200'}`}>
            <div className="flex flex-col">
              <span className={`text-[10px] font-bold uppercase tracking-widest ${dark ? 'text-zinc-500' : 'text-gray-400'}`}>Objective Duration</span>
              <span className={`text-[22px] leading-none mt-1 font-mono font-black tracking-tight ${txt}`}>{formatTimerString(secondsLeft)}</span>
            </div>
            <button onClick={() => setTimerRunning(!timerRunning)} disabled={secondsLeft === 0}
              className={`px-5 py-2.5 rounded-[12px] text-[13px] font-bold tracking-wide transition-colors shadow-sm ${
                secondsLeft === 0 ? 'bg-zinc-200 text-zinc-400 cursor-not-allowed' : timerRunning ? 'bg-rose-500 text-white active:bg-rose-600' : 'bg-[#34C759] text-white active:bg-green-600'
              }`}>
              {secondsLeft === 0 ? 'Completed' : timerRunning ? 'Pause Activity' : 'Start Timer'}
            </button>
          </div>
        )}

        <div className="px-4 pt-4 pb-2">
          {phase === 'live' && (
            <div className="space-y-3">
              <button onClick={captureAndConfirm} disabled={!confirmed}
                className={`w-full py-4 rounded-[18px] text-[15px] font-bold tracking-wide transition-all duration-300 ${
                  confirmed ? 'bg-[#007AFF] text-white shadow-[0_8px_20px_rgba(0,122,255,0.4)] active:scale-[0.98]' : `${pill} ${sub} cursor-not-allowed`
                }`}>
                {confirmed ? 'Complete Quest' : instructionText}
              </button>
              <div className="flex gap-3">
                <button onClick={flipCamera} className={`${quest.reps ? 'w-full' : 'flex-1'} py-3.5 rounded-[16px] text-[14px] font-semibold ${pill} ${txt} active:opacity-70 transition-opacity`}>
                  Flip Camera
                </button>
                {!quest.reps && (
                <label className={`flex-1 py-3.5 rounded-[16px] text-[14px] font-semibold ${pill} ${txt} text-center cursor-pointer active:opacity-70 transition-opacity ${questType === 'map' ? 'ring-2 ring-[#007AFF] bg-[#007AFF]/10 text-[#007AFF]' : ''}`}>
                  Upload Photo
                  <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
                </label>
                )}
              </div>
            </div>
          )}

          {(phase === 'error' || phase === 'starting') && !quest.reps && (
            <label className={`flex items-center justify-center w-full py-4 rounded-[18px] text-[15px] font-bold ${pill} ${txt} cursor-pointer active:opacity-70 transition-opacity`}>
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
export default function QuestDailyApp() {
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem('sq_dark');
    if (saved !== null) return saved === 'true';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  const [showInstallPrompt, setShowInstallPrompt] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('sq_dark', dark);
  }, [dark]);

  useEffect(() => {
    // Show install prompt once per user session
    const hasSeen = localStorage.getItem('sq_has_seen_install');
    if (!hasSeen) {
      setShowInstallPrompt(true);
    }
  }, []);

  const handleDismissInstall = () => {
    localStorage.setItem('sq_has_seen_install', 'true');
    setShowInstallPrompt(false);
  };

  const [level,     setLevel]     = useState(() => parseInt(localStorage.getItem('sq_level'))    || 1);
  const [xp,        setXp]        = useState(() => parseInt(localStorage.getItem('sq_xp'))       || 0);
  const [quests,    setQuests]    = useState(() => {
    const saved     = JSON.parse(localStorage.getItem('sq_quests')) || [];
    const anyDone   = saved.some(q => q.completed);
    const lastReset = parseInt(localStorage.getItem('sq_lastReset')) || 0;
    const expired   = Date.now() - lastReset >= ONE_DAY_MS;
    if (!expired && saved.length >= 5) return saved;
    if (!expired && anyDone) return saved;
    const n = Math.floor(Math.random() * 3) + 5;
    return [...QUEST_POOL].sort(() => 0.5 - Math.random()).slice(0, n).map(randomizeQuest);
  });
  
  const [lastReset, setLastReset] = useState(() => parseInt(localStorage.getItem('sq_lastReset')) || 0);
  const [timeLeft,  setTimeLeft]  = useState('--:--:--');
  const [proofModalId,   setProofModalId]   = useState(null);
  const [proofImages,  setProofImages]  = useState(() => JSON.parse(localStorage.getItem('sq_proofs')) || {});
  const [viewingProof, setViewingProof] = useState(null);
  const [detailQuestId,     setDetailQuestId]     = useState(null);
  const [completionQuest, setCompletionQuest] = useState(null);

  // Always derive these from the live quests array so saved progress (or a
  // completion) made while a modal is open is reflected the next time it opens.
  const proofModal  = quests.find(q => q.id === proofModalId)  || null;
  const detailQuest = quests.find(q => q.id === detailQuestId) || null;

  const xpRef    = useRef(xp);
  const levelRef = useRef(level);
  useEffect(() => { xpRef.current    = xp;    }, [xp]);
  useEffect(() => { levelRef.current = level; }, [level]);

  const xpRequired = level * 100;

  const generateNewQuests = useCallback((ts) => {
    const n = Math.floor(Math.random() * 3) + 5;
    const selected = [...QUEST_POOL].sort(() => 0.5 - Math.random()).slice(0, n).map(randomizeQuest);
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
    localStorage.setItem('sq_level', level);
    localStorage.setItem('sq_xp', xp);
    localStorage.setItem('sq_quests', JSON.stringify(quests));
    localStorage.setItem('sq_lastReset', lastReset);
  }, [level, xp, quests, lastReset]);
  useEffect(() => { localStorage.setItem('sq_proofs', JSON.stringify(proofImages)); }, [proofImages]);

  const applyXpChange = useCallback((amount) => {
    let newXp = xpRef.current + amount, newLevel = levelRef.current;
    while (newXp >= newLevel * 100) { newXp -= newLevel * 100; newLevel++; }
    while (newXp < 0 && newLevel > 1) { newLevel--; newXp += newLevel * 100; }
    if (newLevel === 1 && newXp < 0) newXp = 0;
    setXp(newXp); setLevel(newLevel);
  }, []);

 const handleQuestClick = (quest) => {
  if (quest.completed) {
    return;
  }

  setDetailQuestId(quest.id);
};

  const handleProofConfirm = (questId, img) => {
    const quest = quests.find(q => q.id === questId);
    setProofImages(prev => ({ ...prev, [questId]: img }));
    setQuests(prev => prev.map(q => q.id === questId ? { ...q, completed: true, progress: 0 } : q));
    applyXpChange(quest?.xp ?? 0);
    setProofModalId(null);
    setDetailQuestId(null);
    setCompletionQuest(quest || null);
  };

  // Saves how far along the quest was (reps done / seconds left) so the
  // person can pick back up where they left off next time they open it.
  const handleCancelProof = (progress) => {
    if (proofModalId && progress !== undefined) {
      setQuests(prev => prev.map(q => q.id === proofModalId ? { ...q, progress } : q));
    }
    setProofModalId(null);
  };

  const xpPct  = Math.min(100, Math.max(0, (xp / xpRequired) * 100));
  const allDone = quests.length > 0 && quests.every(q => q.completed);

  const bg       = dark ? 'bg-black'      : 'bg-[#F2F2F7]';
  const cardBg   = dark ? 'bg-zinc-900'   : 'bg-white';
  const txt      = dark ? 'text-white'    : 'text-gray-900';
  const sub      = dark ? 'text-zinc-400' : 'text-gray-500';
  const sep      = dark ? 'border-zinc-800' : 'border-gray-100';
  const secLabel = dark ? 'text-zinc-500' : 'text-gray-400';
  const completedCount = quests.filter(q => q.completed).length;
  const dailyPct = quests.length ? (completedCount / quests.length) * 100 : 0;

  return (
    <div className={`${bg} min-h-screen flex flex-col items-center transition-colors duration-200`}>
      <div className="relative w-full max-w-[430px] flex flex-col min-h-screen overflow-hidden shadow-2xl bg-inherit">
        
        <AnimatedBackground dark={dark} />

        {showInstallPrompt && <IOSInstallPrompt onDismiss={handleDismissInstall} dark={dark} />}

        {proofModal && (
          <CameraModal quest={proofModal} dark={dark}
            onConfirm={img => handleProofConfirm(proofModal.id, img)}
            onCancel={handleCancelProof} />
        )}

        {viewingProof && (
          <div className="fixed inset-0 z-[70] flex items-center justify-center p-4 sq-anim-pop"
            style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)' }}
            onClick={() => setViewingProof(null)}>
            <img src={viewingProof} alt="proof" className="max-w-full max-h-full rounded-[24px] shadow-2xl" />
          </div>
        )}

        {completionQuest ? (
          <CompletionScreen quest={completionQuest} dark={dark} timeLeft={timeLeft}
            onToggleTheme={() => setDark(d => !d)}
            onBack={() => setCompletionQuest(null)} />
        ) : detailQuest ? (
          <QuestDetailScreen quest={detailQuest} dark={dark} timeLeft={timeLeft}
            onToggleTheme={() => setDark(d => !d)}
            onBack={() => setDetailQuestId(null)}
            onMarkComplete={() => setProofModalId(detailQuest.id)} />
        ) : (
          <>
            <div className={`relative z-10 safe-top px-3 pb-3 transition-colors duration-200`}>
              <div className="flex items-center justify-between pt-2">
                <div className="flex items-center gap-2.5">
                  <LogoIcon size={34} dark={dark} />
                  <h1 className={`text-[22px] font-bold tracking-tight ${txt}`}>QuestDaily</h1>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`flex items-center gap-1 px-2.5 py-1 rounded-full ${dark ? 'bg-zinc-800/80' : 'bg-gray-100'}`}>
                    <span className="text-[10px]">⏱</span>
                    <span className={`text-[11px] font-mono font-medium ${dark ? 'text-zinc-300' : 'text-gray-600'}`}>{timeLeft}</span>
                  </div>
                  <button onClick={() => setDark(d => !d)}
                    className={`w-8 h-8 rounded-full flex items-center justify-center ${dark ? 'bg-zinc-800/80 text-zinc-300' : 'bg-gray-100 text-gray-600'} active:opacity-70 transition-colors`}>
                    {dark ? <SunIcon /> : <MoonIcon />}
                  </button>
                </div>
              </div>
              <div className="mt-3 flex items-center gap-1.5">
                <span className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white text-[10px] font-bold px-2 py-0.5 rounded-full">Lv {level}</span>
                <span className={`text-[11px] font-medium ${sub}`}>Novice Adventurer</span>
                <span className={`ml-auto text-[11px] font-semibold ${dark ? 'text-zinc-500' : 'text-gray-400'}`}>{xp} / {xpRequired} XP</span>
              </div>
            </div>

            <div className="relative z-10 flex-1 scroll-ios px-3 pt-1 pb-8 space-y-5">
              <div className={`${dark ? 'bg-zinc-900/70 border border-white/5' : 'bg-white border border-gray-100'} rounded-[20px] px-4 py-4 flex items-center justify-between sq-anim-pop shadow-sm`}>
                <div>
                  <p className={`text-[14px] font-bold ${txt}`}>Daily Progress</p>
                  <p className={`text-[12px] mt-0.5 font-medium ${sub}`}>{completedCount} / {quests.length} completed</p>
                </div>
                <ProgressRing pct={dailyPct} dark={dark} />
              </div>

              <div>
                <p className={`text-[11px] font-bold uppercase tracking-widest ${secLabel} mb-2.5 px-1`}>
                  Today's Objectives
                </p>

                <div className={`${dark ? 'bg-zinc-900/70 border border-white/5' : 'bg-white border border-gray-100 shadow-sm'} rounded-[20px] overflow-hidden`}>
                  {quests.map((quest, i) => (
                    <div key={quest.id}>
                      {i > 0 && <div className={`border-t ${sep} ml-[58px]`} />}
                      <button onClick={() => {
  if (!quest.completed) handleQuestClick(quest);
}}
                        className={`w-full flex items-center gap-3.5 px-4 py-[18px] text-left active:bg-black/5 transition-all`}>
                        <div className={`w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center border-2 transition-all duration-300 ${quest.completed ? 'bg-[#34C759] border-[#34C759] shadow-[0_0_10px_rgba(52,199,89,0.4)]' : dark ? 'border-zinc-600' : 'border-gray-300'}`}>
                          {quest.completed && <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>}
                        </div>
                        <span className={`flex-1 text-[15px] font-semibold leading-snug transition-colors ${quest.completed ? (dark ? 'text-zinc-600 line-through' : 'text-gray-400 line-through') : txt}`}>
                          {quest.text}
                          {!quest.completed && quest.progress > 0 && (
                            <span className="ml-2 align-middle text-[10px] font-bold uppercase tracking-wide text-[#007AFF] bg-[#007AFF]/10 px-1.5 py-0.5 rounded-full">
                              In progress
                            </span>
                          )}
                        </span>
                        <div className="flex items-center gap-2 flex-shrink-0">
                          {quest.completed && proofImages[quest.id] && (
                            <button onClick={e => { e.stopPropagation(); setViewingProof(proofImages[quest.id]); }}
                              className="w-7 h-7 rounded-md overflow-hidden flex-shrink-0 ring-1 ring-black/10 shadow-sm active:scale-95 transition-transform">
                              <img src={proofImages[quest.id]} alt="proof" className="w-full h-full object-cover" />
                            </button>
                          )}
                          <span className={`text-[13px] font-bold ${quest.completed ? 'text-[#34C759]' : dark ? 'text-zinc-500' : 'text-gray-400'}`}>+{quest.xp}</span>
                          {quest.completed ? (
                            <span className={`text-[10px] font-bold ${dark ? 'text-zinc-600' : 'text-gray-300'}`}>XP</span>
                          ) : (
                            <span className={`text-[11px] ${dark ? 'text-zinc-600' : 'text-gray-300'}`}>
                              <svg width="8" height="14" viewBox="0 0 8 14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="1 1 7 7 1 13"/></svg>
                            </span>
                          )}
                        </div>
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              {allDone && (
                <div className={`${cardBg} rounded-[20px] p-6 text-center border ${dark ? 'border-white/5' : 'border-gray-100 shadow-sm'}`}>
                  <p className="text-3xl mb-3">🏆</p>
                  <p className={`font-bold text-[16px] ${txt}`}>All Quests Complete</p>
                  <p className={`text-sm mt-1.5 font-medium ${sub}`}>Rest up. New quests when the timer hits zero.</p>
                </div>
              )}

              {quests.length > 0 && (
                <p className={`text-[11px] font-medium text-center ${secLabel} px-4 pb-2 uppercase tracking-wide`}>
                  All quests verified by on-device AI <br/> no data leaves your phone
                </p>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
