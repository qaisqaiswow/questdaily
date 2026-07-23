"use client";

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { pipeline, env } from '@huggingface/transformers';

// ─── UPGRADED AI SETUP ────────────────────────────────────────────────────────
env.allowLocalModels = false;
let classifierPromise = null;

function getClassifier(onProgress) {
  if (!classifierPromise) {
    // Attempt WebGPU acceleration for instant inference, fallback to WASM
    classifierPromise = pipeline(
      'zero-shot-image-classification',
      'Xenova/clip-vit-base-patch32',
      {
        device: 'webgpu',
        ...(onProgress ? { progress_callback: onProgress } : {})
      }
    ).catch(async (error) => {
      console.warn("WebGPU not available, falling back to WASM:", error);
      return pipeline(
        'zero-shot-image-classification',
        'Xenova/clip-vit-base-patch32',
        onProgress ? { progress_callback: onProgress } : undefined
      );
    }).catch(error => {
      classifierPromise = null;
      throw error;
    });
  }
  return classifierPromise;
}

// ─── UPGRADED LABEL ENGINEERING (PROMPT ENSEMBLING) ──────────────────────────
// Using multi-angle and varied lighting descriptors significantly boosts CLIP accuracy
const QUEST_LABELS = {
  q1:  { type: 'reps', activity: ['a clear photo of a person exercising doing pushups on the floor', 'an athlete performing pushups in a room', 'action shot of pushup workout'], label: 'doing pushups', bodyParts: ['Chest', 'Triceps', 'Shoulders', 'Core'] },
  q2:  { type: 'reps', activity: ['a clear photo of a person exercising doing deep squats', 'an athlete performing leg squats workout', 'action shot of squat exercise'], label: 'doing squats', bodyParts: ['Quads', 'Hamstrings', 'Glutes', 'Core'] },
  q4:  { type: 'reps', activity: ['a clear photo of a person doing pullups on a bar', 'an athlete pulling their chin over a pullup bar', 'upper body back workout pullups'], label: 'doing pullups', bodyParts: ['Lats', 'Upper Back', 'Biceps', 'Forearms'] },
  q12: { type: 'reps', activity: ['a clear photo of a person exercising doing situps or crunches', 'an athlete performing core abdominal situps on the floor'], label: 'doing situps', bodyParts: ['Abs', 'Obliques', 'Hip Flexors'] },
  q17: { type: 'reps', activity: ['a clear photo of a person jumping rope fast', 'an athlete skipping rope in mid-air cardio workout'], label: 'jumping rope', bodyParts: ['Calves', 'Quads', 'Shoulders', 'Cardio'] },
  q23: { type: 'reps', activity: ['a clear photo of a person doing walking lunges exercise', 'an athlete performing deep leg split lunges'], label: 'doing lunges', bodyParts: ['Quads', 'Glutes', 'Hamstrings'] },

  q3:  { type: 'map', activity: ['strava running workout map dashboard with duration and speed metrics', 'gps running pace distance tracking workout summary screen on phone'], label: 'running metrics map' },
  q16: { type: 'map', activity: ['cycling route ride summary dashboard with speed and time logs', 'bicycle fitness tracking gps map workout summary'], label: 'cycling metrics map' },
  q5:  { type: 'map', activity: ['gps walking tracking map route screenshot', 'outdoor walking route map tracker app screen'], label: 'walking map screenshot' },
  q22: { type: 'map', activity: ['10000 step counter fitness app screenshot dashboard', 'daily step tracking goal completed screen'], label: 'step tracking map' },

  q9:  { type: 'food', activity: ['a high quality photo of a healthy nutritious food meal salad vegetables on a plate', 'balanced clean meal bowl with protein and greens'], label: 'plate of healthy food' },
  q19: { type: 'food', activity: ['clean sugar-free healthy meal on plate', 'plate of fresh vegetables and whole unprocessed foods'], label: 'plate of clean food' },
  q20: { type: 'food', activity: ['freshly cooked homemade food meal on a dining plate', 'warm homemade meal prepared from scratch in a bowl'], label: 'cooked meal' },
  q14: { type: 'food', activity: ['a close up photo of a glass filled with green smoothie juice', 'blended vegetable green detox juice drink in a glass or bottle'], label: 'green smoothie' },
  q6:  { type: 'food', activity: ['a large full glass of clear drinking water', 'a reusable sports water bottle filled to the top with water'], label: 'water bottle' },

  q7:  { type: 'action', activity: ['a person sitting meditating cross-legged peacefully', 'mindfulness Zen meditation exercise eyes closed'], label: 'meditating' },
  q8:  { type: 'action', activity: ['a person stretching their muscles flexibility workout', 'athlete holding a yoga stretch pose on a mat'], label: 'stretching' },
  q10: { type: 'action', activity: ['a person resting peacefully sleeping in bed under blankets', 'dark bedroom with person resting in bed eyes closed'], label: 'getting good sleep' },
  q11: { type: 'action', activity: ['a person performing active jumping jacks cardio workout', 'high intensity burpees or jumping jacks exercise'], label: 'doing cardio' },
  q13: { type: 'action', activity: ['close up of a person handwriting with a pen in a notebook or journal', 'writing thoughts in a daily diary planner'], label: 'journaling' },
  q15: { type: 'action', activity: ['a person holding a static forearm plank exercise position on the floor', 'core endurance plank position workout'], label: 'holding a plank' },
  q18: { type: 'action', activity: ['refreshing shower running water droplets in bathroom', 'clean bathroom shower head spraying cold water'], label: 'in the shower' },
  q21: { type: 'action', activity: ['a person breathing deeply with focus and calm posture', 'deep pranayama breathing relaxation exercise'], label: 'deep breathing' },
  q24: { type: 'action', activity: ['a bedroom at night prepared for early sleep', 'sleeping peacefully in a dark quiet bedroom at night'], label: 'sleeping early' },
  q25: { type: 'action', activity: ['a person sitting inside an ice bath tub cold plunge', 'cold water immersion therapy tub filled with floating ice cubes'], label: 'in a cold plunge' },
};

const getNegativeLabels = (type) => {
  const base = [
    'an empty room with no people', 
    'a blurry dark out-of-focus abstract background', 
    'a close up selfie of a person looking directly into the camera doing nothing',
    'a person sitting passively on a sofa or chair looking at a screen'
  ];
  if (type === 'map') return [...base, 'google maps navigation driving directions screen', 'blank city map with no workout data', 'sweaty face selfie', 'shoes on the floor'];
  if (type === 'food') return [...base, 'empty ceramic plate or dirty bowl', 'restaurant paper menu or receipt', 'store grocery product barcode wrapper'];
  return [...base, 'a person holding a smartphone taking a picture of the mirror'];
};

const QUEST_POOL = [
  { id: 'q1',  text: 'Do 20 pushups',                      xp: 50, reps: 20 },
  { id: 'q2',  text: 'Do 30 squats',                       xp: 45, reps: 30 },
  { id: 'q3',  text: 'Go for a 15-minute run',             xp: 75, duration: 900 },
  { id: 'q4',  text: 'Do 10 pullups',                      xp: 60, reps: 10 },
  { id: 'q16', text: 'Do 15 minutes of cycling',           xp: 55, duration: 900 },
  { id: 'q17', text: 'Do 50 jumping rope reps',            xp: 40, reps: 50 },
  { id: 'q18', text: 'Take a cold shower',                 xp: 50 },
  { id: 'q19', text: 'Eat no sugar today',                 xp: 65 },
  { id: 'q20', text: 'Cook a meal from scratch',           xp: 45 },
  { id: 'q21', text: 'Do 10 minutes of deep breathing',    xp: 30, duration: 600 },
  { id: 'q22', text: 'Take 10,000 steps',                  xp: 70 },
  { id: 'q23', text: 'Do 3 sets of lunges',                xp: 40, reps: 30 },
  { id: 'q24', text: 'Go to bed before 11pm',              xp: 35 },
  { id: 'q25', text: 'Do a 5-minute ice bath or cold plunge', xp: 80, duration: 300 },
  { id: 'q5',  text: 'Walk outside for 20 minutes',        xp: 35, duration: 1200 },
  { id: 'q6',  text: 'Drink 2 liters of water today',          xp: 25 },
  { id: 'q7',  text: 'Meditate for 10 minutes',            xp: 45, duration: 600 },
  { id: 'q8',  text: 'Stretch for 10 minutes',             xp: 35, duration: 600 },
  { id: 'q9',  text: 'Eat a healthy meal',                 xp: 40 },
  { id: 'q10', text: 'Get 8 hours of sleep',                   xp: 55 },
  { id: 'q11', text: 'Do 3 minutes of jumping jacks',          xp: 30, duration: 180 },
  { id: 'q12', text: 'Do 20 situps',                       xp: 40, reps: 20 },
  { id: 'q13', text: 'Write in your journal',                  xp: 20 },
  { id: 'q14', text: 'Drink a green smoothie',                 xp: 30 },
  { id: 'q15', text: 'Hold a plank for 60 seconds',            xp: 50, duration: 60 },
];

const ONE_DAY_MS = 24 * 60 * 60 * 1000;
const PASS_THRESHOLD  = 0.38; 
const REP_SCAN_INTERVAL_MS = 180; 
const REP_MIN_DURATION_MS = 550;
const REP_FORM_CONFIDENCE = 0.38;

const REP_PROFILES = {
  q1: {
    target: ['a photo of a person at the very bottom of a pushup with chest inches from the floor', 'a person doing a pushup with elbows deeply bent at 90 degrees'],
    reset: ['a photo of a person in a high straight-arm plank pushup position', 'a person resting at the top of a pushup with arms fully extended'],
    cue: 'Lower your chest down, then push completely back up.',
  },
  q2: {
    target: ['a photo of a person at the bottom of a deep squat with bent knees and hips back', 'a person squatting low with thighs parallel to the floor'],
    reset: ['a photo of a person standing upright after completing a squat', 'a person standing tall with legs fully straight'],
    cue: 'Drop into a deep squat, then return to a full upright stand.',
  },
  q4: {
    target: ['a photo of a person at the top of a pullup with chin cleared over the bar', 'a person pulling their body upwards on a pullup bar'],
    reset: ['a photo of a person hanging freely from a pullup bar with straight arms', 'a person hanging at the bottom dead hang of a pullup'],
    cue: 'Pull up clear to the bar, then drop back to straight arms.',
  },
  q12: {
    target: ['a photo of a person at the top contraction of a situp with torso lifted off the ground', 'a person crunching their abs upward towards knees'],
    reset: ['a photo of a person lying completely flat on their back on the floor', 'a person resting flat on the exercise mat'],
    cue: 'Crunch your torso all the way up, then lie completely flat.',
  },
  q17: {
    target: ['a photo of a person airborne feet off the ground while jumping rope', 'a person jumping high in the air skipping rope'],
    reset: ['a photo of a person standing flat on the ground holding a jump rope', 'a person standing still between jump rope skips'],
    cue: 'Keep continuous active jumps inside the camera frame.',
  },
  q23: {
    target: ['a photo of a person at the bottom of a deep lunge stance with back knee near floor', 'a person in a deep split-stance leg lunge'],
    reset: ['a photo of a person standing upright after finishing a lunge step', 'a person standing tall with feet together'],
    cue: 'Step deep down into the lunge stance, then rise all the way up.',
  },
};

function getMaxLabelScore(results, candidates) {
  if (!candidates?.length) return 0;
  return results.reduce(
    (best, result) => (candidates.includes(result.label) ? Math.max(best, result.score) : best),
    0
  );
}

// Upgraded rep tracker with noise gating and velocity checks
function createRepTracker() {
  return {
    phase: 'seek-target',
    reps: 0,
    smoothedTarget: 0,
    smoothedReset: 0,
    lastSampleAt: 0,
    lastRepAt: 0,
  };
}

function advanceRepTracker(tracker, { targetScore, resetScore, now }) {
  const alpha = 0.70; // Slightly higher responsiveness for faster rep feedback
  tracker.lastSampleAt = now;
  tracker.smoothedTarget = tracker.smoothedTarget * (1 - alpha) + targetScore * alpha;
  tracker.smoothedReset = tracker.smoothedReset * (1 - alpha) + resetScore * alpha;

  const formScore = Math.max(tracker.smoothedTarget, tracker.smoothedReset);
  let counted = false;
  let phaseChanged = false;

  // Noise gate: ensure there is a clear delta between target and reset confidence
  if (tracker.phase === 'seek-target') {
    if (tracker.smoothedTarget >= REP_FORM_CONFIDENCE && tracker.smoothedTarget > tracker.smoothedReset + 0.08) {
      tracker.phase = 'seek-reset';
      phaseChanged = true;
    }
  } else if (tracker.phase === 'seek-reset') {
    if (tracker.smoothedReset >= REP_FORM_CONFIDENCE && tracker.smoothedReset > tracker.smoothedTarget + 0.08) {
      const isNewRep = now - tracker.lastRepAt >= REP_MIN_DURATION_MS;
      if (isNewRep) {
        tracker.reps += 1;
        tracker.lastRepAt = now;
        tracker.phase = 'seek-target';
        counted = true;
        phaseChanged = true;
      }
    }
  }

  return { counted, phaseChanged, phase: tracker.phase, confidence: formScore };
}

// ─── ICONOGRAPHY & UI HELPERS ──────────────────────────────────────────────────
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
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
);
const ShareIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/>
  </svg>
);
const PlusSquareIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="4" ry="4"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/>
  </svg>
);

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
        .sq-streak {
          position: absolute; width: 180%; height: 1.5px; left: -40%;
          background: linear-gradient(90deg, transparent, var(--sq-streak-c1), var(--sq-streak-c2), transparent);
          transform-origin: center; filter: blur(0.5px); opacity: 0.7;
        }
        .sq-streak-1 { top: 14%;  transform: rotate(-18deg); animation: sq-drift-a 9s ease-in-out infinite; }
        .sq-streak-2 { top: 42%;  transform: rotate(-12deg); animation: sq-drift-b 13s ease-in-out infinite; opacity: 0.45; }
        .sq-streak-3 { top: 68%;  transform: rotate(-22deg); animation: sq-drift-a 11s ease-in-out infinite reverse; opacity: 0.35; }
        @keyframes sq-drift-a {
          0%   { transform: translateX(-6%) rotate(-18deg); opacity: 0.35; }
          50%  { transform: translateX(6%)  rotate(-16deg); opacity: 0.8; }
          100% { transform: translateX(-6%) rotate(-18deg); opacity: 0.35; }
        }
        @keyframes sq-drift-b {
          0%   { transform: translateX(5%)  rotate(-12deg); opacity: 0.25; }
          50%  { transform: translateX(-5%) rotate(-10deg); opacity: 0.55; }
          100% { transform: translateX(5%)  rotate(-12deg); opacity: 0.25; }
        }
        .sq-glow {
          position: absolute; width: 260px; height: 260px; border-radius: 999px;
          filter: blur(70px); opacity: var(--sq-glow-o);
        }
        .sq-glow-1 { top: -60px; left: -60px; background: var(--sq-glow-1); animation: sq-float 10s ease-in-out infinite; }
        .sq-glow-2 { bottom: -80px; right: -60px; background: var(--sq-glow-2); animation: sq-float 12s ease-in-out infinite reverse; }
        @keyframes sq-float {
          0%, 100% { transform: translate(0,0) scale(1); }
          50% { transform: translate(20px, -15px) scale(1.15); }
        }
        @keyframes sq-pop-in {
          0% { opacity: 0; transform: scale(0.9) translateY(8px); }
          100% { opacity: 1; transform: scale(1) translateY(0); }
        }
        @keyframes sq-check-in {
          0% { opacity: 0; transform: scale(0.4); }
          60% { opacity: 1; transform: scale(1.15); }
          100% { opacity: 1; transform: scale(1); }
        }
        @keyframes sq-icon-float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-4px); }
        }
        .sq-anim-pop { animation: sq-pop-in 0.45s cubic-bezier(0.22, 1, 0.36, 1) both; }
        .sq-anim-check { animation: sq-check-in 0.55s cubic-bezier(0.22, 1, 0.36, 1) both; }
        .sq-anim-float { animation: sq-icon-float 3.2s ease-in-out infinite; }
      `}</style>
    </div>
  );
};

// ─── PER-QUEST ICONOGRAPHY & THEMES ───────────────────────────────────────────
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
  wind: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 8h11a2.5 2.5 0 1 0-2.5-2.5"/><path d="M3 13h15a2.5 2.5 0 1 1-2.5 2.5"/><path d="M3 18h9a2 2 0 1 0-2-2"/></svg>),
  rope: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M4 20c4-8 12-8 16 0"/><path d="M4 4c4 8 12 8 16 0"/></svg>),
  target: (p) => (<svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="0.6" fill="currentColor"/></svg>),
};

const QUEST_THEME = {
  q1:  { icon: 'dumbbell', grad: 'from-fuchsia-500 to-purple-600' },
  q2:  { icon: 'legs',     grad: 'from-violet-500 to-indigo-600' },
  q3:  { icon: 'run',      grad: 'from-orange-400 to-rose-500' },
  q4:  { icon: 'bar',      grad: 'from-purple-500 to-blue-600' },
  q5:  { icon: 'footsteps',grad: 'from-teal-400 to-cyan-600' },
  q6:  { icon: 'cup',      grad: 'from-indigo-400 to-violet-600' },
  q7:  { icon: 'lotus',    grad: 'from-emerald-400 to-teal-600' },
  q8:  { icon: 'stretch',  grad: 'from-sky-400 to-indigo-600' },
  q9:  { icon: 'bowl',     grad: 'from-lime-400 to-emerald-600' },
  q10: { icon: 'moon',     grad: 'from-indigo-500 to-slate-700' },
  q11: { icon: 'bolt',     grad: 'from-yellow-400 to-orange-600' },
  q12: { icon: 'core',     grad: 'from-rose-500 to-red-600' },
  q13: { icon: 'pencil',   grad: 'from-amber-400 to-orange-600' },
  q14: { icon: 'smoothie', grad: 'from-green-400 to-emerald-600' },
  q15: { icon: 'stopwatch',grad: 'from-cyan-400 to-blue-600' },
  q16: { icon: 'bike',     grad: 'from-blue-400 to-indigo-600' },
  q17: { icon: 'rope',     grad: 'from-pink-500 to-fuchsia-600' },
  q18: { icon: 'droplet',  grad: 'from-sky-400 to-blue-600' },
  q19: { icon: 'leaf',     grad: 'from-emerald-400 to-green-600' },
  q20: { icon: 'pot',      grad: 'from-orange-400 to-amber-600' },
  q21: { icon: 'wind',     grad: 'from-cyan-300 to-teal-600' },
  q22: { icon: 'footsteps',grad: 'from-violet-400 to-purple-600' },
  q23: { icon: 'legs',     grad: 'from-fuchsia-500 to-rose-600' },
  q24: { icon: 'moon',     grad: 'from-indigo-400 to-blue-700' },
  q25: { icon: 'snowflake',grad: 'from-cyan-300 to-blue-600' },
};

const QUEST_ABOUT = {
  q1: "Pushups build raw upper-body strength and core stability in one clean movement — no equipment required.",
  q2: "Squats fire up your biggest muscle groups and reinforce the mechanics behind almost every athletic movement.",
  q3: "A steady run gets your heart rate up, clears your head, and builds endurance over time.",
  q4: "Pullups are one of the purest tests of back and grip strength — a few reps go a long way.",
  q5: "A brisk walk outside boosts circulation, mood, and gives your eyes a break from the screen.",
  q6: "Staying hydrated helps your body perform better and keeps your mind sharp.",
  q7: "A short meditation resets your focus and lowers stress before it builds up.",
  q8: "Stretching keeps your muscles loose and your joints moving through their full range.",
  q9: "A balanced, whole-food meal fuels recovery and keeps your energy steady.",
  q10: "Consistent, sufficient sleep is the single biggest lever for recovery and focus.",
  q11: "A quick cardio burst spikes your heart rate and wakes up your whole body fast.",
  q12: "Situps target your core and build the stability everything else is built on.",
  q13: "Journaling for a few minutes helps you process the day and plan the next one.",
  q14: "A green smoothie is an easy way to pack in nutrients when you're short on time.",
  q15: "Holding a plank builds isometric core strength that carries over to everything else.",
  q16: "Cycling is easy on the joints while still building serious cardio endurance.",
  q17: "Jump rope sharpens coordination and torches calories in a small amount of time.",
  q18: "A cold shower is a quick way to train discipline and wake up your nervous system.",
  q19: "Cutting added sugar for a day gives your energy levels a noticeably steadier baseline.",
  q20: "Cooking from scratch puts you in control of what goes into your body.",
  q21: "A few minutes of deep breathing calms your nervous system and sharpens focus.",
  q22: "Hitting your step count keeps your body moving steadily throughout the day.",
  q23: "Lunges build single-leg strength and balance that squats alone don't cover.",
  q24: "An earlier bedtime compounds — better sleep tonight means a better day tomorrow.",
  q25: "Cold exposure trains resilience and gives your recovery a real boost.",
};

const QUEST_QUOTE = {
  q1:  "Strength grows one rep at a time.",
  q2:  "Every squat builds a stronger foundation.",
  q3:  "Miles don't lie — you earned this one.",
  q4:  "Small steps every day lead to big changes.",
  q5:  "One step at a time is still progress.",
  q6:  "Small steps every day lead to big changes.",
  q7:  "A quiet mind carries the loudest strength.",
  q8:  "Flexibility today, resilience tomorrow.",
  q9:  "You fueled the body that carries you.",
  q10: "Rest is where the real gains happen.",
  q11: "Energy in motion stays in motion.",
  q12: "A strong core holds everything else together.",
  q13: "The pen remembers what the mind forgets.",
  q14: "Good fuel, good day.",
  q15: "Stillness can be the hardest work of all.",
  q16: "Every mile ridden is a mile earned.",
  q17: "Rhythm builds more than just your legs.",
  q18: "Discomfort today, discipline for life.",
  q19: "Progress, not perfection.",
  q20: "What you cook is what you become.",
  q21: "Breathe in control, breathe out doubt.",
  q22: "One step at a time is still progress.",
  q23: "Balance is built one side at a time.",
  q24: "Tonight's rest is tomorrow's edge.",
  q25: "You didn't come this far to only come this far.",
};

const QuestIconBadge = ({ questId, size = 96, dark, floating = false }) => {
  const theme = QUEST_THEME[questId] || { icon: 'target', grad: 'from-indigo-500 to-purple-600' };
  const Icon = QuestSvg[theme.icon] || QuestSvg.target;
  return (
    <div
      className={`relative flex items-center justify-center rounded-full bg-gradient-to-br ${theme.grad} ${floating ? 'sq-anim-float' : ''}`}
      style={{
        width: size, height: size,
        boxShadow: `0 0 0 1px rgba(255,255,255,0.15) inset, 0 8px 30px -8px rgba(139,92,246,0.65)`,
      }}
    >
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
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none"
          stroke="url(#sq-ring-gradient)" strokeWidth={stroke} strokeLinecap="round"
          strokeDasharray={c}
          style={{ strokeDashoffset: offset, transition: 'stroke-dashoffset 0.8s cubic-bezier(0.22,1,0.36,1)' }}
        />
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

// ─── NEW: FIRST-TIME IOS HOME SCREEN INSTALL MODAL ────────────────────────────
function IOSInstallModal({ dark, onClose }) {
  const cardBg = dark ? 'bg-[#181226]/95 border-white/10' : 'bg-white/95 border-gray-200';
  const txt = dark ? 'text-white' : 'text-gray-900';
  const sub = dark ? 'text-zinc-400' : 'text-gray-600';
  const stepBg = dark ? 'bg-white/5 border-white/5' : 'bg-gray-50 border-gray-100';

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-4 bg-black/60 backdrop-blur-md sq-anim-pop">
      <div className={`w-full max-w-sm rounded-[32px] p-6 border shadow-2xl ${cardBg} flex flex-col items-center text-center relative overflow-hidden`}>
        <AnimatedBackground dark={dark} />
        
        <div className="relative z-10 flex flex-col items-center w-full">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-purple-500/30 mb-4">
            <span className="text-2xl">⚡️</span>
          </div>

          <h3 className={`text-[20px] font-bold leading-tight ${txt}`}>
            Install Side Quests
          </h3>
          <p className={`text-[13px] mt-1.5 px-2 leading-relaxed ${sub}`}>
            Add this app to your iPhone Home Screen for full-screen experience and daily quest tracking.
          </p>

          <div className="w-full mt-6 space-y-3 text-left">
            {/* Step 1 */}
            <div className={`flex items-center gap-3 p-3.5 rounded-2xl border ${stepBg}`}>
              <div className="w-9 h-9 rounded-xl bg-blue-500/10 flex items-center justify-center shrink-0">
                <ShareIcon />
              </div>
              <div>
                <p className={`text-[13px] font-semibold ${txt}`}>1. Tap the Share Button</p>
                <p className={`text-[11px] ${sub}`}>At the bottom navigation bar in Safari</p>
              </div>
            </div>

            {/* Step 2 */}
            <div className={`flex items-center gap-3 p-3.5 rounded-2xl border ${stepBg}`}>
              <div className="w-9 h-9 rounded-xl bg-purple-500/10 text-purple-400 flex items-center justify-center shrink-0 font-bold">
                <PlusSquareIcon />
              </div>
              <div>
                <p className={`text-[13px] font-semibold ${txt}`}>2. Add to Home Screen</p>
                <p className={`text-[11px] ${sub}`}>Scroll down the action list and tap</p>
              </div>
            </div>

            {/* Step 3 */}
            <div className={`flex items-center gap-3 p-3.5 rounded-2xl border ${stepBg}`}>
              <div className="w-9 h-9 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center shrink-0 font-bold text-sm">
                Add
              </div>
              <div>
                <p className={`text-[13px] font-semibold ${txt}`}>3. Tap &quot;Add&quot; Top Right</p>
                <p className={`text-[11px] ${sub}`}>Launch anytime directly from your screen</p>
              </div>
            </div>
          </div>

          <button
            onClick={onClose}
            className="w-full mt-6 py-3.5 rounded-2xl bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-semibold text-[15px] shadow-lg shadow-indigo-500/25 active:scale-[0.98] transition-transform"
          >
            Got it, Let&apos;s Go!
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── QUEST DETAIL SCREEN (WITH FIXED LIGHT/DARK BUTTON) ──────────────────────
function QuestDetailScreen({ quest, dark, setDark, onToggleDark, timeLeft, onBack, onMarkComplete }) {
  const txt = dark ? 'text-white' : 'text-gray-900';
  const sub = dark ? 'text-zinc-400' : 'text-gray-500';
  const pill = dark ? 'bg-zinc-800/80' : 'bg-gray-100';
  const cardBg = dark ? 'bg-zinc-900/70' : 'bg-white';
  const theme = QUEST_THEME[quest.id] || { grad: 'from-indigo-500 to-purple-600' };
  const about = QUEST_ABOUT[quest.id] || 'Stay consistent — every quest you complete adds up to real progress.';

  const handleThemeToggle = () => {
    if (onToggleDark) onToggleDark();
    else if (setDark) setDark(!dark);
  };

  return (
    <div className="sq-anim-pop relative z-10 flex flex-col h-full min-h-screen">
      <div className="flex items-center justify-between px-4 pb-3" style={{ paddingTop: 'max(env(safe-area-inset-top), 18px)' }}>
        <button onClick={onBack}
          className={`w-9 h-9 rounded-full flex items-center justify-center ${dark ? 'bg-zinc-800/80 text-zinc-300' : 'bg-gray-100 text-gray-600'} active:opacity-70 transition-colors`}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6"/>
          </svg>
        </button>
        <div className={`flex items-center gap-1 px-2.5 py-1 rounded-full ${pill}`}>
          <span className="text-[10px]">⏱</span>
          <span className={`text-[11px] font-mono font-medium ${dark ? 'text-zinc-300' : 'text-gray-600'}`}>{timeLeft}</span>
        </div>
        
        {/* Fixed interactive Light/Dark switch button matching the back button styling exactly */}
        <button 
          onClick={handleThemeToggle}
          aria-label="Toggle theme"
          className={`w-9 h-9 rounded-full flex items-center justify-center ${dark ? 'bg-zinc-800/80 text-zinc-300' : 'bg-gray-100 text-gray-600'} active:opacity-70 transition-colors`}
        >
          {dark ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>

      <div className="px-4 flex-1 flex flex-col justify-between pb-8">
        <div>
          <div
            className={`relative overflow-hidden rounded-[26px] px-6 pt-10 pb-8 flex flex-col items-center text-center bg-gradient-to-b ${dark ? 'from-[#1c1530] to-[#0d0a17]' : 'from-indigo-50 to-white'} border ${dark ? 'border-white/10' : 'border-gray-100'}`}
          >
            <AnimatedBackground dark={dark} />
            <div className="relative z-10 flex flex-col items-center">
              <QuestIconBadge questId={quest.id} size={92} dark={dark} floating />
              <h2 className={`mt-5 text-[22px] font-bold leading-tight max-w-xs ${txt}`}>
                {quest.text}
              </h2>
              <div className="mt-3.5 inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-semibold text-[12px]">
                <span>+{quest.xp} XP</span>
              </div>
            </div>
          </div>

          <div className="mt-6">
            <h3 className={`text-[11px] font-bold tracking-wider uppercase ${sub} px-1`}>
              About This Quest
            </h3>
            <div className={`mt-2 p-4 rounded-2xl border ${dark ? 'bg-white/[0.03] border-white/5 text-zinc-300' : 'bg-gray-50 border-gray-100 text-gray-600'} text-[13px] leading-relaxed`}>
              {about}
            </div>
          </div>
        </div>

        <button
          onClick={() => onMarkComplete(quest.id)}
          className="w-full mt-6 py-4 rounded-2xl bg-gradient-to-r from-emerald-500 to-teal-600 text-white font-bold text-[16px] shadow-lg shadow-emerald-500/25 active:scale-[0.99] transition-all flex items-center justify-center gap-2"
        >
          <span>Mark as Completed</span>
          <CheckIcon />
        </button>
      </div>
    </div>
  );
}

// ─── MAIN ROOT APP COMPONENT ──────────────────────────────────────────────────
export default function App() {
  const [dark, setDark] = useState(true);
  const [activeQuestId, setActiveQuestId] = useState(null);
  const [completedQuests, setCompletedQuests] = useState(['q6']); // Example starting state
  const [showA2HS, setShowA2HS] = useState(false);
  const [timeLeft, setTimeLeft] = useState('23:59:01');

  // Trigger iOS Install Modal ONLY ONCE on initial launch
  useEffect(() => {
    const hasSeenA2HS = localStorage.getItem('sq_has_seen_a2hs_v1');
    if (!hasSeenA2HS) {
      // Small delay for smooth entry animation after app renders
      const timer = setTimeout(() => setShowA2HS(true), 800);
      return () => clearTimeout(timer);
    }
  }, []);

  const handleCloseA2HS = () => {
    localStorage.setItem('sq_has_seen_a2hs_v1', 'true');
    setShowA2HS(false);
  };

  const handleToggleDark = () => setDark(!dark);

  const handleMarkComplete = (id) => {
    if (!completedQuests.includes(id)) {
      setCompletedQuests([...completedQuests, id]);
    }
    setActiveQuestId(null);
  };

  const activeQuest = useMemo(() => 
    QUEST_POOL.find(q => q.id === activeQuestId), [activeQuestId]
  );

  const totalXP = useMemo(() => 
    completedQuests.reduce((sum, id) => {
      const q = QUEST_POOL.find(item => item.id === id);
      return sum + (q ? q.xp : 0);
    }, 0), [completedQuests]
  );

  const progressPct = Math.min(100, (completedQuests.length / 5) * 100);
  const bgClass = dark ? 'bg-[#09070f] text-white' : 'bg-[#f8fafc] text-gray-900';

  return (
    <div className={`min-h-screen w-full relative selection:bg-purple-500 selection:text-white font-sans ${bgClass}`}>
      <AnimatedBackground dark={dark} />

      {/* First-Time iOS Add to Home Screen Onboarding Modal */}
      {showA2HS && <IOSInstallModal dark={dark} onClose={handleCloseA2HS} />}

      <div className="max-w-md mx-auto relative z-10 min-h-screen flex flex-col">
        {activeQuest ? (
          <QuestDetailScreen
            quest={activeQuest}
            dark={dark}
            setDark={setDark}
            onToggleDark={handleToggleDark}
            timeLeft={timeLeft}
            onBack={() => setActiveQuestId(null)}
            onMarkComplete={handleMarkComplete}
          />
        ) : (
          /* Main Dashboard View */
          <div className="p-4 flex-1 flex flex-col sq-anim-pop" style={{ paddingTop: 'max(env(safe-area-inset-top), 20px)' }}>
            {/* Top Header */}
            <div className="flex items-center justify-between pb-4">
              <div className="flex items-center gap-2.5">
                <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center shadow-md shadow-indigo-500/20 font-black text-white text-lg">
                  ⚡️
                </div>
                <div>
                  <h1 className="text-lg font-bold leading-none">Side Quests</h1>
                  <span className={`text-[11px] font-mono ${dark ? 'text-purple-400' : 'text-purple-600'}`}>+{totalXP} XP EARNED</span>
                </div>
              </div>

              <button
                onClick={handleToggleDark}
                aria-label="Toggle theme"
                className={`w-9 h-9 rounded-full flex items-center justify-center ${dark ? 'bg-zinc-800/80 text-zinc-300' : 'bg-gray-200/80 text-gray-700'} active:scale-95 transition-all`}
              >
                {dark ? <SunIcon /> : <MoonIcon />}
              </button>
            </div>

            {/* Daily Progress Banner */}
            <div className={`mt-2 p-5 rounded-[24px] border flex items-center justify-between ${dark ? 'bg-white/[0.03] border-white/10 shadow-2xl shadow-black/40' : 'bg-white border-gray-100 shadow-sm'}`}>
              <div>
                <span className={`text-[11px] font-bold uppercase tracking-wider ${dark ? 'text-zinc-400' : 'text-gray-400'}`}>Daily Progress</span>
                <h2 className="text-[18px] font-bold mt-0.5">You&apos;re crushing it today!</h2>
                <p className={`text-[12px] mt-1 ${dark ? 'text-zinc-400' : 'text-gray-500'}`}>{completedQuests.length} of 5 daily goals met</p>
              </div>
              <ProgressRing pct={progressPct} dark={dark} />
            </div>

            {/* Quests List */}
            <div className="mt-6 flex-1">
              <h3 className={`text-[12px] font-bold uppercase tracking-wider mb-3 px-1 ${dark ? 'text-zinc-400' : 'text-gray-400'}`}>
                Available Quests
              </h3>
              
              <div className="space-y-2.5 pb-12">
                {QUEST_POOL.slice(0, 7).map((q) => {
                  const isDone = completedQuests.includes(q.id);
                  const theme = QUEST_THEME[q.id] || { icon: 'target', grad: 'from-indigo-500 to-purple-600' };
                  
                  return (
                    <div
                      key={q.id}
                      onClick={() => !isDone && setActiveQuestId(q.id)}
                      className={`group p-3.5 rounded-[20px] border flex items-center justify-between transition-all duration-200 ${
                        isDone 
                          ? dark ? 'bg-white/[0.01] border-white/5 opacity-50' : 'bg-gray-50 border-gray-100 opacity-60'
                          : dark ? 'bg-white/[0.04] border-white/10 hover:bg-white/[0.07] cursor-pointer active:scale-[0.98]' : 'bg-white border-gray-100 hover:border-gray-200 shadow-sm cursor-pointer active:scale-[0.98]'
                      }`}
                    >
                      <div className="flex items-center gap-3.5">
                        <QuestIconBadge questId={q.id} size={44} dark={dark} />
                        <div>
                          <p className={`text-[14px] font-bold leading-snug ${isDone ? 'line-through' : ''}`}>
                            {q.text}
                          </p>
                          <span className={`text-[11px] font-semibold ${dark ? 'text-purple-400' : 'text-purple-600'}`}>
                            +{q.xp} XP {q.reps ? `• ${q.reps} Reps` : q.duration ? `• ${q.duration / 60} Mins` : ''}
                          </span>
                        </div>
                      </div>

                      <div className={`w-7 h-7 rounded-full flex items-center justify-center border transition-colors ${
                        isDone 
                          ? 'bg-emerald-500 border-emerald-500 text-white' 
                          : dark ? 'border-white/20 group-hover:border-white/40 text-transparent' : 'border-gray-300 group-hover:border-gray-400 text-transparent'
                      }`}>
                        <CheckIcon />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
