"use client";

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { pipeline, env } from '@huggingface/transformers';
import { Analytics } from '@vercel/analytics/react';

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

// Highly descriptive labels improve CLIP zero-shot accuracy drastically
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
const REP_MIN_DURATION_MS = 600;
const REP_FORM_CONFIDENCE = 0.35;

const REP_PROFILES = {
  q1: {
    target: ['a photo of a person at the bottom of a pushup with chest touching the floor', 'a person doing a pushup with elbows deeply bent'],
    reset: ['a photo of a person in a straight arm plank position', 'a person at the top of a pushup with straight arms'],
    cue: 'Lower your chest down, then push completely back up.',
  },
  q2: {
    target: ['a photo of a person at the bottom of a deep squat with bent knees', 'a person squatting with thighs parallel to the floor'],
    reset: ['a photo of a person standing tall and upright after a squat', 'a person standing normally with straight legs'],
    cue: 'Drop into a deep squat, then return to a full upright stand.',
  },
  q4: {
    target: ['a photo of a person at the top of a pullup with chin over the bar', 'a person pulling their body up on a bar'],
    reset: ['a photo of a person hanging freely from a pullup bar with straight arms', 'a person hanging from a bar extending arms'],
    cue: 'Pull up clear to the bar, then drop back to straight arms.',
  },
  q12: {
    target: ['a photo of a person at the top of a situp with torso lifted off the ground', 'a person crunching their abs upward'],
    reset: ['a photo of a person lying flat on their back on the floor', 'a person resting on their back'],
    cue: 'Crunch your torso all the way up, then lie completely flat.',
  },
  q17: {
    target: ['a photo of a person airborne while jumping rope', 'a person jumping in the air skipping rope'],
    reset: ['a photo of a person standing on the ground holding a jump rope', 'a person standing still with a skipping rope'],
    cue: 'Keep continuous active jumps inside the camera frame.',
  },
  q23: {
    target: ['a photo of a person at the bottom of a deep lunge stance with bent knees', 'a person in a deep split-stance lunge'],
    reset: ['a photo of a person standing upright after finishing a lunge', 'a person standing tall with feet together'],
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
  const alpha = 0.65; 
  tracker.lastSampleAt = now;
  tracker.smoothedTarget = tracker.smoothedTarget * (1 - alpha) + targetScore * alpha;
  tracker.smoothedReset = tracker.smoothedReset * (1 - alpha) + resetScore * alpha;

  const formScore = Math.max(tracker.smoothedTarget, tracker.smoothedReset);
  let counted = false;
  let phaseChanged = false;

  if (tracker.phase === 'seek-target') {
    if (tracker.smoothedTarget >= REP_FORM_CONFIDENCE && tracker.smoothedTarget > tracker.smoothedReset + 0.05) {
      tracker.phase = 'seek-reset';
      phaseChanged = true;
    }
  } else if (tracker.phase === 'seek-reset') {
    if (tracker.smoothedReset >= REP_FORM_CONFIDENCE && tracker.smoothedReset > tracker.smoothedTarget + 0.05) {
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

// ─── AMBIENT ANIMATED BACKGROUND ───────────────────────────────────────────────
// Soft diagonal light streaks that drift slowly behind the whole app, like the
// reference design. Pure CSS keyframes, no canvas — cheap to run.
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
          position: absolute;
          width: 180%;
          height: 1.5px;
          left: -40%;
          background: linear-gradient(90deg, transparent, var(--sq-streak-c1), var(--sq-streak-c2), transparent);
          transform-origin: center;
          filter: blur(0.5px);
          opacity: 0.7;
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
          position: absolute;
          width: 260px; height: 260px;
          border-radius: 999px;
          filter: blur(70px);
          opacity: var(--sq-glow-o);
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

// ─── PER-QUEST ICONOGRAPHY ─────────────────────────────────────────────────────
// Every quest gets its own glyph + gradient theme, echoing the "cup" treatment
// in the reference: a soft gradient disc, an inner ring, and a centered icon.
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

// id -> { icon, gradient (tailwind classes), ring (border/glow color) }
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

const QUEST_QUOTES = [
  "Small steps every day lead to big changes.",
  "Discipline is choosing between what you want now and what you want most.",
  "Progress, not perfection.",
  "The body achieves what the mind believes.",
  "One quest at a time.",
  "Consistency beats intensity.",
  "You didn't come this far to only come this far.",
];

// A specific, on-theme quote for each quest — shown on its completion screen.
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

// Circular ring used for the "Daily Progress" card
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

// ─── QUEST DETAIL SCREEN ────────────────────────────────────────────────────────
function QuestDetailScreen({ quest, dark, timeLeft, onBack, onMarkComplete }) {
  const txt = dark ? 'text-white' : 'text-gray-900';
  const sub = dark ? 'text-zinc-400' : 'text-gray-500';
  const pill = dark ? 'bg-zinc-800/80' : 'bg-gray-100';
  const cardBg = dark ? 'bg-zinc-900/70' : 'bg-white';
  const theme = QUEST_THEME[quest.id] || { grad: 'from-indigo-500 to-purple-600' };
  const about = QUEST_ABOUT[quest.id] || 'Stay consistent — every quest you complete adds up to real progress.';

  return (
    <div className="sq-anim-pop relative z-10">
      <div className="flex items-center justify-between px-4 pb-3" style={{ paddingTop: 'max(env(safe-area-inset-top), 18px)' }}>
        <button onClick={onBack}
          className={`w-9 h-9 rounded-full flex items-center justify-center ${dark ? 'bg-zinc-800/80 text-zinc-300' : 'bg-gray-100 text-gray-600'} active:opacity-70`}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6"/>
          </svg>
        </button>
        <div className={`flex items-center gap-1 px-2.5 py-1 rounded-full ${pill}`}>
          <span className="text-[10px]">⏱</span>
          <span className={`text-[11px] font-mono font-medium ${dark ? 'text-zinc-300' : 'text-gray-600'}`}>{timeLeft}</span>
        </div>
        <div className={`w-9 h-9 rounded-full flex items-center justify-center ${pill}`}>
          <span className="text-[15px]">☀️</span>
        </div>
      </div>

      <div className="px-4">
        <div
          className={`relative overflow-hidden rounded-[26px] px-6 pt-10 pb-8 flex flex-col items-center text-center bg-gradient-to-b ${dark ? 'from-[#1c1530] to-[#0d0a17]' : 'from-indigo-50 to-white'} border ${dark ? 'border-white/10' : 'border-gray-100'}`}
        >
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

      <div className="px-4 mt-5">
        <p className={`text-[11px] font-semibold uppercase tracking-widest mb-1.5 px-1 ${dark ? 'text-zinc-500' : 'text-gray-400'}`}>
          About this quest
        </p>
        <div className={`${cardBg} rounded-[16px] p-4 border ${dark ? 'border-white/5' : 'border-gray-100'}`}>
          <p className={`text-[14px] leading-relaxed ${sub}`}>{about}</p>
        </div>
      </div>

      <div className="px-4 mt-6">
        <button
          onClick={onMarkComplete}
          className={`w-full py-4 rounded-[16px] text-[15px] font-semibold text-white bg-gradient-to-r ${theme.grad} shadow-lg active:scale-[0.97] transition-transform`}
          style={{ boxShadow: '0 10px 30px -10px rgba(139,92,246,0.6)' }}
        >
          Mark as Completed
        </button>
      </div>
    </div>
  );
}

// ─── COMPLETION SCREEN ───────────────────────────────────────────────────────────
function CompletionScreen({ quest, dark, timeLeft, onBack }) {
  const txt = dark ? 'text-white' : 'text-gray-900';
  const sub = dark ? 'text-zinc-400' : 'text-gray-500';
  const pill = dark ? 'bg-zinc-800/80' : 'bg-gray-100';
  const cardBg = dark ? 'bg-zinc-900/70' : 'bg-white';
  const quote = useMemo(
    () => QUEST_QUOTE[quest?.id] || QUEST_QUOTES[Math.floor(Math.random() * QUEST_QUOTES.length)],
    [quest?.id]
  );

  return (
    <div className="sq-anim-pop relative z-10">
      <div className="flex items-center justify-between px-4 pb-3" style={{ paddingTop: 'max(env(safe-area-inset-top), 18px)' }}>
        <button onClick={onBack}
          className={`w-9 h-9 rounded-full flex items-center justify-center ${dark ? 'bg-zinc-800/80 text-zinc-300' : 'bg-gray-100 text-gray-600'} active:opacity-70`}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6"/>
          </svg>
        </button>
        <div className={`flex items-center gap-1 px-2.5 py-1 rounded-full ${pill}`}>
          <span className="text-[10px]">⏱</span>
          <span className={`text-[11px] font-mono font-medium ${dark ? 'text-zinc-300' : 'text-gray-600'}`}>{timeLeft}</span>
        </div>
        <div className={`w-9 h-9 rounded-full flex items-center justify-center ${pill}`}>
          <span className="text-[15px]">☀️</span>
        </div>
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

      <div className="px-4 mt-5">
        <div className={`${cardBg} rounded-[16px] p-5 border ${dark ? 'border-white/5' : 'border-gray-100'} relative`}>
          <span className={`absolute top-2 left-3 text-3xl leading-none ${dark ? 'text-zinc-700' : 'text-gray-200'}`}>&ldquo;</span>
          <p className={`text-[15px] font-medium text-center leading-relaxed px-3 ${txt}`}>{quote}</p>
          <span className={`absolute bottom-1 right-3 text-3xl leading-none ${dark ? 'text-zinc-700' : 'text-gray-200'}`}>&rdquo;</span>
        </div>
      </div>

      <div className="px-4 mt-6">
        <button
          onClick={onBack}
          className={`w-full py-4 rounded-[16px] text-[15px] font-semibold ${dark ? 'bg-zinc-800 text-white' : 'bg-gray-100 text-gray-800'} active:opacity-70`}
        >
          Back to Quests
        </button>
      </div>
    </div>
  );
}

// ─── CAMERA / AI MODAL ────────────────────────────────────────────────────────
function CameraModal({ quest, onConfirm, onCancel, dark }) {
  const videoRef     = useRef(null);
  const aiCanvasRef  = useRef(null); 
  const streamRef    = useRef(null);
  const scanTimerRef = useRef(null);
  const isScanningRef = useRef(false);
  const scanRunRef = useRef(0);
  const cameraSessionRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const confirmedRef = useRef(false);
  const passStreakRef = useRef(0);
  const smoothedActionRef = useRef(0);
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

  const [secondsLeft, setSecondsLeft] = useState(quest.duration || 0);
  const [timerRunning, setTimerRunning] = useState(false);

  const labels = QUEST_LABELS[quest.id];
  const questType = labels?.type || 'action';
  const repProfile = questType === 'reps' ? REP_PROFILES[quest.id] : null;

  const activeNegatives = useMemo(() => getNegativeLabels(questType), [questType]);
  
  const classifierLabels = useMemo(() => {
    if (questType === 'reps' && repProfile) {
      return [...new Set([...repProfile.target, ...repProfile.reset, ...activeNegatives])];
    }
    return [...new Set([...(labels?.activity ?? []), ...activeNegatives])];
  }, [questType, labels, repProfile, activeNegatives]);

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

        const negScore = getMaxLabelScore(results, activeNegatives);
        setLastLabel(results[0]?.label ?? '');

        if (quest.reps && repProfile) {
          const update = advanceRepTracker(repTrackerRef.current, {
            targetScore: getMaxLabelScore(results, repProfile.target),
            resetScore: getMaxLabelScore(results, repProfile.reset),
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
          const actScore = getMaxLabelScore(results, labels.activity);
          smoothedActionRef.current = smoothedActionRef.current * 0.5 + actScore * 0.5;
          const displayScore = smoothedActionRef.current;
          setLiveScore(Math.round(displayScore * 100));
          const passed = displayScore > negScore && displayScore >= PASS_THRESHOLD;
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
    smoothedActionRef.current = 0;
    confirmedRef.current = false;
    lastVideoTimeRef.current = -1;
    resetRepTracking();
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

  const meterColor = liveScore >= (quest.reps ? REP_FORM_CONFIDENCE : PASS_THRESHOLD) * 100 ? 'bg-emerald-500' : liveScore >= 15 ? 'bg-amber-400' : 'bg-rose-500';

  const bg     = dark ? 'bg-zinc-950'  : 'bg-white';
  const border = dark ? 'border-white/10' : 'border-gray-200';
  const txt    = dark ? 'text-white'   : 'text-gray-900';
  const sub    = dark ? 'text-zinc-400' : 'text-gray-500';
  const pill   = dark ? 'bg-zinc-800/80'  : 'bg-gray-100';
  const accentText = dark ? 'text-violet-400' : 'text-violet-600';

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center" style={{ background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}>
      <div className={`relative ${bg} w-full max-w-lg rounded-t-[28px] overflow-hidden shadow-2xl border-t ${border}`} style={{ paddingBottom: 'max(env(safe-area-inset-bottom), 16px)' }}>
        <div className="pointer-events-none absolute inset-x-0 top-0 h-40 overflow-hidden opacity-60">
          <AnimatedBackground dark={dark} />
        </div>

        <div className="relative flex justify-center pt-3 pb-1">
          <div className={`w-10 h-1 rounded-full ${dark ? 'bg-zinc-600' : 'bg-gray-300'}`} />
        </div>

        <div className={`relative flex items-center justify-between px-5 py-3 border-b ${border}`}>
          <button onClick={onCancel} className={`text-sm font-medium ${accentText}`}>Cancel</button>
          <div className="text-center">
            <p className={`text-sm font-semibold ${txt}`}>AI Verification</p>
            <p className={`text-xs ${sub} mt-0.5 max-w-[240px] truncate`}>{uiSubtext}</p>
          </div>
          <div className="w-14" />
        </div>

        <div className="relative bg-black" style={{ aspectRatio: '4/3' }}>
          <video ref={videoRef} autoPlay playsInline muted
              className={`w-full h-full object-cover ${phase === 'live' && !uploadedProof ? 'opacity-100' : 'opacity-0'}`} />

          {phase === 'live' && !uploadedProof && labels?.bodyParts && (
            <div className="absolute inset-0 pointer-events-none border-[3px] border-dashed border-violet-500/30 m-4 rounded-xl animate-pulse z-10">
              <div className="absolute top-3 left-3 bg-black/70 backdrop-blur-md px-2.5 py-1.5 rounded-lg border border-violet-500/40 text-[10px] font-mono tracking-wider text-violet-300">
                <div className="flex items-center gap-1.5 mb-1 text-white uppercase font-bold text-[11px]">
                  <span className="w-2 h-2 rounded-full bg-violet-400 animate-ping inline-block" />
                  Biometric Engine Active
                </div>
                <div className="text-white/60 text-[9px] mb-0.5">Tracking Matrix Focus Points:</div>
                <div className="flex flex-wrap gap-1 max-w-[180px] mt-1">
                  {labels.bodyParts.map((part) => (
                    <span key={part} className="bg-violet-950 text-violet-300 px-1.5 py-0.5 rounded border border-violet-800/60 font-semibold">
                      {part}
                    </span>
                  ))}
                </div>
              </div>
              
              <div className="absolute top-0 left-0 w-4 h-4 border-t-4 border-l-4 border-violet-400" />
              <div className="absolute top-0 right-0 w-4 h-4 border-t-4 border-r-4 border-violet-400" />
              <div className="absolute bottom-0 left-0 w-4 h-4 border-b-4 border-l-4 border-violet-400" />
              <div className="absolute bottom-0 right-0 w-4 h-4 border-b-4 border-r-4 border-violet-400" />
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
                        <p className="text-white text-xs font-bold tracking-wide bg-green-600/80 px-2 py-0.5 rounded-md">
                            {repsDone} / {quest.reps} REPS
                        </p>
                      ) : (
                        Array.from({ length: REQUIRED_PASSES }).map((_, i) => (
                            <div key={i} className={`w-1.5 h-1.5 rounded-full transition-colors duration-300 ${i < passStreak ? 'bg-green-400' : 'bg-white/30'}`} />
                        ))
                      )}
                    </div>
                  </div>
                  
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
                    <p className="text-white/80 font-medium text-[11px] mt-2 bg-black/40 p-1.5 rounded border border-white/10">
                      {repPhase === 'seek-target'
                        ? `Target: ${repProfile?.cue}`
                        : 'Position Locked! Return to complete this rep.'}
                    </p>
                  )}
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
                <div className="h-full bg-gradient-to-r from-indigo-400 to-violet-500 transition-all duration-300 rounded-full"
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
                    : 'bg-green-500 text-white active:bg-green-600'
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
                    ? 'bg-gradient-to-r from-indigo-500 to-violet-600 text-white shadow-lg active:scale-[0.97]'
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
                <label className={`flex-1 py-3 rounded-[14px] text-sm font-medium ${pill} ${sub} text-center cursor-pointer active:opacity-70 ${questType === 'map' ? 'ring-2 ring-violet-500' : ''}`}>
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
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem('sq_dark');
    if (saved !== null) return saved === 'true';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('sq_dark', dark);
  }, [dark]);

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
  const [detailQuest,     setDetailQuest]     = useState(null);
  const [completionQuest, setCompletionQuest] = useState(null);

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
      const m = Math.floor((remaining /     60_000) % 60);
      const s = Math.floor((remaining /      1_000) % 60);
      setTimeLeft(`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`);
    }, 1000);
    return () => clearInterval(id);
  }, [lastReset, quests.length, generateNewQuests]);

  useEffect(() => {
    localStorage.setItem('sq_level',       level);
    localStorage.setItem('sq_xp',          xp);
    localStorage.setItem('sq_quests',    JSON.stringify(quests));
    localStorage.setItem('sq_lastReset', lastReset);
  }, [level, xp, quests, lastReset]);
  useEffect(() => { localStorage.setItem('sq_proofs', JSON.stringify(proofImages)); }, [proofImages]);

  const applyXpChange = useCallback((amount) => {
    let newXp = xpRef.current + amount, newLevel = levelRef.current;
    while (newXp >= newLevel * 100)    { newXp -= newLevel * 100; newLevel++; }
    while (newXp < 0 && newLevel > 1) { newLevel--; newXp += newLevel * 100; }
    if (newLevel === 1 && newXp < 0)    newXp = 0;
    setXp(newXp); setLevel(newLevel);
  }, []);

  const handleQuestClick = (quest) => {
    if (quest.completed) {
      setQuests(prev => prev.map(q => q.id === quest.id ? { ...q, completed: false } : q));
      applyXpChange(-quest.xp);
      setProofImages(prev => { const n = { ...prev }; delete n[quest.id]; return n; });
    } else {
      setDetailQuest(quest);
    }
  };

  const handleProofConfirm = (questId, img) => {
    const quest = quests.find(q => q.id === questId);
    setProofImages(prev => ({ ...prev, [questId]: img }));
    setQuests(prev => prev.map(q => q.id === questId ? { ...q, completed: true } : q));
    applyXpChange(quest?.xp ?? 0);
    setProofModal(null);
    setDetailQuest(null);
    setCompletionQuest(quest || null);
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
      <div className="relative w-full max-w-[430px] flex flex-col min-h-screen overflow-hidden">

      <AnimatedBackground dark={dark} />

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

      {completionQuest ? (
        <CompletionScreen quest={completionQuest} dark={dark} timeLeft={timeLeft}
          onBack={() => setCompletionQuest(null)} />
      ) : detailQuest ? (
        <QuestDetailScreen quest={detailQuest} dark={dark} timeLeft={timeLeft}
          onBack={() => setDetailQuest(null)}
          onMarkComplete={() => setProofModal(detailQuest)} />
      ) : (
      <>
      {/* Header */}
      <div className={`relative z-10 safe-top px-3 pb-3 transition-colors duration-200`}>
        <div className="flex items-center justify-between pt-2">
          <div className="flex items-center gap-2.5">
            <LogoIcon size={34} dark={dark} />
            <h1 className={`text-[22px] font-bold tracking-tight ${txt}`}>Side Quests</h1>
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
          <span className={`text-[11px] ${sub}`}>Novice Adventurer</span>
          <span className={`ml-auto text-[11px] ${dark ? 'text-zinc-500' : 'text-gray-400'}`}>{xp} / {xpRequired} XP</span>
        </div>
      </div>

      {/* Quest List Containers */}
      <div className="relative z-10 flex-1 scroll-ios px-3 pt-1 pb-6 space-y-5">

        {/* Daily Progress card */}
        <div className={`${dark ? 'bg-zinc-900/70 border border-white/5' : 'bg-white border border-gray-100'} rounded-[16px] px-4 py-4 flex items-center justify-between sq-anim-pop`}>
          <div>
            <p className={`text-[14px] font-semibold ${txt}`}>Daily Progress</p>
            <p className={`text-[12px] mt-0.5 ${sub}`}>{completedCount} / {quests.length} completed</p>
          </div>
          <ProgressRing pct={dailyPct} dark={dark} />
        </div>

        <div>
          <p className={`text-[11px] font-semibold uppercase tracking-widest ${secLabel} mb-1.5 px-1`}>
            Today's Objectives
          </p>

          <div className={`${dark ? 'bg-zinc-900/70 border border-white/5' : 'bg-[#F2F2F7]'} rounded-[14px] overflow-hidden`}>
            {quests.map((quest, i) => (
              <div key={quest.id}>
                {i > 0 && <div className={`border-t ${sep} ml-[58px]`} />}
                <button
                  onClick={() => handleQuestClick(quest)}
                  className={`w-full flex items-center gap-3.5 px-4 py-[18px] text-left active:opacity-60 transition-opacity`}
                >
                  <div className={`w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center border-2 transition-all duration-200 ${
                    quest.completed
                      ? 'bg-[#34C759] border-[#34C759]'
                      : dark ? 'border-zinc-600' : 'border-gray-300'
                  }`}>
                    {quest.completed && (
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round">
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
                        ? 'text-[#34C759]'
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
      </>
      )}

      <div className="safe-bottom" />
      </div>
      <Analytics />
    </div>
  );
}
