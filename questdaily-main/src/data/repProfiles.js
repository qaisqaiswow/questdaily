// ─── REP PROFILES ──────────────────────────────────────────────────────────
// Each profile describes, in plain joint-angle terms, what "fully down" and
// "fully up" look like for one rep of an exercise. src/lib/poseMath.js turns
// these into a continuous 0-1 completion score per frame; the angle numbers
// below are sane starting points and are worth re-tuning against real footage
// once you can run the app locally.
//
// mode: 'angle'    — angle at the middle joint of `triplet`, combined across
//                     left/right via `side` ('min' | 'max' | 'avg').
// mode: 'vertical' — normalized vertical displacement of the hips from a
//                     slow-moving baseline (for exercises that aren't a joint
//                     bend, like jump rope).
export const REP_PROFILES = {
  q1: { // pushups — elbow angle
    mode: 'angle',
    triplet: ['shoulder', 'elbow', 'wrist'],
    side: 'min',
    downAngle: 95,
    upAngle: 165,
    cue: 'Lower your chest down, then push completely back up.',
  },
  q2: { // squats — knee angle
    mode: 'angle',
    triplet: ['hip', 'knee', 'ankle'],
    side: 'min',
    downAngle: 100,
    upAngle: 165,
    cue: 'Drop into a deep squat, then return to a full upright stand.',
  },
  q4: { // pullups — elbow angle (chin near bar = bent elbow)
    mode: 'angle',
    triplet: ['shoulder', 'elbow', 'wrist'],
    side: 'min',
    downAngle: 80,
    upAngle: 160,
    cue: 'Pull up clear to the bar, then drop back to straight arms.',
  },
  q12: { // situps — torso angle at the hip
    mode: 'angle',
    triplet: ['shoulder', 'hip', 'knee'],
    side: 'avg',
    downAngle: 75,
    upAngle: 140,
    cue: 'Crunch your torso all the way up, then lie back down with control.',
  },
  q23: { // lunges — front knee angle
    mode: 'angle',
    triplet: ['hip', 'knee', 'ankle'],
    side: 'min',
    downAngle: 100,
    upAngle: 165,
    cue: 'Step deep down into the lunge stance, then rise all the way up.',
  },
  q17: { // jump rope — vertical hip oscillation, not a joint bend
    mode: 'vertical',
    jumpRatio: 0.12, // fraction of torso length the hips must rise to count as airborne
    cue: 'Keep continuous, springy jumps inside the camera frame.',
  },
};
