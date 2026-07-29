// ─── POSE MATH ENGINE ──────────────────────────────────────────────────────
// Turns raw MediaPipe PoseLandmarker output into rep counts. Runs inside
// poseWorker.js, off the main thread. Framework-agnostic on purpose so it
// stays cheap to unit-test in isolation.

// BlazePose 33-point layout (the subset we actually use).
const POSE_LANDMARKS = {
  nose: 0,
  left_shoulder: 11, right_shoulder: 12,
  left_elbow: 13, right_elbow: 14,
  left_wrist: 15, right_wrist: 16,
  left_hip: 23, right_hip: 24,
  left_knee: 25, right_knee: 26,
  left_ankle: 27, right_ankle: 28,
};

export const POSE_VISIBILITY_THRESHOLD = 0.55;
const PHASE_ENTER_THRESHOLD = 0.55;
const PHASE_MARGIN = 0.08;
const REP_MIN_DURATION_MS = 380;
const SMOOTH_ALPHA = 0.45;

export const REP_SCAN_INTERVAL_MS = 130;

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// Smooth 0→1 ease so scores near the threshold ramp gently instead of
// flipping on a hard cutoff — this is what makes the rep counter forgiving
// without losing precision at the extremes.
function smoothstep(edge0, edge1, x) {
  if (edge0 === edge1) return x >= edge1 ? 1 : 0;
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function angleBetween(a, b, c) {
  const abx = a.x - b.x, aby = a.y - b.y;
  const cbx = c.x - b.x, cby = c.y - b.y;
  const magA = Math.hypot(abx, aby);
  const magC = Math.hypot(cbx, cby);
  if (magA === 0 || magC === 0) return null;
  const cos = clamp((abx * cbx + aby * cby) / (magA * magC), -1, 1);
  return Math.acos(cos) * (180 / Math.PI);
}

function getPoint(landmarks, name, side) {
  const idx = POSE_LANDMARKS[`${side}_${name}`];
  const point = idx != null ? landmarks[idx] : null;
  return point && (point.visibility ?? 1) >= POSE_VISIBILITY_THRESHOLD ? point : null;
}

function angleForSide(landmarks, triplet, side) {
  const a = getPoint(landmarks, triplet[0], side);
  const b = getPoint(landmarks, triplet[1], side);
  const c = getPoint(landmarks, triplet[2], side);
  if (!a || !b || !c) return null;
  return angleBetween(a, b, c);
}

function resolveAngle(landmarks, triplet, strategy) {
  const left = angleForSide(landmarks, triplet, 'left');
  const right = angleForSide(landmarks, triplet, 'right');
  if (left == null && right == null) return null;
  if (left == null) return right;
  if (right == null) return left;
  if (strategy === 'min') return Math.min(left, right);
  if (strategy === 'max') return Math.max(left, right);
  return (left + right) / 2;
}

function midpointVisible(landmarks, name) {
  const left = getPoint(landmarks, name, 'left');
  const right = getPoint(landmarks, name, 'right');
  if (!left || !right) return null;
  return { x: (left.x + right.x) / 2, y: (left.y + right.y) / 2 };
}

// Returns { targetScore, resetScore, visible } — both scores in [0, 1],
// where "target" is the effortful/flexed position (bottom of a pushup, top
// of a pullup, peak of a jump) and "reset" is the relaxed/extended one.
export function computePoseSignal(landmarks, profile, tracker) {
  if (profile.mode === 'angle') {
    const angle = resolveAngle(landmarks, profile.triplet, profile.side);
    if (angle == null) return { targetScore: 0, resetScore: 0, visible: false };
    const t = smoothstep(profile.downAngle, profile.upAngle, angle);
    return { targetScore: 1 - t, resetScore: t, visible: true };
  }

  // Vertical mode (jump rope): track hip height against a slow rolling
  // baseline so it self-calibrates to however far the camera is standing.
  const shoulderMid = midpointVisible(landmarks, 'shoulder');
  const hipMid = midpointVisible(landmarks, 'hip');
  if (!shoulderMid || !hipMid) return { targetScore: 0, resetScore: 0, visible: false };

  const torsoLength = Math.hypot(shoulderMid.x - hipMid.x, shoulderMid.y - hipMid.y) || 0.0001;
  if (tracker.verticalBaseline == null) tracker.verticalBaseline = hipMid.y;
  else tracker.verticalBaseline = tracker.verticalBaseline * 0.98 + hipMid.y * 0.02;

  // Image y grows downward, so a rise (jump) makes hipMid.y smaller than baseline.
  const displacement = (tracker.verticalBaseline - hipMid.y) / torsoLength;
  const { jumpRatio } = profile;
  const targetScore = smoothstep(jumpRatio * 0.3, jumpRatio, displacement);
  const resetScore = 1 - smoothstep(0, jumpRatio * 0.3, displacement);
  return { targetScore, resetScore, visible: true };
}

export function createRepTracker() {
  return {
    phase: 'seek-target',
    reps: 0,
    smoothedTarget: 0,
    smoothedReset: 0,
    lastRepAt: 0,
    verticalBaseline: null,
  };
}

export function advanceRepTracker(tracker, { targetScore, resetScore, visible, now }) {
  if (!visible) {
    return {
      counted: false,
      phaseChanged: false,
      phase: tracker.phase,
      confidence: Math.max(tracker.smoothedTarget, tracker.smoothedReset),
      visible: false,
    };
  }

  tracker.smoothedTarget = tracker.smoothedTarget * (1 - SMOOTH_ALPHA) + targetScore * SMOOTH_ALPHA;
  tracker.smoothedReset = tracker.smoothedReset * (1 - SMOOTH_ALPHA) + resetScore * SMOOTH_ALPHA;

  let counted = false;
  let phaseChanged = false;

  if (tracker.phase === 'seek-target') {
    if (tracker.smoothedTarget >= PHASE_ENTER_THRESHOLD && tracker.smoothedTarget > tracker.smoothedReset + PHASE_MARGIN) {
      tracker.phase = 'seek-reset';
      phaseChanged = true;
    }
  } else if (tracker.smoothedReset >= PHASE_ENTER_THRESHOLD && tracker.smoothedReset > tracker.smoothedTarget + PHASE_MARGIN) {
    if (now - tracker.lastRepAt >= REP_MIN_DURATION_MS) {
      tracker.reps += 1;
      tracker.lastRepAt = now;
      counted = true;
    }
    tracker.phase = 'seek-target';
    phaseChanged = true;
  }

  return {
    counted,
    phaseChanged,
    phase: tracker.phase,
    confidence: Math.max(tracker.smoothedTarget, tracker.smoothedReset),
    visible: true,
  };
}
