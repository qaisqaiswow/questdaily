// Runs MediaPipe PoseLandmarker entirely off the main thread. The main thread
// captures an ImageBitmap from the <video> element and transfers ownership of
// it here (zero-copy), we run pose detection + the rep state machine, and
// post back just the small result the UI needs to render.
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import { REP_PROFILES } from '../data/repProfiles.js';
import { computePoseSignal, createRepTracker, advanceRepTracker } from '../lib/poseMath.js';

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

let landmarkerPromise = null;
let tracker = createRepTracker();
let currentQuestId = null;
let targetReps = 0;

async function createLandmarker(delegate) {
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numPoses: 1,
  });
}

function ensureLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = createLandmarker('GPU').catch(() => createLandmarker('CPU')).catch(err => {
      landmarkerPromise = null;
      throw err;
    });
  }
  return landmarkerPromise;
}

self.onmessage = async (event) => {
  const msg = event.data;

  if (msg.type === 'init') {
    try {
      await ensureLandmarker();
      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'error', message: err?.message || 'The pose model could not be loaded.' });
    }
    return;
  }

  if (msg.type === 'configure') {
    currentQuestId = msg.questId;
    targetReps = msg.targetReps || 0;
    tracker = createRepTracker();
    return;
  }

  if (msg.type === 'reset') {
    tracker = createRepTracker();
    return;
  }

  if (msg.type === 'frame') {
    const { bitmap, timestamp, requestId } = msg;
    try {
      const profile = REP_PROFILES[currentQuestId];
      if (!profile) {
        self.postMessage({ type: 'result', requestId, reps: tracker.reps, phase: tracker.phase, confidence: 0, visible: false, done: false });
        return;
      }

      const model = await ensureLandmarker();
      const result = model.detectForVideo(bitmap, timestamp);
      const landmarks = result?.landmarks?.[0] ?? null;

      const update = landmarks
        ? advanceRepTracker(tracker, { ...computePoseSignal(landmarks, profile, tracker), now: timestamp })
        : { counted: false, phase: tracker.phase, confidence: 0, visible: false };

      self.postMessage({
        type: 'result',
        requestId,
        reps: tracker.reps,
        phase: update.phase,
        confidence: update.confidence,
        visible: update.visible,
        done: targetReps > 0 && tracker.reps >= targetReps,
      });
    } catch (err) {
      // Still resolve this frame's pending promise (with a neutral result) so
      // the main-thread frame loop never stalls waiting on a dropped frame —
      // the separate 'error' message surfaces the message for the UI banner.
      self.postMessage({ type: 'result', requestId, reps: tracker.reps, phase: tracker.phase, confidence: 0, visible: false, done: false });
      self.postMessage({ type: 'error', message: err?.message || 'Pose analysis failed.' });
    } finally {
      bitmap.close();
    }
  }
};
