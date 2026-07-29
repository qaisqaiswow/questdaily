import { useCallback, useEffect, useRef, useState } from 'react';

// Owns the pose worker's lifecycle: creates it lazily when `active` turns
// true, tears it down when the modal closes or the quest changes type, and
// exposes a backpressure-safe `sendFrame` (call it, then wait for the next
// result before sending another — the caller decides that cadence).
export function usePoseWorker({ active, questId, targetReps, onResult, onError, retryKey = 0 }) {
  const workerRef = useRef(null);
  const onResultRef = useRef(onResult);
  const onErrorRef = useRef(onError);
  const [ready, setReady] = useState(false);

  onResultRef.current = onResult;
  onErrorRef.current = onError;

  useEffect(() => {
    if (!active) return undefined;
    setReady(false);
    const worker = new Worker(new URL('../workers/poseWorker.js', import.meta.url), { type: 'module' });
    workerRef.current = worker;

    worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'ready') setReady(true);
      else if (msg.type === 'result') onResultRef.current?.(msg);
      else if (msg.type === 'error') onErrorRef.current?.(msg.message);
    };
    worker.onerror = (event) => onErrorRef.current?.(event.message || 'Pose worker crashed.');
    worker.postMessage({ type: 'init' });

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, [active, retryKey]);

  useEffect(() => {
    if (!ready) return;
    workerRef.current?.postMessage({ type: 'configure', questId, targetReps });
  }, [ready, questId, targetReps]);

  const sendFrame = useCallback((bitmap, timestamp, requestId) => {
    workerRef.current?.postMessage({ type: 'frame', bitmap, timestamp, requestId }, [bitmap]);
  }, []);

  const reset = useCallback(() => {
    workerRef.current?.postMessage({ type: 'reset' });
  }, []);

  return { ready, sendFrame, reset };
}
