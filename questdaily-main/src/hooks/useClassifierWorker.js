import { useCallback, useEffect, useRef, useState } from 'react';

// Same idea as usePoseWorker, for the zero-shot classifier. Kept as a
// separate hook/worker so a rep quest never has to load this model at all.
export function useClassifierWorker({ active, onResult, onError, retryKey = 0 }) {
  const workerRef = useRef(null);
  const onResultRef = useRef(onResult);
  const onErrorRef = useRef(onError);
  const [ready, setReady] = useState(false);
  const [progress, setProgress] = useState(0);

  onResultRef.current = onResult;
  onErrorRef.current = onError;

  useEffect(() => {
    if (!active) return undefined;
    setReady(false);
    setProgress(0);
    const worker = new Worker(new URL('../workers/classifierWorker.js', import.meta.url), { type: 'module' });
    workerRef.current = worker;

    worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'ready') setReady(true);
      else if (msg.type === 'progress') setProgress(msg.percent);
      else if (msg.type === 'result') onResultRef.current?.(msg);
      else if (msg.type === 'error') onErrorRef.current?.(msg.message, msg.requestId);
    };
    worker.onerror = (event) => onErrorRef.current?.(event.message || 'Classifier worker crashed.');
    worker.postMessage({ type: 'init' });

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, [active, retryKey]);

  const classify = useCallback((dataUrl, labels, requestId) => {
    workerRef.current?.postMessage({ type: 'classify', dataUrl, labels, requestId });
  }, []);

  return { ready, progress, classify };
}
