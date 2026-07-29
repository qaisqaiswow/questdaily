import { useEffect, useRef } from 'react';

// Drives a "capture a video frame, hand it off, wait for it to be processed,
// repeat" loop without ever queuing up frames faster than the AI model (in
// its worker) can actually consume them. `onFrame` does the capture + send
// and returns a promise that resolves once a result comes back.
export function useFrameLoop({ active, videoRef, minDelayMs = 60, idleDelayMs = 250, onFrame }) {
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;
  const lastTimeRef = useRef(-1);

  useEffect(() => {
    if (!active) return undefined;
    let disposed = false;
    let timer = null;
    lastTimeRef.current = -1;

    const tick = async () => {
      if (disposed) return;
      const video = videoRef.current;

      if (document.visibilityState === 'hidden') {
        timer = setTimeout(tick, 1000);
        return;
      }
      if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) {
        timer = setTimeout(tick, idleDelayMs);
        return;
      }
      if (video.currentTime === lastTimeRef.current) {
        timer = setTimeout(tick, 80);
        return;
      }
      lastTimeRef.current = video.currentTime;

      const start = performance.now();
      try {
        await onFrameRef.current(video);
      } catch {
        // Swallowed here — callers surface failures via their own onError path.
      }
      if (disposed) return;
      timer = setTimeout(tick, Math.max(minDelayMs - (performance.now() - start), 0));
    };

    tick();
    return () => { disposed = true; clearTimeout(timer); };
  }, [active, videoRef, minDelayMs, idleDelayMs]);
}
