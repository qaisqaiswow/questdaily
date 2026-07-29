import { useCallback, useRef } from 'react';

// Bridges fire-and-forget worker postMessage calls back into awaitable
// promises, keyed by requestId, so useFrameLoop can hold off sending the
// next frame until the current one has actually been processed.
export function usePendingRequests() {
  const resolversRef = useRef(new Map());

  const wait = useCallback((requestId) => new Promise((resolve) => {
    resolversRef.current.set(requestId, resolve);
  }), []);

  const resolve = useCallback((requestId, value) => {
    const resolver = resolversRef.current.get(requestId);
    if (resolver) {
      resolversRef.current.delete(requestId);
      resolver(value);
    }
  }, []);

  return { wait, resolve };
}
