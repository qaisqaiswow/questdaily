import { useCallback, useRef, useState } from 'react';

export function useToast(durationMs = 1800) {
  const [toast, setToast] = useState(null);
  const timeoutRef = useRef(null);

  const showToast = useCallback((next) => {
    clearTimeout(timeoutRef.current);
    setToast(next);
    timeoutRef.current = setTimeout(() => setToast(null), durationMs);
  }, [durationMs]);

  return [toast, showToast];
}
