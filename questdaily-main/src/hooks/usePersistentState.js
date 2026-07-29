import { useCallback, useEffect, useRef, useState } from 'react';
import { readStorage, writeStorage } from '../lib/storage.js';

// Debounces writes so rapid state changes (toggling several quests in a row)
// collapse into one localStorage write instead of one per change, and always
// flushes immediately when the tab is hidden/closed so nothing is lost.
export function usePersistedValue(key, value, { debounceMs = 250 } = {}) {
  const valueRef = useRef(value);
  valueRef.current = value;

  const flush = useCallback(() => {
    writeStorage(key, valueRef.current);
  }, [key]);

  useEffect(() => {
    const id = setTimeout(flush, debounceMs);
    return () => clearTimeout(id);
  }, [value, debounceMs, flush]);

  useEffect(() => {
    const onVisibility = () => { if (document.visibilityState === 'hidden') flush(); };
    window.addEventListener('pagehide', flush);
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      window.removeEventListener('pagehide', flush);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [flush]);
}

export function usePersistentState(key, initialValue, options) {
  const [value, setValue] = useState(() => readStorage(key, initialValue));
  usePersistedValue(key, value, options);
  return [value, setValue];
}
