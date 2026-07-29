import { useEffect } from 'react';
import { usePersistentState } from './usePersistentState.js';

export function useDarkMode() {
  const [dark, setDark] = usePersistentState('sq_dark', () => window.matchMedia('(prefers-color-scheme: dark)').matches);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
  }, [dark]);

  return [dark, setDark];
}
