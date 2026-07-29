export function readStorage(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    if (raw == null) return typeof fallback === 'function' ? fallback() : fallback;
    return JSON.parse(raw);
  } catch {
    return typeof fallback === 'function' ? fallback() : fallback;
  }
}

export function writeStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage full/unavailable (e.g. private browsing) — fail silently,
    // in-memory state still works for the rest of the session.
  }
}
