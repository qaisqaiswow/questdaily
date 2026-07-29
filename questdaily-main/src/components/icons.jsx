import { memo } from 'react';

export const LogoIcon = memo(function LogoIcon({ size = 34, dark }) {
  return (
    <img
      src="/logo-transparent.png"
      alt="Side Quests"
      width={size}
      height={size}
      style={{ filter: dark ? 'invert(0)' : 'invert(1)', opacity: 0.9 }}
    />
  );
});

export const SunIcon = memo(function SunIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
    </svg>
  );
});

export const MoonIcon = memo(function MoonIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
});

export const CheckIcon = memo(function CheckIcon({ size = 12, strokeWidth = 3 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
});

export const ChevronIcon = memo(function ChevronIcon() {
  return (
    <svg width="8" height="14" viewBox="0 0 8 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="1 1 7 7 1 13" />
    </svg>
  );
});
