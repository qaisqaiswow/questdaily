/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          '-apple-system', 'BlinkMacSystemFont', 'SF Pro Display',
          'SF Pro Text', 'Helvetica Neue', 'Arial', 'sans-serif'
        ],
      },
      colors: {
        apple: {
          blue:   '#007AFF',
          green:  '#34C759',
          orange: '#FF9500',
          red:    '#FF3B30',
          gray1:  '#8E8E93',
          gray2:  '#AEAEB2',
          gray3:  '#C7C7CC',
          gray4:  '#D1D1D6',
          gray5:  '#E5E5EA',
          gray6:  '#F2F2F7',
        },
        glow: {
          blue:   '#4DA3FF',
          violet: '#8B7CFF',
          cyan:   '#22D3EE',
          green:  '#4ADE80',
        },
      },
      borderRadius: {
        ios: '12px',
        'ios-lg': '16px',
        'ios-xl': '20px',
      },
      boxShadow: {
        'card-light': '0 1px 2px rgba(0,0,0,0.04), 0 8px 24px -12px rgba(0,0,0,0.08)',
        'card-dark': '0 1px 2px rgba(0,0,0,0.3), 0 12px 28px -12px rgba(0,0,0,0.5)',
        'glow-blue': '0 0 0 1px rgba(0,122,255,0.35), 0 8px 24px -6px rgba(0,122,255,0.45)',
        'glow-green': '0 0 0 1px rgba(52,199,89,0.4), 0 8px 20px -6px rgba(52,199,89,0.5)',
      },
      transitionTimingFunction: {
        spring: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
        smooth: 'cubic-bezier(0.4, 0, 0.2, 1)',
      },
      keyframes: {
        blob: {
          '0%, 100%': { transform: 'translate3d(0,0,0) scale(1)' },
          '33%':      { transform: 'translate3d(4%, -6%, 0) scale(1.08)' },
          '66%':      { transform: 'translate3d(-3%, 4%, 0) scale(0.95)' },
        },
        'pop-in': {
          '0%':   { transform: 'scale(0.6)', opacity: '0' },
          '60%':  { transform: 'scale(1.12)', opacity: '1' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        'rise-in': {
          '0%':   { transform: 'translate3d(-50%, 12px, 0)', opacity: '0' },
          '100%': { transform: 'translate3d(-50%, 0, 0)', opacity: '1' },
        },
        shimmer: {
          '0%':   { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        blob: 'blob 18s ease-in-out infinite',
        'blob-slow': 'blob 26s ease-in-out infinite reverse',
        'pop-in': 'pop-in 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
        'rise-in': 'rise-in 0.35s cubic-bezier(0.34, 1.56, 0.64, 1)',
        shimmer: 'shimmer 1.8s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
