"use client";

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { pipeline, env } from '@huggingface/transformers';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import * as exifr from 'exifr';
import { initializeApp, getApps } from 'firebase/app';
import {
  getAuth, onAuthStateChanged, signOut, deleteUser,
  createUserWithEmailAndPassword, signInWithEmailAndPassword,
  GoogleAuthProvider, signInWithPopup, signInWithRedirect, getRedirectResult,
  updateEmail, updatePassword, reauthenticateWithCredential, EmailAuthProvider,
} from 'firebase/auth';
import { getFirestore, doc, getDoc, setDoc, deleteDoc, getDocs, onSnapshot, collection, query, orderBy, limit } from 'firebase/firestore';
import {
  Sun, Moon, CheckSquare, History as HistoryIcon, Settings as SettingsIcon,
  Dumbbell, PersonStanding, Bike, Footprints, CircleDot, Timer, ChevronsUp,
  Droplet, Leaf, Utensils, CookingPot, Flower2, Move, Zap, Flame,
  Pencil, Snowflake, Wind, Waves, Target, GlassWater, CupSoda,
  LogOut, Eye, EyeOff, Lock, AtSign, Camera, Trash2, ShieldCheck, Sparkles, ListChecks,
  Bell, Download, UserCog, KeyRound, ChevronRight, CircleUserRound, X, RotateCcw
} from 'lucide-react';

const firebaseConfig = {
  apiKey: "AIzaSyDWaoCQjCc8mpF9jE8FIZvSKDKkxgTUELA",
  authDomain: "quest-daily-8debb.firebaseapp.com",
  projectId: "quest-daily-8debb",
  storageBucket: "quest-daily-8debb.firebasestorage.app",
  messagingSenderId: "628879821292",
  appId: "1:628879821292:web:997e8155a24b806eb2d8c9",
  measurementId: "G-C6S86XMFRZ",
};

const firebaseApp = getApps().length ? getApps()[0] : initializeApp(firebaseConfig);
const auth = getAuth(firebaseApp);
const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({ prompt: 'select_account' });
const db = getFirestore(firebaseApp);
const LEADERBOARD_COLLECTION = 'leaderboard';
const LEADERBOARD_SIZE = 100;

async function syncLeaderboardEntry(uid, { username, level, xp, totalXpEarned, streak }) {
  if (!uid || !username) return { ok: false, error: 'Not signed in yet — try again in a moment.' };
  try {
    await setDoc(doc(db, LEADERBOARD_COLLECTION, uid), {
      username: username.slice(0, 20),
      level,
      xp,
      totalXpEarned,
      ...(streak !== undefined ? { streak } : {}),
      updatedAt: Date.now(),
    }, { merge: true });
    return { ok: true };
  } catch (err) {
    console.error('Leaderboard sync failed:', err);
    return { ok: false, error: `${err.code || 'error'}: ${err.message}` };
  }
}

async function fetchRemoteProfile(uid) {
  if (!uid) return null;
  try {
    const snap = await getDoc(doc(db, LEADERBOARD_COLLECTION, uid));
    return snap.exists() ? snap.data() : null;
  } catch (err) {
    console.error('Profile fetch failed:', err);
    return null;
  }
}

async function syncHistoryEntry(uid, entry) {
  if (!uid) return;
  try {
    await setDoc(doc(db, 'users', uid, 'history', entry.id), entry);
  } catch (err) {
    console.error('History sync failed:', err);
  }
}

async function fetchRemoteHistory(uid) {
  if (!uid) return [];
  try {
    const q = query(collection(db, 'users', uid, 'history'), orderBy('ts', 'desc'), limit(500));
    const snap = await getDocs(q);
    return snap.docs.map(d => d.data());
  } catch (err) {
    console.error('History fetch failed:', err);
    return [];
  }
}

const USERNAME_EMAIL_DOMAIN = 'questdaily-users.app';
const USERNAME_RE = /^[a-zA-Z0-9_]{3,20}$/;

function usernameToEmail(usernameLower) {
  return `${usernameLower}@${USERNAME_EMAIL_DOMAIN}`;
}

function validateUsername(raw) {
  const trimmed = raw.trim();
  if (!USERNAME_RE.test(trimmed)) return 'Username must be 3-20 characters: letters, numbers, and underscores only.';
  return null;
}

function validatePassword(raw) {
  if (raw.length < 6) return 'Password must be at least 6 characters.';
  if (raw.length > 128) return 'Password is too long.';
  return null;
}

function friendlyAuthError(err) {
  const code = err?.code || '';
  if (code === 'auth/email-already-in-use') return 'That username is already taken.';
  if (code === 'auth/weak-password') return 'Password must be at least 6 characters.';
  if (code === 'auth/wrong-password' || code === 'auth/invalid-credential' || code === 'auth/invalid-login-credentials') return 'Incorrect username or password.';
  if (code === 'auth/user-not-found') return 'No account found with that username.';
  if (code === 'auth/too-many-requests') return 'Too many attempts — try again in a bit.';
  if (code === 'auth/network-request-failed') return "Can't reach the server. Check your connection.";
  if (code === 'auth/requires-recent-login') return 'For security, please log out and log back in, then try again.';
  if (code === 'auth/invalid-email') return 'That username can\'t be used. Try a different one.';
  if (code === 'auth/popup-blocked') return 'Your browser blocked the sign-in popup. Popup fallback activated.';
  if (code === 'auth/account-exists-with-different-credential') return 'That account is already tied to a different sign-in method.';
  if (code === 'permission-denied' || code === 'firestore/permission-denied') {
    return "Database rules blocking access. Ensure 'usernames', 'users', and 'leaderboard' security rules are set.";
  }
  return err?.message ? `Error: ${err.message}` : 'Something went wrong. Please try again.';
}

async function signUpAccount(usernameRaw, password) {
  const username = usernameRaw.trim();
  const usernameLower = username.toLowerCase();
  const usernameDoc = doc(db, 'usernames', usernameLower);

  const existing = await getDoc(usernameDoc);
  if (existing.exists()) {
    const err = new Error('That username is already taken.');
    err.code = 'auth/email-already-in-use';
    throw err;
  }

  const cred = await createUserWithEmailAndPassword(auth, usernameToEmail(usernameLower), password);
  await Promise.all([
    setDoc(usernameDoc, { uid: cred.user.uid, username }),
    setDoc(doc(db, 'users', cred.user.uid), { username, createdAt: Date.now() }),
  ]);
  return { uid: cred.user.uid, username };
}

async function logInAccount(usernameRaw, password) {
  const username = usernameRaw.trim();
  const usernameLower = username.toLowerCase();
  const cred = await signInWithEmailAndPassword(auth, usernameToEmail(usernameLower), password);

  let displayUsername = username;
  try {
    const snap = await getDoc(doc(db, 'usernames', usernameLower));
    if (snap.exists() && snap.data()?.username) displayUsername = snap.data().username;
  } catch {}

  return { uid: cred.user.uid, username: displayUsername };
}

async function logOutAccount() {
  await signOut(auth);
}

async function mintUsernameForGoogleUser(user) {
  if (!user) return;
  const snapUser = await getDoc(doc(db, 'users', user.uid));
  if (snapUser.exists() && snapUser.data()?.username) return snapUser.data().username;

  const rawBase = user.displayName || (user.email ? user.email.split('@')[0] : 'Player');
  let base = rawBase.replace(/[^a-zA-Z0-9_]/g, '').slice(0, 16);
  if (base.length < 3) base = (base + 'Player').slice(0, 16);

  let display = base;
  let usernameLower = base.toLowerCase();
  for (let attempt = 0; attempt < 8; attempt++) {
    const snap = await getDoc(doc(db, 'usernames', usernameLower));
    if (!snap.exists()) break;
    const suffix = String(Math.floor(1000 + Math.random() * 9000));
    display = `${base}${suffix}`.slice(0, 20);
    usernameLower = display.toLowerCase();
  }

  await Promise.all([
    setDoc(doc(db, 'usernames', usernameLower), { uid: user.uid, username: display }),
    setDoc(doc(db, 'users', user.uid), { username: display, createdAt: Date.now(), photoURL: user.photoURL || null }),
  ]);
  if (user.photoURL) {
    await setDoc(doc(db, 'leaderboard', user.uid), { photoURL: user.photoURL }, { merge: true }).catch(() => {});
  }
  return display;
}

// Improved Google Login: Tries Popup with Seamless Fallback to Redirect
async function signInWithGoogle() {
  try {
    const result = await signInWithPopup(auth, googleProvider);
    if (result?.user) {
      await mintUsernameForGoogleUser(result.user);
    }
    return { ok: true, user: result.user };
  } catch (err) {
    console.warn('Google popup sign-in encountered issue, switching to redirect:', err);
    if (
      err.code === 'auth/popup-blocked' ||
      err.code === 'auth/popup-closed-by-user' ||
      err.code === 'auth/cancelled-popup-request' ||
      err.code === 'auth/operation-not-allowed'
    ) {
      await signInWithRedirect(auth, googleProvider);
      return { ok: true, redirecting: true };
    }
    throw err;
  }
}

function getAuthProviderLabel(user) {
  if (!user) return null;
  const ids = (user.providerData || []).map(p => p.providerId);
  if (ids.includes('google.com')) return 'google';
  if (ids.includes('password')) return 'password';
  return 'other';
}

async function updateUsername({ newUsernameRaw, oldUsername, currentPassword }) {
  const currentUser = auth.currentUser;
  if (!currentUser) throw Object.assign(new Error('You need to be signed in to do that.'), { code: 'auth/no-current-user' });

  const newUsername = newUsernameRaw.trim();
  const newLower = newUsername.toLowerCase();
  const oldLower = (oldUsername || '').toLowerCase();
  const usesPassword = getAuthProviderLabel(currentUser) === 'password';

  if (newLower === oldLower) {
    await Promise.all([
      setDoc(doc(db, 'usernames', oldLower), { uid: currentUser.uid, username: newUsername }, { merge: true }),
      setDoc(doc(db, 'users', currentUser.uid), { username: newUsername }, { merge: true }),
      setDoc(doc(db, 'leaderboard', currentUser.uid), { username: newUsername }, { merge: true }),
    ]);
    return newUsername;
  }

  const takenSnap = await getDoc(doc(db, 'usernames', newLower));
  if (takenSnap.exists()) {
    throw Object.assign(new Error('That username is already taken.'), { code: 'auth/email-already-in-use' });
  }

  if (usesPassword) {
    const newEmail = usernameToEmail(newLower);
    try {
      await updateEmail(currentUser, newEmail);
    } catch (err) {
      if (err?.code === 'auth/requires-recent-login') {
        if (!currentPassword) throw err;
        await reauthenticateWithCredential(currentUser, EmailAuthProvider.credential(usernameToEmail(oldLower), currentPassword));
        await updateEmail(currentUser, newEmail);
      } else {
        throw err;
      }
    }
  }

  await Promise.all([
    setDoc(doc(db, 'usernames', newLower), { uid: currentUser.uid, username: newUsername }),
    setDoc(doc(db, 'users', currentUser.uid), { username: newUsername }, { merge: true }),
    setDoc(doc(db, 'leaderboard', currentUser.uid), { username: newUsername }, { merge: true }),
  ]);
  await deleteDoc(doc(db, 'usernames', oldLower)).catch(() => {});

  return newUsername;
}

async function changeAccountPassword(currentPassword, newPassword) {
  const currentUser = auth.currentUser;
  if (!currentUser?.email) throw Object.assign(new Error('You need to be signed in to do that.'), { code: 'auth/no-current-user' });
  await reauthenticateWithCredential(currentUser, EmailAuthProvider.credential(currentUser.email, currentPassword));
  await updatePassword(currentUser, newPassword);
}

const AVATAR_SIZE = 256;
const AVATAR_QUALITY = 0.8;

function resizeImageToSquareDataUrl(file) {
  return new Promise((resolve, reject) => {
    if (!file.type?.startsWith('image/')) { reject(new Error('Please choose an image file.')); return; }
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const side = Math.min(img.width, img.height);
        const sx = (img.width - side) / 2;
        const sy = (img.height - side) / 2;
        const canvas = document.createElement('canvas');
        canvas.width = AVATAR_SIZE; canvas.height = AVATAR_SIZE;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, sx, sy, side, side, 0, 0, AVATAR_SIZE, AVATAR_SIZE);
        resolve(canvas.toDataURL('image/jpeg', AVATAR_QUALITY));
      };
      img.onerror = () => reject(new Error('That file doesn\'t look like a valid image.'));
      img.src = e.target.result;
    };
    reader.onerror = () => reject(new Error('Could not read that file.'));
    reader.readAsDataURL(file);
  });
}

async function updateProfilePhoto(uid, dataUrl) {
  await Promise.all([
    setDoc(doc(db, 'users', uid), { photoURL: dataUrl }, { merge: true }),
    setDoc(doc(db, 'leaderboard', uid), { photoURL: dataUrl }, { merge: true }),
  ]);
}

async function deleteAccountData(uid, usernameLower) {
  let historyDocs = [];
  try {
    const historySnap = await getDocs(collection(db, 'users', uid, 'history'));
    historyDocs = historySnap.docs;
  } catch (err) {
    console.error('Could not list history for deletion:', err);
  }
  await Promise.all(historyDocs.map(d => deleteDoc(d.ref).catch(() => {})));

  const results = await Promise.allSettled([
    deleteDoc(doc(db, 'users', uid)),
    deleteDoc(doc(db, 'leaderboard', uid)),
    usernameLower ? deleteDoc(doc(db, 'usernames', usernameLower)) : Promise.resolve(),
  ]);
  const failure = results.find(r => r.status === 'rejected');
  if (failure) throw failure.reason;
}

function useAuthUser() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    // Process Google redirect result if page reloaded from redirect
    getRedirectResult(auth)
      .then(async (res) => {
        if (res?.user) {
          await mintUsernameForGoogleUser(res.user);
        }
      })
      .catch((err) => console.error('Redirect result error:', err));

    const unsub = onAuthStateChanged(auth, async (u) => { 
      if (u) await mintUsernameForGoogleUser(u);
      setUser(u); 
      setLoading(false); 
    });
    return unsub;
  }, []);
  return { user, loading };
}

function useLeaderboard() {
  const [entries, setEntries] = useState([]);
  const [status, setStatus] = useState('loading');
  const [errorDetail, setErrorDetail] = useState(null);
  useEffect(() => {
    const q = query(collection(db, LEADERBOARD_COLLECTION), orderBy('totalXpEarned', 'desc'), limit(LEADERBOARD_SIZE));
    const timeout = setTimeout(() => {
      setStatus(prev => prev === 'loading' ? 'error' : prev);
      setErrorDetail(prev => prev || 'Timed out waiting for response.');
    }, 8000);
    const unsub = onSnapshot(
        q,
        snap => { clearTimeout(timeout); setEntries(snap.docs.map(d => ({ id: d.id, ...d.data() }))); setStatus('ready'); setErrorDetail(null); },
        err => { clearTimeout(timeout); console.error('Leaderboard error:', err); setStatus('error'); setErrorDetail(`${err.code || 'error'}: ${err.message}`); }
    );
    return () => { clearTimeout(timeout); unsub(); };
  }, []);
  return { entries, status, errorDetail };
}

let hapticsEnabled = true;
function setHapticsPref(enabled) {
  hapticsEnabled = enabled;
  try { localStorage.setItem('sq_haptics_enabled', enabled ? 'true' : 'false'); } catch {}
}
function haptic(pattern = 10) {
  if (!hapticsEnabled) return;
  try { if (typeof navigator !== 'undefined' && navigator.vibrate) navigator.vibrate(pattern); } catch {}
}

function getLevelTitle(level) {
  if (level < 5)  return 'Novice';
  if (level < 10) return 'Apprentice';
  if (level < 20) return 'Skilled';
  if (level < 35) return 'Veteran';
  return 'Legendary';
}

// System Theme Palette
const C = {
  light: {
    bg: '#F2F2F7', bgElevated: '#FFFFFF', bgSecondary: '#F2F2F7', bgTertiary: '#E5E5EA',
    label: '#000000', labelSecondary: 'rgba(60,60,67,0.60)', labelTertiary: 'rgba(60,60,67,0.30)',
    separator: 'rgba(60,60,67,0.29)', fill: 'rgba(120,120,128,0.12)',
    blue: '#007AFF', green: '#34C759', orange: '#FF9500', red: '#FF3B30',
    indigo: '#5856D6', teal: '#30B0C7', purple: '#AF52DE', yellow: '#FFCC00', pink: '#FF2D55', gray: '#8E8E93',
    glass: 'rgba(255, 255, 255, 0.42)', glassStrong: 'rgba(255, 255, 255, 0.68)',
    glassBorder: 'rgba(255, 255, 255, 0.45)', glassHighlight: 'rgba(255, 255, 255, 0.85)',
    glassShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.08), 0 1px 2px 0 rgba(0,0,0,0.02)',
  },
  dark: {
    bg: '#000000', bgElevated: '#1C1C1E', bgSecondary: '#1C1C1E', bgTertiary: '#2C2C2E',
    label: '#FFFFFF', labelSecondary: 'rgba(235,235,245,0.60)', labelTertiary: 'rgba(235,235,245,0.30)',
    separator: 'rgba(84,84,88,0.65)', fill: 'rgba(120,120,128,0.24)',
    blue: '#0A84FF', green: '#30D158', orange: '#FF9F0A', red: '#FF453A',
    indigo: '#5E5CE6', teal: '#40C8E0', purple: '#BF5AF2', yellow: '#FFD60A', pink: '#FF375F', gray: '#98989D',
    glass: 'rgba(22, 22, 26, 0.52)', glassStrong: 'rgba(32, 32, 36, 0.75)',
    glassBorder: 'rgba(255, 255, 255, 0.12)', glassHighlight: 'rgba(255, 255, 255, 0.16)',
    glassShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.45), inset 0 0 0 1px rgba(255,255,255,0.05)',
  },
};

// EXPO LIQUID GLASS STYLING (Ultra-clean VisionOS / Expo BlurView glass)
const glassStyle = (c, strong = false) => ({
  background: strong
    ? `linear-gradient(135deg, ${c.glassHighlight} 0%, ${c.glassStrong} 100%)`
    : `linear-gradient(135deg, ${c.glassHighlight} 0%, ${c.glass} 100%)`,
  backdropFilter: 'blur(30px) saturate(190%)',
  WebkitBackdropFilter: 'blur(30px) saturate(190%)',
  border: `1px solid ${c.glassBorder}`,
  boxShadow: `inset 0 1px 0 0 ${c.glassHighlight}, ${c.glassShadow}`,
  borderRadius: '24px',
  transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
});

// SYSTEM CSS & SMOOTH ANIMATIONS
const SystemType = () => (
    <style>{`
    .sq-root, .sq-root * { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Arial, sans-serif; }
    .sq-mono, .sq-mono * { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-variant-numeric: tabular-nums; }
    
    .sq-root {
      -webkit-user-select: none; user-select: none;
      -webkit-touch-callout: none;
      touch-action: pan-x pan-y;
    }
    .sq-root input, .sq-root textarea {
      -webkit-user-select: text; user-select: text;
      touch-action: manipulation;
    }

    /* Expo Glass Cards Interactive Spring Hover & Tap */
    .expo-glass-card {
      transition: transform 0.28s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s ease, background 0.3s ease;
      will-change: transform;
    }
    .expo-glass-card:active {
      transform: scale(0.975);
    }

    /* Siri-Style Thinking Orbs Keyframe Animations */
    @keyframes sq-thinking-orb-1 {
      0%, 100% { transform: translate3d(-10%, -10%, 0) scale(1) rotate(0deg); opacity: 0.5; }
      33% { transform: translate3d(20%, 15%, 0) scale(1.3) rotate(120deg); opacity: 0.75; }
      66% { transform: translate3d(-15%, 25%, 0) scale(0.85) rotate(240deg); opacity: 0.45; }
    }
    @keyframes sq-thinking-orb-2 {
      0%, 100% { transform: translate3d(15%, 10%, 0) scale(1.1) rotate(0deg); opacity: 0.45; }
      33% { transform: translate3d(-20%, -15%, 0) scale(0.9) rotate(-120deg); opacity: 0.7; }
      66% { transform: translate3d(10%, -25%, 0) scale(1.25) rotate(-240deg); opacity: 0.55; }
    }
    @keyframes sq-thinking-orb-3 {
      0%, 100% { transform: translate3d(5%, -20%, 0) scale(0.95) rotate(0deg); opacity: 0.4; }
      50% { transform: translate3d(-10%, 20%, 0) scale(1.35) rotate(180deg); opacity: 0.65; }
    }
    @keyframes sq-orb-pulse {
      0%, 100% { filter: blur(75px) contrast(120%); }
      50% { filter: blur(95px) contrast(150%); }
    }

    .sq-thinking-orb-1 { animation: sq-thinking-orb-1 18s ease-in-out infinite, sq-orb-pulse 12s ease-in-out infinite; will-change: transform, opacity; }
    .sq-thinking-orb-2 { animation: sq-thinking-orb-2 22s ease-in-out infinite, sq-orb-pulse 14s ease-in-out infinite; will-change: transform, opacity; }
    .sq-thinking-orb-3 { animation: sq-thinking-orb-3 16s ease-in-out infinite, sq-orb-pulse 10s ease-in-out infinite; will-change: transform, opacity; }

    /* Fluid UI Animations */
    @keyframes sq-fade-scale-in {
      from { opacity: 0; transform: scale(0.94) translate3d(0, 10px, 0); }
      to { opacity: 1; transform: scale(1) translate3d(0, 0, 0); }
    }
    .sq-anim-in { animation: sq-fade-scale-in 0.38s cubic-bezier(0.16, 1, 0.3, 1) both; }
    
    button { transition: opacity 0.2s ease, transform 0.2s cubic-bezier(0.16, 1, 0.3, 1); }
    button:active { transform: scale(0.95); }

    @media (prefers-reduced-motion: reduce) {
      .sq-thinking-orb-1, .sq-thinking-orb-2, .sq-thinking-orb-3 { animation: none; }
    }
  `}</style>
);

// Siri-style "Thinking Orbs" Background
const AmbientBackground = React.memo(function AmbientBackground({ c }) {
  return (
      <div className="fixed inset-0 pointer-events-none z-0" aria-hidden="true" style={{ overflow: 'hidden' }}>
        <div className="sq-thinking-orb-1" style={{ position: 'absolute', top: '-15%', left: '-10%', width: '80vw', height: '50vh', borderRadius: '50%', background: `radial-gradient(circle, ${c.blue} 0%, ${c.indigo} 60%, transparent 100%)`, opacity: 0.5 }} />
        <div className="sq-thinking-orb-2" style={{ position: 'absolute', top: '25%', right: '-15%', width: '85vw', height: '55vh', borderRadius: '50%', background: `radial-gradient(circle, ${c.purple} 0%, ${c.pink} 60%, transparent 100%)`, opacity: 0.45 }} />
        <div className="sq-thinking-orb-3" style={{ position: 'absolute', bottom: '-15%', left: '10%', width: '75vw', height: '48vh', borderRadius: '50%', background: `radial-gradient(circle, ${c.teal} 0%, ${c.green} 60%, transparent 100%)`, opacity: 0.4 }} />
      </div>
  );
});

// Camera View with Modernized, Dependable Expo-Glass Cancel Button
const CameraModal = ({ isOpen, onClose, onCapture }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-xl sq-anim-in">
      {/* Reliable, Beautiful Expo Glass Close / Cancel Button */}
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          haptic(15);
          onClose();
        }}
        className="absolute top-6 right-6 z-50 flex items-center justify-center w-12 h-12 rounded-full text-white bg-white/15 backdrop-blur-2xl border border-white/30 active:scale-90 transition-all duration-200 shadow-2xl cursor-pointer pointer-events-auto hover:bg-white/25"
        aria-label="Close Camera"
      >
        <X size={22} className="stroke-[2.5]" />
      </button>

      <div className="relative w-full max-w-md h-[80vh] mx-4 rounded-3xl overflow-hidden border border-white/20 shadow-2xl bg-black flex flex-col items-center justify-center">
        <div className="text-white/70 text-sm flex flex-col items-center gap-3">
          <Camera size={44} className="animate-pulse text-white/90" />
          <span>Camera Viewfinder Ready</span>
        </div>

        {/* Action Controls */}
        <div className="absolute bottom-8 left-0 right-0 flex items-center justify-center gap-6 px-6">
          <button
            type="button"
            onClick={() => { haptic(20); onCapture?.(); }}
            className="w-20 h-20 rounded-full border-4 border-white flex items-center justify-center bg-white/20 backdrop-blur-md active:scale-90 transition-transform shadow-xl"
          >
            <div className="w-16 h-16 rounded-full bg-white" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default function App() {
  const { user, loading: authLoading } = useAuthUser();
  const [theme, setTheme] = useState('dark');
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const c = C[theme];

  return (
    <div className={`sq-root min-h-screen relative overflow-x-hidden ${theme === 'dark' ? 'bg-black text-white' : 'bg-[#F2F2F7] text-black'}`}>
      <SystemType />
      <AmbientBackground c={c} />

      {/* Main Container */}
      <main className="relative z-10 max-w-lg mx-auto px-4 py-8 flex flex-col gap-6">
        {/* Header Glass Card */}
        <header style={glassStyle(c, true)} className="expo-glass-card p-6 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Quest Daily</h1>
            <p className="text-xs text-neutral-400 mt-1">
              {user ? `Logged in as ${user.displayName || 'Player'}` : 'Sign in to sync stats'}
            </p>
          </div>
          <button
            onClick={() => { haptic(); setTheme(t => t === 'dark' ? 'light' : 'dark'); }}
            className="p-3 rounded-2xl bg-white/10 border border-white/20 active:scale-95 transition-all"
          >
            {theme === 'dark' ? <Sun size={20} className="text-yellow-400" /> : <Moon size={20} className="text-indigo-600" />}
          </button>
        </header>

        {/* Auth / Action Glass Card */}
        <section style={glassStyle(c)} className="expo-glass-card p-6 flex flex-col gap-4">
          <h2 className="text-lg font-semibold">Account & Verification</h2>
          {!user ? (
            <button
              onClick={async () => {
                haptic();
                try {
                  await signInWithGoogle();
                } catch (err) {
                  alert(friendlyAuthError(err));
                }
              }}
              className="flex items-center justify-center gap-3 w-full py-3.5 px-4 rounded-2xl bg-white text-black font-medium shadow-lg hover:bg-neutral-100 active:scale-95 transition-all"
            >
              <GoogleIcon size={20} />
              <span>Continue with Google</span>
            </button>
          ) : (
            <button
              onClick={() => { haptic(); logOutAccount(); }}
              className="flex items-center justify-center gap-2 w-full py-3 px-4 rounded-2xl bg-red-500/20 border border-red-500/30 text-red-400 font-medium active:scale-95 transition-all"
            >
              <LogOut size={18} />
              <span>Sign Out</span>
            </button>
          )}

          <div className="h-[1px] bg-white/10 my-1" />

          {/* Camera Trigger */}
          <button
            onClick={() => { haptic(15); setIsCameraOpen(true); }}
            className="flex items-center justify-center gap-2 w-full py-3.5 px-4 rounded-2xl bg-blue-600 text-white font-semibold shadow-lg active:scale-95 transition-all"
          >
            <Camera size={20} />
            <span>Open Verification Camera</span>
          </button>
        </section>
      </main>

      {/* Modernized Camera Modal with Reliable Glass Cancel Button */}
      <CameraModal
        isOpen={isCameraOpen}
        onClose={() => setIsCameraOpen(false)}
        onCapture={() => {
          setIsCameraOpen(false);
          alert('Verification frame captured!');
        }}
      />
    </div>
  );
}
