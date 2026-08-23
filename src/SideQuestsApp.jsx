"use client";

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { pipeline, env } from '@huggingface/transformers';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import * as exifr from 'exifr';
import { initializeApp, getApps } from 'firebase/app';
import {
  getAuth, onAuthStateChanged, signOut, deleteUser,
  createUserWithEmailAndPassword, signInWithEmailAndPassword,
  GoogleAuthProvider, signInWithPopup,
  updateEmail, updatePassword, reauthenticateWithCredential, EmailAuthProvider,
} from 'firebase/auth';
import { getFirestore, doc, getDoc, setDoc, deleteDoc, getDocs, onSnapshot, collection, query, orderBy, limit } from 'firebase/firestore';
import {
  Sun, Moon, CheckSquare, History as HistoryIcon, Settings as SettingsIcon,
  Dumbbell, PersonStanding, Bike, Footprints, CircleDot, Timer, ChevronsUp,
  Droplet, Leaf, Utensils, CookingPot, Flower2, Move, Zap, Flame,
  Pencil, Snowflake, Wind, Waves, Target, GlassWater, CupSoda,
  LogOut, Eye, EyeOff, Lock, AtSign, Camera, Trash2, ShieldCheck, Sparkles, ListChecks,
  Bell, Download, UserCog, KeyRound, ChevronRight, CircleUserRound,
} from 'lucide-react';

// firebase / leaderboard stuff
// swap in your own project config (Firebase console > project settings > your apps)
// this is fine to ship client-side btw, it's not secret — the real access control
// is in the firestore security rules
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
const db = getFirestore(firebaseApp);
const LEADERBOARD_COLLECTION = 'leaderboard';
const LEADERBOARD_SIZE = 100;

// pushes the player's stats up to their leaderboard doc. ranked by lifetime
// xp earned (not current xp) so leveling up never makes your rank go down.
// this same doc also acts as a per-account profile backup — see
// fetchRemoteProfile below — so progress can survive a cleared localStorage
// as long as the anonymous auth session (and its uid) is still intact.
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

// reads the profile doc back — used on load to restore level/xp/streak if
// localStorage came back empty but this uid's Firestore data didn't.
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

// completed-quest history, stored per-account under users/{uid}/history/{id}
// so it isn't stranded on a single browser's localStorage.
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

// account system
// -----------------------------------------------------------------------
// Firebase Auth doesn't have native "username + password" accounts — it
// wants an email. So a username is mapped to a deterministic, fake-but-
// validly-formatted email (`somebody` -> `somebody@questdaily-users.app`)
// and Firebase Auth is used normally underneath. This means:
//   - usernames are case-insensitive (the email is always lowercased)
//   - there's no real inbox behind that address, so Firebase's built-in
//     "forgot password" email reset can't work here — if that's needed
//     later it requires collecting a real email address at signup instead
// A `usernames/{usernameLower}` doc is the source of truth for "is this
// username taken" and for looking up the display-cased username on login;
// the account's own profile lives in `users/{uid}`.
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
  if (code === 'auth/popup-blocked') return 'Your browser blocked the sign-in popup. Please allow popups for this site and try again.';
  if (code === 'auth/account-exists-with-different-credential') return 'That Google account\'s email is already tied to a different sign-in method here.';
  if (code === 'permission-denied' || code === 'firestore/permission-denied') {
    return "Your database's security rules are blocking this — the 'usernames', 'users', and 'leaderboard' collections in Firestore need read/write rules set up. This is a Firebase Console configuration step, not something wrong with what you typed.";
  }
  return err?.message ? `Something went wrong: ${err.message}` : 'Something went wrong. Please try again.';
}

// Creates a brand-new account. Reserves the username first (so two people
// racing on the same name get a clean "taken" error instead of a confusing
// Firebase Auth error), then creates the Auth user, then writes the
// username reservation + profile doc. If the profile writes fail after the
// Auth user was already created, the reservation is rolled back so the
// username isn't permanently stuck on a half-created account.
async function signUpAccount(usernameRaw, password) {
  const username = usernameRaw.trim();
  const usernameLower = username.toLowerCase();
  const usernameDoc = doc(db, 'usernames', usernameLower);

  let existing;
  try {
    existing = await getDoc(usernameDoc);
  } catch (err) {
    console.error('Username availability check failed:', err);
    throw err; // surfaced via friendlyAuthError, including the permission-denied case
  }
  if (existing.exists()) {
    const err = new Error('That username is already taken.');
    err.code = 'auth/email-already-in-use';
    throw err;
  }

  const cred = await createUserWithEmailAndPassword(auth, usernameToEmail(usernameLower), password);
  try {
    await Promise.all([
      setDoc(usernameDoc, { uid: cred.user.uid, username }),
      setDoc(doc(db, 'users', cred.user.uid), { username, createdAt: Date.now() }),
    ]);
  } catch (err) {
    console.error('Failed to finish account setup:', err);
    throw err;
  }
  return { uid: cred.user.uid, username };
}

// Logs into an existing account. Looks up the reservation doc first purely
// to recover the original display-cased username (login itself only needs
// the deterministic email, so a typo'd case still signs in fine).
async function logInAccount(usernameRaw, password) {
  const username = usernameRaw.trim();
  const usernameLower = username.toLowerCase();
  const cred = await signInWithEmailAndPassword(auth, usernameToEmail(usernameLower), password);

  let displayUsername = username;
  try {
    const snap = await getDoc(doc(db, 'usernames', usernameLower));
    if (snap.exists() && snap.data()?.username) displayUsername = snap.data().username;
  } catch { /* non-fatal — fall back to what they typed */ }

  return { uid: cred.user.uid, username: displayUsername };
}

async function logOutAccount() {
  await signOut(auth);
}

// Signs in with a Google popup. Google accounts don't come with a username
// the person chose, so on a person's very first Google sign-in one is
// minted automatically from their Google display name (falling back to
// their email), retried with a random numeric suffix if it's taken — same
// `usernames` reservation + `users/{uid}` profile doc that the username/
// password flow creates, so everything downstream (hydration, leaderboard,
// settings) treats a Google account exactly like any other. Their Google
// avatar is carried over as the initial profile photo too. Returning users
// don't need any of this — their `users/{uid}` doc already exists, and the
// app's normal auth-state hydration effect picks it up the same way a page
// reload does.
async function mintUsernameForGoogleUser(user) {
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
}

async function signInWithGoogle() {
  const result = await signInWithPopup(auth, googleProvider);
  const existing = await getDoc(doc(db, 'users', result.user.uid)).catch(() => null);
  if (!existing?.exists()) {
    await mintUsernameForGoogleUser(result.user);
  }
  return result.user;
}

// editing an existing account
// -----------------------------------------------------------------------
function getAuthProviderLabel(user) {
  if (!user) return null;
  const ids = (user.providerData || []).map(p => p.providerId);
  if (ids.includes('google.com')) return 'google';
  if (ids.includes('password')) return 'password';
  return 'other';
}

// Renames the account's username. For password accounts this is more than
// a display change: login derives the (fake) Auth email straight from the
// username, so the underlying Firebase Auth email has to move in lockstep
// or the person would be locked out under their new name. That rename can
// require a recent login — if so, this reauthenticates with the password
// the caller supplied and retries once. Google accounts skip all of that;
// their real Google email never changes, so renaming is purely a Firestore
// metadata update.
async function updateUsername({ newUsernameRaw, oldUsername, currentPassword }) {
  const currentUser = auth.currentUser;
  if (!currentUser) throw Object.assign(new Error('You need to be signed in to do that.'), { code: 'auth/no-current-user' });

  const newUsername = newUsernameRaw.trim();
  const newLower = newUsername.toLowerCase();
  const oldLower = (oldUsername || '').toLowerCase();
  const usesPassword = getAuthProviderLabel(currentUser) === 'password';

  if (newLower === oldLower) {
    // same underlying name, only casing changed (or literally unchanged) —
    // no Auth email involved, just update the display casing everywhere
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
        if (!currentPassword) throw err; // caller prompts for the password and retries with it
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

// Password-account only. Requires the current password to reauthenticate
// (Firebase demands a recent login for this regardless of session age).
async function changeAccountPassword(currentPassword, newPassword) {
  const currentUser = auth.currentUser;
  if (!currentUser?.email) throw Object.assign(new Error('You need to be signed in to do that.'), { code: 'auth/no-current-user' });
  await reauthenticateWithCredential(currentUser, EmailAuthProvider.credential(currentUser.email, currentPassword));
  await updatePassword(currentUser, newPassword);
}

// profile pictures
// -----------------------------------------------------------------------
// There's no Firebase Storage bucket wired up here, so photos are stored
// as compressed, center-cropped JPEG data URLs directly on the Firestore
// profile doc (and mirrored onto the leaderboard doc so avatars show up
// there too). Firestore caps a document at ~1MB, so the image is resized
// down hard first — 256x256 at moderate JPEG quality lands well under
// 100KB in practice, comfortably inside that limit with room to spare.
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

// Best-effort cleanup of everything an account owns before the Auth user
// itself is deleted: the full history subcollection (deleted doc-by-doc,
// since Firestore has no client-side "delete a collection" call), the
// profile doc, the leaderboard entry, and the username reservation so the
// name becomes available again.
async function deleteAccountData(uid, usernameLower) {
  let historyDocs = [];
  try {
    const historySnap = await getDocs(collection(db, 'users', uid, 'history'));
    historyDocs = historySnap.docs;
  } catch (err) {
    console.error('Could not list history for deletion (continuing with the rest):', err);
  }
  await Promise.all(historyDocs.map(d => deleteDoc(d.ref).catch(() => {})));

  const results = await Promise.allSettled([
    deleteDoc(doc(db, 'users', uid)),
    deleteDoc(doc(db, 'leaderboard', uid)),
    usernameLower ? deleteDoc(doc(db, 'usernames', usernameLower)) : Promise.resolve(),
  ]);
  const failure = results.find(r => r.status === 'rejected');
  if (failure) {
    // At least one core doc failed to delete — surface it rather than
    // silently leaving orphaned data, so the person sees why the account
    // wasn't fully removed instead of it just quietly not working.
    console.error('Account data cleanup partially failed:', failure.reason);
    throw failure.reason;
  }
}

// tracks the signed-in Firebase user (or null once auth has resolved and
// nobody's signed in) — the account system's version of the old anonymous
// uid hook.
function useAuthUser() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const unsub = onAuthStateChanged(auth, u => { setUser(u); setLoading(false); });
    return unsub;
  }, []);
  return { user, loading };
}

// live top-100, onSnapshot keeps it updated in realtime so no polling needed
function useLeaderboard() {
  const [entries, setEntries] = useState([]);
  const [status, setStatus] = useState('loading'); // 'loading' | 'ready' | 'error'
  const [errorDetail, setErrorDetail] = useState(null);
  useEffect(() => {
    const q = query(collection(db, LEADERBOARD_COLLECTION), orderBy('totalXpEarned', 'desc'), limit(LEADERBOARD_SIZE));
    const timeout = setTimeout(() => {
      setStatus(prev => prev === 'loading' ? 'error' : prev);
      setErrorDetail(prev => prev || 'Timed out waiting for a response. Firestore Database and Anonymous Auth may not be enabled yet in the Firebase console.');
    }, 8000);
    const unsub = onSnapshot(
        q,
        snap => { clearTimeout(timeout); setEntries(snap.docs.map(d => ({ id: d.id, ...d.data() }))); setStatus('ready'); setErrorDetail(null); },
        err => { clearTimeout(timeout); console.error('Leaderboard listener error:', err); setStatus('error'); setErrorDetail(`${err.code || 'error'}: ${err.message}`); }
    );
    return () => { clearTimeout(timeout); unsub(); };
  }, []);
  return { entries, status, errorDetail };
}

// haptics — gated behind a user preference (Settings → Notifications), so
// turning it off actually silences every haptic() call app-wide rather than
// just hiding a toggle that does nothing.
let hapticsEnabled = true;
function setHapticsPref(enabled) {
  hapticsEnabled = enabled;
  try { localStorage.setItem('sq_haptics_enabled', enabled ? 'true' : 'false'); } catch { /* storage unavailable */ }
}
function haptic(pattern = 10) {
  if (!hapticsEnabled) return;
  try { if (typeof navigator !== 'undefined' && navigator.vibrate) navigator.vibrate(pattern); } catch { /* unsupported */ }
}

function getLevelTitle(level) {
  if (level < 5)  return 'Novice';
  if (level < 10) return 'Apprentice';
  if (level < 20) return 'Skilled';
  if (level < 35) return 'Veteran';
  return 'Legendary';
}

// ai model setup
env.allowLocalModels = false;

// fast model, used by the live scan loop while the camera is open (~every 1.2s).
// quantized so it's fast enough to run on-device — we make up for the accuracy
// hit with multi-frame smoothing + requiring a streak of passes (see below)
let classifierPromise = null;
function getClassifier(onProgress) {
  if (!classifierPromise) {
    classifierPromise = pipeline(
        'zero-shot-image-classification',
        'Xenova/siglip-base-patch16-224',
        { dtype: 'q8', ...(onProgress ? { progress_callback: onProgress } : {}) }
    ).catch(error => {
      classifierPromise = null;
      throw error;
    });
  }
  return classifierPromise;
}

// bigger/slower model, only used for one-shot checks (uploaded photos, the final
// confirm frame) — basically a second opinion before we actually mark it verified
let staticClassifierPromise = null;
function getStaticClassifier(onProgress) {
  if (!staticClassifierPromise) {
    staticClassifierPromise = pipeline(
        'zero-shot-image-classification',
        'Xenova/siglip-large-patch16-256',
        onProgress ? { progress_callback: onProgress } : undefined
    ).catch(error => {
      staticClassifierPromise = null;
      throw error;
    });
  }
  return staticClassifierPromise;
}

// photo source labels (for the "is this actually your photo" check)
const SOURCE_CAMERA_LABEL = 'a real unedited photo taken with a phone camera';
const SOURCE_DOWNLOADED_LABELS = [
  'a professional stock photography image',
  'a screenshot or image saved from a website',
  'an image downloaded from a search engine or social media',
];
const SOURCE_LABELS = [SOURCE_CAMERA_LABEL, ...SOURCE_DOWNLOADED_LABELS];

// pose model — this is what actually counts reps
let landmarkerPromise = null;
function getPoseLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      );
      return PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
      });
    })().catch(error => {
      landmarkerPromise = null;
      throw error;
    });
  }
  return landmarkerPromise;
}

const LM = {
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_ELBOW: 13, R_ELBOW: 14,
  L_WRIST: 15, R_WRIST: 16,
  L_HIP: 23, R_HIP: 24,
  L_KNEE: 25, R_KNEE: 26,
  L_ANKLE: 27, R_ANKLE: 28,
};

const POSE_CONNECTIONS = [
  [11, 12], [11, 23], [12, 24], [23, 24], // Torso
  [11, 13], [13, 15],                     // Left Arm
  [12, 14], [14, 16],                     // Right Arm
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31], // Left Leg
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32], // Right Leg
  [0, 1], [1, 2], [2, 3], [3, 7],         // Left Face
  [0, 4], [4, 5], [5, 6], [6, 8],         // Right Face
  [9, 10]                                 // Mouth
];

function angleBetween(a, b, c) {
  if (!a || !b || !c) return null;
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.hypot(v1.x, v1.y);
  const mag2 = Math.hypot(v2.x, v2.y);
  if (mag1 === 0 || mag2 === 0) return null;
  const cos = Math.min(1, Math.max(-1, dot / (mag1 * mag2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

function visiblePt(pt, minVis = 0.5) {
  return pt && (pt.visibility === undefined || pt.visibility >= minVis);
}

function pickSide(lms, leftKeys, rightKeys) {
  const left = leftKeys.map(k => lms[k]);
  const right = rightKeys.map(k => lms[k]);
  const leftOk = left.every(p => visiblePt(p));
  const rightOk = right.every(p => visiblePt(p));
  if (leftOk && rightOk) {
    return left.map((p, i) => ({
      x: (p.x + right[i].x) / 2,
      y: (p.y + right[i].y) / 2,
      visibility: Math.min(p.visibility ?? 1, right[i].visibility ?? 1),
    }));
  }
  if (leftOk) return left;
  if (rightOk) return right;
  return null;
}

const REP_METRICS = {
  q1: (lms) => { const p = pickSide(lms, [LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST], [LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST]); return p ? angleBetween(p[0], p[1], p[2]) : null; },
  q2: (lms) => { const p = pickSide(lms, [LM.L_HIP, LM.L_KNEE, LM.L_ANKLE], [LM.R_HIP, LM.R_KNEE, LM.R_ANKLE]); return p ? angleBetween(p[0], p[1], p[2]) : null; },
  q4: (lms) => { const p = pickSide(lms, [LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST], [LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST]); return p ? angleBetween(p[0], p[1], p[2]) : null; },
  q12:(lms) => { const p = pickSide(lms, [LM.L_SHOULDER, LM.L_HIP, LM.L_KNEE], [LM.R_SHOULDER, LM.R_HIP, LM.R_KNEE]); return p ? angleBetween(p[0], p[1], p[2]) : null; },
  q17:(lms) => { const hips = [lms[LM.L_HIP], lms[LM.R_HIP]].filter(p => visiblePt(p)); if (!hips.length) return null; return hips.reduce((s, p) => s + p.y, 0) / hips.length; },
  q23:(lms) => { const p = pickSide(lms, [LM.L_HIP, LM.L_KNEE, LM.L_ANKLE], [LM.R_HIP, LM.R_KNEE, LM.R_ANKLE]); return p ? angleBetween(p[0], p[1], p[2]) : null; },
};

const REP_CONFIG = {
  q1:  { mode: 'angle',  downThreshold: 100,  upThreshold: 155,   cueDown: 'Lower into the pushup', cueUp: 'Push back up to full extension' },
  q2:  { mode: 'angle',  downThreshold: 110,  upThreshold: 160,   cueDown: 'Squat down',             cueUp: 'Stand back up' },
  q4:  { mode: 'angle',  downThreshold: 80,   upThreshold: 150,   cueDown: 'Pull up to the bar',     cueUp: 'Lower to a full hang' },
  q12: { mode: 'angle',  downThreshold: 110,  upThreshold: 150,   cueDown: 'Crunch up',              cueUp: 'Lower back down' },
  q17: { mode: 'height', downThreshold: 0.02, upThreshold: 0.005, cueDown: 'Jump!',                  cueUp: 'Land' },
  q23: { mode: 'angle',  downThreshold: 110,  upThreshold: 160,   cueDown: 'Lower into the lunge',   cueUp: 'Return to standing' },
};

const MIN_REP_MS = 500;

function createPoseRepState(initialReps = 0) {
  return { phase: 'up', reps: initialReps, baselineY: null, lastRepAt: 0, smoothed: null };
}

function updatePoseRepState(state, questId, landmarks, now) {
  const config = REP_CONFIG[questId];
  const metricFn = REP_METRICS[questId];
  if (!config || !metricFn || !landmarks) {
    return { reps: state.reps, phase: state.phase, cue: 'Move into frame', counted: false };
  }
  const raw = metricFn(landmarks);
  if (raw === null) {
    return { reps: state.reps, phase: state.phase, cue: 'Move fully into frame', counted: false };
  }
  state.smoothed = state.smoothed === null ? raw : state.smoothed * 0.6 + raw * 0.4;
  const value = state.smoothed;
  let counted = false;

  if (config.mode === 'height') {
    if (state.baselineY === null) state.baselineY = value;
    else state.baselineY = state.baselineY * 0.98 + value * 0.02;
    const jumpHeight = state.baselineY - value;
    if (state.phase === 'up' && jumpHeight >= config.downThreshold) {
      state.phase = 'down';
    } else if (state.phase === 'down' && jumpHeight <= config.upThreshold) {
      if (now - state.lastRepAt >= MIN_REP_MS) { state.reps += 1; state.lastRepAt = now; counted = true; }
      state.phase = 'up';
    }
  } else {
    if (state.phase === 'up' && value <= config.downThreshold) {
      state.phase = 'down';
    } else if (state.phase === 'down' && value >= config.upThreshold) {
      if (now - state.lastRepAt >= MIN_REP_MS) { state.reps += 1; state.lastRepAt = now; counted = true; }
      state.phase = 'up';
    }
  }

  const cue = state.phase === 'up' ? config.cueDown : config.cueUp;
  return { reps: state.reps, phase: state.phase, cue, counted };
}

// quest labels
const QUEST_LABELS = {
  q1:  { type: 'reps', activity: ['person doing pushups on floor', 'pushup exercise'], label: 'doing pushups', bodyParts: ['Chest', 'Triceps', 'Shoulders', 'Core'] },
  q2:  { type: 'reps', activity: ['person doing squats exercise', 'squat workout legs bent'], label: 'doing squats', bodyParts: ['Quads', 'Hamstrings', 'Glutes', 'Core'] },
  q4:  { type: 'reps', activity: ['person doing pullups on bar', 'pullup bar exercise'], label: 'doing pullups', bodyParts: ['Lats', 'Upper Back', 'Biceps', 'Forearms'] },
  q12: { type: 'reps', activity: ['person doing situps or crunches', 'abdominal exercise on floor'], label: 'doing situps', bodyParts: ['Abs', 'Obliques', 'Hip Flexors'] },
  q17: { type: 'reps', activity: ['person jumping rope', 'skipping rope exercise'], label: 'jumping rope', bodyParts: ['Calves', 'Quads', 'Shoulders', 'Cardio'] },
  q23: { type: 'reps', activity: ['person doing lunges exercise', 'lunge workout legs split stance'], label: 'doing lunges', bodyParts: ['Quads', 'Glutes', 'Hamstrings'] },
  q3:  { type: 'map', activity: ['gps tracking map route screenshot', 'fitness tracker map running route'], label: 'running map screenshot' },
  q16: { type: 'map', activity: ['gps tracking map route screenshot', 'cycling route map on phone screen'], label: 'cycling map screenshot' },
  q5:  { type: 'map', activity: ['gps tracking map route screenshot', 'walking route map tracker'], label: 'walking map screenshot' },
  q22: { type: 'map', activity: ['gps tracking map route screenshot', 'step counter fitness app screenshot'], label: 'step tracking map' },
  q9:  { type: 'food', activity: ['healthy food meal salad vegetables on a plate', 'nutritious meal in a bowl'], label: 'plate of healthy food' },
  q19: { type: 'food', activity: ['clean healthy meal on plate', 'plate of vegetables and whole foods'], label: 'plate of clean food' },
  q20: { type: 'food', activity: ['cooked food on a plate', 'homemade meal in a bowl or plate'], label: 'cooked meal' },
  q14: { type: 'food', activity: ['glass of green smoothie', 'blended green juice drink'], label: 'green smoothie' },
  q6:  { type: 'food', activity: ['glass of water', 'reusable water bottle filled'], label: 'water bottle' },
  q7:  { type: 'action', activity: ['person meditating cross-legged', 'mindfulness exercise'], label: 'meditating' },
  q8:  { type: 'action', activity: ['person stretching muscles', 'yoga stretch pose'], label: 'stretching', bodyParts: ['Full Body', 'Flexibility'] },
  q10: { type: 'action', activity: ['person sleeping in bed', 'person resting in bed eyes closed'], label: 'getting good sleep' },
  q11: { type: 'action', activity: ['person doing jumping jacks or burpees'], label: 'doing cardio', bodyParts: ['Cardio', 'Full Body'] },
  q13: { type: 'action', activity: ['handwriting in notebook or journal'], label: 'journaling' },
  q15: { type: 'action', activity: ['person doing plank exercise', 'plank position core exercise'], label: 'holding a plank', bodyParts: ['Core', 'Shoulders'] },
  q18: { type: 'action', activity: ['shower running water', 'bathroom shower head with water'], label: 'in the shower' },
  q21: { type: 'action', activity: ['person breathing deeply eyes closed'], label: 'deep breathing' },
  q24: { type: 'action', activity: ['person sleeping in bed at night', 'sleeping in dark bedroom'], label: 'sleeping early' },
  q25: { type: 'action', activity: ['person in ice bath tub', 'cold plunge tub with ice'], label: 'in a cold plunge' },
  q26: { type: 'action', activity: ['person doing a wall sit exercise against a wall'], label: 'holding a wall sit', bodyParts: ['Quads', 'Core'] },
  q27: { type: 'action', activity: ['person doing a glute bridge exercise on floor'], label: 'holding a glute bridge', bodyParts: ['Glutes', 'Core'] },
  q28: { type: 'action', activity: ['person doing high knees exercise'], label: 'doing high knees', bodyParts: ['Cardio', 'Quads'] },
  q29: { type: 'action', activity: ['person doing mountain climbers exercise'], label: 'doing mountain climbers', bodyParts: ['Core', 'Cardio'] },
  q30: { type: 'action', activity: ['person doing a superman back exercise lying face down'], label: 'holding a superman pose', bodyParts: ['Lower Back', 'Glutes'] },
  q31: { type: 'action', activity: ['person doing burpees exercise'], label: 'doing burpees', bodyParts: ['Full Body', 'Cardio'] },
};

const MOVEMENT_ACTION_IDS = new Set(['q8', 'q11', 'q15', 'q26', 'q27', 'q28', 'q29', 'q30', 'q31']);

// labels that catch the most common ways people try to fake a check-in:
// holding a printed photo up to the lens, or pointing the camera at a
// second screen (phone/tablet/laptop) that's playing a photo or video of
// someone else doing the activity. These are added to every negative set
// so the classifier is actively working against replay attempts, not just
// scoring "does this look vaguely like the activity".
const ANTI_SPOOF_LABELS = [
  'a photo or video playing on a phone or tablet screen',
  'a laptop or computer monitor screen',
  'a printed photograph being held up to the camera',
  'a picture of a picture',
];
const SPOOF_BLOCK_THRESHOLD = 0.30;
const PASS_MARGIN = 0.12; // how much the activity score must clear the negative/spoof score by

const getNegativeLabels = (type) => {
  const base = ['person sitting doing nothing', 'person standing still straight', 'random everyday object', ...ANTI_SPOOF_LABELS];
  if (type === 'map') return [...base, 'sweaty selfie face', 'picture of running shoes', 'treadmill machine indoors', 'person running outside'];
  if (type === 'food') return [...base, 'empty plate or bowl', 'restaurant paper menu', 'store product barcode', 'person eating face'];
  return [...base, 'phone or computer screen'];
};

const QUEST_POOL = [
  { id: 'q1',  textTemplate: 'Do {n} pushups',                       xp: 50, reps: 20 },
  { id: 'q2',  textTemplate: 'Do {n} squats',                        xp: 45, reps: 30 },
  { id: 'q3',  textTemplate: 'Go for a {n}-minute run',              xp: 75, duration: 180 },
  { id: 'q4',  textTemplate: 'Do {n} pullups',                       xp: 60, reps: 10 },
  { id: 'q16', textTemplate: 'Do {n} minutes of cycling',            xp: 55, duration: 180 },
  { id: 'q17', textTemplate: 'Do {n} jumping rope reps',             xp: 40, reps: 50 },
  { id: 'q21', textTemplate: 'Do {n} minutes of deep breathing',     xp: 30, duration: 180 },
  { id: 'q23', textTemplate: 'Do {n} lunges',                        xp: 40, reps: 30 },
  { id: 'q25', textTemplate: 'Do a {n}-minute ice bath or cold plunge', xp: 80, duration: 180 },
  { id: 'q5',  textTemplate: 'Walk outside for {n} minutes',         xp: 35, duration: 180 },
  { id: 'q7',  textTemplate: 'Meditate for {n} minutes',             xp: 45, duration: 180 },
  { id: 'q8',  textTemplate: 'Stretch for {n} minutes',              xp: 35, duration: 180 },
  { id: 'q11', textTemplate: 'Do {n} minutes of jumping jacks',      xp: 30, duration: 180 },
  { id: 'q12', textTemplate: 'Do {n} situps',                        xp: 40, reps: 20 },
  { id: 'q15', textTemplate: 'Hold a plank for {n} minutes',         xp: 50, duration: 180 },
  { id: 'q26', textTemplate: 'Hold a wall sit for {n} minutes',      xp: 40, duration: 180 },
  { id: 'q27', textTemplate: 'Hold a glute bridge for {n} minutes',  xp: 35, duration: 180 },
  { id: 'q28', textTemplate: 'Do {n} minutes of high knees',         xp: 35, duration: 180 },
  { id: 'q29', textTemplate: 'Do {n} minutes of mountain climbers',  xp: 40, duration: 180 },
  { id: 'q30', textTemplate: 'Hold a superman pose for {n} minutes', xp: 35, duration: 180 },
  { id: 'q31', textTemplate: 'Do {n} minutes of burpees',            xp: 45, duration: 180 },
];

function randomizeQuest(pool) {
  if (pool.reps) {
    const n = Math.floor(Math.random() * 55) + 1; // 1-55 reps
    return { ...pool, reps: n, text: pool.textTemplate.replace('{n}', n), completed: false, progress: 0 };
  }
  if (pool.duration) {
    const minutes = Math.floor(Math.random() * 5) + 1; // 1-5 minutes
    return { ...pool, duration: minutes * 60, text: pool.textTemplate.replace('{n}', minutes), completed: false, progress: 0 };
  }
  return { ...pool, text: pool.textTemplate, completed: false, progress: 0 };
}

const ONE_DAY_MS = 24 * 60 * 60 * 1000;
const REQUIRED_PASSES = 4;
const PASS_THRESHOLD  = 0.42;
const SCORE_SMOOTHING = 0.38; // weight given to each new frame when smoothing confidence — lower = slower to react, harder to fool with a quick flash

function getMaxLabelScore(results, candidates) {
  if (!candidates?.length) return 0;
  return results.reduce((best, result) => (candidates.includes(result.label) ? Math.max(best, result.score) : best), 0);
}

async function checkPhotoAuthenticity(file, dataUrl, classifierLabels) {
  let hasCameraMetadata = false;
  try {
    const tags = await exifr.parse(file, ['Make', 'Model']);
    hasCameraMetadata = Boolean(tags && (tags.Make || tags.Model));
  } catch { /* no/unreadable EXIF — treated as absent, not fatal */ }

  let cameraScore = 0, downloadedScore = 0;
  try {
    const classifier = await getStaticClassifier();
    const results = await classifier(dataUrl, classifierLabels);
    cameraScore = getMaxLabelScore(results, [SOURCE_CAMERA_LABEL]);
    downloadedScore = getMaxLabelScore(results, SOURCE_DOWNLOADED_LABELS);
  } catch { /* if the model call fails, fall back to EXIF alone */ }

  const margin = downloadedScore - cameraScore;
  const modelThinksDownloaded = margin > 0.05;
  const suspicious = !hasCameraMetadata && modelThinksDownloaded;
  const confidence = Math.round(Math.max(cameraScore, downloadedScore) * 100);
  return { hasCameraMetadata, cameraScore, downloadedScore, suspicious, confidence };
}

// colors
// just the standard ios system colors, light + dark. using plain objects instead
// of tailwind's dark: classes since we're already passing `dark` around as state
const C = {
  light: {
    bg: '#F2F2F7', bgElevated: '#FFFFFF', bgSecondary: '#F2F2F7', bgTertiary: '#E5E5EA',
    label: '#000000', labelSecondary: 'rgba(60,60,67,0.60)', labelTertiary: 'rgba(60,60,67,0.30)',
    separator: 'rgba(60,60,67,0.29)', fill: 'rgba(120,120,128,0.12)',
    blue: '#007AFF', green: '#34C759', orange: '#FF9500', red: '#FF3B30',
    indigo: '#5856D6', teal: '#30B0C7', purple: '#AF52DE', yellow: '#FFCC00', pink: '#FF2D55', gray: '#8E8E93',
    glass: 'rgba(255,255,255,0.55)', glassStrong: 'rgba(255,255,255,0.72)',
    glassBorder: 'rgba(255,255,255,0.6)', glassHighlight: 'rgba(255,255,255,0.9)',
    glassShadow: '0 1px 1px rgba(0,0,0,0.03), 0 8px 24px -12px rgba(0,0,0,0.14)',
  },
  dark: {
    bg: '#000000', bgElevated: '#1C1C1E', bgSecondary: '#1C1C1E', bgTertiary: '#2C2C2E',
    label: '#FFFFFF', labelSecondary: 'rgba(235,235,245,0.60)', labelTertiary: 'rgba(235,235,245,0.30)',
    separator: 'rgba(84,84,88,0.65)', fill: 'rgba(120,120,128,0.24)',
    blue: '#0A84FF', green: '#30D158', orange: '#FF9F0A', red: '#FF453A',
    indigo: '#5E5CE6', teal: '#40C8E0', purple: '#BF5AF2', yellow: '#FFD60A', pink: '#FF375F', gray: '#98989D',
    glass: 'rgba(28,28,30,0.55)', glassStrong: 'rgba(28,28,30,0.72)',
    glassBorder: 'rgba(255,255,255,0.1)', glassHighlight: 'rgba(255,255,255,0.14)',
    glassShadow: '0 1px 1px rgba(0,0,0,0.2), 0 8px 24px -12px rgba(0,0,0,0.5)',
  },
};

// one helper so every glass surface looks the same — real "liquid glass":
// heavy blur + saturation boost, a soft glossy wash top-to-bottom, and a
// thin specular highlight traced along the inner top edge like light
// catching the rim of actual glass.
const glassStyle = (c, strong = false) => ({
  background: `linear-gradient(165deg, ${c.glassHighlight} 0%, ${strong ? c.glassStrong : c.glass} 22%, ${strong ? c.glassStrong : c.glass} 100%)`,
  backdropFilter: 'blur(22px) saturate(165%)',
  WebkitBackdropFilter: 'blur(22px) saturate(165%)',
  border: `0.5px solid ${c.glassBorder}`,
  boxShadow: `inset 0 1px 0 ${c.glassHighlight}, inset 0 0 0 0.5px rgba(255,255,255,0.04), ${c.glassShadow}`,
  transform: 'translateZ(0)',
});

// system font stack, no google fonts import — applied to every descendant so
// nothing inside the app can accidentally fall back to a non-system typeface
const SystemType = () => (
    <style>{`
    .sq-root, .sq-root * { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif; }
    .sq-mono, .sq-mono * { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-variant-numeric: tabular-nums; }
    .sq-large-title { letter-spacing: -0.6px; }
    .sq-title { letter-spacing: -0.2px; }
    * { -webkit-tap-highlight-color: transparent; }
    /* No text selection, no pinch/double-tap zoom — this is meant to feel
       like a native app, not a webpage. Inputs stay selectable/editable so
       usernames etc. still work normally. touch-action still allows normal
       scrolling (pan-y); it only blocks the pinch/double-tap zoom gestures —
       the viewport meta tag in index.html should also set
       maximum-scale=1, user-scalable=no as the primary safeguard, since
       some browsers only respect that and ignore touch-action for zoom. */
    .sq-root {
      -webkit-user-select: none; -moz-user-select: none; user-select: none;
      -webkit-touch-callout: none;
      touch-action: pan-x pan-y;
    }
    .sq-root input, .sq-root textarea {
      -webkit-user-select: text; -moz-user-select: text; user-select: text;
      touch-action: manipulation;
    }
    /* Theme crossfade — kept to cheap, compositor-friendly properties only.
       box-shadow/backdrop-filter are deliberately excluded here: animating
       those on many overlapping glass surfaces at once is what causes janky,
       "unpolished" frame drops, so those stay instant while color/background
       still crossfade smoothly. */
    .sq-root, .sq-root :where(div, span, p, h1, h2, h3, button, a, input) {
      transition: background-color 0.3s ease, border-color 0.3s ease, color 0.25s ease;
    }
    svg { transition: stroke 0.2s ease, fill 0.2s ease; }
    .sq-scroll { -webkit-overflow-scrolling: touch; }
    .sq-icon-fff svg { color: #fff; }
    @keyframes sq-fade-up { from { opacity: 0; transform: translate3d(0,7px,0); } to { opacity: 1; transform: translate3d(0,0,0); } }
    @keyframes sq-check-in { 0% { opacity: 0; transform: scale(0.5); } 70% { opacity: 1; transform: scale(1.06); } 100% { opacity: 1; transform: scale(1); } }
    .sq-anim-in { animation: sq-fade-up 0.32s cubic-bezier(0.22,1,0.36,1) both; will-change: transform, opacity; }
    .sq-anim-check { animation: sq-check-in 0.4s cubic-bezier(0.22,1,0.36,1) both; }
    button, a { transition: opacity 0.2s ease-out, transform 0.2s cubic-bezier(0.22,1,0.36,1), background-color 0.25s ease; }
    button:active { transform: scale(0.96); }
    @keyframes sq-icon-pop { 0% { opacity: 0; transform: scale(0.6) rotate(-8deg); } 60% { opacity: 1; transform: scale(1.12) rotate(2deg); } 100% { opacity: 1; transform: scale(1) rotate(0deg); } }
    .sq-icon-tap { transition: transform 0.2s cubic-bezier(0.34,1.56,0.64,1); display: inline-flex; will-change: transform; }
    button:active .sq-icon-tap { transform: scale(1.2) rotate(-6deg); }
    .sq-icon-pop-in { animation: sq-icon-pop 0.4s cubic-bezier(0.22,1,0.36,1) both; }
    .sq-tab-icon { transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1); will-change: transform; }
    .sq-tab-icon-active { transform: scale(1.14) translateY(-1px); }
    @keyframes sq-drift-a { 0%,100% { transform: translate3d(-6%,-4%,0) scale(1); } 50% { transform: translate3d(8%,10%,0) scale(1.15); } }
    @keyframes sq-drift-b { 0%,100% { transform: translate3d(10%,6%,0) scale(1.1); } 50% { transform: translate3d(-8%,-8%,0) scale(0.95); } }
    @keyframes sq-drift-c { 0%,100% { transform: translate3d(-4%,8%,0) scale(0.95); } 50% { transform: translate3d(6%,-10%,0) scale(1.1); } }
    .sq-orb-a { animation: sq-drift-a 26s ease-in-out infinite; will-change: transform; }
    .sq-orb-b { animation: sq-drift-b 32s ease-in-out infinite; will-change: transform; }
    .sq-orb-c { animation: sq-drift-c 22s ease-in-out infinite; will-change: transform; }
    .sq-confetti-piece { position: absolute; top: -6%; border-radius: 2px; opacity: 0.9; animation-name: sq-confetti-fall; animation-timing-function: cubic-bezier(0.35,0,0.65,1); animation-fill-mode: forwards; }
    @keyframes sq-confetti-fall { 0% { transform: translate(0,0) rotate(0deg); opacity: 1; } 85% { opacity: 1; } 100% { transform: translate(var(--sq-drift), 115vh) rotate(var(--sq-rot)); opacity: 0; } }
    @media (prefers-reduced-motion: reduce) {
      .sq-orb-a, .sq-orb-b, .sq-orb-c { animation: none; }
    }
  `}</style>
);

// background blobs
// 3 slow-drifting blurred color fields, fixed behind everything. this is what
// the glass cards are actually blurring/tinting — without it the "glass" is
// just a translucent gray box, which looks flat
const AmbientBackground = React.memo(function AmbientBackground({ c }) {
  return (
      <div className="fixed inset-0 pointer-events-none z-0" aria-hidden="true" style={{ overflow: 'hidden' }}>
        <div className="sq-orb-a" style={{ position: 'absolute', top: '-10%', left: '-15%', width: '75%', height: '42%', borderRadius: '50%', background: c.blue, opacity: 0.16, filter: 'blur(70px)' }} />
        <div className="sq-orb-b" style={{ position: 'absolute', top: '30%', right: '-20%', width: '70%', height: '46%', borderRadius: '50%', background: c.purple, opacity: 0.13, filter: 'blur(80px)' }} />
        <div className="sq-orb-c" style={{ position: 'absolute', bottom: '-14%', left: '5%', width: '65%', height: '40%', borderRadius: '50%', background: c.teal, opacity: 0.13, filter: 'blur(75px)' }} />
      </div>
  );
});

// icons
// SF-Symbols-style icons via lucide-react — clean, rounded, consistent 1.5–2px strokes
const CheckIcon = React.memo(function CheckIcon({ size = 12 }) {
  return (
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="20 6 9 17 4 12"/>
      </svg>
  );
});
const ChevronIcon = React.memo(function ChevronIcon({ color }) {
  return (
      <svg width="8" height="14" viewBox="0 0 8 14" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="1 1 7 7 1 13"/></svg>
  );
});
const BackChevron = React.memo(function BackChevron({ color }) {
  return (
      <svg width="11" height="18" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
  );
});
const IOSShareIcon = React.memo(function IOSShareIcon() {
  return (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/>
      </svg>
  );
});
const IOSAddIcon = React.memo(function IOSAddIcon() {
  return (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="4" ry="4"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/>
      </svg>
  );
});
const AndroidMenuIcon = React.memo(function AndroidMenuIcon() {
  return (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="5" r="2.2"/><circle cx="12" cy="12" r="2.2"/><circle cx="12" cy="19" r="2.2"/></svg>
  );
});
const AndroidAddIcon = React.memo(function AndroidAddIcon() {
  return (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <rect x="5" y="2" width="14" height="20" rx="2" ry="2"/><line x1="12" y1="18" x2="12.01" y2="18"/><line x1="9" y1="11" x2="15" y2="11"/><line x1="12" y1="8" x2="12" y2="14"/>
      </svg>
  );
});
const TrophyIcon = React.memo(function TrophyIcon({ size = 16 }) {
  return (
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <path d="M8 4h8v5a4 4 0 0 1-8 0V4Z"/><path d="M8 5H5a3 3 0 0 0 3 4"/><path d="M16 5h3a3 3 0 0 1-3 4"/><path d="M12 13v3"/><path d="M9 20h6"/><path d="M10 16h4l.5 4h-5l.5-4Z"/>
      </svg>
  );
});
const WarningIcon = React.memo(function WarningIcon({ size = 15 }) {
  return (
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10.3 3.6 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.6a2 2 0 0 0-3.4 0Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
  );
});
const SignalOffIcon = React.memo(function SignalOffIcon({ size = 24 }) {
  return (
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M5 12.5a11 11 0 0 1 4-2.5"/><path d="M9.5 8.8A11 11 0 0 1 19 10.5"/><path d="M12.5 15a4 4 0 0 1 3 1.8"/><circle cx="8" cy="19" r="1"/><line x1="2" y1="2" x2="22" y2="22"/>
      </svg>
  );
});
const FlagIcon = React.memo(function FlagIcon({ size = 24 }) {
  return (
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M5 21V4"/><path d="M5 4h13l-3 4 3 4H5"/>
      </svg>
  );
});
const GoogleIcon = React.memo(function GoogleIcon({ size = 18 }) {
  return (
      <svg width={size} height={size} viewBox="0 0 48 48">
        <path fill="#FFC107" d="M43.611,20.083H42V20H24v8h11.303c-1.649,4.657-6.08,8-11.303,8c-6.627,0-12-5.373-12-12s5.373-12,12-12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C12.955,4,4,12.955,4,24s8.955,20,20,20s20-8.955,20-20C44,22.659,43.862,21.35,43.611,20.083z"/>
        <path fill="#FF3D00" d="M6.306,14.691l6.571,4.819C14.655,15.108,18.961,12,24,12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C16.318,4,9.656,8.337,6.306,14.691z"/>
        <path fill="#4CAF50" d="M24,44c5.166,0,9.86-1.977,13.409-5.192l-6.19-5.238C29.211,35.091,26.715,36,24,36c-5.202,0-9.619-3.317-11.283-7.946l-6.522,5.025C9.505,39.556,16.227,44,24,44z"/>
        <path fill="#1976D2" d="M43.611,20.083H42V20H24v8h11.303c-0.792,2.237-2.231,4.166-4.087,5.571c0.001-0.001,0.002-0.001,0.003-0.002l6.19,5.238C36.971,39.205,44,34,44,24C44,22.659,43.862,21.35,43.611,20.083z"/>
      </svg>
  );
});

// quest icons — mapped straight onto Lucide (SF-Symbols-style) components.
// Each is a drop-in React component, so existing call sites (which pass
// size/color/style props) work unchanged.
const QuestSvg = {
  cup: GlassWater,
  smoothie: CupSoda,
  dumbbell: Dumbbell,
  run: PersonStanding,
  bike: Bike,
  footsteps: Footprints,
  legs: Footprints,
  core: CircleDot,
  stopwatch: Timer,
  bar: ChevronsUp,
  droplet: Droplet,
  leaf: Leaf,
  bowl: Utensils,
  pot: CookingPot,
  moon: Moon,
  lotus: Flower2,
  stretch: Move,
  bolt: Zap,
  flame: Flame,
  pencil: Pencil,
  snowflake: Snowflake,
  wind: Wind,
  rope: Waves,
  target: Target,
};

// Quest categories map to system colors, not a rotating decorative palette —
// each color means the same category everywhere in the app.
// brand logo — embedded as a data URI (resized + palette-optimized from the
// uploaded artwork) so the app icon works standalone with no separate asset
// file or hosting path to manage. White line-art on transparent background,
// designed to sit inside a solid-color rounded badge the same way the old
// placeholder trophy icon did.
const BRAND_LOGO_SRC = 'data:image/png;base64,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';
const BrandLogo = React.memo(function BrandLogo({ size = 24 }) {
  return <img src={BRAND_LOGO_SRC} alt="" width={size} height={size} style={{ display: 'block', objectFit: 'contain' }} />;
});

const CAT = { strength: 'orange', cardio: 'red', mind: 'indigo', recover: 'purple', fuel: 'green', cold: 'teal', track: 'blue' };
const QUEST_THEME = {
  q1:  { icon: 'dumbbell', cat: CAT.strength }, q2:  { icon: 'legs',      cat: CAT.strength },
  q3:  { icon: 'run',      cat: CAT.cardio },    q4:  { icon: 'bar',       cat: CAT.strength },
  q5:  { icon: 'footsteps',cat: CAT.track },      q6:  { icon: 'cup',       cat: CAT.fuel },
  q7:  { icon: 'lotus',    cat: CAT.mind },       q8:  { icon: 'stretch',   cat: CAT.recover },
  q9:  { icon: 'bowl',     cat: CAT.fuel },        q10: { icon: 'moon',      cat: CAT.recover },
  q11: { icon: 'bolt',     cat: CAT.cardio },      q12: { icon: 'core',      cat: CAT.strength },
  q13: { icon: 'pencil',   cat: CAT.mind },        q14: { icon: 'smoothie',  cat: CAT.fuel },
  q15: { icon: 'stopwatch',cat: CAT.recover },     q16: { icon: 'bike',      cat: CAT.cardio },
  q17: { icon: 'rope',     cat: CAT.cardio },      q18: { icon: 'droplet',   cat: CAT.recover },
  q19: { icon: 'leaf',     cat: CAT.fuel },        q20: { icon: 'pot',       cat: CAT.fuel },
  q21: { icon: 'wind',     cat: CAT.mind },        q22: { icon: 'footsteps', cat: CAT.track },
  q23: { icon: 'legs',     cat: CAT.strength },    q24: { icon: 'moon',      cat: CAT.recover },
  q25: { icon: 'snowflake',cat: CAT.cold },
  q26: { icon: 'legs',     cat: CAT.strength },    q27: { icon: 'core',      cat: CAT.strength },
  q28: { icon: 'bolt',     cat: CAT.cardio },      q29: { icon: 'core',      cat: CAT.cardio },
  q30: { icon: 'stretch',  cat: CAT.recover },     q31: { icon: 'flame',     cat: CAT.cardio },
};

const QUEST_ABOUT = {
  q1: "Pushups build raw upper-body strength and core stability in one clean movement.", q2: "Squats fire up your biggest muscle groups and reinforce the mechanics behind athletic movement.",
  q3: "A steady run gets your heart rate up, clears your head, and builds endurance.", q4: "Pullups are one of the purest tests of back and grip strength.",
  q5: "A brisk walk outside boosts circulation, mood, and gives your eyes a break.", q6: "Staying hydrated helps your body perform better and keeps your mind sharp.",
  q7: "A short meditation resets your focus and lowers stress before it builds up.", q8: "Stretching keeps your muscles loose and your joints moving.",
  q9: "A balanced, whole-food meal fuels recovery and keeps your energy steady.", q10: "Consistent, sufficient sleep is the single biggest lever for recovery.",
  q11: "A quick cardio burst spikes your heart rate and wakes up your whole body fast.", q12: "Situps target your core and build the stability everything else is built on.",
  q13: "Journaling for a few minutes helps you process the day and plan the next one.", q14: "A green smoothie is an easy way to pack in nutrients when short on time.",
  q15: "Holding a plank builds isometric core strength that carries over to everything else.", q16: "Cycling is easy on the joints while still building serious cardio endurance.",
  q17: "Jump rope sharpens coordination and torches calories in a small amount of time.", q18: "A cold shower is a quick way to train discipline and wake up your nervous system.",
  q19: "Cutting added sugar for a day gives your energy levels a noticeably steadier baseline.", q20: "Cooking from scratch puts you in control of what goes into your body.",
  q21: "A few minutes of deep breathing calms your nervous system and sharpens focus.", q22: "Hitting your step count keeps your body moving steadily throughout the day.",
  q23: "Lunges build single-leg strength and balance that squats alone don't cover.", q24: "An earlier bedtime compounds — better sleep tonight means a better day tomorrow.",
  q25: "Cold exposure trains resilience and gives your recovery a real boost.",
  q26: "Wall sits build isometric leg strength without any equipment needed.",
  q27: "Glute bridges wake up and strengthen the muscles that keep your hips stable.",
  q28: "High knees spike your heart rate fast and sharpen coordination.",
  q29: "Mountain climbers combine cardio and core work in one fluid movement.",
  q30: "The superman hold strengthens your lower back and posterior chain.",
  q31: "Burpees are a full-body blast that builds strength and conditioning together.",
};

const QUEST_QUOTES = ["Small steps every day lead to big changes.", "Discipline is choosing between what you want now and what you want most.", "Progress, not perfection.", "The body achieves what the mind believes.", "One quest at a time.", "Consistency beats intensity.", "You didn't come this far to only come this far."];
const QUEST_QUOTE = { q1: "Strength grows one rep at a time.", q2: "Every squat builds a stronger foundation.", q3: "Miles don't lie — you earned this one.", q4: "Small steps every day lead to big changes.", q5: "One step at a time is still progress.", q6: "Small steps every day lead to big changes.", q7: "A quiet mind carries the loudest strength.", q8: "Flexibility today, resilience tomorrow.", q9: "You fueled the body that carries you.", q10: "Rest is where the real gains happen.", q11: "Energy in motion stays in motion.", q12: "A strong core holds everything else together.", q13: "The pen remembers what the mind forgets.", q14: "Good fuel, good day.", q15: "Stillness can be the hardest work of all.", q16: "Every mile ridden is a mile earned.", q17: "Rhythm builds more than just your legs.", q18: "Discomfort today, discipline for life.", q19: "Progress, not perfection.", q20: "What you cook is what you become.", q21: "Breathe in control, breathe out doubt.", q22: "One step at a time is still progress.", q23: "Balance is built one side at a time.", q24: "Tonight's rest is tomorrow's edge.", q25: "You didn't come this far to only come this far." };

// flat icon tile, one color one glyph, no gradient/gloss nonsense
const QuestIconBadge = React.memo(function QuestIconBadge({ questId, size = 88, c }) {
  const theme = QUEST_THEME[questId] || { icon: 'target', cat: 'blue' };
  const Icon = QuestSvg[theme.icon] || QuestSvg.target;
  const color = c[theme.cat];
  return (
      <div
          style={{
            width: size, height: size, borderRadius: size * 0.36,
            background: color, display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
          }}
      >
        <Icon className="sq-icon-pop-in sq-icon-tap" width={Math.round(size * 0.44)} height={Math.round(size * 0.44)} strokeWidth={1.75} color="#FFFFFF" style={{ color: '#FFFFFF' }} />
      </div>
  );
});

// smaller version for list rows
const QuestRowIcon = React.memo(function QuestRowIcon({ questId, c, muted }) {
  const theme = QUEST_THEME[questId] || { icon: 'target', cat: 'blue' };
  const Icon = QuestSvg[theme.icon] || QuestSvg.target;
  const color = c[theme.cat];
  return (
      <div style={{ width: 30, height: 30, borderRadius: 12, background: muted ? c.fill : color, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
        <Icon className="sq-icon-pop-in sq-icon-tap" width={16} height={16} strokeWidth={1.75} color={muted ? c.labelTertiary : '#FFFFFF'} style={{ color: muted ? c.labelTertiary : '#FFFFFF' }} />
      </div>
  );
});

// like the activity rings but just one flat stroke, no glow
const ProgressRing = React.memo(function ProgressRing({ pct, size = 46, stroke = 5, c }) {
  const r = (size - stroke) / 2;
  const circumference = 2 * Math.PI * r;
  const offset = circumference - (pct / 100) * circumference;
  return (
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={c.fill} strokeWidth={stroke} />
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={c.blue} strokeWidth={stroke} strokeLinecap="round"
                  strokeDasharray={circumference} style={{ strokeDashoffset: offset, transition: 'stroke-dashoffset 0.6s cubic-bezier(0.22,1,0.36,1)' }} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="sq-mono" style={{ fontSize: 11, fontWeight: 600, color: c.label }}>{Math.round(pct)}%</span>
        </div>
      </div>
  );
});

// little stat tile
const StatChip = React.memo(function StatChip({ label, value, c, accentColor }) {
  return (
      <div style={{ ...glassStyle(c), borderRadius: 20, padding: '14px 8px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
        <span className="sq-mono" style={{ fontSize: 20, fontWeight: 600, color: accentColor || c.label, lineHeight: 1 }}>{value}</span>
        <span style={{ fontSize: 11, fontWeight: 500, color: c.labelSecondary }}>{label}</span>
      </div>
  );
});

// profile picture — shows the account's photo if it has one, otherwise a
// flat colored circle with the username's first letter. Used in the
// header, the leaderboard, and the settings screen so avatars look
// identical everywhere.
const Avatar = React.memo(function Avatar({ photoURL, username, size = 36, c, ring }) {
  const initial = (username || '?').trim().charAt(0).toUpperCase() || '?';
  return (
      <div style={{
        width: size, height: size, borderRadius: '50%', flexShrink: 0, overflow: 'hidden',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: photoURL ? 'transparent' : c.blue,
        boxShadow: ring ? `0 0 0 2px ${c.bgElevated}, 0 0 0 3.5px ${c.blue}` : 'none',
      }}>
        {photoURL
            ? <img src={photoURL} alt={username || 'Profile'} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            : <span style={{ color: '#fff', fontWeight: 700, fontSize: size * 0.42 }}>{initial}</span>}
      </div>
  );
});

// confetti
const Confetti = ({ count = 22, big = false, c }) => {
  const colors = [c.blue, c.green, c.orange, c.purple, c.teal];
  const pieces = useMemo(() => {
    const total = big ? Math.round(count * 1.6) : count;
    return Array.from({ length: total }).map((_, i) => ({
      id: i,
      left: Math.random() * 100,
      delay: Math.random() * 0.3,
      duration: 1.3 + Math.random() * 1.1,
      color: colors[i % colors.length],
      w: 5 + Math.random() * 5,
      h: 3 + Math.random() * 4,
      rot: Math.round(Math.random() * 360),
      drift: Math.round((Math.random() - 0.5) * 140),
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [count, big]);

  // Keyframes/class for confetti pieces live once in <SystemType/> instead of
  // being re-injected as a fresh <style> tag every time a quest completes.
  return (
      <div className="pointer-events-none absolute inset-0 overflow-hidden z-20" aria-hidden="true">
        {pieces.map(p => (
            <span key={p.id} className="sq-confetti-piece" style={{
              left: `${p.left}%`, '--sq-drift': `${p.drift}px`, '--sq-rot': `${p.rot}deg`,
              animationDelay: `${p.delay}s`, animationDuration: `${p.duration}s`,
              background: p.color, width: p.w, height: p.h,
            }} />
        ))}
      </div>
  );
};

// Ticking countdown — isolated into its own component with local state so a
// 1-second tick doesn't force the entire quest list / camera / ambient
// background tree above it to re-render. Parents only re-render when the
// underlying `lastReset` timestamp actually changes (i.e. once a day).
function CountdownDisplay({ lastReset, className, style }) {
  const [text, setText] = useState('--:--:--');
  useEffect(() => {
    const tick = () => {
      const remaining = ONE_DAY_MS - (Date.now() - lastReset);
      if (remaining <= 0) { setText('00:00:00'); return; }
      const h = Math.floor((remaining / 3_600_000) % 24);
      const m = Math.floor((remaining /    60_000) % 60);
      const s = Math.floor((remaining /     1_000) % 60);
      setText(`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [lastReset]);
  return <span className={className} style={style}>{text}</span>;
}

// first-run welcome screen — shown once, before any account exists on this
// device, then never again (a localStorage flag remembers it's been seen).
// Explains the core loop and specifically calls out where to add a profile
// photo later, since that lives inside Settings rather than being part of
// account creation itself.
function WelcomeScreen({ onContinue, c }) {
  const steps = [
    { Icon: ListChecks, title: 'Get a fresh set of quests every day', body: 'A new mix of strength, cardio, mindfulness, and recovery quests unlocks every 24 hours.' },
    { Icon: ShieldCheck, title: 'Verify with your camera', body: 'On-device AI and pose tracking confirm you actually did it — nothing you record ever leaves your phone.' },
    { Icon: Sparkles, title: 'Level up and climb the leaderboard', body: 'Earn XP for every quest, build a daily streak, and see how you rank against everyone else.' },
    { Icon: Camera, title: 'Add a profile photo anytime', body: "Once you're signed in, open Settings → tap your avatar → choose a photo. It syncs to your account automatically." },
  ];
  return (
      <div className="fixed inset-0 z-[100] flex flex-col sq-anim-in" style={{ background: c.bg }}>
        <div className="flex-1 sq-scroll overflow-y-auto flex flex-col justify-center px-6" style={{ paddingTop: 'max(env(safe-area-inset-top), 32px)', paddingBottom: 24 }}>
          <div className="text-center mb-8">
            <div style={{ width: 76, height: 76, borderRadius: 30, background: c.blue, display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 18px' }}>
              <BrandLogo size={40} />
            </div>
            <h1 className="sq-large-title" style={{ fontSize: 30, fontWeight: 800, color: c.label, marginBottom: 8 }}>Welcome to QuestDaily</h1>
            <p style={{ fontSize: 15, color: c.labelSecondary, lineHeight: 1.4, maxWidth: 320, margin: '0 auto' }}>
              A few things to know before you dive in.
            </p>
          </div>

          <div className="flex flex-col gap-3">
            {steps.map(({ Icon, title, body }, i) => (
                <div key={title} className="sq-anim-in" style={{ ...glassStyle(c), borderRadius: 20, padding: 16, display: 'flex', gap: 14, animationDelay: `${i * 0.06}s` }}>
                  <div className="sq-icon-fff" style={{ width: 40, height: 40, borderRadius: 14, background: c.blue, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <Icon size={19} strokeWidth={1.9} />
                  </div>
                  <div style={{ minWidth: 0 }}>
                    <p style={{ fontSize: 14, fontWeight: 600, color: c.label, lineHeight: 1.3 }}>{title}</p>
                    <p style={{ fontSize: 12.5, color: c.labelSecondary, marginTop: 3, lineHeight: 1.45 }}>{body}</p>
                  </div>
                </div>
            ))}
          </div>
        </div>

        <div className="px-6" style={{ paddingBottom: 'max(env(safe-area-inset-bottom), 24px)' }}>
          <button onClick={onContinue} style={{ width: '100%', padding: '16px', borderRadius: 999, fontSize: 16, fontWeight: 600, color: '#fff', background: c.blue }}>
            Get started
          </button>
        </div>
      </div>
  );
}


// rest of the app until resolved, same as the old name-picker did, but now
// backs onto real Firebase Auth accounts instead of a purely local name.
function AuthModal({ onSignedUp, onLoggedIn, c }) {
  const [mode, setMode] = useState('signup'); // 'signup' | 'login'
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [googleSubmitting, setGoogleSubmitting] = useState(false);

  const switchMode = (next) => {
    setMode(next); setError(null); setPassword(''); setConfirmPassword(''); setShowPassword(false);
  };

  const handleGoogleClick = async () => {
    if (submitting || googleSubmitting) return;
    setError(null);
    setGoogleSubmitting(true);
    haptic(10);
    try {
      await signInWithGoogle();
      // nothing else to do — the resulting auth-state change is picked up
      // by the app's normal hydration effect, same as any other sign-in
    } catch (err) {
      // a closed/cancelled popup isn't an error worth showing
      if (err?.code !== 'auth/popup-closed-by-user' && err?.code !== 'auth/cancelled-popup-request') {
        console.error('Google sign-in failed:', err);
        setError(friendlyAuthError(err));
      }
      setGoogleSubmitting(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (submitting) return;
    setError(null);

    const usernameError = validateUsername(username);
    if (usernameError) { setError(usernameError); return; }

    if (mode === 'signup') {
      const passwordError = validatePassword(password);
      if (passwordError) { setError(passwordError); return; }
      if (password !== confirmPassword) { setError('Passwords don\'t match.'); return; }
    } else if (!password) {
      setError('Enter your password.'); return;
    }

    setSubmitting(true);
    haptic(10);
    try {
      if (mode === 'signup') {
        const { username: finalUsername } = await signUpAccount(username, password);
        onSignedUp(finalUsername);
      } else {
        const { username: finalUsername } = await logInAccount(username, password);
        onLoggedIn(finalUsername);
      }
    } catch (err) {
      console.error(`${mode} failed:`, err);
      setError(friendlyAuthError(err));
      setSubmitting(false);
    }
  };

  const inputStyle = { width: '100%', borderRadius: 14, border: `1px solid ${c.separator}`, background: c.bgSecondary, color: c.label, padding: '11px 40px 11px 40px', fontSize: 15, fontWeight: 500, outline: 'none' };

  return (
      <div className="fixed inset-0 z-[100] flex items-center justify-center p-5 sq-anim-in" style={{ background: 'rgba(0,0,0,0.4)', backdropFilter: 'blur(6px)', WebkitBackdropFilter: 'blur(6px)' }}>
        <div style={{ ...glassStyle(c, true), borderRadius: 26, width: '100%', maxWidth: 360, padding: '28px 24px 22px', textAlign: 'center' }}>
          <div style={{ width: 52, height: 52, borderRadius: 26, background: c.blue, display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
            <BrandLogo size={28} />
          </div>
          <h2 className="sq-title" style={{ fontSize: 17, fontWeight: 600, color: c.label }}>
            {mode === 'signup' ? 'Create your account' : 'Welcome back'}
          </h2>
          <p style={{ fontSize: 13, lineHeight: 1.4, color: c.labelSecondary, marginTop: 6 }}>
            {mode === 'signup'
                ? 'Your progress syncs to this account on any device.'
                : 'Log in to pick up where you left off.'}
          </p>

          <button type="button" onClick={handleGoogleClick} disabled={submitting || googleSubmitting}
                  style={{ width: '100%', marginTop: 18, padding: '12px', borderRadius: 14, fontSize: 14, fontWeight: 600, color: c.label, background: c.bgSecondary, border: `1px solid ${c.separator}`, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, opacity: googleSubmitting ? 0.7 : 1 }}>
            {googleSubmitting
                ? <span style={{ width: 16, height: 16, border: `2px solid ${c.fill}`, borderTopColor: c.label, borderRadius: '50%' }} className="animate-spin" />
                : <GoogleIcon size={17} />}
            {googleSubmitting ? 'Connecting…' : 'Continue with Google'}
          </button>

          <div className="flex items-center gap-3" style={{ margin: '16px 0' }}>
            <div style={{ flex: 1, height: 1, background: c.separator }} />
            <span style={{ fontSize: 11, fontWeight: 500, color: c.labelTertiary, textTransform: 'uppercase', letterSpacing: 0.4 }}>or</span>
            <div style={{ flex: 1, height: 1, background: c.separator }} />
          </div>

          <form onSubmit={handleSubmit} style={{ textAlign: 'left' }}>
            <div style={{ position: 'relative' }}>
              <AtSign size={16} strokeWidth={2} color={c.labelTertiary} style={{ position: 'absolute', left: 13, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }} />
              <input
                  type="text" value={username} autoFocus autoCapitalize="none" autoCorrect="off"
                  maxLength={20}
                  onChange={e => { setUsername(e.target.value); setError(null); }}
                  placeholder="Username"
                  style={{ ...inputStyle, paddingRight: 14 }}
              />
            </div>

            <div style={{ position: 'relative', marginTop: 10 }}>
              <Lock size={16} strokeWidth={2} color={c.labelTertiary} style={{ position: 'absolute', left: 13, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }} />
              <input
                  type={showPassword ? 'text' : 'password'} value={password}
                  autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
                  onChange={e => { setPassword(e.target.value); setError(null); }}
                  placeholder="Password"
                  style={inputStyle}
              />
              <button type="button" onClick={() => setShowPassword(s => !s)} aria-label={showPassword ? 'Hide password' : 'Show password'}
                      style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', color: c.labelTertiary, padding: 4 }}>
                {showPassword ? <EyeOff size={16} strokeWidth={2} /> : <Eye size={16} strokeWidth={2} />}
              </button>
            </div>

            {mode === 'signup' && (
                <div style={{ position: 'relative', marginTop: 10 }}>
                  <Lock size={16} strokeWidth={2} color={c.labelTertiary} style={{ position: 'absolute', left: 13, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }} />
                  <input
                      type={showPassword ? 'text' : 'password'} value={confirmPassword}
                      autoComplete="new-password"
                      onChange={e => { setConfirmPassword(e.target.value); setError(null); }}
                      placeholder="Confirm password"
                      style={{ ...inputStyle, paddingRight: 14 }}
                  />
                </div>
            )}

            {error && <p style={{ color: c.red, fontSize: 12, fontWeight: 500, marginTop: 10 }}>{error}</p>}

            <button type="submit" disabled={submitting}
                    style={{ width: '100%', marginTop: 14, padding: '13px', borderRadius: 999, fontSize: 15, fontWeight: 600, color: '#FFFFFF', background: c.blue, opacity: submitting ? 0.7 : 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
              {submitting && <span style={{ width: 15, height: 15, border: '2px solid rgba(255,255,255,0.35)', borderTopColor: '#fff', borderRadius: '50%' }} className="animate-spin" />}
              {submitting ? (mode === 'signup' ? 'Creating account…' : 'Logging in…') : (mode === 'signup' ? 'Create account' : 'Log in')}
            </button>
          </form>

          <button onClick={() => switchMode(mode === 'signup' ? 'login' : 'signup')} disabled={submitting}
                  style={{ marginTop: 16, fontSize: 13, fontWeight: 500, color: c.blue }}>
            {mode === 'signup' ? 'Already have an account? Log in' : 'New here? Create an account'}
          </button>
        </div>
      </div>
  );
}

// leaderboard
function LeaderboardScreen({ dark, onBack, myUid, myUsername, myPhotoURL, myLevel, myXp, syncError, onRetrySync, c }) {
  const { entries, status, errorDetail } = useLeaderboard();

  // don't surface a name until they've actually earned XP — someone who's
  // just picked a username but hasn't completed anything yet stays
  // anonymous. Checked against level too (not just totalXpEarned) so an
  // older or partially-synced doc missing that field doesn't wrongly hide
  // someone who's clearly made progress.
  const rankedEntries = useMemo(
      () => entries.filter(e => (e.totalXpEarned ?? 0) > 0 || (e.level ?? 1) > 1),
      [entries]
  );

  const myRank = useMemo(() => {
    const idx = rankedEntries.findIndex(e => e.id === myUid);
    return idx === -1 ? null : idx + 1;
  }, [rankedEntries, myUid]);

  const RANK_COLOR = { 1: c.yellow, 2: c.gray, 3: c.orange };

  return (
      <div className="sq-anim-in relative z-10 min-h-screen flex flex-col">
        <div className="flex items-center px-4 pb-3" style={{ paddingTop: 'max(env(safe-area-inset-top), 16px)' }}>
          <button onClick={onBack} className="flex items-center gap-1" style={{ color: c.blue }}>
            <BackChevron color={c.blue} />
            <span style={{ fontSize: 17 }}>Quests</span>
          </button>
        </div>
        <div className="px-4 pb-2">
          <h1 className="sq-large-title" style={{ fontSize: 30, fontWeight: 700, color: c.label }}>Leaderboard</h1>
        </div>

        {syncError && (
            <div className="px-4 mb-3">
              <div style={{ borderRadius: 16, padding: '10px 14px', display: 'flex', gap: 10, background: dark ? 'rgba(255,69,58,0.14)' : 'rgba(255,59,48,0.08)' }}>
                <WarningIcon size={15} />
                <div style={{ flex: 1, color: c.red }}>
                  <p style={{ fontSize: 12, fontWeight: 500, lineHeight: 1.4 }}>Your last score didn't save: {syncError}</p>
                  {onRetrySync && <button onClick={onRetrySync} style={{ fontSize: 11, fontWeight: 600, textDecoration: 'underline', marginTop: 4 }}>Try again</button>}
                </div>
              </div>
            </div>
        )}

        {myRank && (
            <div className="px-4 mb-3">
              <div style={{ borderRadius: 20, padding: '12px 14px', display: 'flex', alignItems: 'center', gap: 12, background: c.blue }}>
                <div style={{ width: 34, height: 34, borderRadius: 999, background: 'rgba(255,255,255,0.22)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontWeight: 700, fontSize: 13 }} className="sq-mono">
                  #{myRank}
                </div>
                <Avatar photoURL={myPhotoURL} username={myUsername} size={34} c={c} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{ color: '#fff', fontWeight: 600, fontSize: 14, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{myUsername} (You)</p>
                  <p className="sq-mono" style={{ color: 'rgba(255,255,255,0.75)', fontSize: 11, fontWeight: 500 }}>Level {myLevel} · {myXp} XP this level</p>
                </div>
              </div>
            </div>
        )}

        <div className="flex-1 sq-scroll overflow-y-auto px-4 pb-10">
          {status === 'loading' && (
              <div className="flex flex-col items-center justify-center py-16 gap-3">
                <div style={{ width: 26, height: 26, border: `2.5px solid ${c.fill}`, borderTopColor: c.blue, borderRadius: '50%' }} className="animate-spin" />
                <p style={{ fontSize: 13, color: c.labelSecondary }}>Loading rankings…</p>
              </div>
          )}

          {status === 'error' && (
              <div className="flex flex-col items-center justify-center py-16 gap-2 text-center px-6" style={{ color: c.labelTertiary }}>
                <SignalOffIcon size={24} />
                <p style={{ fontSize: 14, fontWeight: 600, color: c.label }}>Can't reach the leaderboard</p>
                <p style={{ fontSize: 12, color: c.labelSecondary }}>{errorDetail || 'Check your connection, or the backend may not be configured yet.'}</p>
              </div>
          )}

          {status === 'ready' && rankedEntries.length === 0 && (
              <div className="flex flex-col items-center justify-center py-16 gap-2 text-center px-6" style={{ color: c.labelTertiary }}>
                <FlagIcon size={24} />
                <p style={{ fontSize: 14, fontWeight: 600, color: c.label }}>No rankings yet</p>
                <p style={{ fontSize: 12, color: c.labelSecondary }}>
                  {myLevel > 1 || myXp > 0 ? "Your progress hasn't reached the server yet. Complete another quest and check back." : 'Complete a quest to be the first on the board.'}
                </p>
              </div>
          )}

          {status === 'ready' && rankedEntries.length > 0 && (
              <div style={{ ...glassStyle(c), borderRadius: 20, overflow: 'hidden' }}>
                {rankedEntries.map((entry, i) => {
                  const rank = i + 1;
                  const isMe = entry.id === myUid;
                  return (
                      <div key={entry.id}>
                        {i > 0 && <div style={{ marginLeft: 58, borderTop: `1px solid ${c.separator}` }} />}
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 14px', background: isMe ? c.fill : 'transparent' }}>
                          <div className="sq-mono" style={{ width: 30, height: 30, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, fontWeight: 700, background: RANK_COLOR[rank] || c.fill, color: RANK_COLOR[rank] ? '#fff' : c.labelSecondary }}>
                            {rank}
                          </div>
                          <Avatar photoURL={entry.photoURL} username={entry.username} size={30} c={c} />
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <p style={{ fontSize: 14, fontWeight: 600, color: c.label, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              {entry.username || 'Adventurer'}{isMe ? ' (You)' : ''}
                            </p>
                            <p style={{ fontSize: 11, color: c.labelSecondary }}>Level {entry.level ?? 1}</p>
                          </div>
                          <span className="sq-mono" style={{ fontSize: 12, fontWeight: 600, color: c.labelSecondary, flexShrink: 0 }}>
                      {(entry.totalXpEarned ?? 0).toLocaleString()} XP
                    </span>
                        </div>
                      </div>
                  );
                })}
              </div>
          )}
        </div>
      </div>
  );
}

// history tab
function groupLabelForDate(ts) {
  const d = new Date(ts);
  const startOf = (date) => { const x = new Date(date); x.setHours(0, 0, 0, 0); return x.getTime(); };
  const today = startOf(Date.now());
  const yesterday = today - ONE_DAY_MS;
  const day = startOf(ts);
  if (day === today) return 'Today';
  if (day === yesterday) return 'Yesterday';
  const sameYear = d.getFullYear() === new Date().getFullYear();
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: sameYear ? undefined : 'numeric' });
}

function HistoryScreen({ history, dark, onToggleTheme, c }) {
  const groups = useMemo(() => {
    const sorted = [...history].sort((a, b) => b.ts - a.ts);
    const out = [];
    let current = null;
    for (const entry of sorted) {
      const label = groupLabelForDate(entry.ts);
      if (!current || current.label !== label) { current = { label, items: [] }; out.push(current); }
      current.items.push(entry);
    }
    return out;
  }, [history]);

  const totalXp = useMemo(() => history.reduce((sum, e) => sum + (e.xp || 0), 0), [history]);

  return (
      <div className="relative z-10 flex flex-col flex-1 sq-anim-in">
        <div className="px-4 pb-2" style={{ paddingTop: 'max(env(safe-area-inset-top), 16px)' }}>
          <div className="flex items-center justify-between">
            <h1 className="sq-large-title" style={{ fontSize: 30, fontWeight: 700, color: c.label }}>History</h1>
            <button onClick={onToggleTheme} aria-label="Toggle theme"
                    style={{ ...glassStyle(c), width: 32, height: 32, borderRadius: 999, display: 'flex', alignItems: 'center', justifyContent: 'center', color: c.label }}>
              <span className="sq-icon-tap">{dark ? <Sun key="sun" size={16} strokeWidth={1.75} className="sq-icon-pop-in" /> : <Moon key="moon" size={16} strokeWidth={1.75} className="sq-icon-pop-in" />}</span>
            </button>
          </div>
          <p style={{ fontSize: 13, color: c.labelSecondary, marginTop: 2 }}>Every quest you've ever completed</p>
        </div>

        <div className="flex-1 sq-scroll overflow-y-auto px-4 pt-2 space-y-5" style={{ paddingBottom: 'calc(100px + env(safe-area-inset-bottom))' }}>
          <div className="grid grid-cols-3 gap-2.5 sq-anim-in">
            <StatChip label="Completed" value={history.length} c={c} />
            <StatChip label="Total XP" value={totalXp.toLocaleString()} c={c} />
            <StatChip label="Days logged" value={groups.length} c={c} />
          </div>

          {groups.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 gap-2 text-center px-6" style={{ color: c.labelTertiary }}>
                <FlagIcon size={24} />
                <p style={{ fontSize: 14, fontWeight: 600, color: c.label }}>No history yet</p>
                <p style={{ fontSize: 12, color: c.labelSecondary }}>Quests you complete will show up here, forever.</p>
              </div>
          ) : (
              groups.map(group => (
                  <div key={group.label}>
                    <p style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, color: c.labelSecondary, marginBottom: 6, paddingLeft: 2 }}>
                      {group.label}
                    </p>
                    <div style={{ ...glassStyle(c), borderRadius: 20, overflow: 'hidden' }}>
                      {group.items.map((entry, i) => (
                          <div key={entry.id}>
                            {i > 0 && <div style={{ marginLeft: 58, borderTop: `1px solid ${c.separator}` }} />}
                            <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '14px 14px' }}>
                              <QuestRowIcon questId={entry.questId} c={c} />
                              <div style={{ flex: 1, minWidth: 0 }}>
                                <p style={{ fontSize: 15, fontWeight: 500, color: c.label, lineHeight: 1.3 }}>{entry.text}</p>
                                <p className="sq-mono" style={{ fontSize: 11, color: c.labelTertiary, marginTop: 2 }}>
                                  {new Date(entry.ts).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
                                  {entry.leveledUp ? ' · Level up' : ''}
                                </p>
                              </div>
                              <span className="sq-mono" style={{ fontSize: 12, fontWeight: 700, color: c.green, flexShrink: 0 }}>+{entry.xp}</span>
                            </div>
                          </div>
                      ))}
                    </div>
                  </div>
              ))
          )}
        </div>
      </div>
  );
}

// settings tab — profile photo, appearance, account actions, notifications,
// and data export. A real settings screen instead of scattering account
// controls across other screens.
function SettingsScreen({
                          username, photoURL, uid, authProviderLabel, dark, onToggleTheme,
                          onPhotoFile, photoUploading, photoError, onDismissPhotoError,
                          onUsernameChanged, onLogout, onDeleteAccount,
                          hapticsOn, onToggleHaptics,
                          dailyReminderOn, onToggleDailyReminder, notificationsSupported,
                          onExportData, c,
                        }) {
  // null | 'username' | 'password' | 'delete' — only one inline editor
  // open at a time, inside the Account card.
  const [editing, setEditing] = useState(null);

  const [usernameInput, setUsernameInput] = useState(username || '');
  const [usernameReauthPassword, setUsernameReauthPassword] = useState('');
  const [needsReauthForUsername, setNeedsReauthForUsername] = useState(false);
  const [usernameSaving, setUsernameSaving] = useState(false);
  const [usernameError, setUsernameError] = useState(null);

  const [currentPasswordInput, setCurrentPasswordInput] = useState('');
  const [newPasswordInput, setNewPasswordInput] = useState('');
  const [confirmNewPasswordInput, setConfirmNewPasswordInput] = useState('');
  const [passwordSaving, setPasswordSaving] = useState(false);
  const [passwordError, setPasswordError] = useState(null);
  const [passwordSaved, setPasswordSaved] = useState(false);

  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState(null);

  const rowStyle = { width: '100%', display: 'flex', alignItems: 'center', gap: 12, padding: '13px 14px', textAlign: 'left' };
  const smallInputStyle = { width: '100%', borderRadius: 12, border: `1px solid ${c.separator}`, background: c.bgSecondary, color: c.label, padding: '10px 12px', fontSize: 14, fontWeight: 500, outline: 'none' };

  const openEditor = (which) => {
    setEditing(which);
    if (which === 'username') { setUsernameInput(username || ''); setUsernameError(null); setNeedsReauthForUsername(false); setUsernameReauthPassword(''); }
    if (which === 'password') { setCurrentPasswordInput(''); setNewPasswordInput(''); setConfirmNewPasswordInput(''); setPasswordError(null); setPasswordSaved(false); }
    if (which === 'delete') setDeleteError(null);
  };
  const closeEditor = () => setEditing(null);

  const handleSaveUsername = async () => {
    const validationError = validateUsername(usernameInput);
    if (validationError) { setUsernameError(validationError); return; }
    setUsernameSaving(true); setUsernameError(null);
    try {
      const finalUsername = await updateUsername({
        newUsernameRaw: usernameInput,
        oldUsername: username,
        currentPassword: needsReauthForUsername ? usernameReauthPassword : undefined,
      });
      onUsernameChanged(finalUsername);
      setEditing(null);
    } catch (err) {
      if (err?.code === 'auth/requires-recent-login') {
        setNeedsReauthForUsername(true);
        setUsernameError('For security, enter your password to confirm this change.');
      } else {
        console.error('Username update failed:', err);
        setUsernameError(friendlyAuthError(err));
      }
    } finally {
      setUsernameSaving(false);
    }
  };

  const handleSavePassword = async () => {
    if (!currentPasswordInput) { setPasswordError('Enter your current password.'); return; }
    const validationError = validatePassword(newPasswordInput);
    if (validationError) { setPasswordError(validationError); return; }
    if (newPasswordInput !== confirmNewPasswordInput) { setPasswordError("New passwords don't match."); return; }
    setPasswordSaving(true); setPasswordError(null);
    try {
      await changeAccountPassword(currentPasswordInput, newPasswordInput);
      setPasswordSaved(true);
      setCurrentPasswordInput(''); setNewPasswordInput(''); setConfirmNewPasswordInput('');
    } catch (err) {
      const code = err?.code;
      if (code === 'auth/wrong-password' || code === 'auth/invalid-credential' || code === 'auth/invalid-login-credentials') {
        setPasswordError('Current password is incorrect.');
      } else {
        console.error('Password update failed:', err);
        setPasswordError(friendlyAuthError(err));
      }
    } finally {
      setPasswordSaving(false);
    }
  };

  const handleDeleteConfirm = async () => {
    setDeleting(true); setDeleteError(null);
    const res = await onDeleteAccount();
    if (res && !res.ok) { setDeleting(false); setDeleteError(res.error); }
    // on success this component unmounts (auth state flips to signed-out), nothing more to do
  };

  const usesPassword = authProviderLabel === 'password';

  return (
      <div className="relative z-10 flex flex-col flex-1 sq-anim-in">
        <div className="px-4 pb-2" style={{ paddingTop: 'max(env(safe-area-inset-top), 16px)' }}>
          <div className="flex items-center justify-between">
            <h1 className="sq-large-title" style={{ fontSize: 30, fontWeight: 700, color: c.label }}>Settings</h1>
            <button onClick={onToggleTheme} aria-label="Toggle theme"
                    style={{ ...glassStyle(c), width: 32, height: 32, borderRadius: 999, display: 'flex', alignItems: 'center', justifyContent: 'center', color: c.label }}>
              <span className="sq-icon-tap">{dark ? <Sun key="sun" size={16} strokeWidth={1.75} className="sq-icon-pop-in" /> : <Moon key="moon" size={16} strokeWidth={1.75} className="sq-icon-pop-in" />}</span>
            </button>
          </div>
        </div>

        <div className="flex-1 sq-scroll overflow-y-auto px-4 pt-3 space-y-6" style={{ paddingBottom: 'calc(100px + env(safe-area-inset-bottom))' }}>
          {/* Profile */}
          <div style={{ ...glassStyle(c), borderRadius: 20, padding: 20, display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
            <label style={{ position: 'relative', cursor: 'pointer', display: 'inline-block' }}>
              <Avatar photoURL={photoURL} username={username} size={76} c={c} ring />
              <div className="sq-icon-fff" style={{ position: 'absolute', bottom: -2, right: -2, width: 26, height: 26, borderRadius: '50%', background: c.blue, border: `2px solid ${c.bgElevated}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {photoUploading
                    ? <span style={{ width: 11, height: 11, border: '2px solid rgba(255,255,255,0.4)', borderTopColor: '#fff', borderRadius: '50%' }} className="animate-spin" />
                    : <Camera size={12} strokeWidth={2} />}
              </div>
              <input type="file" accept="image/*" className="hidden" disabled={photoUploading}
                     onChange={e => { const f = e.target.files?.[0]; if (f) onPhotoFile(f); e.target.value = ''; }} />
            </label>
            <p style={{ fontSize: 16, fontWeight: 600, color: c.label, marginTop: 12 }}>{username}</p>
            <p style={{ fontSize: 12, color: c.labelTertiary, marginTop: 2 }}>Tap your photo to change it</p>
            {photoError && (
                <div style={{ marginTop: 10, borderRadius: 12, padding: '8px 12px', background: dark ? 'rgba(255,69,58,0.14)' : 'rgba(255,59,48,0.08)' }}>
                  <p style={{ fontSize: 12, fontWeight: 500, color: c.red }}>{photoError}</p>
                  <button onClick={onDismissPhotoError} style={{ fontSize: 11, fontWeight: 700, color: c.red, opacity: 0.75, marginTop: 4, textTransform: 'uppercase', letterSpacing: 0.3 }}>Dismiss</button>
                </div>
            )}
          </div>

          {/* Account */}
          <div>
            <p style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, color: c.labelSecondary, marginBottom: 8, paddingLeft: 2 }}>Account</p>
            <div style={{ ...glassStyle(c), borderRadius: 20, overflow: 'hidden' }}>

              {/* signed-in-with indicator */}
              <div style={rowStyle}>
                <div className="sq-icon-fff" style={{ width: 30, height: 30, borderRadius: 10, background: c.teal, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <CircleUserRound size={15} strokeWidth={1.9} />
                </div>
                <span style={{ flex: 1, fontSize: 15, fontWeight: 500, color: c.label }}>Signed in with</span>
                <span style={{ fontSize: 13, fontWeight: 500, color: c.labelSecondary }}>{authProviderLabel === 'google' ? 'Google' : 'Username & password'}</span>
              </div>
              <div style={{ marginLeft: 58, borderTop: `1px solid ${c.separator}` }} />

              {/* username */}
              {editing !== 'username' ? (
                  <button onClick={() => openEditor('username')} style={rowStyle}>
                    <div className="sq-icon-fff" style={{ width: 30, height: 30, borderRadius: 10, background: c.blue, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                      <UserCog size={14} strokeWidth={1.9} />
                    </div>
                    <span style={{ flex: 1, fontSize: 15, fontWeight: 500, color: c.label }}>Edit username</span>
                    <ChevronIcon color={c.labelTertiary} />
                  </button>
              ) : (
                  <div style={{ padding: '14px' }}>
                    <p style={{ fontSize: 13, fontWeight: 600, color: c.label, marginBottom: 8 }}>Edit username</p>
                    <input value={usernameInput} maxLength={20} autoCapitalize="none" autoCorrect="off"
                           onChange={e => { setUsernameInput(e.target.value); setUsernameError(null); }}
                           style={smallInputStyle} />
                    {needsReauthForUsername && (
                        <input type="password" value={usernameReauthPassword} placeholder="Current password"
                               onChange={e => { setUsernameReauthPassword(e.target.value); setUsernameError(null); }}
                               style={{ ...smallInputStyle, marginTop: 8 }} />
                    )}
                    {usernameError && <p style={{ fontSize: 12, fontWeight: 500, color: c.red, marginTop: 8 }}>{usernameError}</p>}
                    <div className="flex gap-2.5" style={{ marginTop: 12 }}>
                      <button onClick={closeEditor} disabled={usernameSaving}
                              style={{ flex: 1, padding: '11px', borderRadius: 999, fontSize: 13, fontWeight: 600, background: c.fill, color: c.label }}>
                        Cancel
                      </button>
                      <button onClick={handleSaveUsername} disabled={usernameSaving || (needsReauthForUsername && !usernameReauthPassword)}
                              style={{ flex: 1, padding: '11px', borderRadius: 999, fontSize: 13, fontWeight: 600, background: c.blue, color: '#fff', opacity: usernameSaving ? 0.7 : 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
                        {usernameSaving && <span style={{ width: 12, height: 12, border: '2px solid rgba(255,255,255,0.4)', borderTopColor: '#fff', borderRadius: '50%' }} className="animate-spin" />}
                        {usernameSaving ? 'Saving…' : 'Save'}
                      </button>
                    </div>
                  </div>
              )}
              <div style={{ marginLeft: 58, borderTop: `1px solid ${c.separator}` }} />

              {/* password — password accounts only */}
              {usesPassword && (
                  <>
                    {editing !== 'password' ? (
                        <button onClick={() => openEditor('password')} style={rowStyle}>
                          <div className="sq-icon-fff" style={{ width: 30, height: 30, borderRadius: 10, background: c.purple, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                            <KeyRound size={14} strokeWidth={1.9} />
                          </div>
                          <span style={{ flex: 1, fontSize: 15, fontWeight: 500, color: c.label }}>Change password</span>
                          <ChevronIcon color={c.labelTertiary} />
                        </button>
                    ) : (
                        <div style={{ padding: '14px' }}>
                          <p style={{ fontSize: 13, fontWeight: 600, color: c.label, marginBottom: 8 }}>Change password</p>
                          {passwordSaved ? (
                              <div style={{ borderRadius: 12, padding: '10px 12px', background: dark ? 'rgba(48,209,88,0.14)' : 'rgba(52,199,89,0.1)' }}>
                                <p style={{ fontSize: 12.5, fontWeight: 600, color: c.green }}>Password updated.</p>
                              </div>
                          ) : (
                              <>
                                <input type="password" value={currentPasswordInput} placeholder="Current password" autoComplete="current-password"
                                       onChange={e => { setCurrentPasswordInput(e.target.value); setPasswordError(null); }}
                                       style={smallInputStyle} />
                                <input type="password" value={newPasswordInput} placeholder="New password" autoComplete="new-password"
                                       onChange={e => { setNewPasswordInput(e.target.value); setPasswordError(null); }}
                                       style={{ ...smallInputStyle, marginTop: 8 }} />
                                <input type="password" value={confirmNewPasswordInput} placeholder="Confirm new password" autoComplete="new-password"
                                       onChange={e => { setConfirmNewPasswordInput(e.target.value); setPasswordError(null); }}
                                       style={{ ...smallInputStyle, marginTop: 8 }} />
                              </>
                          )}
                          {passwordError && <p style={{ fontSize: 12, fontWeight: 500, color: c.red, marginTop: 8 }}>{passwordError}</p>}
                          <div className="flex gap-2.5" style={{ marginTop: 12 }}>
                            <button onClick={closeEditor} disabled={passwordSaving}
                                    style={{ flex: 1, padding: '11px', borderRadius: 999, fontSize: 13, fontWeight: 600, background: c.fill, color: c.label }}>
                              {passwordSaved ? 'Done' : 'Cancel'}
                            </button>
                            {!passwordSaved && (
                                <button onClick={handleSavePassword} disabled={passwordSaving}
                                        style={{ flex: 1, padding: '11px', borderRadius: 999, fontSize: 13, fontWeight: 600, background: c.blue, color: '#fff', opacity: passwordSaving ? 0.7 : 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
                                  {passwordSaving && <span style={{ width: 12, height: 12, border: '2px solid rgba(255,255,255,0.4)', borderTopColor: '#fff', borderRadius: '50%' }} className="animate-spin" />}
                                  {passwordSaving ? 'Saving…' : 'Save'}
                                </button>
                            )}
                          </div>
                        </div>
                    )}
                    <div style={{ marginLeft: 58, borderTop: `1px solid ${c.separator}` }} />
                  </>
              )}

              {/* log out */}
              <button onClick={onLogout} style={rowStyle}>
                <div className="sq-icon-fff" style={{ width: 30, height: 30, borderRadius: 10, background: c.gray, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <LogOut size={14} strokeWidth={1.9} />
                </div>
                <span style={{ flex: 1, fontSize: 15, fontWeight: 500, color: c.label }}>Log out</span>
                <ChevronIcon color={c.labelTertiary} />
              </button>
              <div style={{ marginLeft: 58, borderTop: `1px solid ${c.separator}` }} />

              {/* delete account */}
              {editing !== 'delete' ? (
                  <button onClick={() => openEditor('delete')} style={rowStyle}>
                    <div style={{ width: 30, height: 30, borderRadius: 10, background: dark ? 'rgba(255,69,58,0.16)' : 'rgba(255,59,48,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, color: c.red }}>
                      <Trash2 size={14} strokeWidth={1.9} />
                    </div>
                    <span style={{ flex: 1, fontSize: 15, fontWeight: 500, color: c.red }}>Delete account</span>
                    <ChevronIcon color={c.labelTertiary} />
                  </button>
              ) : (
                  <div style={{ padding: '14px' }}>
                    <p style={{ fontSize: 13, fontWeight: 600, color: c.label }}>Delete your account?</p>
                    <p style={{ fontSize: 12, color: c.labelSecondary, marginTop: 4, lineHeight: 1.4 }}>
                      This permanently removes your profile, history, and leaderboard entry. This can't be undone.
                    </p>
                    {deleteError && <p style={{ fontSize: 12, fontWeight: 500, color: c.red, marginTop: 8 }}>{deleteError}</p>}
                    <div className="flex gap-2.5" style={{ marginTop: 12 }}>
                      <button onClick={closeEditor} disabled={deleting}
                              style={{ flex: 1, padding: '11px', borderRadius: 999, fontSize: 13, fontWeight: 600, background: c.fill, color: c.label }}>
                        Cancel
                      </button>
                      <button onClick={handleDeleteConfirm} disabled={deleting}
                              style={{ flex: 1, padding: '11px', borderRadius: 999, fontSize: 13, fontWeight: 600, background: c.red, color: '#fff', opacity: deleting ? 0.7 : 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
                        {deleting && <span style={{ width: 12, height: 12, border: '2px solid rgba(255,255,255,0.4)', borderTopColor: '#fff', borderRadius: '50%' }} className="animate-spin" />}
                        {deleting ? 'Deleting…' : 'Delete'}
                      </button>
                    </div>
                  </div>
              )}
            </div>
          </div>

          {/* Notifications */}
          <div>
            <p style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, color: c.labelSecondary, marginBottom: 8, paddingLeft: 2 }}>Notifications</p>
            <div style={{ ...glassStyle(c), borderRadius: 20, overflow: 'hidden' }}>
              <div style={rowStyle}>
                <div className="sq-icon-fff" style={{ width: 30, height: 30, borderRadius: 10, background: c.orange, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <Bell size={14} strokeWidth={1.9} />
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{ fontSize: 15, fontWeight: 500, color: c.label }}>Daily reminder</p>
                  {!notificationsSupported && <p style={{ fontSize: 11, color: c.labelTertiary, marginTop: 2 }}>Not supported in this browser</p>}
                </div>
                <button onClick={onToggleDailyReminder} disabled={!notificationsSupported} aria-label="Toggle daily reminder"
                        style={{ width: 46, height: 27, borderRadius: 999, background: dailyReminderOn ? c.blue : c.fill, position: 'relative', flexShrink: 0, opacity: notificationsSupported ? 1 : 0.5, transition: 'background-color 0.25s ease' }}>
                  <span style={{ position: 'absolute', top: 2, left: dailyReminderOn ? 21 : 2, width: 23, height: 23, borderRadius: '50%', background: '#fff', boxShadow: '0 1px 3px rgba(0,0,0,0.25)', transition: 'left 0.22s cubic-bezier(0.34,1.56,0.64,1)' }} />
                </button>
              </div>
              {dailyReminderOn && (
                  <p style={{ fontSize: 11, color: c.labelTertiary, padding: '0 14px 12px 58px', lineHeight: 1.4 }}>
                    Reminds you around 6 PM if you haven't checked in — only while QuestDaily is open in a tab.
                  </p>
              )}
              <div style={{ marginLeft: 58, borderTop: `1px solid ${c.separator}` }} />
              <div style={rowStyle}>
                <div className="sq-icon-fff" style={{ width: 30, height: 30, borderRadius: 10, background: c.pink, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <Sparkles size={14} strokeWidth={1.9} />
                </div>
                <span style={{ flex: 1, fontSize: 15, fontWeight: 500, color: c.label }}>Haptic feedback</span>
                <button onClick={onToggleHaptics} aria-label="Toggle haptic feedback"
                        style={{ width: 46, height: 27, borderRadius: 999, background: hapticsOn ? c.blue : c.fill, position: 'relative', flexShrink: 0, transition: 'background-color 0.25s ease' }}>
                  <span style={{ position: 'absolute', top: 2, left: hapticsOn ? 21 : 2, width: 23, height: 23, borderRadius: '50%', background: '#fff', boxShadow: '0 1px 3px rgba(0,0,0,0.25)', transition: 'left 0.22s cubic-bezier(0.34,1.56,0.64,1)' }} />
                </button>
              </div>
            </div>
          </div>

          {/* Appearance */}
          <div>
            <p style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, color: c.labelSecondary, marginBottom: 8, paddingLeft: 2 }}>Appearance</p>
            <div style={{ ...glassStyle(c), borderRadius: 20, overflow: 'hidden' }}>
              <div style={rowStyle}>
                <div className="sq-icon-fff" style={{ width: 30, height: 30, borderRadius: 10, background: c.indigo, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  {dark ? <Moon size={15} strokeWidth={1.9} /> : <Sun size={15} strokeWidth={1.9} />}
                </div>
                <span style={{ flex: 1, fontSize: 15, fontWeight: 500, color: c.label }}>Dark mode</span>
                <button onClick={onToggleTheme} aria-label="Toggle dark mode"
                        style={{ width: 46, height: 27, borderRadius: 999, background: dark ? c.blue : c.fill, position: 'relative', flexShrink: 0, transition: 'background-color 0.25s ease' }}>
                  <span style={{ position: 'absolute', top: 2, left: dark ? 21 : 2, width: 23, height: 23, borderRadius: '50%', background: '#fff', boxShadow: '0 1px 3px rgba(0,0,0,0.25)', transition: 'left 0.22s cubic-bezier(0.34,1.56,0.64,1)' }} />
                </button>
              </div>
            </div>
          </div>

          {/* Privacy & data */}
          <div>
            <p style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, color: c.labelSecondary, marginBottom: 8, paddingLeft: 2 }}>Privacy & data</p>
            <div style={{ ...glassStyle(c), borderRadius: 20, overflow: 'hidden' }}>
              <button onClick={onExportData} style={rowStyle}>
                <div className="sq-icon-fff" style={{ width: 30, height: 30, borderRadius: 10, background: c.green, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <Download size={14} strokeWidth={1.9} />
                </div>
                <span style={{ flex: 1, fontSize: 15, fontWeight: 500, color: c.label }}>Export my data</span>
                <ChevronIcon color={c.labelTertiary} />
              </button>
            </div>
            <p style={{ fontSize: 11, color: c.labelTertiary, marginTop: 8, paddingLeft: 2, lineHeight: 1.4 }}>
              Downloads your level, XP, streak, and full quest history as a JSON file.
            </p>
          </div>

          {/* About */}
          <div>
            <p style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, color: c.labelSecondary, marginBottom: 8, paddingLeft: 2 }}>About</p>
            <div style={{ ...glassStyle(c), borderRadius: 20, padding: '14px 16px' }}>
              <p style={{ fontSize: 12.5, color: c.labelSecondary, lineHeight: 1.5 }}>
                Level, XP, streak, quest history, and your profile photo sync to your account.
                Quest check-in photos and today's quest list stay on this device only.
              </p>
              <p className="sq-mono" style={{ fontSize: 10, color: c.labelTertiary, marginTop: 10 }}>QuestDaily · v1.0</p>
            </div>
          </div>
        </div>
      </div>
  );
}


// pill highlight, iOS system blue for the active tab and iOS system gray
// for inactive tabs.
const TabBar = React.memo(function TabBar({ activeTab, onChange, c }) {
  const items = [
    { key: 'quests', label: 'Quests', Icon: CheckSquare },
    { key: 'history', label: 'History', Icon: HistoryIcon },
    { key: 'settings', label: 'Settings', Icon: SettingsIcon },
  ];
  return (
      <div style={{ position: 'fixed', left: '50%', bottom: 'max(env(safe-area-inset-bottom), 16px)', transform: 'translateX(-50%)', zIndex: 50 }}>
        <div
            className="backdrop-blur-2xl"
            style={{
              ...glassStyle(c, true),
              borderRadius: 999,
              padding: '6px 6px',
              display: 'flex',
              width: 300,
            }}
        >
          {items.map(({ key, label, Icon }) => {
            const active = activeTab === key;
            const color = active ? c.blue : c.gray;
            return (
                <button
                    key={key}
                    onClick={() => onChange(key)}
                    style={{
                      flex: 1,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: 3,
                      padding: '8px 0 7px',
                      borderRadius: 999,
                      color,
                      background: active ? c.fill : 'transparent',
                      transition: 'background-color 0.25s ease',
                    }}
                >
                  <span className={`sq-tab-icon ${active ? 'sq-tab-icon-active' : ''}`}>
                    <Icon size={21} strokeWidth={active ? 2.1 : 1.75} color={color} />
                  </span>
                  <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: -0.1, color }}>{label}</span>
                </button>
            );
          })}
        </div>
      </div>
  );
});

// add to home screen prompt
function DeviceInstallPrompt({ onDismiss, c }) {
  const [step, setStep] = useState(0);
  const [os, setOs] = useState(null);

  const nextStep = () => { if (step === 1) setStep(2); else onDismiss(); };

  const rowStyle = { width: '100%', padding: '16px 18px', borderRadius: 18, fontSize: 16, fontWeight: 500, textAlign: 'left', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: c.bgSecondary, color: c.label };

  return (
      <div className="fixed inset-0 z-[100] flex flex-col sq-anim-in" style={{ background: c.bg }}>
        <div className="flex-1 flex flex-col justify-center px-6" style={{ paddingTop: 'max(env(safe-area-inset-top), 24px)', paddingBottom: 24 }}>
          {step === 0 && (
              <>
                <div className="text-center mb-8">
                  <div className="sq-icon-fff" style={{ width: 84, height: 84, borderRadius: 32, background: c.blue, display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 20px' }}>
                    <IOSAddIcon style={{ width: 40, height: 40 }} />
                  </div>
                  <h3 className="sq-large-title" style={{ fontSize: 26, fontWeight: 800, color: c.label, marginBottom: 8 }}>Add to Home Screen</h3>
                  <p style={{ fontSize: 15, color: c.labelSecondary, lineHeight: 1.4 }}>Which device are you using?</p>
                </div>
                <div className="flex flex-col gap-3">
                  <button onClick={() => { setOs('ios'); setStep(1); }} style={rowStyle}>
                    <span>iPhone or iPad</span><ChevronIcon color={c.labelTertiary} />
                  </button>
                  <button onClick={() => { setOs('android'); setStep(1); }} style={rowStyle}>
                    <span>Android</span><ChevronIcon color={c.labelTertiary} />
                  </button>
                </div>
              </>
          )}

          {step > 0 && os === 'ios' && (
              <>
                <div className="text-center mb-2">
                  <h3 className="sq-large-title" style={{ fontSize: 26, fontWeight: 800, color: c.label, marginBottom: 6 }}>Install on iOS</h3>
                  <p style={{ fontSize: 14, color: c.labelSecondary, marginBottom: 28 }}>Step {step} of 2</p>
                </div>
                <div style={{ ...glassStyle(c), borderRadius: 28, padding: '40px 24px', marginBottom: 20, textAlign: 'center' }}>
                  <div className="sq-icon-fff" style={{ width: 72, height: 72, borderRadius: 999, background: c.blue, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 20px' }}>
                    {step === 1 ? <IOSShareIcon style={{ width: 30, height: 30 }} /> : <IOSAddIcon style={{ width: 30, height: 30 }} />}
                  </div>
                  <p style={{ fontSize: 18, fontWeight: 600, color: c.label, marginBottom: 6 }}>
                    {step === 1 ? 'Tap the Share button' : 'Tap Add to Home Screen'}
                  </p>
                  <p style={{ fontSize: 14, color: c.labelSecondary }}>
                    {step === 1 ? 'Find it in the Safari toolbar.' : 'Scroll down the share sheet to find it.'}
                  </p>
                </div>
              </>
          )}

          {step > 0 && os === 'android' && (
              <>
                <div className="text-center mb-2">
                  <h3 className="sq-large-title" style={{ fontSize: 26, fontWeight: 800, color: c.label, marginBottom: 6 }}>Install on Android</h3>
                  <p style={{ fontSize: 14, color: c.labelSecondary, marginBottom: 28 }}>Step {step} of 2</p>
                </div>
                <div style={{ ...glassStyle(c), borderRadius: 28, padding: '40px 24px', marginBottom: 20, textAlign: 'center' }}>
                  <div className="sq-icon-fff" style={{ width: 72, height: 72, borderRadius: 999, background: c.green, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 20px' }}>
                    {step === 1 ? <AndroidMenuIcon style={{ width: 30, height: 30 }} /> : <AndroidAddIcon style={{ width: 30, height: 30 }} />}
                  </div>
                  <p style={{ fontSize: 18, fontWeight: 600, color: c.label, marginBottom: 6 }}>
                    {step === 1 ? 'Tap the menu icon' : 'Tap Add to Home Screen'}
                  </p>
                  <p style={{ fontSize: 14, color: c.labelSecondary }}>
                    {step === 1 ? 'Three dots, usually top right in Chrome.' : 'Select it from the menu, or "Install app".'}
                  </p>
                </div>
              </>
          )}
        </div>

        <div className="px-6" style={{ paddingBottom: 'max(env(safe-area-inset-bottom), 24px)' }}>
          {step === 0 ? (
              <button onClick={onDismiss} style={{ width: '100%', padding: '16px', borderRadius: 999, fontSize: 16, fontWeight: 600, color: c.blue, background: c.fill }}>
                Not Now
              </button>
          ) : (
              <div className="flex gap-3">
                {step === 2 && <button onClick={() => setStep(1)} style={{ flex: 1, padding: '16px', borderRadius: 999, fontSize: 16, fontWeight: 600, background: c.fill, color: c.label }}>Back</button>}
                <button onClick={nextStep} style={{ flex: 2, padding: '16px', borderRadius: 999, fontSize: 16, fontWeight: 600, background: os === 'ios' ? c.blue : c.green, color: '#fff' }}>
                  {step === 1 ? 'Next' : 'Done'}
                </button>
              </div>
          )}
        </div>
      </div>
  );
}

// quest detail
function QuestDetailScreen({ quest, dark, onToggleTheme, lastReset, onBack, onMarkComplete, c }) {
  const about = QUEST_ABOUT[quest.id] || 'Stay consistent — every quest you complete adds up to real progress.';
  const theme = QUEST_THEME[quest.id] || { cat: 'blue' };
  const accent = c[theme.cat];

  return (
      <div className="sq-anim-in relative z-10 min-h-screen flex flex-col">
        <div className="flex items-center justify-between px-4 pb-2" style={{ paddingTop: 'max(env(safe-area-inset-top), 16px)' }}>
          <button onClick={onBack} className="flex items-center gap-1" style={{ color: c.blue }}>
            <BackChevron color={c.blue} /><span style={{ fontSize: 17 }}>Quests</span>
          </button>
          <div className="flex items-center gap-3">
            <CountdownDisplay lastReset={lastReset} className="sq-mono" style={{ fontSize: 13, fontWeight: 500, color: c.labelSecondary }} />
            <button onClick={onToggleTheme} style={{ ...glassStyle(c), width: 30, height: 30, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: c.label }}>
              <span className="sq-icon-tap">{dark ? <Sun key="sun" size={16} strokeWidth={1.75} className="sq-icon-pop-in" /> : <Moon key="moon" size={16} strokeWidth={1.75} className="sq-icon-pop-in" />}</span>
            </button>
          </div>
        </div>

        <div className="px-4 mt-3 flex flex-col items-center text-center">
          <QuestIconBadge questId={quest.id} size={84} c={c} />
          <h2 style={{ marginTop: 16, fontSize: 22, fontWeight: 700, color: c.label, maxWidth: 260 }}>{quest.text}</h2>
          <span className="sq-mono" style={{ marginTop: 8, fontSize: 13, fontWeight: 700, color: accent }}>+{quest.xp} XP</span>
        </div>

        <div className="px-4 mt-8 flex-1">
          <p style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, color: c.labelSecondary, marginBottom: 6, paddingLeft: 2 }}>About</p>
          <div style={{ ...glassStyle(c), borderRadius: 20, padding: 16 }}>
            <p style={{ fontSize: 14, lineHeight: 1.5, color: c.labelSecondary }}>{about}</p>
          </div>
        </div>

        <div className="px-4 pb-8 mt-6">
          <button onClick={onMarkComplete}
                  style={{ width: '100%', padding: '15px', borderRadius: 999, fontSize: 16, fontWeight: 600, color: '#fff', background: c.blue }}>
            Start
          </button>
        </div>
      </div>
  );
}

// completion screen
function CompletionScreen({ quest, dark, onToggleTheme, lastReset, onBack, c }) {
  const quote = useMemo(() => QUEST_QUOTE[quest?.id] || QUEST_QUOTES[Math.floor(Math.random() * QUEST_QUOTES.length)], [quest?.id]);
  const leveledUp = Boolean(quest?.leveledUp);
  const accent = leveledUp ? c.orange : c.green;

  useEffect(() => { haptic(leveledUp ? [25, 40, 25, 40, 70] : [15, 30, 15]); }, [leveledUp]);

  return (
      <div className="sq-anim-in relative z-10 min-h-screen flex flex-col">
        <div className="flex items-center justify-between px-4 pb-2" style={{ paddingTop: 'max(env(safe-area-inset-top), 16px)' }}>
          <button onClick={onBack} className="flex items-center gap-1" style={{ color: c.blue }}>
            <BackChevron color={c.blue} /><span style={{ fontSize: 17 }}>Quests</span>
          </button>
          <div className="flex items-center gap-3">
            <CountdownDisplay lastReset={lastReset} className="sq-mono" style={{ fontSize: 13, fontWeight: 500, color: c.labelSecondary }} />
            <button onClick={onToggleTheme} style={{ ...glassStyle(c), width: 30, height: 30, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: c.label }}>
              <span className="sq-icon-tap">{dark ? <Sun key="sun" size={16} strokeWidth={1.75} className="sq-icon-pop-in" /> : <Moon key="moon" size={16} strokeWidth={1.75} className="sq-icon-pop-in" />}</span>
            </button>
          </div>
        </div>

        <div className="relative px-4 mt-6 flex flex-col items-center text-center">
          {leveledUp && <Confetti big c={c} />}
          {leveledUp && (
              <span className="sq-mono" style={{ marginBottom: 12, fontSize: 11, fontWeight: 700, letterSpacing: 0.6, textTransform: 'uppercase', color: '#fff', background: c.orange, padding: '5px 14px', borderRadius: 999 }}>
            Level up
          </span>
          )}
          <div className="sq-anim-check sq-icon-fff" style={{ width: 88, height: 88, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: accent }}>
            <CheckIcon size={38} />
          </div>
          <h2 style={{ marginTop: 18, fontSize: 22, fontWeight: 700, color: c.label }}>Quest complete</h2>
          <span className="sq-mono" style={{ marginTop: 8, fontSize: 13, fontWeight: 700, color: '#fff', background: accent, padding: '4px 12px', borderRadius: 999 }}>
          +{quest?.xp ?? 0} XP
        </span>
        </div>

        <div className="px-4 mt-8 flex-1">
          <div style={{ ...glassStyle(c), borderRadius: 20, padding: 20, textAlign: 'center' }}>
            <p style={{ fontSize: 15, fontWeight: 500, lineHeight: 1.5, color: c.label }}>{quote}</p>
          </div>
        </div>

        <div className="px-4 pb-8 mt-6">
          <button onClick={onBack} style={{ width: '100%', padding: '15px', borderRadius: 999, fontSize: 16, fontWeight: 600, color: c.label, background: c.fill }}>
            Back to Quests
          </button>
        </div>
      </div>
  );
}

// camera modal (the big one)
function CameraModal({ quest, onConfirm, onCancel, c }) {
  const videoRef          = useRef(null);
  const aiCanvasRef       = useRef(null);
  const skeletonCanvasRef = useRef(null);
  const streamRef         = useRef(null);
  const scanTimerRef      = useRef(null);
  const isScanningRef     = useRef(false);
  const scanRunRef        = useRef(0);
  const cameraSessionRef  = useRef(0);
  const lastVideoTimeRef  = useRef(-1);
  const confirmedRef      = useRef(false);
  const passStreakRef     = useRef(0);
  const poseStateRef      = useRef(createPoseRepState(quest.reps ? (quest.progress || 0) : 0));
  const poseRafRef        = useRef(null);
  const poseLastVideoTimeRef = useRef(-1);
  const smoothedActRef    = useRef(null);
  const smoothedNegRef    = useRef(null);
  const smoothedSpoofRef  = useRef(null);
  const spoofNoticeActiveRef = useRef(false);
  const verifyingRef      = useRef(false);
  const repSpoofSuspectedRef = useRef(false);
  const smoothedRepSpoofRef  = useRef(null);
  const repSpoofNoticeActiveRef = useRef(false);

  const [phase,         setPhase]         = useState('starting');
  const [camError,      setCamError]      = useState(null);
  const [facingMode,    setFacingMode]    = useState('environment');
  const [cameraVersion, setCameraVersion] = useState(0);
  const [modelReady,    setModelReady]    = useState(false);
  const [modelProgress, setModelProgress] = useState(null);
  const [modelError,    setModelError]    = useState(null);
  const [modelVersion,  setModelVersion]  = useState(0);
  const [poseReady,     setPoseReady]     = useState(false);
  const [poseError,     setPoseError]     = useState(null);
  const [repCue,        setRepCue]        = useState('Get in frame');
  const [notice,        setNotice]        = useState(() => {
    if (quest.reps && quest.progress > 0) return `Resuming — you already logged ${quest.progress} of ${quest.reps} reps.`;
    if (quest.duration && quest.progress != null && quest.progress < quest.duration) {
      const m = Math.floor(quest.progress / 60), s = quest.progress % 60;
      return `Resuming — ${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')} left on the clock.`;
    }
    return null;
  });
  const [scanning,      setScanning]      = useState(false);
  const [uploading,     setUploading]     = useState(false);
  const [liveScore,     setLiveScore]     = useState(0);
  const [passStreak,    setPassStreak]    = useState(0);
  const [repsDone,      setRepsDone]      = useState(quest.reps ? (quest.progress || 0) : 0);
  const [repPhase,      setRepPhase]      = useState('up');
  const [confirmed,     setConfirmed]     = useState(false);
  const [verifying,     setVerifying]     = useState(false);
  const [lastLabel,     setLastLabel]     = useState('');
  const [uploadedProof, setUploadedProof] = useState(null);

  const [secondsLeft, setSecondsLeft] = useState(quest.duration ? (quest.progress ?? quest.duration) : 0);
  const [timerRunning, setTimerRunning] = useState(false);

  const labels = QUEST_LABELS[quest.id];
  const questType = labels?.type || 'action';
  const hasSkeletonTracking = Boolean(quest.reps) || MOVEMENT_ACTION_IDS.has(quest.id);

  const activeNegatives = useMemo(() => getNegativeLabels(questType), [questType]);
  const classifierLabels = useMemo(
      () => [...new Set([...(labels?.activity ?? []), ...activeNegatives])],
      [labels, activeNegatives]
  );
  const [sourceWarning, setSourceWarning] = useState(null);

  let instructionText = `Show the camera you're ${labels?.label ?? 'doing it'}…`;
  let uiSubtext = quest.text;

  if (questType === 'map') { instructionText = "Upload a metric summary screenshot (Strava, Nike, Garmin etc.)"; uiSubtext = "Must clearly state distance metrics & active time duration summary logs."; }
  else if (questType === 'food') { instructionText = "Take a clear picture of the food on your plate."; uiSubtext = "Must be a real photo of a prepared meal or plate."; }
  else if (questType === 'reps') { instructionText = `Position the camera for full-body tracking: ${quest.reps} reps.`; uiSubtext = "Keep your entire working frame visible to log movements."; }

  useEffect(() => {
    let intervalId = null;
    if (timerRunning && secondsLeft > 0) {
      intervalId = setInterval(() => {
        setSecondsLeft(prev => {
          if (prev <= 1) {
            setTimerRunning(false);
            if (!quest.reps && questType === 'action') {
              setConfirmed(true);
              confirmedRef.current = true;
            }
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(intervalId);
  }, [timerRunning, secondsLeft, questType, quest.reps]);

  const formatTimerString = (secs) => `${String(Math.floor(secs / 60)).padStart(2, '0')}:${String(secs % 60).padStart(2, '0')}`;

  const stopCamera = useCallback(() => {
    const stream = streamRef.current;
    streamRef.current = null;
    stream?.getTracks().forEach(track => track.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const startCamera = useCallback(async (mode) => {
    const session = ++cameraSessionRef.current;
    stopCamera();
    setCamError(null); setNotice(null); setUploadedProof(null); setPhase('starting');
    const attempts = [
      { video: { facingMode: { ideal: mode }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
      { video: { facingMode: { ideal: mode } }, audio: false },
      { video: true, audio: false },
    ];
    for (const constraints of attempts) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (session !== cameraSessionRef.current) { stream.getTracks().forEach(track => track.stop()); return; }
        streamRef.current = stream;
        const video = videoRef.current;
        if (!video) { stopCamera(); return; }
        video.srcObject = stream;
        await video.play();
        if (session !== cameraSessionRef.current) { stream.getTracks().forEach(track => track.stop()); return; }
        stream.getVideoTracks().forEach(track => {
          track.addEventListener('ended', () => { if (session === cameraSessionRef.current) { setCamError('unavailable'); setPhase('error'); } });
        });
        setPhase('live');
        return;
      } catch (err) {
        if (session !== cameraSessionRef.current) return;
        stopCamera();
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError' || err.name === 'SecurityError') {
          setCamError('permission'); setPhase('error'); return;
        }
      }
    }
    if (session === cameraSessionRef.current) { setCamError('unavailable'); setPhase('error'); }
  }, [stopCamera]);

  useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia) { setCamError('unsupported'); setPhase('error'); return undefined; }
    startCamera(facingMode);
    return () => { cameraSessionRef.current += 1; stopCamera(); };
  }, [facingMode, cameraVersion, startCamera, stopCamera]);

  useEffect(() => {
    if (quest.reps) { setModelReady(true); return undefined; }
    let cancelled = false;
    (async () => {
      try {
        setModelError(null);
        await getClassifier(evt => {
          if (cancelled) return;
          if (evt.status === 'progress' && evt.total) setModelProgress(Math.round((evt.loaded / evt.total) * 100));
        });
        if (!cancelled) { setModelReady(true); setModelProgress(null); }
      } catch (err) {
        if (!cancelled) { setModelError('The AI model could not be loaded. Check your connection and try again.'); setModelProgress(null); }
      }
    })();
    return () => { cancelled = true; };
  }, [modelVersion, quest.reps]);

  useEffect(() => {
    if (quest.reps || !modelReady) return undefined;
    let cancelled = false;
    getStaticClassifier().catch(() => { /* falls back gracefully in runFinalVerification */ });
    return () => { cancelled = true; };
  }, [modelReady, quest.reps]);

  const resetRepTracking = useCallback(() => {
    const baseline = quest.reps ? (quest.progress || 0) : 0;
    poseStateRef.current = createPoseRepState(baseline);
    setRepsDone(baseline);
    setRepPhase('up');
    setRepCue('Get in frame');
  }, [quest.reps, quest.progress]);

  // NOTE (perf fix): rep-based quests (pushups/squats/pullups/etc.) are
  // verified entirely by the MediaPipe pose-angle loop below, which already
  // runs on every video frame via requestAnimationFrame — that's the real
  // rep counter. This generic zero-shot classifier scan loop is only needed
  // to confirm non-rep quest types (action/food/map), so it now bails out
  // immediately for quest.reps. Previously it ran *concurrently* with the
  // pose loop for rep quests too: doubling model inference load exactly when
  // the device is already busiest (running pose tracking every frame), and
  // — since `confirmed` was a single shared piece of state — it could flip
  // the "complete quest" button to enabled just from a generic "doing
  // pushups" label match, without the rep counter having reached the target.
  // A lightweight, rep-specific anti-spoof check (below) already covers the
  // "photo/video of someone else" case for this quest type.
  useEffect(() => {
    if (quest.reps) return undefined;
    if (phase !== 'live' || !modelReady || confirmed || uploading || !labels || !classifierLabels.length) return undefined;
    const runId = ++scanRunRef.current;
    let disposed = false;
    let consecutiveErrors = 0;
    const isCurrentRun = () => !disposed && runId === scanRunRef.current && !confirmedRef.current;
    const scheduleNext = (delay) => { if (!isCurrentRun()) return; clearTimeout(scanTimerRef.current); scanTimerRef.current = window.setTimeout(scanLoop, delay); };

    const scanLoop = async () => {
      const video = videoRef.current, canvas = aiCanvasRef.current;
      if (!isCurrentRun()) return;
      if (document.visibilityState === 'hidden') { scheduleNext(1000); return; }
      if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) { scheduleNext(250); return; }
      if (video.currentTime === lastVideoTimeRef.current || isScanningRef.current) { scheduleNext(80); return; }
      lastVideoTimeRef.current = video.currentTime;
      isScanningRef.current = true;
      setScanning(true);

      try {
        if (canvas.width !== 224 || canvas.height !== 224) { canvas.width = 224; canvas.height = 224; }
        const context = canvas.getContext('2d');
        if (!context) throw new Error('Canvas 2D context unavailable');

        const scale = Math.min(224 / video.videoWidth, 224 / video.videoHeight);
        const width = video.videoWidth * scale;
        const height = video.videoHeight * scale;
        context.fillStyle = '#000';
        context.fillRect(0, 0, 224, 224);
        context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, (224 - width) / 2, (224 - height) / 2, width, height);

        const dataUrl = canvas.toDataURL('image/jpeg', 0.55);
        const classifier = await getClassifier();
        const results = await classifier(dataUrl, classifierLabels);
        if (!isCurrentRun()) return;
        consecutiveErrors = 0;

        const actScore = getMaxLabelScore(results, labels.activity);
        const negScore = getMaxLabelScore(results, activeNegatives);
        const spoofScore = getMaxLabelScore(results, ANTI_SPOOF_LABELS);
        setLastLabel(results[0]?.label ?? '');

        smoothedActRef.current = smoothedActRef.current === null ? actScore : smoothedActRef.current * (1 - SCORE_SMOOTHING) + actScore * SCORE_SMOOTHING;
        smoothedNegRef.current = smoothedNegRef.current === null ? negScore : smoothedNegRef.current * (1 - SCORE_SMOOTHING) + negScore * SCORE_SMOOTHING;
        smoothedSpoofRef.current = smoothedSpoofRef.current === null ? spoofScore : smoothedSpoofRef.current * (1 - SCORE_SMOOTHING) + spoofScore * SCORE_SMOOTHING;
        const smoothedAct = smoothedActRef.current;
        const smoothedNeg = smoothedNegRef.current;
        const smoothedSpoof = smoothedSpoofRef.current;

        setLiveScore(Math.round(smoothedAct * 100));
        const spoofSuspected = smoothedSpoof >= SPOOF_BLOCK_THRESHOLD;
        const passed = !spoofSuspected && (smoothedAct - smoothedNeg) >= PASS_MARGIN && smoothedAct >= PASS_THRESHOLD;
        passStreakRef.current = passed
            ? Math.min(REQUIRED_PASSES, passStreakRef.current + 1)
            : Math.max(0, passStreakRef.current - 1);
        setPassStreak(passStreakRef.current);
        if (spoofSuspected) {
          spoofNoticeActiveRef.current = true;
          setNotice('This looks like it\'s coming from a screen or printed photo rather than you, live. Point the camera at yourself doing it in person.');
        } else if (spoofNoticeActiveRef.current) {
          spoofNoticeActiveRef.current = false;
          setNotice(null);
        }
        if (passStreakRef.current >= REQUIRED_PASSES && !verifyingRef.current && !confirmedRef.current) {
          verifyingRef.current = true;
          setVerifying(true);
          runFinalVerification();
        } else if (passed) {
          haptic(8);
        }
      } catch (err) {
        consecutiveErrors += 1;
        if (consecutiveErrors >= 3 && isCurrentRun()) setModelError('AI analysis is temporarily unavailable. Try closing and reopening the camera.');
      } finally {
        isScanningRef.current = false;
        if (isCurrentRun()) { setScanning(false); scheduleNext(consecutiveErrors ? 1000 : 1200); }
      }
    };

    const runFinalVerification = async () => {
      try {
        const video = videoRef.current;
        if (!video || !video.videoWidth) throw new Error('no live frame available');
        const size = 256;
        const canvas = document.createElement('canvas');
        canvas.width = size; canvas.height = size;
        const ctx = canvas.getContext('2d');
        const scale = Math.min(size / video.videoWidth, size / video.videoHeight);
        const w = video.videoWidth * scale, h = video.videoHeight * scale;
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, size, size);
        ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, (size - w) / 2, (size - h) / 2, w, h);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

        const staticClassifier = await getStaticClassifier();
        const results = await staticClassifier(dataUrl, classifierLabels);
        const actScore = getMaxLabelScore(results, labels.activity);
        const negScore = getMaxLabelScore(results, activeNegatives);
        const spoofScore = getMaxLabelScore(results, ANTI_SPOOF_LABELS);
        const finalPass = spoofScore < SPOOF_BLOCK_THRESHOLD && (actScore - negScore) >= PASS_MARGIN && actScore >= PASS_THRESHOLD;

        if (!isCurrentRun()) return;
        if (finalPass) {
          confirmedRef.current = true;
          setConfirmed(true);
          haptic([15, 30, 15]);
        } else {
          // fail closed: a second, higher-resolution model disagreed with the
          // live estimate, so drop most of the streak instead of confirming
          if (spoofScore >= SPOOF_BLOCK_THRESHOLD) {
            spoofNoticeActiveRef.current = true;
            setNotice('This looks like it\'s coming from a screen or printed photo rather than you, live. Point the camera at yourself doing it in person.');
          }
          passStreakRef.current = Math.max(0, REQUIRED_PASSES - 2);
          setPassStreak(passStreakRef.current);
        }
      } catch {
        // model call failed — fail closed rather than auto-confirming, so a
        // dropped network request or model hiccup can't be used as a free pass
        if (isCurrentRun()) {
          passStreakRef.current = Math.max(0, REQUIRED_PASSES - 2);
          setPassStreak(passStreakRef.current);
        }
      } finally {
        verifyingRef.current = false;
        setVerifying(false);
        if (isCurrentRun()) scheduleNext(400);
      }
    };
    scanLoop();
    return () => { disposed = true; scanRunRef.current += 1; clearTimeout(scanTimerRef.current); };
  }, [phase, modelReady, confirmed, uploading, labels, classifierLabels, quest.reps, activeNegatives]);

  useEffect(() => {
    if (!hasSkeletonTracking || phase !== 'live' || confirmed) return undefined;
    let cancelled = false;
    (async () => {
      try {
        setPoseError(null);
        const landmarker = await getPoseLandmarker();
        if (cancelled) return;
        setPoseReady(true);

        const loop = () => {
          if (cancelled) return;
          const video = videoRef.current;
          const canvas = skeletonCanvasRef.current;
          if (video && video.readyState >= 2 && video.currentTime !== poseLastVideoTimeRef.current) {
            poseLastVideoTimeRef.current = video.currentTime;
            const now = performance.now();
            try {
              const result = landmarker.detectForVideo(video, now);
              const landmarks = result?.landmarks?.[0] ?? null;
              if (landmarks && quest.reps) {
                const update = updatePoseRepState(poseStateRef.current, quest.id, landmarks, now);
                setRepPhase(update.phase);
                setRepCue(update.cue);
                if (update.counted) {
                  setRepsDone(update.reps);
                  if (update.reps >= quest.reps) {
                    if (repSpoofSuspectedRef.current) {
                      // hold one rep short until the screen/photo signal clears —
                      // this stops a looped video of someone else exercising
                      // (which pose angles alone can't distinguish from the real thing)
                      poseStateRef.current.reps = quest.reps - 1;
                      setRepsDone(quest.reps - 1);
                      haptic(10);
                    } else {
                      confirmedRef.current = true;
                      setConfirmed(true);
                      haptic([15, 30, 15]);
                    }
                  } else {
                    haptic(10);
                  }
                }
              } else if (!landmarks && quest.reps) {
                setRepCue('Step back so your full body is visible');
              }

              if (canvas && video) {
                const ctx = canvas.getContext('2d');
                if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
                  canvas.width = canvas.clientWidth || 300;
                  canvas.height = canvas.clientHeight || 300;
                }
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                if (landmarks && !confirmedRef.current) {
                  const Wc = canvas.width;
                  const Hc = canvas.height;
                  const Wv = video.videoWidth || 640;
                  const Hv = video.videoHeight || 480;
                  const scale = Math.max(Wc / Wv, Hc / Hv);
                  const Wr = Wv * scale;
                  const Hr = Hv * scale;
                  const Ox = (Wc - Wr) / 2;
                  const Oy = (Hc - Hr) / 2;

                  const mapPt = (pt) => ({ x: Ox + pt.x * Wr, y: Oy + pt.y * Hr, vis: pt.visibility ?? 1 });
                  const mapped = landmarks.map(mapPt);
                  const currentPhase = poseStateRef.current.phase;
                  const lineColor = currentPhase === 'down' ? '#30D158' : '#0A84FF';

                  ctx.save();
                  ctx.lineWidth = 3;
                  ctx.lineCap = 'round';
                  ctx.lineJoin = 'round';
                  ctx.strokeStyle = lineColor;
                  POSE_CONNECTIONS.forEach(([i, j]) => {
                    const p1 = mapped[i]; const p2 = mapped[j];
                    if (p1 && p2 && p1.vis > 0.4 && p2.vis > 0.4) {
                      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
                    }
                  });
                  ctx.restore();

                  ctx.save();
                  mapped.forEach((pt, idx) => {
                    if (pt.vis > 0.4 && (idx === 0 || (idx >= 11 && idx <= 32))) {
                      const isMajorJoint = [11, 12, 13, 14, 23, 24, 25, 26].includes(idx);
                      ctx.beginPath();
                      ctx.arc(pt.x, pt.y, isMajorJoint ? 6 : 4, 0, 2 * Math.PI);
                      ctx.fillStyle = 'rgba(0,0,0,0.65)';
                      ctx.fill();
                      ctx.lineWidth = isMajorJoint ? 2 : 1.5;
                      ctx.strokeStyle = lineColor;
                      ctx.stroke();
                      ctx.beginPath();
                      ctx.arc(pt.x, pt.y, isMajorJoint ? 2.5 : 1.6, 0, 2 * Math.PI);
                      ctx.fillStyle = '#ffffff';
                      ctx.fill();
                    }
                  });
                  ctx.restore();
                }
              }
            } catch (err) { /* transient frame errors are fine, keep looping */ }
          }
          if (!cancelled) poseRafRef.current = requestAnimationFrame(loop);
        };
        poseRafRef.current = requestAnimationFrame(loop);
      } catch (err) {
        if (!cancelled) setPoseError('Could not load the pose tracking model. Check your connection and try again.');
      }
    })();
    return () => {
      cancelled = true;
      if (poseRafRef.current) cancelAnimationFrame(poseRafRef.current);
      if (skeletonCanvasRef.current) {
        const ctx = skeletonCanvasRef.current.getContext('2d');
        ctx?.clearRect(0, 0, skeletonCanvasRef.current.width, skeletonCanvasRef.current.height);
      }
    };
  }, [hasSkeletonTracking, quest.reps, quest.id, phase, confirmed]);

  // rep quests are verified by joint-angle thresholds, which a static photo
  // can't fake (angles never move) — but a *video* of someone else
  // exercising, played on another screen and pointed at the camera, would
  // still cross those thresholds. This runs alongside pose tracking purely
  // to catch that: it doesn't judge the activity, only whether the feed
  // looks like it's coming from a screen or printed photo.
  useEffect(() => {
    if (!quest.reps || phase !== 'live' || confirmed) return undefined;
    let cancelled = false;
    let timer = null;

    const check = async () => {
      try {
        const video = videoRef.current;
        const canvas = aiCanvasRef.current;
        if (video && canvas && video.videoWidth) {
          if (canvas.width !== 224 || canvas.height !== 224) { canvas.width = 224; canvas.height = 224; }
          const ctx = canvas.getContext('2d');
          const scale = Math.min(224 / video.videoWidth, 224 / video.videoHeight);
          const w = video.videoWidth * scale, h = video.videoHeight * scale;
          ctx.fillStyle = '#000';
          ctx.fillRect(0, 0, 224, 224);
          ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, (224 - w) / 2, (224 - h) / 2, w, h);
          const dataUrl = canvas.toDataURL('image/jpeg', 0.5);
          const classifier = await getClassifier();
          const results = await classifier(dataUrl, [...ANTI_SPOOF_LABELS, 'a real person exercising in a room']);
          if (!cancelled) {
            const spoofScore = getMaxLabelScore(results, ANTI_SPOOF_LABELS);
            smoothedRepSpoofRef.current = smoothedRepSpoofRef.current === null ? spoofScore : smoothedRepSpoofRef.current * 0.5 + spoofScore * 0.5;
            const suspected = smoothedRepSpoofRef.current >= SPOOF_BLOCK_THRESHOLD;
            repSpoofSuspectedRef.current = suspected;
            if (suspected) {
              repSpoofNoticeActiveRef.current = true;
              setNotice('This looks like it\'s coming from a screen or printed photo rather than you, live. Point the camera at yourself doing it in person.');
            } else if (repSpoofNoticeActiveRef.current) {
              repSpoofNoticeActiveRef.current = false;
              setNotice(null);
            }
          }
        }
      } catch { /* transient — try again next tick */ }
      finally { if (!cancelled) timer = window.setTimeout(check, 2500); }
    };
    timer = window.setTimeout(check, 1500);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [quest.reps, phase, confirmed]);

  const retryCamera = () => { scanRunRef.current += 1; lastVideoTimeRef.current = -1; setCamError(null); setPhase('starting'); setCameraVersion(v => v + 1); setSourceWarning(null); };
  const flipCamera = () => {
    clearTimeout(scanTimerRef.current); scanRunRef.current += 1; setPhase('starting'); setLiveScore(0); setPassStreak(0); setRepsDone(quest.reps ? (quest.progress || 0) : 0); passStreakRef.current = 0; confirmedRef.current = false; lastVideoTimeRef.current = -1; smoothedActRef.current = null; smoothedNegRef.current = null; smoothedSpoofRef.current = null; spoofNoticeActiveRef.current = false; repSpoofSuspectedRef.current = false; smoothedRepSpoofRef.current = null; repSpoofNoticeActiveRef.current = false; verifyingRef.current = false; setVerifying(false); resetRepTracking(); setConfirmed(false); setSourceWarning(null); setFacingMode(m => m === 'environment' ? 'user' : 'environment');
    if (skeletonCanvasRef.current) {
      const ctx = skeletonCanvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, skeletonCanvasRef.current.width, skeletonCanvasRef.current.height);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0]; if (!file) return;
    if (quest.reps) { setNotice('Rep quests need live camera tracking; a single photo cannot verify a full set.'); e.target.value = ''; return; }
    const reader = new FileReader();
    reader.onload = async (ev) => {
      let accepted = false; scanRunRef.current += 1; setUploading(true); stopCamera(); clearTimeout(scanTimerRef.current);
      setSourceWarning(null);
      const dataUrl = ev.target.result; setUploadedProof(dataUrl);
      if (!labels) { onConfirm(dataUrl); return; }
      setPhase('live'); setScanning(true);
      try {
        if (questType !== 'map') {
          const authCheck = await checkPhotoAuthenticity(file, dataUrl, SOURCE_LABELS);
          if (authCheck.suspicious) {
            setPassStreak(0);
            setNotice(`This looks like a stock photo, screenshot, or image pulled from the internet rather than one you took yourself (${authCheck.confidence}% confidence). Please upload an original photo.`);
            return;
          }
          if (!authCheck.hasCameraMetadata) {
            setSourceWarning("This photo has no camera metadata, so we can't fully confirm it's an original. Live camera capture is the most reliable option.");
          }
        }
        const classifier = await getStaticClassifier();
        const results = await classifier(dataUrl, classifierLabels);
        const actScore = getMaxLabelScore(results, labels.activity);
        const negScore = getMaxLabelScore(results, activeNegatives);
        const spoofScore = getMaxLabelScore(results, ANTI_SPOOF_LABELS);
        setLiveScore(Math.round(actScore * 100)); setLastLabel(results[0]?.label ?? '');
        if (spoofScore >= SPOOF_BLOCK_THRESHOLD) {
          setPassStreak(0);
          setNotice('This looks like a photo of a screen or a printed photo rather than an original shot. Please upload something you photographed directly.');
        } else if ((actScore - negScore) >= PASS_MARGIN && actScore >= PASS_THRESHOLD) {
          setPassStreak(REQUIRED_PASSES); confirmedRef.current = true; accepted = true; setConfirmed(true); haptic([15, 30, 15]);
        } else { setPassStreak(0); setNotice("Verification failed. Please make sure you upload a clear activity summary log showing distance and elapsed time metrics."); }
      } catch (err) { } finally { setScanning(false); setUploading(false); if (!accepted) { setUploadedProof(null); retryCamera(); } }
    };
    reader.readAsDataURL(file);
  };

  const handleCancel = () => {
    stopCamera();
    if (quest.reps) onCancel(repsDone > 0 ? repsDone : undefined);
    else if (quest.duration) onCancel(secondsLeft < quest.duration ? secondsLeft : undefined);
    else onCancel(undefined);
  };

  const confirmingRef = useRef(false);
  const captureAndConfirm = () => {
    if (confirmingRef.current) return;
    confirmingRef.current = true;
    if (uploadedProof) { onConfirm(uploadedProof); return; }
    const video = videoRef.current, canvas = aiCanvasRef.current;
    if (!video || !canvas) { confirmingRef.current = false; return; }
    canvas.width = video.videoWidth || 640; canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);
    onConfirm(canvas.toDataURL('image/jpeg', 0.82));
  };

  const meterColor = liveScore >= PASS_THRESHOLD * 100 ? c.green : liveScore >= 15 ? c.orange : c.red;

  return (
      <div className="fixed inset-0 z-[60] flex items-end justify-center" style={{ background: 'rgba(0,0,0,0.5)' }}>
        <div style={{ background: c.bg, width: '100%', height: '95vh', borderTopLeftRadius: 24, borderTopRightRadius: 24, overflow: 'hidden', display: 'flex', flexDirection: 'column', paddingBottom: 'max(env(safe-area-inset-bottom), 14px)' }}>
          <div className="flex justify-center pt-3 pb-2">
            <div style={{ width: 36, height: 5, borderRadius: 999, background: c.fill }} />
          </div>

          <div className="flex items-center justify-center px-6 pb-3">
            <div className="text-center">
              <p style={{ fontSize: 15, fontWeight: 600, color: c.label }}>Quest check-in</p>
              <p style={{ fontSize: 11, color: c.labelSecondary, marginTop: 2, maxWidth: 240, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{uiSubtext}</p>
            </div>
          </div>

          <div className="relative flex-1 mx-4 rounded-3xl overflow-hidden" style={{ background: '#000' }}>
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" style={{ opacity: phase === 'live' && !uploadedProof ? 1 : 0, transition: 'opacity 0.3s' }} />
            <canvas ref={skeletonCanvasRef} className="absolute inset-0 w-full h-full pointer-events-none object-cover z-10"
                    style={{ opacity: phase === 'live' && !uploadedProof && hasSkeletonTracking ? 1 : 0, transition: 'opacity 0.3s' }} />

            {phase === 'live' && !uploadedProof && labels?.bodyParts && (
                <div className="absolute top-3 left-3 z-10" style={{ background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(10px)', borderRadius: 12, padding: '9px 11px' }}>
                  <div className="flex items-center gap-1.5 mb-1.5" style={{ color: '#fff', fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4 }}>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: c.blue, display: 'inline-block' }} className="animate-pulse" />
                    Tracking
                  </div>
                  <div className="flex flex-wrap gap-1 max-w-[150px]">
                    {labels.bodyParts.map((part) => (
                        <span key={part} style={{ background: 'rgba(255,255,255,0.14)', color: '#fff', padding: '2px 7px', borderRadius: 8, fontSize: 9, fontWeight: 500 }} className="sq-mono">
                    {part}
                  </span>
                    ))}
                  </div>
                </div>
            )}

            {uploadedProof && (<img src={uploadedProof} alt="Uploaded verification" className="absolute inset-0 w-full h-full object-cover" />)}

            {phase === 'starting' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                  <div style={{ width: 30, height: 30, border: '2.5px solid rgba(255,255,255,0.2)', borderTopColor: '#fff', borderRadius: '50%' }} className="animate-spin" />
                  <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: 12, fontWeight: 500 }}>Opening camera…</p>
                </div>
            )}

            {phase === 'error' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-8 text-center">
                  <p style={{ color: '#fff', fontWeight: 700, fontSize: 15 }}>{camError === 'permission' ? 'Camera access blocked' : 'No camera found'}</p>
                  <p style={{ color: 'rgba(255,255,255,0.65)', fontSize: 13, lineHeight: 1.5, marginBottom: 4 }}>
                    {camError === 'permission' ? 'Open this app in a new tab and allow camera access when prompted.' : quest.reps ? 'This quest needs a live camera. Check your camera and try again.' : 'Upload a photo instead.'}
                  </p>
                  <button onClick={retryCamera} style={{ background: c.blue, color: '#fff', fontSize: 13, fontWeight: 600, padding: '9px 22px', borderRadius: 999 }}>Try again</button>
                </div>
            )}

            {(modelError || notice) && (
                <div className="absolute inset-x-3 top-3 z-30" style={{ borderRadius: 12, background: 'rgba(255,59,48,0.92)', backdropFilter: 'blur(10px)', padding: '10px 14px' }}>
                  <p style={{ fontSize: 13, fontWeight: 500, lineHeight: 1.4, color: '#fff' }}>{modelError || notice}</p>
                  <button onClick={() => setNotice(null)} style={{ fontSize: 11, fontWeight: 700, color: 'rgba(255,255,255,0.75)', marginTop: 6, textTransform: 'uppercase', letterSpacing: 0.3 }}>Dismiss</button>
                </div>
            )}

            {!modelError && !notice && sourceWarning && (
                <div className="absolute inset-x-3 top-3 z-30" style={{ borderRadius: 12, background: 'rgba(255,159,10,0.92)', backdropFilter: 'blur(10px)', padding: '10px 14px' }}>
                  <p style={{ fontSize: 13, fontWeight: 500, lineHeight: 1.4, color: '#fff' }}>{sourceWarning}</p>
                  <button onClick={() => setSourceWarning(null)} style={{ fontSize: 11, fontWeight: 700, color: 'rgba(255,255,255,0.75)', marginTop: 6, textTransform: 'uppercase', letterSpacing: 0.3 }}>Dismiss</button>
                </div>
            )}

            {phase === 'live' && quest.reps && (
                <div className="absolute inset-x-0 bottom-0 px-4 pb-4 pt-10 z-20" style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.75), transparent)' }}>
                  {confirmed ? (
                      <div className="flex items-center gap-2 w-max px-3 py-1.5 rounded-full" style={{ background: 'rgba(48,209,88,0.2)' }}>
                        <div style={{ width: 18, height: 18, borderRadius: '50%', background: c.green, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><CheckIcon size={10} /></div>
                        <p style={{ color: '#DFFBE8', fontSize: 12, fontWeight: 600 }}>Verified</p>
                      </div>
                  ) : poseError ? (
                      <p style={{ color: '#FF9F9A', fontSize: 12, fontWeight: 500 }}>{poseError}</p>
                  ) : !poseReady ? (
                      <p style={{ color: 'rgba(255,255,255,0.8)', fontSize: 11, fontWeight: 600 }}>Getting ready to track your reps…</p>
                  ) : (
                      <>
                        <div className="flex items-end justify-between mb-2">
                          <p style={{ color: 'rgba(255,255,255,0.9)', fontSize: 13, fontWeight: 500 }}>{repCue}</p>
                          <p className="sq-mono" style={{ color: '#fff', fontSize: 12, fontWeight: 700, background: c.blue, padding: '4px 10px', borderRadius: 8 }}>
                            {repsDone} / {quest.reps} reps
                          </p>
                        </div>
                        <div style={{ width: '100%', height: 4, borderRadius: 999, background: 'rgba(255,255,255,0.2)', overflow: 'hidden' }}>
                          <div style={{ height: '100%', background: c.green, width: `${Math.min(100, (repsDone / quest.reps) * 100)}%`, transition: 'width 0.3s ease-out' }} />
                        </div>
                      </>
                  )}
                </div>
            )}

            {phase === 'live' && !quest.reps && modelReady && (
                <div className="absolute inset-x-0 bottom-0 px-4 pb-4 pt-10 z-20" style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.75), transparent)' }}>
                  {confirmed ? (
                      <div className="flex items-center gap-2 w-max px-3 py-1.5 rounded-full" style={{ background: 'rgba(48,209,88,0.2)' }}>
                        <div style={{ width: 18, height: 18, borderRadius: '50%', background: c.green, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><CheckIcon size={10} /></div>
                        <p style={{ color: '#DFFBE8', fontSize: 12, fontWeight: 600 }}>Verified</p>
                      </div>
                  ) : verifying ? (
                      <div className="flex items-center gap-2 w-max px-3 py-1.5 rounded-full" style={{ background: 'rgba(10,132,255,0.2)' }}>
                        <div style={{ width: 14, height: 14, border: '2px solid rgba(255,255,255,0.3)', borderTopColor: '#fff', borderRadius: '50%' }} className="animate-spin" />
                        <p style={{ color: '#fff', fontSize: 12, fontWeight: 600 }}>Confirming…</p>
                      </div>
                  ) : (
                      <>
                        <div className="flex items-end justify-between mb-2">
                          <p style={{ color: 'rgba(255,255,255,0.9)', fontSize: 13, fontWeight: 500 }}>{scanning ? 'Scanning…' : `${liveScore}% confidence`}</p>
                          <div className="flex items-center gap-1">
                            {Array.from({ length: REQUIRED_PASSES }).map((_, i) => (
                                <div key={i} style={{ width: 6, height: 6, borderRadius: '50%', background: i < passStreak ? c.green : 'rgba(255,255,255,0.3)', transition: 'background 0.3s' }} />
                            ))}
                          </div>
                        </div>
                        <div style={{ width: '100%', height: 4, borderRadius: 999, background: 'rgba(255,255,255,0.2)', overflow: 'hidden' }}>
                          <div style={{ height: '100%', background: meterColor, width: `${liveScore}%`, transition: 'width 0.5s ease-out' }} />
                        </div>
                        {lastLabel && <p className="sq-mono" style={{ color: 'rgba(255,255,255,0.4)', fontSize: 9, marginTop: 6, textTransform: 'uppercase', letterSpacing: 0.3, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{lastLabel}</p>}
                      </>
                  )}
                </div>
            )}

            {phase === 'live' && !quest.reps && !modelReady && !modelError && (
                <div className="absolute inset-x-0 bottom-0 px-4 pb-4 pt-10 z-20" style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.7), transparent)' }}>
                  <p style={{ color: 'rgba(255,255,255,0.8)', fontSize: 11, fontWeight: 600, marginBottom: 8 }}>Getting ready… {modelProgress ?? 0}%</p>
                  <div style={{ width: '100%', height: 4, borderRadius: 999, background: 'rgba(255,255,255,0.2)', overflow: 'hidden' }}>
                    <div style={{ height: '100%', background: c.blue, width: `${modelProgress ?? 0}%`, transition: 'width 0.3s' }} />
                  </div>
                </div>
            )}
            <canvas ref={aiCanvasRef} className="hidden" />
          </div>

          {quest.duration && (
              <div className="mx-4 mt-4" style={{ padding: '13px 16px', borderRadius: 20, background: c.bgElevated, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <span style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, color: c.labelSecondary }}>Duration</span>
                  <p className="sq-mono" style={{ fontSize: 20, fontWeight: 700, color: c.label, marginTop: 2 }}>{formatTimerString(secondsLeft)}</p>
                </div>
                <button onClick={() => setTimerRunning(!timerRunning)} disabled={secondsLeft === 0}
                        style={{ padding: '10px 20px', borderRadius: 12, fontSize: 13, fontWeight: 600, color: '#fff', background: secondsLeft === 0 ? c.gray : timerRunning ? c.red : c.green, opacity: secondsLeft === 0 ? 0.5 : 1 }}>
                  {secondsLeft === 0 ? 'Done' : timerRunning ? 'Pause' : 'Start'}
                </button>
              </div>
          )}

          <div className="px-4 pt-4 pb-1">
            {phase === 'live' && (
                <div className="space-y-2.5">
                  <button onClick={captureAndConfirm} disabled={!confirmed}
                          style={{ width: '100%', padding: '15px', borderRadius: 999, fontSize: 16, fontWeight: 600, color: confirmed ? '#fff' : c.labelTertiary, background: confirmed ? c.blue : c.fill }}>
                    {confirmed ? 'Complete quest' : verifying ? 'Confirming…' : instructionText}
                  </button>
                  <div className="flex gap-2.5">
                    <button onClick={flipCamera} style={{ flex: quest.reps ? '1 1 100%' : 1, padding: '13px', borderRadius: 999, fontSize: 14, fontWeight: 600, background: c.fill, color: c.label }}>
                      Flip camera
                    </button>
                    {!quest.reps && (
                        <label style={{ flex: 1, padding: '13px', borderRadius: 999, fontSize: 14, fontWeight: 600, textAlign: 'center', cursor: 'pointer', background: questType === 'map' ? c.bgElevated : c.fill, color: questType === 'map' ? c.blue : c.label, border: questType === 'map' ? `1.5px solid ${c.blue}` : 'none' }}>
                          Upload photo
                          <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
                        </label>
                    )}
                  </div>
                </div>
            )}

            {(phase === 'error' || phase === 'starting') && !quest.reps && (
                <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', padding: '15px', borderRadius: 999, fontSize: 16, fontWeight: 600, background: c.fill, color: c.label, cursor: 'pointer' }}>
                  Upload a photo
                  <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
                </label>
            )}

            <button onClick={handleCancel} style={{ width: '100%', marginTop: 10, padding: '15px', borderRadius: 999, fontSize: 16, fontWeight: 600, background: 'transparent', color: c.red }}>
              Cancel
            </button>
          </div>
        </div>
      </div>
  );
}

// main app
export default function QuestDailyApp() {
  const [isMounted, setIsMounted] = useState(false);
  const { user: authUser, loading: authLoading } = useAuthUser();
  const uid = authUser?.uid ?? null;

  const [dark, setDark] = useState(true);
  const [showInstallPrompt, setShowInstallPrompt] = useState(false);
  const [level, setLevel] = useState(1);
  const [xp, setXp] = useState(0);
  const [totalXpEarned, setTotalXpEarned] = useState(0);
  const [streak, setStreak] = useState(0);
  const [username, setUsername] = useState(null);
  const [showLeaderboard, setShowLeaderboard] = useState(false);
  const [leaderboardSyncError, setLeaderboardSyncError] = useState(null);
  const [quests, setQuests] = useState([]);
  const [lastReset, setLastReset] = useState(0);
  const [proofImages, setProofImages] = useState({});
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('quests');
  const [photoURL, setPhotoURL] = useState(null);
  const [photoUploading, setPhotoUploading] = useState(false);
  const [photoError, setPhotoError] = useState(null);
  const [hasSeenWelcome, setHasSeenWelcome] = useState(true); // defaults true until localStorage is checked, so it never flashes for returning users
  const [hapticsOn, setHapticsOnState] = useState(true);
  const [dailyReminderOn, setDailyReminderOn] = useState(false);
  const notificationsSupported = typeof window !== 'undefined' && 'Notification' in window;

  useEffect(() => {
    setIsMounted(true);
    const savedDark = localStorage.getItem('sq_dark');
    setDark(savedDark !== null ? savedDark === 'true' : window.matchMedia('(prefers-color-scheme: dark)').matches);

    if (!localStorage.getItem('sq_has_seen_install_v2')) {
      setShowInstallPrompt(true);
    }

    setLevel(parseInt(localStorage.getItem('sq_level')) || 1);
    setXp(parseInt(localStorage.getItem('sq_xp')) || 0);
    setTotalXpEarned(parseInt(localStorage.getItem('sq_totalXpEarned')) || 0);
    setStreak(parseInt(localStorage.getItem('sq_streak')) || 0);
    setLastReset(parseInt(localStorage.getItem('sq_lastReset')) || 0);
    setHasSeenWelcome(localStorage.getItem('sq_has_seen_welcome') === 'true');
    const savedHaptics = localStorage.getItem('sq_haptics_enabled');
    const hapticsInitial = savedHaptics === null ? true : savedHaptics === 'true';
    setHapticsOnState(hapticsInitial);
    setHapticsPref(hapticsInitial);
    setDailyReminderOn(localStorage.getItem('sq_daily_reminder') === 'true' && notificationsSupported && Notification.permission === 'granted');
    setProofImages(JSON.parse(localStorage.getItem('sq_proofs')) || {});
    setHistory(JSON.parse(localStorage.getItem('sq_history')) || []);

    const savedQuests = JSON.parse(localStorage.getItem('sq_quests')) || [];
    const anyDone = savedQuests.some(q => q.completed);
    const lastResetTs = parseInt(localStorage.getItem('sq_lastReset')) || 0;
    const expired = Date.now() - lastResetTs >= ONE_DAY_MS;

    if (!expired && savedQuests.length >= 5) setQuests(savedQuests);
    else if (!expired && anyDone) setQuests(savedQuests);
    else {
      const n = Math.floor(Math.random() * 3) + 5;
      setQuests([...QUEST_POOL].sort(() => 0.5 - Math.random()).slice(0, n).map(randomizeQuest));
    }
  }, []);

  // Pulls this account's synced profile (level/xp/streak), full history, and
  // profile photo from Firestore and makes it the source of truth locally —
  // used right after logging in, and again on every app load where a
  // session is already persisted (Firebase keeps the user signed in across
  // reloads), so a device always shows whatever the account actually has,
  // not whatever happened to be cached in this browser's localStorage.
  //
  // Importantly, `username` is guaranteed to end up set to *something* once
  // this resolves — falling back from the `users/{uid}` doc, to the
  // `leaderboard/{uid}` doc, to the account's own email local-part, to a
  // generic label — because the app blocks on `username` being set while
  // showing a full-screen loading state. If every source were empty and
  // this left username null, that loading screen would never go away.
  const hydrateAccount = useCallback(async (uidToLoad, fallbackEmail) => {
    const [profile, remoteHistory, profileDoc] = await Promise.all([
      fetchRemoteProfile(uidToLoad),
      fetchRemoteHistory(uidToLoad),
      getDoc(doc(db, 'users', uidToLoad)).catch(() => null),
    ]);
    setLevel(typeof profile?.level === 'number' ? profile.level : 1);
    setXp(typeof profile?.xp === 'number' ? profile.xp : 0);
    setTotalXpEarned(typeof profile?.totalXpEarned === 'number' ? profile.totalXpEarned : 0);
    setStreak(typeof profile?.streak === 'number' ? profile.streak : 0);
    setHistory(remoteHistory);

    const docUsername = profileDoc?.exists?.() ? profileDoc.data()?.username : null;
    const emailLocalPart = fallbackEmail ? fallbackEmail.split('@')[0] : null;
    setUsername(docUsername || profile?.username || emailLocalPart || 'You');
    setPhotoURL(profileDoc?.exists?.() ? (profileDoc.data()?.photoURL ?? null) : (profile?.photoURL ?? null));
  }, []);

  // Tracks which uid we've already hydrated so a re-render (or the profile
  // sync effect below writing back to Firestore) doesn't trigger a refetch
  // loop — only an actual account change re-hydrates.
  const hydratedUidRef = useRef(null);
  useEffect(() => {
    if (!isMounted || authLoading) return;
    if (!authUser) { hydratedUidRef.current = null; return; }
    if (hydratedUidRef.current === authUser.uid) return;
    hydratedUidRef.current = authUser.uid;
    hydrateAccount(authUser.uid, authUser.email);
  }, [authUser, authLoading, isMounted, hydrateAccount]);

  // Defensive safety net: hydrateAccount above should always resolve
  // username to something, but if a Firestore call ever hangs indefinitely
  // instead of resolving or rejecting (a genuine network stall, unlike a
  // normal failure which is already caught), this guarantees the app still
  // becomes usable a few seconds later instead of sitting on the loading
  // screen forever.
  useEffect(() => {
    if (!authUser || username) return undefined;
    const t = setTimeout(() => {
      setUsername(prev => prev || (authUser.email ? authUser.email.split('@')[0] : 'You'));
    }, 8000);
    return () => clearTimeout(t);
  }, [authUser, username]);

  useEffect(() => {
    if (isMounted) {
      document.documentElement.classList.toggle('dark', dark);
      localStorage.setItem('sq_dark', dark);
    }
  }, [dark, isMounted]);

  const handleDismissInstall = () => {
    localStorage.setItem('sq_has_seen_install_v2', 'true');
    setShowInstallPrompt(false);
  };

  // A brand-new account starts from a clean slate — no remote data to fetch,
  // so just mark it hydrated and let the profile-sync effect below write
  // the initial (level 1 / 0 xp) doc up to Firestore.
  const handleSignedUp = (finalUsername) => {
    hydratedUidRef.current = auth.currentUser?.uid ?? hydratedUidRef.current;
    setUsername(finalUsername);
    setLevel(1); setXp(0); setTotalXpEarned(0); setStreak(0); setHistory([]); setProofImages({}); setPhotoURL(null);
    setActiveTab('quests'); setHasSeenWelcome(true); localStorage.setItem('sq_has_seen_welcome', 'true');
  };

  const handleLoggedIn = async (finalUsername) => {
    setUsername(finalUsername); // optimistic — correct casing already resolved during login
    if (auth.currentUser) {
      hydratedUidRef.current = auth.currentUser.uid;
      setProofImages({}); // proof photos are device-local and never synced
      await hydrateAccount(auth.currentUser.uid, auth.currentUser.email);
    }
    setActiveTab('quests'); setHasSeenWelcome(true); localStorage.setItem('sq_has_seen_welcome', 'true');
  };

  const resetLocalAccountState = () => {
    hydratedUidRef.current = null;
    setUsername(null); setPhotoURL(null); setPhotoError(null); setPhotoUploading(false);
    setLevel(1); setXp(0); setTotalXpEarned(0); setStreak(0); setHistory([]); setProofImages({});
    setQuests([]); setLastReset(0);
    setShowLeaderboard(false); setDetailQuestId(null); setCompletionQuest(null); setProofModalId(null);
    setActiveTab('quests');
    ['sq_level','sq_xp','sq_totalXpEarned','sq_streak','sq_quests','sq_lastReset','sq_proofs','sq_history','sq_streak_date']
        .forEach(k => localStorage.removeItem(k));
  };

  const handleLogout = async () => {
    haptic(10);
    try {
      await logOutAccount();
    } catch (err) {
      console.error('Logout failed:', err);
    }
    resetLocalAccountState();
  };

  const handlePhotoFile = async (file) => {
    if (!uid) return;
    setPhotoError(null);
    setPhotoUploading(true);
    try {
      const dataUrl = await resizeImageToSquareDataUrl(file);
      setPhotoURL(dataUrl); // optimistic — update the UI before the write confirms
      await updateProfilePhoto(uid, dataUrl);
    } catch (err) {
      console.error('Profile photo update failed:', err);
      setPhotoError(err?.message || 'Could not update your photo. Please try again.');
    } finally {
      setPhotoUploading(false);
    }
  };

  const handleDeleteAccount = async () => {
    const currentUser = auth.currentUser;
    if (!currentUser) return { ok: false, error: 'You need to be signed in to do that.' };
    const usernameLower = username ? username.toLowerCase() : null;
    try {
      await deleteAccountData(currentUser.uid, usernameLower);
      await deleteUser(currentUser);
      resetLocalAccountState();
      return { ok: true };
    } catch (err) {
      console.error('Account deletion failed:', err);
      return { ok: false, error: friendlyAuthError(err) };
    }
  };

  const retryLeaderboardSync = () => {
    if (uid && username) {
      syncLeaderboardEntry(uid, { username, level, xp, totalXpEarned, streak })
          .then(res => setLeaderboardSyncError(res.ok ? null : res.error));
    }
  };

  // Settings screen calls this after a successful rename — updateUsername()
  // has already synced Firestore (and the Auth email, for password
  // accounts), this just brings the in-app display up to date.
  const handleUsernameChanged = (newUsername) => setUsername(newUsername);

  const handleToggleHaptics = () => {
    const next = !hapticsOn;
    setHapticsOnState(next);
    setHapticsPref(next);
    if (next) haptic(10); // gives immediate feedback that it's back on
  };

  const handleToggleDailyReminder = async () => {
    if (!notificationsSupported) return;
    if (dailyReminderOn) {
      setDailyReminderOn(false);
      localStorage.setItem('sq_daily_reminder', 'false');
      return;
    }
    try {
      const permission = Notification.permission === 'granted' ? 'granted' : await Notification.requestPermission();
      if (permission === 'granted') {
        setDailyReminderOn(true);
        localStorage.setItem('sq_daily_reminder', 'true');
      }
      // denied/dismissed: leave the toggle off, no error needed — the
      // browser's own permission UI already told them what happened
    } catch (err) {
      console.error('Notification permission request failed:', err);
    }
  };

  // While the daily reminder is on, checks once a minute whether it's past
  // the reminder hour and today's reminder hasn't fired yet. This only
  // works while the tab is open — there's no service worker registered
  // here for true background push, which the Settings copy makes clear.
  const REMINDER_HOUR = 18;
  useEffect(() => {
    if (!dailyReminderOn || !notificationsSupported || Notification.permission !== 'granted') return undefined;
    const check = () => {
      const now = new Date();
      const todayKey = now.toDateString();
      if (now.getHours() < REMINDER_HOUR) return;
      if (localStorage.getItem('sq_daily_reminder_last') === todayKey) return;
      try {
        new Notification('QuestDaily', { body: "You've got quests waiting — jump back in!" });
        localStorage.setItem('sq_daily_reminder_last', todayKey);
      } catch (err) { console.error('Could not show reminder notification:', err); }
    };
    check();
    const id = setInterval(check, 60_000);
    return () => clearInterval(id);
  }, [dailyReminderOn, notificationsSupported]);

  const handleExportData = () => {
    const payload = {
      username, level, xp, totalXpEarned, streak,
      history,
      exportedAt: new Date().toISOString(),
    };
    try {
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `questdaily-${(username || 'export').replace(/[^a-zA-Z0-9_-]/g, '')}-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Data export failed:', err);
    }
  };

  const authProviderLabel = getAuthProviderLabel(authUser);


  useEffect(() => {
    if (uid && username) {
      syncLeaderboardEntry(uid, { username, level, xp, totalXpEarned, streak })
          .then(res => setLeaderboardSyncError(res.ok ? null : res.error));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uid, username]);

  const [proofModalId,   setProofModalId]   = useState(null);
  const [viewingProof, setViewingProof] = useState(null);
  const [detailQuestId,     setDetailQuestId]     = useState(null);
  const [completionQuest, setCompletionQuest] = useState(null);

  const proofModal  = quests.find(q => q.id === proofModalId)  || null;
  const detailQuest = quests.find(q => q.id === detailQuestId) || null;

  const xpRef    = useRef(xp);
  const levelRef = useRef(level);
  const totalXpEarnedRef = useRef(totalXpEarned);
  useEffect(() => { xpRef.current    = xp;    }, [xp]);
  useEffect(() => { levelRef.current = level; }, [level]);
  useEffect(() => { totalXpEarnedRef.current = totalXpEarned; }, [totalXpEarned]);

  const xpRequired = level * 100;

  const generateNewQuests = useCallback((ts) => {
    const n = Math.floor(Math.random() * 3) + 5;
    const selected = [...QUEST_POOL].sort(() => 0.5 - Math.random()).slice(0, n).map(randomizeQuest);
    setQuests(selected); setLastReset(ts); setProofImages({});
  }, []);

  // Perf fix: this used to hold `timeLeft` as *state on the top-level App
  // component*, ticking every second and re-rendering the entire tree
  // (quest list, ambient background, tab bar, etc.) once a second even when
  // nothing else changed. The visible countdown now lives in its own
  // <CountdownDisplay> component with local state (see above); this effect
  // only does the actual bookkeeping — checking whether the daily quest set
  // has expired — and never calls setState on a tick unless a reset is
  // actually due, so it no longer causes any per-second re-renders here.
  const lastResetRef = useRef(lastReset);
  const questsLenRef = useRef(quests.length);
  useEffect(() => { lastResetRef.current = lastReset; }, [lastReset]);
  useEffect(() => { questsLenRef.current = quests.length; }, [quests.length]);
  useEffect(() => {
    if (!isMounted) return;
    const id = setInterval(() => {
      const now = Date.now();
      const remaining = ONE_DAY_MS - (now - lastResetRef.current);
      if (remaining <= 0 || questsLenRef.current === 0) generateNewQuests(now);
    }, 1000);
    return () => clearInterval(id);
  }, [generateNewQuests, isMounted]);

  // Perf fix: debounce localStorage writes so a burst of related state
  // updates (e.g. completing a quest touches level/xp/totalXpEarned/streak/
  // quests/proofImages/history all at once) coalesces into a single
  // JSON.stringify + write instead of firing on every intermediate state
  // change during the same batch.
  const persistTimerRef = useRef(null);
  useEffect(() => {
    if (!isMounted) return undefined;
    clearTimeout(persistTimerRef.current);
    persistTimerRef.current = setTimeout(() => {
      localStorage.setItem('sq_level', level);
      localStorage.setItem('sq_xp', xp);
      localStorage.setItem('sq_totalXpEarned', totalXpEarned);
      localStorage.setItem('sq_streak', streak);
      localStorage.setItem('sq_quests', JSON.stringify(quests));
      localStorage.setItem('sq_lastReset', lastReset);
      localStorage.setItem('sq_proofs', JSON.stringify(proofImages));
      localStorage.setItem('sq_history', JSON.stringify(history));
    }, 300);
    return () => clearTimeout(persistTimerRef.current);
  }, [level, xp, totalXpEarned, streak, quests, lastReset, proofImages, history, isMounted]);

  const applyXpChange = useCallback((amount) => {
    const startLevel = levelRef.current;
    let newXp = xpRef.current + amount, newLevel = levelRef.current;
    while (newXp >= newLevel * 100) { newXp -= newLevel * 100; newLevel++; }
    while (newXp < 0 && newLevel > 1) { newLevel--; newXp += newLevel * 100; }
    if (newLevel === 1 && newXp < 0) newXp = 0;
    xpRef.current = newXp; levelRef.current = newLevel;
    setXp(newXp); setLevel(newLevel);

    let newTotal = totalXpEarnedRef.current;
    if (amount > 0) {
      newTotal += amount;
      totalXpEarnedRef.current = newTotal;
      setTotalXpEarned(newTotal);
    }

    return { leveledUp: newLevel > startLevel, newXp, newLevel, newTotal };
  }, []);

  const handleQuestClick = (quest) => {
    if (quest.completed) return;
    haptic(6);
    setDetailQuestId(quest.id);
  };

  const handleProofConfirm = (questId, img) => {
    const quest = quests.find(q => q.id === questId);
    if (!quest || quest.completed) { setProofModalId(null); setDetailQuestId(null); return; }
    setProofImages(prev => ({ ...prev, [questId]: img }));
    setQuests(prev => prev.map(q => q.id === questId ? { ...q, completed: true, progress: 0 } : q));
    const { leveledUp, newXp, newLevel, newTotal } = applyXpChange(quest?.xp ?? 0);

    const entry = { id: `${questId}-${Date.now()}`, questId, text: quest.text, xp: quest.xp, ts: Date.now(), leveledUp };
    setHistory(prev => {
      const next = [entry, ...prev];
      return next.length > 500 ? next.slice(0, 500) : next;
    });
    if (uid) syncHistoryEntry(uid, entry);

    const todayStr = new Date().toDateString();
    const lastStreakDate = localStorage.getItem('sq_streak_date');
    let newStreak = streak;
    if (lastStreakDate !== todayStr) {
      const yesterdayStr = new Date(Date.now() - ONE_DAY_MS).toDateString();
      newStreak = lastStreakDate === yesterdayStr ? streak + 1 : 1;
      setStreak(newStreak);
      localStorage.setItem('sq_streak_date', todayStr);
    }

    syncLeaderboardEntry(uid, { username, level: newLevel, xp: newXp, totalXpEarned: newTotal, streak: newStreak })
        .then(res => setLeaderboardSyncError(res.ok ? null : res.error));

    setProofModalId(null);
    setDetailQuestId(null);
    setCompletionQuest(quest ? { ...quest, leveledUp } : null);
  };

  const handleCancelProof = (progress) => {
    if (proofModalId && progress !== undefined) {
      setQuests(prev => prev.map(q => q.id === proofModalId ? { ...q, progress } : q));
    }
    setProofModalId(null);
  };

  if (!isMounted || authLoading) return null;

  const c = dark ? C.dark : C.light;
  const xpPct  = Math.min(100, Math.max(0, (xp / xpRequired) * 100));
  const allDone = quests.length > 0 && quests.every(q => q.completed);

  const completedCount = quests.filter(q => q.completed).length;
  const dailyPct = quests.length ? (completedCount / quests.length) * 100 : 0;

  return (
      <div className="sq-root" style={{ background: c.bg, minHeight: '100vh', width: '100vw', display: 'flex', flexDirection: 'column', alignItems: 'center', transition: 'background-color 0.3s ease' }}>
        <SystemType />
        <div className="relative w-full flex flex-col min-h-screen" style={{ background: c.bg }}>
          <AmbientBackground c={c} />

          {!hasSeenWelcome && (
              <WelcomeScreen c={c} onContinue={() => { haptic(10); setHasSeenWelcome(true); localStorage.setItem('sq_has_seen_welcome', 'true'); }} />
          )}
          {hasSeenWelcome && showInstallPrompt && <DeviceInstallPrompt onDismiss={handleDismissInstall} c={c} />}
          {hasSeenWelcome && !showInstallPrompt && !authUser && <AuthModal onSignedUp={handleSignedUp} onLoggedIn={handleLoggedIn} c={c} />}
          {authUser && !username && (
              <div className="fixed inset-0 z-[100] flex items-center justify-center" style={{ background: c.bg }}>
                <div style={{ width: 28, height: 28, border: `2.5px solid ${c.fill}`, borderTopColor: c.blue, borderRadius: '50%' }} className="animate-spin" />
              </div>
          )}

          {proofModal && (
              <CameraModal quest={proofModal} c={c}
                           onConfirm={img => handleProofConfirm(proofModal.id, img)}
                           onCancel={handleCancelProof} />
          )}

          {viewingProof && (
              <div className="fixed inset-0 z-[70] flex items-center justify-center p-4 sq-anim-in"
                   style={{ background: 'rgba(0,0,0,0.85)' }}
                   onClick={() => setViewingProof(null)}>
                <img src={viewingProof} alt="proof" className="max-w-full max-h-full rounded-2xl" />
              </div>
          )}

          {showLeaderboard ? (
              <LeaderboardScreen dark={dark} c={c} onBack={() => setShowLeaderboard(false)}
                                 myUid={uid} myUsername={username} myPhotoURL={photoURL} myLevel={level} myXp={xp}
                                 syncError={leaderboardSyncError} onRetrySync={retryLeaderboardSync} />
          ) : completionQuest ? (
              <CompletionScreen quest={completionQuest} dark={dark} c={c} lastReset={lastReset}
                                onToggleTheme={() => setDark(d => !d)} onBack={() => setCompletionQuest(null)} />
          ) : detailQuest ? (
              <QuestDetailScreen quest={detailQuest} dark={dark} c={c} lastReset={lastReset}
                                 onToggleTheme={() => setDark(d => !d)} onBack={() => setDetailQuestId(null)}
                                 onMarkComplete={() => setProofModalId(detailQuest.id)} />
          ) : activeTab === 'history' ? (
              <HistoryScreen history={history} dark={dark} c={c} onToggleTheme={() => setDark(d => !d)} />
          ) : activeTab === 'settings' ? (
              <SettingsScreen
                  username={username} photoURL={photoURL} uid={uid} authProviderLabel={authProviderLabel}
                  dark={dark} onToggleTheme={() => setDark(d => !d)}
                  onPhotoFile={handlePhotoFile} photoUploading={photoUploading} photoError={photoError} onDismissPhotoError={() => setPhotoError(null)}
                  onUsernameChanged={handleUsernameChanged} onLogout={handleLogout} onDeleteAccount={handleDeleteAccount}
                  hapticsOn={hapticsOn} onToggleHaptics={handleToggleHaptics}
                  dailyReminderOn={dailyReminderOn} onToggleDailyReminder={handleToggleDailyReminder} notificationsSupported={notificationsSupported}
                  onExportData={handleExportData}
                  c={c}
              />
          ) : (
              <div className="relative z-10 flex flex-col flex-1 sq-anim-in">
                {/* Header */}
                <div className="px-4 pb-2" style={{ paddingTop: 'max(env(safe-area-inset-top), 16px)' }}>
                  <div className="flex items-center justify-between">
                    <h1 className="sq-large-title" style={{ fontSize: 32, fontWeight: 800, color: c.label }}>QuestDaily</h1>
                    <div className="flex items-center gap-2">
                      <button onClick={() => setShowLeaderboard(true)} aria-label="Leaderboard"
                              style={{ ...glassStyle(c), width: 32, height: 32, borderRadius: 999, display: 'flex', alignItems: 'center', justifyContent: 'center', color: c.blue }}>
                        <span className="sq-icon-tap"><TrophyIcon size={15} /></span>
                      </button>
                      <button onClick={() => setDark(d => !d)} aria-label="Toggle theme"
                              style={{ ...glassStyle(c), width: 32, height: 32, borderRadius: 999, display: 'flex', alignItems: 'center', justifyContent: 'center', color: c.label }}>
                        <span className="sq-icon-tap">{dark ? <Sun key="sun" size={16} strokeWidth={1.75} className="sq-icon-pop-in" /> : <Moon key="moon" size={16} strokeWidth={1.75} className="sq-icon-pop-in" />}</span>
                      </button>
                      <button onClick={() => setActiveTab('settings')} aria-label="Settings">
                        <Avatar photoURL={photoURL} username={username} size={32} c={c} />
                      </button>
                    </div>
                  </div>

                  <div className="mt-3 flex items-center gap-1.5">
                    <span style={{ fontSize: 12, fontWeight: 500, color: c.labelTertiary }}>{username}</span>
                  </div>
                  <div className="mt-1 flex items-center gap-2.5">
                <span className="sq-mono" style={{ fontSize: 12, fontWeight: 700, color: '#fff', background: c.blue, padding: '3px 9px', borderRadius: 8 }}>
                  LV {level}
                </span>
                    <span style={{ fontSize: 12, fontWeight: 500, color: c.labelSecondary }}>{getLevelTitle(level)}</span>
                    <span className="sq-mono ml-auto" style={{ fontSize: 11, fontWeight: 600, color: c.labelTertiary }}>{xp} / {xpRequired} XP</span>
                  </div>
                  <div style={{ marginTop: 8, height: 5, width: '100%', borderRadius: 999, background: c.fill, overflow: 'hidden' }}>
                    <div style={{ height: '100%', borderRadius: 999, width: `${xpPct}%`, background: c.blue, transition: 'width 0.6s cubic-bezier(0.22,1,0.36,1)' }} />
                  </div>
                </div>

                <div className="flex-1 sq-scroll overflow-y-auto px-4 pt-2 space-y-4" style={{ paddingBottom: 'calc(100px + env(safe-area-inset-bottom))' }}>
                  <div style={{ ...glassStyle(c), borderRadius: 20, padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 14 }} className="sq-anim-in">
                    <ProgressRing pct={dailyPct} c={c} size={44} />
                    <div className="flex-1" style={{ minWidth: 0 }}>
                      <p style={{ fontSize: 14, fontWeight: 600, color: c.label }}>Daily progress</p>
                      <p style={{ fontSize: 12, marginTop: 2, color: c.labelSecondary }}>{completedCount} of {quests.length} quests done</p>
                    </div>
                    {streak > 0 && (
                        <div className="flex flex-col items-center flex-shrink-0" style={{ paddingLeft: 12, borderLeft: `1px solid ${c.separator}` }}>
                          <QuestSvg.flame width={15} height={15} strokeWidth={1.75} style={{ color: c.orange }} />
                          <span className="sq-mono" style={{ fontSize: 12, fontWeight: 700, color: c.orange, marginTop: 2 }}>{streak}d</span>
                        </div>
                    )}
                  </div>

                  <div>
                    <p style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, color: c.labelSecondary, marginBottom: 8, paddingLeft: 2 }}>
                      Today
                    </p>
                    <div className="flex flex-col gap-2.5">
                      {quests.map((quest, i) => (
                          <button key={quest.id}
                                  onClick={() => { if (!quest.completed) handleQuestClick(quest); }}
                                  className="sq-anim-in"
                                  style={{
                                    ...glassStyle(c),
                                    width: '100%', display: 'flex', alignItems: 'center', gap: 12,
                                    padding: '14px 14px', textAlign: 'left', borderRadius: 20,
                                    opacity: quest.completed ? 0.7 : 1,
                                    animationDelay: `${Math.min(i, 8) * 0.035}s`,
                                  }}>
                            <div className="relative flex-shrink-0">
                              <QuestRowIcon questId={quest.id} c={c} muted={quest.completed} />
                              {quest.completed && (
                                  <div className="sq-icon-fff" style={{ position: 'absolute', bottom: -3, right: -3, width: 16, height: 16, borderRadius: '50%', background: c.green, border: `2px solid ${c.bgElevated}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <CheckIcon size={8} />
                                  </div>
                              )}
                            </div>
                            <span style={{ flex: 1, fontSize: 15, fontWeight: 500, lineHeight: 1.3, color: quest.completed ? c.labelTertiary : c.label, textDecoration: quest.completed ? 'line-through' : 'none' }}>
                        {quest.text}
                              {!quest.completed && quest.progress > 0 && (
                                  <span className="sq-mono" style={{ marginLeft: 8, fontSize: 10, fontWeight: 700, color: c.blue, background: c.fill, padding: '2px 7px', borderRadius: 999 }}>
                            In progress
                          </span>
                              )}
                      </span>
                            <div className="flex items-center gap-2 flex-shrink-0">
                              {quest.completed && proofImages[quest.id] && (
                                  <button onClick={e => { e.stopPropagation(); setViewingProof(proofImages[quest.id]); }}
                                          style={{ width: 26, height: 26, borderRadius: 10, overflow: 'hidden', flexShrink: 0 }}>
                                    <img src={proofImages[quest.id]} alt="proof" className="w-full h-full object-cover" />
                                  </button>
                              )}
                              <span className="sq-mono" style={{ fontSize: 12, fontWeight: 700, color: quest.completed ? c.green : c.labelTertiary }}>+{quest.xp}</span>
                              {!quest.completed && <span className="sq-icon-tap"><ChevronIcon color={c.labelTertiary} /></span>}
                            </div>
                          </button>
                      ))}
                    </div>
                  </div>

                  {allDone && (
                      <div style={{ ...glassStyle(c), borderRadius: 20, padding: 24, textAlign: 'center' }} className="sq-anim-in">
                        <TrophyIcon size={26} style={{ color: c.orange, margin: '0 auto 10px' }} />
                        <p style={{ fontSize: 15, fontWeight: 600, color: c.label }}>All quests complete</p>
                        <p style={{ fontSize: 13, marginTop: 4, color: c.labelSecondary }}>Rest up. New quests when the timer hits zero.</p>
                      </div>
                  )}

                  {quests.length > 0 && (
                      <p style={{ fontSize: 11, fontWeight: 500, textAlign: 'center', color: c.labelTertiary, padding: '0 16px 8px' }}>
                        Checked on your device — nothing you record ever leaves your phone.
                      </p>
                  )}
                </div>
              </div>
          )}

          {isMounted && username && !showLeaderboard && !completionQuest && !detailQuest && !proofModal && (
              <TabBar activeTab={activeTab} onChange={setActiveTab} c={c} />
          )}
        </div>
      </div>
  );
}
