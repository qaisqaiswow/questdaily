"use client";

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { pipeline, env } from '@huggingface/transformers';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import * as exifr from 'exifr';
import { initializeApp, getApps } from 'firebase/app';
import {
  getAuth, onAuthStateChanged, signOut, deleteUser, getRedirectResult,
  createUserWithEmailAndPassword, signInWithEmailAndPassword,
  GoogleAuthProvider, signInWithRedirect,
  updateEmail, updatePassword, reauthenticateWithCredential, EmailAuthProvider,
} from 'firebase/auth';
import {
  getFirestore,
  doc,
  getDoc,
  setDoc,
  deleteDoc,
  getDocs,
  onSnapshot,
  collection,
  query,
  orderBy,
  limit
} from 'firebase/firestore';
import {
  Sun, Moon, CheckSquare, History as HistoryIcon, Settings as SettingsIcon,
  Dumbbell, PersonStanding, Bike, Footprints, CircleDot, Timer, ChevronsUp,
  Droplet, Leaf, Utensils, CookingPot, Flower2, Move, Zap, Flame,
  Pencil, Snowflake, Wind, Waves, Target, GlassWater, CupSoda,
  LogOut, Eye, EyeOff, Lock, AtSign, Camera, Trash2, ShieldCheck, Sparkles, ListChecks,
  Bell, Download, UserCog, KeyRound, ChevronRight, CircleUserRound,
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

const firebaseApp = getApps().length
  ? getApps()[0]
  : initializeApp(firebaseConfig);

const auth = getAuth(firebaseApp);

const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({
  prompt: 'select_account'
});

const db = getFirestore(firebaseApp);

const LEADERBOARD_COLLECTION = 'leaderboard';
const LEADERBOARD_SIZE = 100;

async function syncLeaderboardEntry(
  uid,
  { username, level, xp, totalXpEarned, streak }
) {
  if (!uid || !username) {
    return {
      ok: false,
      error: 'Not signed in yet — try again in a moment.'
    };
  }

  try {
    await setDoc(
      doc(db, LEADERBOARD_COLLECTION, uid),
      {
        username: username.slice(0, 20),
        level,
        xp,
        totalXpEarned,
        ...(streak !== undefined ? { streak } : {}),
        updatedAt: Date.now(),
      },
      { merge: true }
    );

    return { ok: true };
  } catch (err) {
    console.error('Leaderboard sync failed:', err);

    return {
      ok: false,
      error: `${err.code || 'error'}: ${err.message}`
    };
  }
}

async function fetchRemoteProfile(uid) {
  if (!uid) return null;

  try {
    const snap = await getDoc(
      doc(db, LEADERBOARD_COLLECTION, uid)
    );

    return snap.exists() ? snap.data() : null;
  } catch (err) {
    console.error('Profile fetch failed:', err);
    return null;
  }
}

async function syncHistoryEntry(uid, entry) {
  if (!uid) return;

  try {
    await setDoc(
      doc(db, 'users', uid, 'history', entry.id),
      entry
    );
  } catch (err) {
    console.error('History sync failed:', err);
  }
}

async function fetchRemoteHistory(uid) {
  if (!uid) return [];

  try {
    const q = query(
      collection(db, 'users', uid, 'history'),
      orderBy('ts', 'desc'),
      limit(500)
    );

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

  if (!USERNAME_RE.test(trimmed)) {
    return 'Username must be 3-20 characters: letters, numbers, and underscores only.';
  }

  return null;
}

function validatePassword(raw) {
  if (raw.length < 6) {
    return 'Password must be at least 6 characters.';
  }

  if (raw.length > 128) {
    return 'Password is too long.';
  }

  return null;
}

function friendlyAuthError(err) {
  const code = err?.code || '';

  if (code === 'auth/email-already-in-use') {
    return 'That username is already taken.';
  }

  if (code === 'auth/weak-password') {
    return 'Password must be at least 6 characters.';
  }

  if (
    code === 'auth/wrong-password' ||
    code === 'auth/invalid-credential' ||
    code === 'auth/invalid-login-credentials'
  ) {
    return 'Incorrect username or password.';
  }

  if (code === 'auth/user-not-found') {
    return 'No account found with that username.';
  }

  if (code === 'auth/too-many-requests') {
    return 'Too many attempts — try again in a bit.';
  }

  if (code === 'auth/network-request-failed') {
    return "Can't reach the server. Check your connection.";
  }

  if (code === 'auth/requires-recent-login') {
    return 'For security, please log out and log back in, then try again.';
  }

  if (code === 'auth/invalid-email') {
    return "That username can't be used. Try a different one.";
  }

  if (code === 'auth/unauthorized-domain') {
    return 'Google sign-in is blocked for this site. Add this deployed domain in Firebase Authentication → Settings → Authorized domains.';
  }

  if (code === 'auth/operation-not-allowed') {
    return 'Google sign-in is disabled in Firebase. Enable the Google provider in Authentication → Sign-in method.';
  }

  if (code === 'auth/popup-blocked') {
    return 'Your browser blocked the sign-in window. Try again.';
  }

  if (
    code === 'auth/popup-closed-by-user' ||
    code === 'auth/cancelled-popup-request'
  ) {
    return 'Google sign-in was cancelled. Try again.';
  }

  if (code === 'auth/account-exists-with-different-credential') {
    return "That Google account's email is already tied to a different sign-in method here.";
  }

  if (
    code === 'permission-denied' ||
    code === 'firestore/permission-denied'
  ) {
    return "Your database's security rules are blocking this — the 'usernames', 'users', and 'leaderboard' collections in Firestore need read/write rules set up.";
  }

  return err?.message
    ? `Something went wrong: ${err.message}`
    : 'Something went wrong. Please try again.';
}

async function signUpAccount(usernameRaw, password) {
  const username = usernameRaw.trim();
  const usernameLower = username.toLowerCase();
  const usernameDoc = doc(db, 'usernames', usernameLower);

  let existing;

  try {
    existing = await getDoc(usernameDoc);
  } catch (err) {
    console.error('Username availability check failed:', err);
    throw err;
  }

  if (existing.exists()) {
    const err = new Error('That username is already taken.');
    err.code = 'auth/email-already-in-use';
    throw err;
  }

  const cred = await createUserWithEmailAndPassword(
    auth,
    usernameToEmail(usernameLower),
    password
  );

  try {
    await Promise.all([
      setDoc(usernameDoc, {
        uid: cred.user.uid,
        username
      }),
      setDoc(doc(db, 'users', cred.user.uid), {
        username,
        createdAt: Date.now()
      }),
    ]);
  } catch (err) {
    console.error('Failed to finish account setup:', err);
    throw err;
  }

  return {
    uid: cred.user.uid,
    username
  };
}

async function logInAccount(usernameRaw, password) {
  const username = usernameRaw.trim();
  const usernameLower = username.toLowerCase();

  const cred = await signInWithEmailAndPassword(
    auth,
    usernameToEmail(usernameLower),
    password
  );

  let displayUsername = username;

  try {
    const snap = await getDoc(
      doc(db, 'usernames', usernameLower)
    );

    if (snap.exists() && snap.data()?.username) {
      displayUsername = snap.data().username;
    }
  } catch {
    // non-fatal
  }

  return {
    uid: cred.user.uid,
    username: displayUsername
  };
}

async function logOutAccount() {
  await signOut(auth);
}

async function mintUsernameForGoogleUser(user) {
  const rawBase =
    user.displayName ||
    (user.email ? user.email.split('@')[0] : 'Player');

  let base = rawBase
    .replace(/[^a-zA-Z0-9_]/g, '')
    .slice(0, 16);

  if (base.length < 3) {
    base = (base + 'Player').slice(0, 16);
  }

  let display = base;
  let usernameLower = base.toLowerCase();

  for (let attempt = 0; attempt < 8; attempt++) {
    const snap = await getDoc(
      doc(db, 'usernames', usernameLower)
    );

    if (!snap.exists()) break;

    const suffix = String(
      Math.floor(1000 + Math.random() * 9000)
    );

    display = `${base}${suffix}`.slice(0, 20);
    usernameLower = display.toLowerCase();
  }

  await Promise.all([
    setDoc(
      doc(db, 'usernames', usernameLower),
      {
        uid: user.uid,
        username: display
      }
    ),

    setDoc(
      doc(db, 'users', user.uid),
      {
        username: display,
        createdAt: Date.now(),
        photoURL: user.photoURL || null
      }
    ),
  ]);

  if (user.photoURL) {
    await setDoc(
      doc(db, 'leaderboard', user.uid),
      {
        photoURL: user.photoURL
      },
      { merge: true }
    ).catch(() => {});
  }

  return display;
}

async function signInWithGoogle() {
  googleProvider.setCustomParameters({
    prompt: 'select_account'
  });

  await signInWithRedirect(auth, googleProvider);
}

function getAuthProviderLabel(user) {
  if (!user) return null;

  const ids = (user.providerData || []).map(
    p => p.providerId
  );

  if (ids.includes('google.com')) {
    return 'google';
  }

  if (ids.includes('password')) {
    return 'password';
  }

  return 'other';
}

async function updateUsername({
  newUsernameRaw,
  oldUsername,
  currentPassword
}) {
  const currentUser = auth.currentUser;

  if (!currentUser) {
    throw Object.assign(
      new Error('You need to be signed in to do that.'),
      { code: 'auth/no-current-user' }
    );
  }

  const newUsername = newUsernameRaw.trim();
  const newLower = newUsername.toLowerCase();
  const oldLower = (oldUsername || '').toLowerCase();

  const usesPassword =
    getAuthProviderLabel(currentUser) === 'password';

  if (newLower === oldLower) {
    await Promise.all([
      setDoc(
        doc(db, 'usernames', oldLower),
        {
          uid: currentUser.uid,
          username: newUsername
        },
        { merge: true }
      ),

      setDoc(
        doc(db, 'users', currentUser.uid),
        {
          username: newUsername
        },
        { merge: true }
      ),

      setDoc(
        doc(db, 'leaderboard', currentUser.uid),
        {
          username: newUsername
        },
        { merge: true }
      ),
    ]);

    return newUsername;
  }

  const takenSnap = await getDoc(
    doc(db, 'usernames', newLower)
  );

  if (takenSnap.exists()) {
    throw Object.assign(
      new Error('That username is already taken.'),
      { code: 'auth/email-already-in-use' }
    );
  }

  if (usesPassword) {
    const newEmail = usernameToEmail(newLower);

    try {
      await updateEmail(currentUser, newEmail);
    } catch (err) {
      if (err?.code === 'auth/requires-recent-login') {
        if (!currentPassword) throw err;

        await reauthenticateWithCredential(
          currentUser,
          EmailAuthProvider.credential(
            usernameToEmail(oldLower),
            currentPassword
          )
        );

        await updateEmail(currentUser, newEmail);
      } else {
        throw err;
      }
    }
  }

  await Promise.all([
    setDoc(
      doc(db, 'usernames', newLower),
      {
        uid: currentUser.uid,
        username: newUsername
      }
    ),

    setDoc(
      doc(db, 'users', currentUser.uid),
      {
        username: newUsername
      },
      { merge: true }
    ),

    setDoc(
      doc(db, 'leaderboard', currentUser.uid),
      {
        username: newUsername
      },
      { merge: true }
    ),
  ]);

  await deleteDoc(
    doc(db, 'usernames', oldLower)
  ).catch(() => {});

  return newUsername;
}

async function changeAccountPassword(
  currentPassword,
  newPassword
) {
  const currentUser = auth.currentUser;

  if (!currentUser?.email) {
    throw Object.assign(
      new Error('You need to be signed in to do that.'),
      { code: 'auth/no-current-user' }
    );
  }

  await reauthenticateWithCredential(
    currentUser,
    EmailAuthProvider.credential(
      currentUser.email,
      currentPassword
    )
  );

  await updatePassword(currentUser, newPassword);
}

const AVATAR_SIZE = 256;
const AVATAR_QUALITY = 0.8;

function resizeImageToSquareDataUrl(file) {
  return new Promise((resolve, reject) => {
    if (!file.type?.startsWith('image/')) {
      reject(new Error('Please choose an image file.'));
      return;
    }

    const reader = new FileReader();

    reader.onload = e => {
      const img = new Image();

      img.onload = () => {
        const side = Math.min(
          img.width,
          img.height
        );

        const sx = (img.width - side) / 2;
        const sy = (img.height - side) / 2;

        const canvas = document.createElement('canvas');

        canvas.width = AVATAR_SIZE;
        canvas.height = AVATAR_SIZE;

        const ctx = canvas.getContext('2d');

        ctx.drawImage(
          img,
          sx,
          sy,
          side,
          side,
          0,
          0,
          AVATAR_SIZE,
          AVATAR_SIZE
        );

        resolve(
          canvas.toDataURL(
            'image/jpeg',
            AVATAR_QUALITY
          )
        );
      };

      img.onerror = () =>
        reject(
          new Error(
            "That file doesn't look like a valid image."
          )
        );

      img.src = e.target.result;
    };

    reader.onerror = () =>
      reject(new Error('Could not read that file.'));

    reader.readAsDataURL(file);
  });
}

async function updateProfilePhoto(uid, dataUrl) {
  await Promise.all([
    setDoc(
      doc(db, 'users', uid),
      {
        photoURL: dataUrl
      },
      { merge: true }
    ),

    setDoc(
      doc(db, 'leaderboard', uid),
      {
        photoURL: dataUrl
      },
      { merge: true }
    ),
  ]);
}

async function deleteAccountData(
  uid,
  usernameLower
) {
  let historyDocs = [];

  try {
    const historySnap = await getDocs(
      collection(db, 'users', uid, 'history')
    );

    historyDocs = historySnap.docs;
  } catch (err) {
    console.error(
      'Could not list history for deletion:',
      err
    );
  }

  await Promise.all(
    historyDocs.map(d =>
      deleteDoc(d.ref).catch(() => {})
    )
  );

  const results = await Promise.allSettled([
    deleteDoc(doc(db, 'users', uid)),
    deleteDoc(doc(db, 'leaderboard', uid)),
    usernameLower
      ? deleteDoc(
          doc(db, 'usernames', usernameLower)
        )
      : Promise.resolve(),
  ]);

  const failure = results.find(
    r => r.status === 'rejected'
  );

  if (failure) {
    console.error(
      'Account data cleanup partially failed:',
      failure.reason
    );

    throw failure.reason;
  }
}

function useAuthUser() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [authError, setAuthError] = useState(null);

  useEffect(() => {
    let mounted = true;

    getRedirectResult(auth).catch(err => {
      if (!mounted) return;

      console.error(
        'Google redirect result failed:',
        err
      );

      if (
        err?.code &&
        err.code !== 'auth/no-auth-event'
      ) {
        setAuthError(
          friendlyAuthError(err)
        );
      }
    });

    const unsub = onAuthStateChanged(
      auth,
      u => {
        if (!mounted) return;

        setUser(u);
        setLoading(false);

        if (u) {
          setAuthError(null);
        }
      }
    );

    return () => {
      mounted = false;
      unsub();
    };
  }, []);

  return {
    user,
    loading,
    authError
  };
}

function useLeaderboard() {
  const [entries, setEntries] = useState([]);
  const [status, setStatus] =
    useState('loading');
  const [errorDetail, setErrorDetail] =
    useState(null);

  useEffect(() => {
    const q = query(
      collection(
        db,
        LEADERBOARD_COLLECTION
      ),
      orderBy(
        'totalXpEarned',
        'desc'
      ),
      limit(LEADERBOARD_SIZE)
    );

    const timeout = setTimeout(() => {
      setStatus(prev =>
        prev === 'loading'
          ? 'error'
          : prev
      );

      setErrorDetail(
        prev =>
          prev ||
          'Timed out waiting for a response. Check your Firebase configuration and Firestore rules.'
      );
    }, 8000);

    const unsub = onSnapshot(
      q,
      snap => {
        clearTimeout(timeout);

        setEntries(
          snap.docs.map(d => ({
            id: d.id,
            ...d.data()
          }))
        );

        setStatus('ready');
        setErrorDetail(null);
      },
      err => {
        clearTimeout(timeout);

        console.error(
          'Leaderboard listener error:',
          err
        );

        setStatus('error');

        setErrorDetail(
          `${err.code || 'error'}: ${err.message}`
        );
      }
    );

    return () => {
      clearTimeout(timeout);
      unsub();
    };
  }, []);

  return {
    entries,
    status,
    errorDetail
  };
}

let hapticsEnabled = true;

function setHapticsPref(enabled) {
  hapticsEnabled = enabled;

  try {
    localStorage.setItem(
      'sq_haptics_enabled',
      enabled
        ? 'true'
        : 'false'
    );
  } catch {}
}

function haptic(pattern = 10) {
  if (!hapticsEnabled) return;

  try {
    if (
      typeof navigator !== 'undefined' &&
      navigator.vibrate
    ) {
      navigator.vibrate(pattern);
    }
  } catch {}
}

function getLevelTitle(level) {
  if (level < 5) return 'Novice';
  if (level < 10) return 'Apprentice';
  if (level < 20) return 'Skilled';
  if (level < 35) return 'Veteran';

  return 'Legendary';
}

env.allowLocalModels = false;

let classifierPromise = null;

function getClassifier(onProgress) {
  if (!classifierPromise) {
    classifierPromise = pipeline(
      'zero-shot-image-classification',
      'Xenova/siglip-base-patch16-224',
      {
        dtype: 'q8',
        ...(onProgress
          ? {
              progress_callback:
                onProgress
            }
          : {})
      }
    ).catch(error => {
      classifierPromise = null;
      throw error;
    });
  }

  return classifierPromise;
}

let staticClassifierPromise = null;

function getStaticClassifier(onProgress) {
  if (!staticClassifierPromise) {
    staticClassifierPromise = pipeline(
      'zero-shot-image-classification',
      'Xenova/siglip-large-patch16-256',
      onProgress
        ? {
            progress_callback:
              onProgress
          }
        : undefined
    ).catch(error => {
      staticClassifierPromise = null;
      throw error;
    });
  }

  return staticClassifierPromise;
}

const SOURCE_CAMERA_LABEL =
  'a real unedited photo taken with a phone camera';

const SOURCE_DOWNLOADED_LABELS = [
  'a professional stock photography image',
  'a screenshot or image saved from a website',
  'an image downloaded from a search engine or social media',
];

const SOURCE_LABELS = [
  SOURCE_CAMERA_LABEL,
  ...SOURCE_DOWNLOADED_LABELS
];

let landmarkerPromise = null;

function getPoseLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision =
        await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );

      return PoseLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            delegate: 'GPU',
          },

          runningMode: 'VIDEO',
          numPoses: 1,
        }
      );
    })().catch(error => {
      landmarkerPromise = null;
      throw error;
    });
  }

  return landmarkerPromise;
}

const LM = {
  L_SHOULDER: 11,
  R_SHOULDER: 12,
  L_ELBOW: 13,
  R_ELBOW: 14,
  L_WRIST: 15,
  R_WRIST: 16,
  L_HIP: 23,
  R_HIP: 24,
  L_KNEE: 25,
  R_KNEE: 26,
  L_ANKLE: 27,
  R_ANKLE: 28,
};

const POSE_CONNECTIONS = [
  [11, 12],
  [11, 23],
  [12, 24],
  [23, 24],

  [11, 13],
  [13, 15],

  [12, 14],
  [14, 16],

  [23, 25],
  [25, 27],
  [27, 29],
  [27, 31],
  [29, 31],

  [24, 26],
  [26, 28],
  [28, 30],
  [28, 32],
  [30, 32],

  [0, 1],
  [1, 2],
  [2, 3],
  [3, 7],

  [0, 4],
  [4, 5],
  [5, 6],
  [6, 8],

  [9, 10]
];

function angleBetween(a, b, c) {
  if (!a || !b || !c) return null;

  const v1 = {
    x: a.x - b.x,
    y: a.y - b.y
  };

  const v2 = {
    x: c.x - b.x,
    y: c.y - b.y
  };

  const dot =
    v1.x * v2.x +
    v1.y * v2.y;

  const mag1 = Math.hypot(
    v1.x,
    v1.y
  );

  const mag2 = Math.hypot(
    v2.x,
    v2.y
  );

  if (
    mag1 === 0 ||
    mag2 === 0
  ) {
    return null;
  }

  const cos = Math.min(
    1,
    Math.max(
      -1,
      dot / (mag1 * mag2)
    )
  );

  return (
    (Math.acos(cos) * 180) /
    Math.PI
  );
}

function visiblePt(
  pt,
  minVis = 0.5
) {
  return (
    pt &&
    (
      pt.visibility === undefined ||
      pt.visibility >= minVis
    )
  );
}

function pickSide(
  lms,
  leftKeys,
  rightKeys
) {
  const left =
    leftKeys.map(k => lms[k]);

  const right =
    rightKeys.map(k => lms[k]);

  const leftOk =
    left.every(p =>
      visiblePt(p)
    );

  const rightOk =
    right.every(p =>
      visiblePt(p)
    );

  if (
    leftOk &&
    rightOk
  ) {
    return left.map(
      (p, i) => ({
        x:
          (p.x +
            right[i].x) /
          2,

        y:
          (p.y +
            right[i].y) /
          2,

        visibility: Math.min(
          p.visibility ?? 1,
          right[i].visibility ??
            1
        )
      })
    );
  }

  if (leftOk) return left;
  if (rightOk) return right;

  return null;
}

const REP_METRICS = {
  q1: lms => {
    const p = pickSide(
      lms,
      [
        LM.L_SHOULDER,
        LM.L_ELBOW,
        LM.L_WRIST
      ],
      [
        LM.R_SHOULDER,
        LM.R_ELBOW,
        LM.R_WRIST
      ]
    );

    return p
      ? angleBetween(
          p[0],
          p[1],
          p[2]
        )
      : null;
  },

  q2: lms => {
    const p = pickSide(
      lms,
      [
        LM.L_HIP,
        LM.L_KNEE,
        LM.L_ANKLE
      ],
      [
        LM.R_HIP,
        LM.R_KNEE,
        LM.R_ANKLE
      ]
    );

    return p
      ? angleBetween(
          p[0],
          p[1],
          p[2]
        )
      : null;
  },

  q4: lms => {
    const p = pickSide(
      lms,
      [
        LM.L_SHOULDER,
        LM.L_ELBOW,
        LM.L_WRIST
      ],
      [
        LM.R_SHOULDER,
        LM.R_ELBOW,
        LM.R_WRIST
      ]
    );

    return p
      ? angleBetween(
          p[0],
          p[1],
          p[2]
        )
      : null;
  },

  q12: lms => {
    const p = pickSide(
      lms,
      [
        LM.L_SHOULDER,
        LM.L_HIP,
        LM.L_KNEE
      ],
      [
        LM.R_SHOULDER,
        LM.R_HIP,
        LM.R_KNEE
      ]
    );

    return p
      ? angleBetween(
          p[0],
          p[1],
          p[2]
        )
      : null;
  },

  q17: lms => {
    const hips = [
      lms[LM.L_HIP],
      lms[LM.R_HIP]
    ].filter(p =>
      visiblePt(p)
    );

    if (!hips.length)
      return null;

    return hips.reduce(
      (s, p) =>
        s + p.y,
      0
    ) / hips.length;
  },

  q23: lms => {
    const p = pickSide(
      lms,
      [
        LM.L_HIP,
        LM.L_KNEE,
        LM.L_ANKLE
      ],
      [
        LM.R_HIP,
        LM.R_KNEE,
        LM.R_ANKLE
      ]
    );

    return p
      ? angleBetween(
          p[0],
          p[1],
          p[2]
        )
      : null;
  },
};

const REP_CONFIG = {
  q1: {
    mode: 'angle',
    downThreshold: 100,
    upThreshold: 155,
    cueDown:
      'Lower into the pushup',
    cueUp:
      'Push back up to full extension'
  },

  q2: {
    mode: 'angle',
    downThreshold: 110,
    upThreshold: 160,
    cueDown:
      'Squat down',
    cueUp:
      'Stand back up'
  },

  q4: {
    mode: 'angle',
    downThreshold: 80,
    upThreshold: 150,
    cueDown:
      'Pull up to the bar',
    cueUp:
      'Lower to a full hang'
  },

  q12: {
    mode: 'angle',
    downThreshold: 110,
    upThreshold: 150,
    cueDown:
      'Crunch up',
    cueUp:
      'Lower back down'
  },

  q17: {
    mode: 'height',
    downThreshold: 0.02,
    upThreshold: 0.005,
    cueDown:
      'Jump!',
    cueUp:
      'Land'
  },

  q23: {
    mode: 'angle',
    downThreshold: 110,
    upThreshold: 160,
    cueDown:
      'Lower into the lunge',
    cueUp:
      'Return to standing'
  },
};

const MIN_REP_MS = 500;

function createPoseRepState(
  initialReps = 0
) {
  return {
    phase: 'up',
    reps: initialReps,
    baselineY: null,
    lastRepAt: 0,
    smoothed: null
  };
}

function updatePoseRepState(
  state,
  questId,
  landmarks,
  now
) {
  const config =
    REP_CONFIG[questId];

  const metricFn =
    REP_METRICS[questId];

  if (
    !config ||
    !metricFn ||
    !landmarks
  ) {
    return {
      reps: state.reps,
      phase: state.phase,
      cue: 'Move into frame',
      counted: false
    };
  }

  const raw =
    metricFn(landmarks);

  if (raw === null) {
    return {
      reps: state.reps,
      phase: state.phase,
      cue:
        'Move fully into frame',
      counted: false
    };
  }

  state.smoothed =
    state.smoothed === null
      ? raw
      : state.smoothed * 0.6 +
        raw * 0.4;

  const value =
    state.smoothed;

  let counted = false;

  if (
    config.mode === 'height'
  ) {
    if (
      state.baselineY ===
      null
    ) {
      state.baselineY = value;
    } else {
      state.baselineY =
        state.baselineY *
          0.98 +
        value * 0.02;
    }

    const jumpHeight =
      state.baselineY -
      value;

    if (
      state.phase === 'up' &&
      jumpHeight >=
        config.downThreshold
    ) {
      state.phase = 'down';
    } else if (
      state.phase === 'down' &&
      jumpHeight <=
        config.upThreshold
    ) {
      if (
        now -
          state.lastRepAt >=
        MIN_REP_MS
      ) {
        state.reps += 1;
        state.lastRepAt =
          now;
        counted = true;
      }

      state.phase = 'up';
    }
  } else {
    if (
      state.phase === 'up' &&
      value <=
        config.downThreshold
    ) {
      state.phase = 'down';
    } else if (
      state.phase === 'down' &&
      value >=
        config.upThreshold
    ) {
      if (
        now -
          state.lastRepAt >=
        MIN_REP_MS
      ) {
        state.reps += 1;
        state.lastRepAt =
          now;
        counted = true;
      }

      state.phase = 'up';
    }
  }

  const cue =
    state.phase === 'up'
      ? config.cueDown
      : config.cueUp;

  return {
    reps: state.reps,
    phase: state.phase,
    cue,
    counted
  };
}

const QUEST_LABELS = {
  q1: {
    type: 'reps',
    activity: [
      'person doing pushups on floor',
      'pushup exercise'
    ],
    label: 'doing pushups',
    bodyParts: [
      'Chest',
      'Triceps',
      'Shoulders',
      'Core'
    ]
  },

  q2: {
    type: 'reps',
    activity: [
      'person doing squats exercise',
      'squat workout legs bent'
    ],
    label: 'doing squats',
    bodyParts: [
      'Quads',
      'Hamstrings',
      'Glutes',
      'Core'
    ]
  },

  q4: {
    type: 'reps',
    activity: [
      'person doing pullups on bar',
      'pullup bar exercise'
    ],
    label: 'doing pullups',
    bodyParts: [
      'Lats',
      'Upper Back',
      'Biceps',
      'Forearms'
    ]
  },

  q12: {
    type: 'reps',
    activity: [
      'person doing situps or crunches',
      'abdominal exercise on floor'
    ],
    label: 'doing situps',
    bodyParts: [
      'Abs',
      'Obliques',
      'Hip Flexors'
    ]
  },

  q17: {
    type: 'reps',
    activity: [
      'person jumping rope',
      'skipping rope exercise'
    ],
    label: 'jumping rope',
    bodyParts: [
      'Calves',
      'Quads',
      'Shoulders',
      'Cardio'
    ]
  },

  q23: {
    type: 'reps',
    activity: [
      'person doing lunges exercise',
      'lunge workout legs split stance'
    ],
    label: 'doing lunges',
    bodyParts: [
      'Quads',
      'Glutes',
      'Hamstrings'
    ]
  },

  q3: {
    type: 'map',
    activity: [
      'gps tracking map route screenshot',
      'fitness tracker map running route'
    ],
    label: 'running map screenshot'
  },

  q16: {
    type: 'map',
    activity: [
      'gps tracking map route screenshot',
      'cycling route map on phone screen'
    ],
    label: 'cycling map screenshot'
  },

  q5: {
    type: 'map',
    activity: [
      'gps tracking map route screenshot',
      'walking route map tracker'
    ],
    label: 'walking map screenshot'
  },

  q22: {
    type: 'map',
    activity: [
      'gps tracking map route screenshot',
      'step counter fitness app screenshot'
    ],
    label: 'step tracking map'
  },

  q9: {
    type: 'food',
    activity: [
      'healthy food meal salad vegetables on a plate',
      'nutritious meal in a bowl'
    ],
    label: 'plate of healthy food'
  },

  q19: {
    type: 'food',
    activity: [
      'clean healthy meal on plate',
      'plate of vegetables and whole foods'
    ],
    label: 'plate of clean food'
  },

  q20: {
    type: 'food',
    activity: [
      'cooked food on a plate',
      'homemade meal in a bowl or plate'
    ],
    label: 'cooked meal'
  },

  q14: {
    type: 'food',
    activity: [
      'glass of green smoothie',
      'blended green juice drink'
    ],
    label: 'green smoothie'
  },

  q6: {
    type: 'food',
    activity: [
      'glass of water',
      'reusable water bottle filled'
    ],
    label: 'water bottle'
  },

  q7: {
    type: 'action',
    activity: [
      'person meditating cross-legged',
      'mindfulness exercise'
    ],
    label: 'meditating'
  },

  q8: {
    type: 'action',
    activity: [
      'person stretching muscles',
      'yoga stretch pose'
    ],
    label: 'stretching',
    bodyParts: [
      'Full Body',
      'Flexibility'
    ]
  },

  q10: {
    type: 'action',
    activity: [
      'person sleeping in bed',
      'person resting in bed eyes closed'
    ],
    label: 'getting good sleep'
  },

  q11: {
    type: 'action',
    activity: [
      'person doing jumping jacks or burpees'
    ],
    label: 'doing cardio',
    bodyParts: [
      'Cardio',
      'Full Body'
    ]
  },

  q13: {
    type: 'action',
    activity: [
      'handwriting in notebook or journal'
    ],
    label: 'journaling'
  },

  q15: {
    type: 'action',
    activity: [
      'person doing plank exercise',
      'plank position core exercise'
    ],
    label: 'holding a plank',
    bodyParts: [
      'Core',
      'Shoulders'
    ]
  },

  q18: {
    type: 'action',
    activity: [
      'shower running water',
      'bathroom shower head with water'
    ],
    label: 'in the shower'
  },

  q21: {
    type: 'action',
    activity: [
      'person breathing deeply eyes closed'
    ],
    label: 'deep breathing'
  },

  q24: {
    type: 'action',
    activity: [
      'person sleeping in bed at night',
      'sleeping in dark bedroom'
    ],
    label: 'sleeping early'
  },

  q25: {
    type: 'action',
    activity: [
      'person in ice bath tub',
      'cold plunge tub with ice'
    ],
    label: 'in a cold plunge'
  },

  q26: {
    type: 'action',
    activity: [
      'person doing a wall sit exercise against a wall'
    ],
    label: 'holding a wall sit',
    bodyParts: [
      'Quads',
      'Core'
    ]
  },

  q27: {
    type: 'action',
    activity: [
      'person doing a glute bridge exercise on floor'
    ],
    label: 'holding a glute bridge',
    bodyParts: [
      'Glutes',
      'Core'
    ]
  },

  q28: {
    type: 'action',
    activity: [
      'person doing high knees exercise'
    ],
    label: 'doing high knees',
    bodyParts: [
      'Cardio',
      'Quads'
    ]
  },

  q29: {
    type: 'action',
    activity: [
      'person doing mountain climbers exercise'
    ],
    label: 'doing mountain climbers',
    bodyParts: [
      'Core',
      'Cardio'
    ]
  },

  q30: {
    type: 'action',
    activity: [
      'person doing a superman back exercise lying face down'
    ],
    label: 'holding a superman pose',
    bodyParts: [
      'Lower Back',
      'Glutes'
    ]
  },

  q31: {
    type: 'action',
    activity: [
      'person doing burpees exercise'
    ],
    label: 'doing burpees',
    bodyParts: [
      'Full Body',
      'Cardio'
    ]
  },
};

const MOVEMENT_ACTION_IDS =
  new Set([
    'q8',
    'q11',
    'q15',
    'q26',
    'q27',
    'q28',
    'q29',
    'q30',
    'q31'
  ]);

const ANTI_SPOOF_LABELS = [
  'a photo or video playing on a phone or tablet screen',
  'a laptop or computer monitor screen',
  'a printed photograph being held up to the camera',
  'a picture of a picture',
];

const SPOOF_BLOCK_THRESHOLD = 0.30;
const PASS_MARGIN = 0.12;

const getNegativeLabels = type => {
  const base = [
    'person sitting doing nothing',
    'person standing still straight',
    'random everyday object',
    ...ANTI_SPOOF_LABELS
  ];

  if (type === 'map') {
    return [
      ...base,
      'sweaty selfie face',
      'picture of running shoes',
      'treadmill machine indoors',
      'person running outside'
    ];
  }

  if (type === 'food') {
    return [
      ...base,
      'empty plate or bowl',
      'restaurant paper menu',
      'store product barcode',
      'person eating face'
    ];
  }

  return [
    ...base,
    'phone or computer screen'
  ];
};

const QUEST_POOL = [
  {
    id: 'q1',
    textTemplate: 'Do {n} pushups',
    xp: 50,
    reps: 20
  },

  {
    id: 'q2',
    textTemplate: 'Do {n} squats',
    xp: 45,
    reps: 30
  },

  {
    id: 'q3',
    textTemplate: 'Go for a {n}-minute run',
    xp: 75,
    duration: 180
  },

  {
    id: 'q4',
    textTemplate: 'Do {n} pullups',
    xp: 60,
    reps: 10
  },

  {
    id: 'q16',
    textTemplate: 'Do {n} minutes of cycling',
    xp: 55,
    duration: 180
  },

  {
    id: 'q17',
    textTemplate: 'Do {n} jumping rope reps',
    xp: 40,
    reps: 50
  },

  {
    id: 'q21',
    textTemplate: 'Do {n} minutes of deep breathing',
    xp: 30,
    duration: 180
  },

  {
    id: 'q23',
    textTemplate: 'Do {n} lunges',
    xp: 40,
    reps: 30
  },

  {
    id: 'q25',
    textTemplate: 'Do a {n}-minute ice bath or cold plunge',
    xp: 80,
    duration: 180
  },

  {
    id: 'q5',
    textTemplate: 'Walk outside for {n} minutes',
    xp: 35,
    duration: 180
  },

  {
    id: 'q7',
    textTemplate: 'Meditate for {n} minutes',
    xp: 45,
    duration: 180
  },

  {
    id: 'q8',
    textTemplate: 'Stretch for {n} minutes',
    xp: 35,
    duration: 180
  },

  {
    id: 'q11',
    textTemplate: 'Do {n} minutes of jumping jacks',
    xp: 30,
    duration: 180
  },

  {
    id: 'q12',
    textTemplate: 'Do {n} situps',
    xp: 40,
    reps: 20
  },

  {
    id: 'q15',
    textTemplate: 'Hold a plank for {n} minutes',
    xp: 50,
    duration: 180
  },

  {
    id: 'q26',
    textTemplate: 'Hold a wall sit for {n} minutes',
    xp: 40,
    duration: 180
  },

  {
    id: 'q27',
    textTemplate: 'Hold a glute bridge for {n} minutes',
    xp: 35,
    duration: 180
  },

  {
    id: 'q28',
    textTemplate: 'Do {n} minutes of high knees',
    xp: 35,
    duration: 180
  },

  {
    id: 'q29',
    textTemplate: 'Do {n} minutes of mountain climbers',
    xp: 40,
    duration: 180
  },

  {
    id: 'q30',
    textTemplate: 'Hold a superman pose for {n} minutes',
    xp: 35,
    duration: 180
  },

  {
    id: 'q31',
    textTemplate: 'Do {n} minutes of burpees',
    xp: 45,
    duration: 180
  },
];

function randomizeQuest(pool) {
  if (pool.reps) {
    const n =
      Math.floor(
        Math.random() * 55
      ) + 1;

    return {
      ...pool,
      reps: n,
      text:
        pool.textTemplate.replace(
          '{n}',
          n
        ),
      completed: false,
      progress: 0
    };
  }

  if (pool.duration) {
    const minutes =
      Math.floor(
        Math.random() * 5
      ) + 1;

    return {
      ...pool,
      duration:
        minutes * 60,
      text:
        pool.textTemplate.replace(
          '{n}',
          minutes
        ),
      completed: false,
      progress: 0
    };
  }

  return {
    ...pool,
    text: pool.textTemplate,
    completed: false,
    progress: 0
  };
}

const ONE_DAY_MS =
  24 * 60 * 60 * 1000;

const REQUIRED_PASSES = 4;
const PASS_THRESHOLD = 0.42;
const SCORE_SMOOTHING = 0.38;

function getMaxLabelScore(
  results,
  candidates
) {
  if (!candidates?.length)
    return 0;

  return results.reduce(
    (best, result) =>
      candidates.includes(
        result.label
      )
        ? Math.max(
            best,
            result.score
          )
        : best,
    0
  );
}

async function checkPhotoAuthenticity(
  file,
  dataUrl,
  classifierLabels
) {
  let hasCameraMetadata =
    false;

  try {
    const tags =
      await exifr.parse(
        file,
        [
          'Make',
          'Model'
        ]
      );

    hasCameraMetadata =
      Boolean(
        tags &&
        (
          tags.Make ||
          tags.Model
        )
      );
  } catch {}

  let cameraScore = 0;
  let downloadedScore = 0;

  try {
    const classifier =
      await getStaticClassifier();

    const results =
      await classifier(
        dataUrl,
        classifierLabels
      );

    cameraScore =
      getMaxLabelScore(
        results,
        [
          SOURCE_CAMERA_LABEL
        ]
      );

    downloadedScore =
      getMaxLabelScore(
        results,
        SOURCE_DOWNLOADED_LABELS
      );
  } catch {}

  const margin =
    downloadedScore -
    cameraScore;

  const modelThinksDownloaded =
    margin > 0.05;

  const suspicious =
    !hasCameraMetadata &&
    modelThinksDownloaded;

  const confidence =
    Math.round(
      Math.max(
        cameraScore,
        downloadedScore
      ) * 100
    );

  return {
    hasCameraMetadata,
    cameraScore,
    downloadedScore,
    suspicious,
    confidence
  };
}

const C = {
  light: {
    bg: '#F2F2F7',
    bgElevated: '#FFFFFF',
    bgSecondary: '#F2F2F7',
    bgTertiary: '#E5E5EA',

    label: '#000000',
    labelSecondary:
      'rgba(60,60,67,0.60)',
    labelTertiary:
      'rgba(60,60,67,0.30)',

    separator:
      'rgba(60,60,67,0.29)',

    fill:
      'rgba(120,120,128,0.12)',

    blue: '#007AFF',
    green: '#34C759',
    orange: '#FF9500',
    red: '#FF3B30',
    indigo: '#5856D6',
    teal: '#30B0C7',
    purple: '#AF52DE',
    yellow: '#FFCC00',
    pink: '#FF2D55',
    gray: '#8E8E93',

    glass:
      'rgba(255,255,255,0.40)',

    glassStrong:
      'rgba(255,255,255,0.62)',

    glassBorder:
      'rgba(255,255,255,0.72)',

    glassHighlight:
      'rgba(255,255,255,0.95)',

    glassShadow:
      '0 18px 50px -28px rgba(0,0,0,0.22)',
  },

  dark: {
    bg: '#000000',
    bgElevated: '#1C1C1E',
    bgSecondary: '#1C1C1E',
    bgTertiary: '#2C2C2E',

    label: '#FFFFFF',
    labelSecondary:
      'rgba(235,235,245,0.60)',

    labelTertiary:
      'rgba(235,235,245,0.30)',

    separator:
      'rgba(84,84,88,0.65)',

    fill:
      'rgba(120,120,128,0.20)',

    blue: '#0A84FF',
    green: '#30D158',
    orange: '#FF9F0A',
    red: '#FF453A',
    indigo: '#5E5CE6',
    teal: '#40C8E0',
    purple: '#BF5AF2',
    yellow: '#FFD60A',
    pink: '#FF375F',
    gray: '#98989D',

    glass:
      'rgba(28,28,30,0.42)',

    glassStrong:
      'rgba(28,28,30,0.66)',

    glassBorder:
      'rgba(255,255,255,0.12)',

    glassHighlight:
      'rgba(255,255,255,0.16)',

    glassShadow:
      '0 20px 55px -30px rgba(0,0,0,0.70)',
  },
};

const glassStyle = (
  c,
  strong = false
) => ({
  position: 'relative',
  overflow: 'hidden',

  background: `
    linear-gradient(
      145deg,
      ${c.glassHighlight} 0%,
      ${strong
        ? c.glassStrong
        : c.glass} 34%,
      ${strong
        ? c.glassStrong
        : c.glass} 100%
    )
  `,

  backdropFilter:
    'blur(30px) saturate(180%) contrast(105%)',

  WebkitBackdropFilter:
    'blur(30px) saturate(180%) contrast(105%)',

  border:
    `1px solid ${c.glassBorder}`,

  boxShadow: `
    inset 0 1px 0 ${c.glassHighlight},
    inset 0 -1px 0 rgba(255,255,255,0.06),
    ${c.glassShadow}
  `,

  transform:
    'translateZ(0)',
});

const SystemType = () => (
  <style>{`
    .sq-root,
    .sq-root * {
      font-family:
        -apple-system,
        BlinkMacSystemFont,
        "SF Pro Text",
        "Helvetica Neue",
        Arial,
        sans-serif;
    }

    .sq-mono,
    .sq-mono * {
      font-family:
        ui-monospace,
        "SF Mono",
        Menlo,
        monospace;

      font-variant-numeric:
        tabular-nums;
    }

    .sq-large-title {
      letter-spacing: -0.6px;
    }

    .sq-title {
      letter-spacing: -0.2px;
    }

    * {
      -webkit-tap-highlight-color:
        transparent;
    }

    .sq-root {
      -webkit-user-select: none;
      -moz-user-select: none;
      user-select: none;

      -webkit-touch-callout: none;

      touch-action:
        pan-x pan-y;
    }

    .sq-root input,
    .sq-root textarea {
      -webkit-user-select: text;
      -moz-user-select: text;
      user-select: text;
      touch-action: manipulation;
    }

    .sq-root,
    .sq-root :where(
      div,
      span,
      p,
      h1,
      h2,
      h3,
      button,
      a,
      input
    ) {
      transition:
        background-color .3s ease,
        border-color .3s ease,
        color .25s ease;
    }

    svg {
      transition:
        stroke .2s ease,
        fill .2s ease;
    }

    .sq-scroll {
      -webkit-overflow-scrolling:
        touch;
    }

    .sq-icon-fff svg {
      color: #fff;
    }

    @keyframes sq-fade-up {
      from {
        opacity: 0;
        transform:
          translate3d(0,7px,0);
      }

      to {
        opacity: 1;
        transform:
          translate3d(0,0,0);
      }
    }

    @keyframes sq-check-in {
      0% {
        opacity: 0;
        transform:
          scale(.5);
      }

      70% {
        opacity: 1;
        transform:
          scale(1.06);
      }

      100% {
        opacity: 1;
        transform:
          scale(1);
      }
    }

    .sq-anim-in {
      animation:
        sq-fade-up
        .32s
        cubic-bezier(.22,1,.36,1)
        both;

      will-change:
        transform,
        opacity;
    }

    .sq-anim-check {
      animation:
        sq-check-in
        .4s
        cubic-bezier(.22,1,.36,1)
        both;
    }

    button,
    a {
      transition:
        opacity .2s ease-out,
        transform .2s
          cubic-bezier(.22,1,.36,1),
        background-color .25s ease;
    }

    button:active {
      transform:
        scale(.96);
    }

    @keyframes sq-icon-pop {
      0% {
        opacity: 0;
        transform:
          scale(.6)
          rotate(-8deg);
      }

      60% {
        opacity: 1;
        transform:
          scale(1.12)
          rotate(2deg);
      }

      100% {
        opacity: 1;
        transform:
          scale(1)
          rotate(0deg);
      }
    }

    .sq-icon-tap {
      transition:
        transform .2s
        cubic-bezier(.34,1.56,.64,1);

      display:
        inline-flex;

      will-change:
        transform;
    }

    button:active
    .sq-icon-tap {
      transform:
        scale(1.2)
        rotate(-6deg);
    }

    .sq-icon-pop-in {
      animation:
        sq-icon-pop
        .4s
        cubic-bezier(.22,1,.36,1)
        both;
    }

    .sq-tab-icon {
      transition:
        transform .3s
        cubic-bezier(.34,1.56,.64,1);

      will-change:
        transform;
    }

    .sq-tab-icon-active {
      transform:
        scale(1.14)
        translateY(-1px);
    }

    @keyframes sq-drift-a {
      0%,100% {
        transform:
          translate3d(-6%,-4%,0)
          scale(1);
      }

      50% {
        transform:
          translate3d(8%,10%,0)
          scale(1.15);
      }
    }

    @keyframes sq-drift-b {
      0%,100% {
        transform:
          translate3d(10%,6%,0)
          scale(1.1);
      }

      50% {
        transform:
          translate3d(-8%,-8%,0)
          scale(.95);
      }
    }

    @keyframes sq-drift-c {
      0%,100% {
        transform:
          translate3d(-4%,8%,0)
          scale(.95);
      }

      50% {
        transform:
          translate3d(6%,-10%,0)
          scale(1.1);
      }
    }

    .sq-orb-a {
      animation:
        sq-drift-a
        26s
        ease-in-out
        infinite;

      will-change:
        transform;
    }

    .sq-orb-b {
      animation:
        sq-drift-b
        32s
        ease-in-out
        infinite;

      will-change:
        transform;
    }

    .sq-orb-c {
      animation:
        sq-drift-c
        22s
        ease-in-out
        infinite;

      will-change:
        transform;
    }

    @keyframes sq-glass-shimmer {
      0% {
        transform:
          translateX(-120%)
          skewX(-18deg);

        opacity: 0;
      }

      18% {
        opacity: .18;
      }

      45%,100% {
        transform:
          translateX(145%)
          skewX(-18deg);

        opacity: 0;
      }
    }

    .sq-glass {
      position:
        relative;

      isolation:
        isolate;
    }

    .sq-glass::after {
      content: '';

      position:
        absolute;

      inset:
        -40% -20%;

      z-index:
        -1;

      pointer-events:
        none;

      width:
        32%;

      background:
        linear-gradient(
          90deg,
          transparent,
          rgba(255,255,255,.42),
          transparent
        );

      filter:
        blur(10px);

      transform:
        translateX(-120%)
        skewX(-18deg);

      animation:
        sq-glass-shimmer
        8s ease-in-out
        infinite;
    }

    @keyframes sq-modal-in {
      from {
        opacity: 0;

        transform:
          translate3d(0,18px,0)
          scale(.985);
      }

      to {
        opacity: 1;

        transform:
          translate3d(0,0,0)
          scale(1);
      }
    }

    .sq-modal-in {
      animation:
        sq-modal-in
        .42s
        cubic-bezier(.22,1,.36,1)
        both;

      will-change:
        transform,
        opacity;
    }

    @keyframes sq-pulse-soft {
      0%,100% {
        transform:
          scale(1);

        opacity:
          .52;
      }

      50% {
        transform:
          scale(1.18);

        opacity:
          1;
      }
    }

    @keyframes sq-thinking-glow {
      0%,100% {
        box-shadow:
          0 0 0 0
          rgba(10,132,255,0),
          0 0 10px
          rgba(10,132,255,.12);
      }

      50% {
        box-shadow:
          0 0 0 7px
          rgba(10,132,255,0),
          0 0 18px
          rgba(10,132,255,.24);
      }
    }

    .sq-thinking-orbs {
      display:
        inline-flex;

      align-items:
        center;

      gap:
        6px;

      padding:
        7px 10px;

      border-radius:
        999px;

      background:
        rgba(255,255,255,.09);

      border:
        1px solid
        rgba(255,255,255,.13);

      backdrop-filter:
        blur(14px)
        saturate(170%);

      -webkit-backdrop-filter:
        blur(14px)
        saturate(170%);
    }

    .sq-thinking-orb {
      width:
        6px;

      height:
        6px;

      border-radius:
        50%;

      background:
        rgba(255,255,255,.95);

      animation:
        sq-pulse-soft
        1.15s
        ease-in-out
        infinite,

        sq-thinking-glow
        2.3s
        ease-in-out
        infinite;
    }

    .sq-thinking-orb:nth-child(2) {
      animation-delay:
        .16s,
        .16s;
    }

    .sq-thinking-orb:nth-child(3) {
      animation-delay:
        .32s,
        .32s;
    }

    .sq-control {
      min-height:
        46px;

      transition:
        transform .24s
          cubic-bezier(.22,1,.36,1),
        opacity .2s ease,
        background-color .2s ease,
        box-shadow .24s ease;
    }

    .sq-control:hover {
      transform:
        translateY(-1px);
    }

    .sq-control:active {
      transform:
        translateY(1px)
        scale(.985);
    }

    .sq-cam-shell {
      box-shadow:
        inset 0 1px 0
        rgba(255,255,255,.12),

        0 22px 55px -32px
        rgba(0,0,0,.85);
    }

    @keyframes sq-confetti-fall {
      0% {
        transform:
          translate(0,0)
          rotate(0deg);

        opacity: 1;
      }

      85% {
        opacity: 1;
      }

      100% {
        transform:
          translate(
            var(--sq-drift),
            115vh
          )
          rotate(
            var(--sq-rot)
          );

        opacity: 0;
      }
    }

    @media (
      prefers-reduced-motion: reduce
    ) {
      .sq-orb-a,
      .sq-orb-b,
      .sq-orb-c,
      .sq-thinking-orb {
        animation:
          none;
      }
    }
  `}</style>
);

const AmbientBackground =
  React.memo(
    function AmbientBackground({
      c
    }) {
      return (
        <div
          className="fixed inset-0 pointer-events-none z-0"
          aria-hidden="true"
          style={{
            overflow:
              'hidden'
          }}
        >
          <div
            className="sq-orb-a"
            style={{
              position:
                'absolute',
              top: '-10%',
              left: '-15%',
              width: '75%',
              height: '42%',
              borderRadius:
                '50%',
              background:
                c.blue,
              opacity: .16,
              filter:
                'blur(70px)'
            }}
          />

          <div
            className="sq-orb-b"
            style={{
              position:
                'absolute',
              top: '30%',
              right: '-20%',
              width: '70%',
              height: '46%',
              borderRadius:
                '50%',
              background:
                c.purple,
              opacity: .13,
              filter:
                'blur(80px)'
            }}
          />

          <div
            className="sq-orb-c"
            style={{
              position:
                'absolute',
              bottom: '-14%',
              left: '5%',
              width: '65%',
              height: '40%',
              borderRadius:
                '50%',
              background:
                c.teal,
              opacity: .13,
              filter:
                'blur(75px)'
            }}
          />
        </div>
      );
    }
  );

const ThinkingOrbs =
  React.memo(
    function ThinkingOrbs({
      label = 'Thinking…'
    }) {
      return (
        <div className="flex items-center gap-2">
          <div
            className="sq-thinking-orbs"
            aria-label={label}
          >
            <span className="sq-thinking-orb" />
            <span className="sq-thinking-orb" />
            <span className="sq-thinking-orb" />
          </div>

          <span
            style={{
              color:
                'rgba(255,255,255,.72)',
              fontSize: 11,
              fontWeight: 600
            }}
          >
            {label}
          </span>
        </div>
      );
    }
  );

const CheckIcon =
  React.memo(
    function CheckIcon({
      size = 12
    }) {
      return (
        <svg
          width={size}
          height={size}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="20 6 9 17 4 12" />
        </svg>
      );
    }
  );

const ChevronIcon =
  React.memo(
    function ChevronIcon({
      color
    }) {
      return (
        <svg
          width="8"
          height="14"
          viewBox="0 0 8 14"
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="1 1 7 7 1 13" />
        </svg>
      );
    }
  );

const BackChevron =
  React.memo(
    function BackChevron({
      color
    }) {
      return (
        <svg
          width="11"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke={color}
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="15 18 9 12 15 6" />
        </svg>
      );
    }
  );

const IOSShareIcon =
  React.memo(
    function IOSShareIcon() {
      return (
        <svg
          width="22"
          height="22"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8" />
          <polyline points="16 6 12 2 8 6" />
          <line
            x1="12"
            y1="2"
            x2="12"
            y2="15"
          />
        </svg>
      );
    }
  );

const IOSAddIcon =
  React.memo(
    function IOSAddIcon() {
      return (
        <svg
          width="22"
          height="22"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <rect
            x="3"
            y="3"
            width="18"
            height="18"
            rx="4"
            ry="4"
          />
          <line
            x1="12"
            y1="8"
            x2="12"
            y2="16"
          />
          <line
            x1="8"
            y1="12"
            x2="16"
            y2="12"
          />
        </svg>
      );
    }
  );

const AndroidMenuIcon =
  React.memo(
    function AndroidMenuIcon() {
      return (
        <svg
          width="22"
          height="22"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <circle
            cx="12"
            cy="5"
            r="2.2"
          />
          <circle
            cx="12"
            cy="12"
            r="2.2"
          />
          <circle
            cx="12"
            cy="19"
            r="2.2"
          />
        </svg>
      );
    }
  );

const AndroidAddIcon =
  React.memo(
    function AndroidAddIcon() {
      return (
        <svg
          width="22"
          height="22"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <rect
            x="5"
            y="2"
            width="14"
            height="20"
            rx="2"
            ry="2"
          />
          <line
            x1="12"
            y1="18"
            x2="12.01"
            y2="18"
          />
          <line
            x1="9"
            y1="11"
            x2="15"
            y2="11"
          />
          <line
            x1="12"
            y1="8"
            x2="12"
            y2="14"
          />
        </svg>
      );
    }
  );

const TrophyIcon =
  React.memo(
    function TrophyIcon({
      size = 16
    }) {
      return (
        <svg
          width={size}
          height={size}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.7"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M8 4h8v5a4 4 0 0 1-8 0V4Z" />
          <path d="M8 5H5a3 3 0 0 0 3 4" />
          <path d="M16 5h3a3 3 0 0 1-3 4" />
          <path d="M12 13v3" />
          <path d="M9 20h6" />
          <path d="M10 16h4l.5 4h-5l.5-4Z" />
        </svg>
      );
    }
  );

function GoogleIcon({
  size = 18
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <path
        fill="#4285F4"
        d="M21.35 12.27c0-.68-.06-1.34-.18-1.97H12v3.73h5.23a4.48 4.48 0 0 1-1.94 2.94v2.45h3.14c1.84-1.69 2.92-4.18 2.92-7.15Z"
      />
      <path
        fill="#34A853"
        d="M12 21.75c2.63 0 4.84-.87 6.45-2.33l-3.14-2.45c-.87.58-1.98.93-3.31.93-2.54 0-4.69-1.72-5.46-4.03H3.3v2.53A9.75 9.75 0 0 0 12 21.75Z"
      />
      <path
        fill="#FBBC05"
        d="M6.54 13.87A5.86 5.86 0 0 1 6.23 12c0-.65.11-1.29.31-1.87V7.6H3.3A9.76 9.76 0 0 0 2.25 12c0 1.58.38 3.07 1.05 4.4l3.24-2.53Z"
      />
      <path
        fill="#EA4335"
        d="M12 6.1c1.43 0 2.71.49 3.72 1.45l2.79-2.79C16.83 3.2 14.62 2.25 12 2.25a9.75 9.75 0 0 0-8.7 5.35l3.24 2.53C7.31 7.82 9.46 6.1 12 6.1Z"
      />
    </svg>
  );
}

function CameraModal({
  quest,
  onConfirm,
  onCancel,
  c
}) {
  const videoRef =
    useRef(null);

  const aiCanvasRef =
    useRef(null);

  const skeletonCanvasRef =
    useRef(null);

  const streamRef =
    useRef(null);

  const scanTimerRef =
    useRef(null);

  const isScanningRef =
    useRef(false);

  const scanRunRef =
    useRef(0);

  const cameraSessionRef =
    useRef(0);

  const lastVideoTimeRef =
    useRef(-1);

  const confirmedRef =
    useRef(false);

  const passStreakRef =
    useRef(0);

  const poseStateRef =
    useRef(
      createPoseRepState(
        quest.reps
          ? (quest.progress || 0)
          : 0
      )
    );

  const poseRafRef =
    useRef(null);

  const poseLastVideoTimeRef =
    useRef(-1);

  const smoothedActRef =
    useRef(null);

  const smoothedNegRef =
    useRef(null);

  const smoothedSpoofRef =
    useRef(null);

  const spoofNoticeActiveRef =
    useRef(false);

  const verifyingRef =
    useRef(false);

  const repSpoofSuspectedRef =
    useRef(false);

  const smoothedRepSpoofRef =
    useRef(null);

  const repSpoofNoticeActiveRef =
    useRef(false);

  const cancelHandledRef =
    useRef(false);

  const confirmingRef =
    useRef(false);

  const [phase, setPhase] =
    useState('starting');

  const [camError, setCamError] =
    useState(null);

  const [facingMode, setFacingMode] =
    useState('environment');

  const [cameraVersion, setCameraVersion] =
    useState(0);

  const [repPhase, setRepPhase] =
    useState('up');

  const [confirmed, setConfirmed] =
    useState(false);

  const [verifying, setVerifying] =
    useState(false);

  const [lastLabel, setLastLabel] =
    useState('');

  const [uploadedProof, setUploadedProof] =
    useState(null);

  const [secondsLeft, setSecondsLeft] =
    useState(
      quest.duration
        ? (quest.progress ??
            quest.duration)
        : 0
    );

  const [timerRunning, setTimerRunning] =
    useState(false);

  const [liveScore, setLiveScore] =
    useState(0);

  const [passStreak, setPassStreak] =
    useState(0);

  const [repsDone, setRepsDone] =
    useState(
      quest.reps
        ? (quest.progress || 0)
        : 0
    );

  const [repCue, setRepCue] =
    useState('Move into frame');

  const [poseReady, setPoseReady] =
    useState(false);

  const [poseError, setPoseError] =
    useState(null);

  const [modelReady, setModelReady] =
    useState(false);

  const [modelError, setModelError] =
    useState(null);

  const [modelProgress, setModelProgress] =
    useState(0);

  const [scanning, setScanning] =
    useState(false);

  const [notice, setNotice] =
    useState(null);

  const [sourceWarning, setSourceWarning] =
    useState(null);

  const labels =
    QUEST_LABELS[quest.id];

  const questType =
    labels?.type || 'action';

  const hasSkeletonTracking =
    Boolean(quest.reps) ||
    MOVEMENT_ACTION_IDS.has(
      quest.id
    );

  const activeNegatives =
    useMemo(
      () =>
        getNegativeLabels(
          questType
        ),
      [questType]
    );

  const classifierLabels =
    useMemo(
      () =>
        [
          ...(labels?.activity ??
            []),
          ...activeNegatives
        ],
      [
        labels,
        activeNegatives
      ]
    );

  let instructionText =
    `Show the camera you're ${
      labels?.label ??
      'doing it'
    }…`;

  let uiSubtext =
    quest.text;

  if (questType === 'map') {
    instructionText =
      "Upload a metric summary screenshot (Strava, Nike, Garmin etc.)";

    uiSubtext =
      "Must clearly state distance metrics & active time duration summary logs.";
  } else if (
    questType === 'food'
  ) {
    instructionText =
      "Take a clear picture of the food on your plate.";

    uiSubtext =
      "Must be a real photo of a prepared meal or plate.";
  } else if (
    questType === 'reps'
  ) {
    instructionText =
      `Position the camera for full-body tracking: ${quest.reps} reps.`;

    uiSubtext =
      "Keep your entire working frame visible to log movements.";
  }

  useEffect(() => {
    let intervalId = null;

    if (
      timerRunning &&
      secondsLeft > 0
    ) {
      intervalId =
        setInterval(() => {
          setSecondsLeft(prev => {
            if (prev <= 1) {
              setTimerRunning(false);

              if (
                !quest.reps &&
                questType === 'action'
              ) {
                setConfirmed(true);
                confirmedRef.current =
                  true;
              }

              return 0;
            }

            return prev - 1;
          });
        }, 1000);
    }

    return () => {
      clearInterval(
        intervalId
      );
    };
  }, [
    timerRunning,
    secondsLeft,
    questType,
    quest.reps
  ]);

  const formatTimerString =
    secs =>
      `${String(
        Math.floor(secs / 60)
      ).padStart(2, '0')}:${String(
        secs % 60
      ).padStart(2, '0')}`;

  const stopCamera =
    useCallback(() => {
      const stream =
        streamRef.current;

      streamRef.current = null;

      stream
        ?.getTracks()
        .forEach(track =>
          track.stop()
        );

      if (
        videoRef.current
      ) {
        videoRef.current.srcObject =
          null;
      }
    }, []);

  const startCamera =
    useCallback(
      async mode => {
        const session =
          ++cameraSessionRef.current;

        stopCamera();

        setCamError(null);
        setNotice(null);
        setUploadedProof(null);
        setPhase('starting');

        const attempts = [
          {
            video: {
              facingMode: {
                ideal: mode
              },
              width: {
                ideal: 1280
              },
              height: {
                ideal: 720
              }
            },
            audio: false
          },

          {
            video: {
              facingMode: {
                ideal: mode
              }
            },
            audio: false
          },

          {
            video: true,
            audio: false
          }
        ];

        for (
          const constraints of attempts
        ) {
          try {
            const stream =
              await navigator.mediaDevices.getUserMedia(
                constraints
              );

            if (
              session !==
              cameraSessionRef.current
            ) {
              stream
                .getTracks()
                .forEach(track =>
                  track.stop()
                );

              return;
            }

            streamRef.current =
              stream;

            const video =
              videoRef.current;

            if (!video) {
              stopCamera();
              return;
            }

            video.srcObject =
              stream;

            video.muted = true;
            video.playsInline = true;

            await video.play();

            if (
              session !==
              cameraSessionRef.current
            ) {
              stopCamera();
              return;
            }

            setPhase('live');

            return;
          } catch (
            err
          ) {
            console.error(
              'Camera attempt failed:',
              err
            );
          }
        }

        if (
          session ===
          cameraSessionRef.current
        ) {
          setCamError(
            'Camera access failed. Check your browser permissions and make sure this site is served over HTTPS.'
          );

          setPhase('error');
        }
      },
      [stopCamera]
    );

  useEffect(() => {
    cancelHandledRef.current =
      false;

    startCamera(
      facingMode
    );

    return () => {
      cameraSessionRef.current += 1;
      scanRunRef.current += 1;

      clearTimeout(
        scanTimerRef.current
      );

      if (
        poseRafRef.current
      ) {
        cancelAnimationFrame(
          poseRafRef.current
        );
      }

      poseRafRef.current =
        null;

      stopCamera();
    };
  }, [
    facingMode,
    cameraVersion,
    startCamera,
    stopCamera
  ]);

  const retryCamera =
    () => {
      cameraSessionRef.current += 1;
      scanRunRef.current += 1;

      clearTimeout(
        scanTimerRef.current
      );

      lastVideoTimeRef.current =
        -1;

      setCamError(null);
      setNotice(null);
      setPhase('starting');
      setCameraVersion(
        v => v + 1
      );

      setSourceWarning(
        null
      );
    };

  const flipCamera =
    () => {
      clearTimeout(
        scanTimerRef.current
      );

      scanRunRef.current += 1;

      setPhase('starting');
      setLiveScore(0);
      setPassStreak(0);

      setRepsDone(
        quest.reps
          ? (quest.progress || 0)
          : 0
      );

      passStreakRef.current =
        0;

      confirmedRef.current =
        false;

      lastVideoTimeRef.current =
        -1;

      smoothedActRef.current =
        null;

      smoothedNegRef.current =
        null;

      smoothedSpoofRef.current =
        null;

      spoofNoticeActiveRef.current =
        false;

      repSpoofSuspectedRef.current =
        false;

      smoothedRepSpoofRef.current =
        null;

      repSpoofNoticeActiveRef.current =
        false;

      verifyingRef.current =
        false;

      setVerifying(false);

      poseStateRef.current =
        createPoseRepState(
          quest.reps
            ? (quest.progress ||
                0)
            : 0
        );

      setRepPhase('up');

      setRepCue(
        'Move into frame'
      );

      setConfirmed(false);
      setSourceWarning(
        null
      );

      setFacingMode(
        m =>
          m === 'environment'
            ? 'user'
            : 'environment'
      );

      if (
        skeletonCanvasRef.current
      ) {
        const ctx =
          skeletonCanvasRef.current.getContext(
            '2d'
          );

        ctx?.clearRect(
          0,
          0,
          skeletonCanvasRef.current
            .width,
          skeletonCanvasRef.current
            .height
        );
      }
    };

  const handleFileUpload =
    e => {
      const file =
        e.target.files?.[0];

      if (!file) return;

      if (quest.reps) {
        setNotice(
          'Rep quests need live camera tracking; a single photo cannot verify a full set.'
        );

        return;
      }

      const reader =
        new FileReader();

      reader.onload =
        async event => {
          try {
            setScanning(true);
            setUploadedProof(
              event.target.result
            );

            const dataUrl =
              event.target.result;

            const authCheck =
              await checkPhotoAuthenticity(
                file,
                dataUrl,
                classifierLabels
              );

            if (
              authCheck.suspicious
            ) {
              setPassStreak(0);

              setNotice(
                `This looks like a stock photo, screenshot, or image pulled from the internet rather than one you took yourself (${authCheck.confidence}% confidence). Please upload an original photo.`
              );

              return;
            }

            if (
              !authCheck.hasCameraMetadata
            ) {
              setSourceWarning(
                "This photo has no camera metadata, so we can't fully confirm that it's an original. Live camera capture is the most reliable option."
              );
            }

            const classifier =
              await getStaticClassifier();

            const results =
              await classifier(
                dataUrl,
                classifierLabels
              );

            const actScore =
              getMaxLabelScore(
                results,
                labels.activity
              );

            const negScore =
              getMaxLabelScore(
                results,
                activeNegatives
              );

            const spoofScore =
              getMaxLabelScore(
                results,
                ANTI_SPOOF_LABELS
              );

            setLiveScore(
              Math.round(
                actScore * 100
              )
            );

            setLastLabel(
              results[0]?.label ??
              ''
            );

            if (
              spoofScore >=
              SPOOF_BLOCK_THRESHOLD
            ) {
              setPassStreak(0);

              setNotice(
                'This looks like a photo of a screen or a printed photo rather than an original shot. Please upload something you photographed directly.'
              );

              return;
            }

            if (
              (actScore -
                negScore) >=
                PASS_MARGIN &&
              actScore >=
                PASS_THRESHOLD
            ) {
              setPassStreak(
                REQUIRED_PASSES
              );

              confirmedRef.current =
                true;

              setConfirmed(true);

              haptic([
                15,
                30,
                15
              ]);
            } else {
              setPassStreak(0);

              setNotice(
                questType === 'map'
                  ? "Verification failed. Please make sure your upload clearly shows distance and elapsed-time metrics."
                  : "Verification failed. Please upload a clearer original photo."
              );
            }
          } catch (err) {
            console.error(
              'Upload verification failed:',
              err
            );

            setNotice(
              'Verification could not be completed. Please try again.'
            );
          } finally {
            setScanning(false);
          }
        };

      reader.readAsDataURL(
        file
      );

      e.target.value = '';
    };

  const handleCancel =
    useCallback(() => {
      if (
        cancelHandledRef.current
      ) {
        return;
      }

      cancelHandledRef.current =
        true;

      cameraSessionRef.current +=
        1;

      scanRunRef.current += 1;

      clearTimeout(
        scanTimerRef.current
      );

      if (
        poseRafRef.current
      ) {
        cancelAnimationFrame(
          poseRafRef.current
        );
      }

      poseRafRef.current =
        null;

      isScanningRef.current =
        false;

      verifyingRef.current =
        false;

      confirmedRef.current =
        false;

      confirmingRef.current =
        false;

      setTimerRunning(false);
      setScanning(false);
      setVerifying(false);

      stopCamera();

      const result =
        quest.reps
          ? (
              repsDone > 0
                ? repsDone
                : undefined
            )
          : quest.duration
            ? (
                secondsLeft <
                quest.duration
                  ? secondsLeft
                  : undefined
              )
            : undefined;

      requestAnimationFrame(
        () => {
          onCancel(result);
        }
      );
    }, [
      onCancel,
      quest.reps,
      quest.duration,
      repsDone,
      secondsLeft,
      stopCamera
    ]);

  const isCurrentRun =
    useCallback(
      run =>
        run ===
        scanRunRef.current,
      []
    );

  const scheduleNext =
    useCallback(
      delay => {
        clearTimeout(
          scanTimerRef.current
        );

        const run =
          scanRunRef.current;

        scanTimerRef.current =
          window.setTimeout(
            () => {
              if (
                run !==
                scanRunRef.current
              ) {
                return;
              }

              setScanning(false);
            },
            delay
          );
      },
      []
    );

  const runFinalVerification =
    async () => {
      if (
        confirmingRef.current
      ) {
        return;
      }

      if (confirmed) {
        return;
      }

      confirmingRef.current =
        true;

      setVerifying(true);

      try {
        const video =
          videoRef.current;

        if (
          !video ||
          !video.videoWidth
        ) {
          throw new Error(
            'no live frame available'
          );
        }

        const size = 256;

        const canvas =
          document.createElement(
            'canvas'
          );

        canvas.width = size;
        canvas.height = size;

        const ctx =
          canvas.getContext(
            '2d'
          );

        const scale =
          Math.min(
            size /
              video.videoWidth,
            size /
              video.videoHeight
          );

        const w =
          video.videoWidth *
          scale;

        const h =
          video.videoHeight *
          scale;

        ctx.fillStyle = '#000';

        ctx.fillRect(
          0,
          0,
          size,
          size
        );

        ctx.drawImage(
          video,
          0,
          0,
          video.videoWidth,
          video.videoHeight,
          (size - w) /
            2,
          (size - h) /
            2,
          w,
          h
        );

        const dataUrl =
          canvas.toDataURL(
            'image/jpeg',
            0.9
          );

        const staticClassifier =
          await getStaticClassifier();

        const results =
          await staticClassifier(
            dataUrl,
            classifierLabels
          );

        const actScore =
          getMaxLabelScore(
            results,
            labels.activity
          );

        const negScore =
          getMaxLabelScore(
            results,
            activeNegatives
          );

        const spoofScore =
          getMaxLabelScore(
            results,
            ANTI_SPOOF_LABELS
          );

        const finalPass =
          spoofScore <
            SPOOF_BLOCK_THRESHOLD &&
          (
            actScore -
            negScore
          ) >= PASS_MARGIN &&
          actScore >=
            PASS_THRESHOLD;

        if (
          !isCurrentRun(
            scanRunRef.current
          )
        ) {
          return;
        }

        if (finalPass) {
          confirmedRef.current =
            true;

          setConfirmed(true);

          haptic([
            15,
            30,
            15
          ]);
        } else {
          if (
            spoofScore >=
            SPOOF_BLOCK_THRESHOLD
          ) {
            spoofNoticeActiveRef.current =
              true;

            setNotice(
              'This looks like it is coming from a screen or printed photo rather than you, live. Point the camera at yourself doing it in person.'
            );
          }

          passStreakRef.current =
            Math.max(
              0,
              REQUIRED_PASSES - 2
            );

          setPassStreak(
            passStreakRef.current
          );
        }
      } catch (err) {
        if (
          isCurrentRun(
            scanRunRef.current
          )
        ) {
          passStreakRef.current =
            Math.max(
              0,
              REQUIRED_PASSES - 2
            );

          setPassStreak(
            passStreakRef.current
          );

          setNotice(
            'The final AI check failed. Please try again.'
          );
        }
      } finally {
        confirmingRef.current =
          false;

        verifyingRef.current =
          false;

        setVerifying(false);
      }
    };

  useEffect(() => {
    if (
      phase !== 'live' ||
      confirmed ||
      quest.reps
    ) {
      return undefined;
    }

    let disposed = false;

    const scanLoop =
      async () => {
        const run =
          scanRunRef.current;

        if (disposed) {
          return;
        }

        const video =
          videoRef.current;

        if (
          !video ||
          video.readyState < 2 ||
          !video.videoWidth
        ) {
          scheduleNext(250);

          return;
        }

        if (
          video.currentTime ===
          lastVideoTimeRef.current
        ) {
          scheduleNext(250);

          return;
        }

        lastVideoTimeRef.current =
          video.currentTime;

        if (
          verifyingRef.current
        ) {
          scheduleNext(400);

          return;
        }

        try {
          isScanningRef.current =
            true;

          setScanning(true);

          const canvas =
            aiCanvasRef.current;

          if (!canvas) {
            scheduleNext(500);
            return;
          }

          const ctx =
            canvas.getContext(
              '2d'
            );

          canvas.width = 224;
          canvas.height = 224;

          const scale =
            Math.min(
              224 /
                video.videoWidth,
              224 /
                video.videoHeight
            );

          const w =
            video.videoWidth *
            scale;

          const h =
            video.videoHeight *
            scale;

          ctx.fillStyle =
            '#000';

          ctx.fillRect(
            0,
            0,
            224,
            224
          );

          ctx.drawImage(
            video,
            0,
            0,
            video.videoWidth,
            video.videoHeight,
            (224 - w) / 2,
            (224 - h) / 2,
            w,
            h
          );

          const dataUrl =
            canvas.toDataURL(
              'image/jpeg',
              0.5
            );

          const classifier =
            await getClassifier();

          const results =
            await classifier(
              dataUrl,
              classifierLabels
            );

          if (
            disposed ||
            run !==
              scanRunRef.current
          ) {
            return;
          }

          const actScore =
            getMaxLabelScore(
              results,
              labels.activity
            );

          const negScore =
            getMaxLabelScore(
              results,
              activeNegatives
            );

          const spoofScore =
            getMaxLabelScore(
              results,
              ANTI_SPOOF_LABELS
            );

          smoothedActRef.current =
            smoothedActRef.current ===
            null
              ? actScore
              : smoothedActRef.current *
                  (1 -
                    SCORE_SMOOTHING) +
                actScore *
                  SCORE_SMOOTHING;

          smoothedNegRef.current =
            smoothedNegRef.current ===
            null
              ? negScore
              : smoothedNegRef.current *
                  (1 -
                    SCORE_SMOOTHING) +
                negScore *
                  SCORE_SMOOTHING;

          smoothedSpoofRef.current =
            smoothedSpoofRef.current ===
            null
              ? spoofScore
              : smoothedSpoofRef.current *
                  0.5 +
                spoofScore *
                  0.5;

          const smoothAct =
            smoothedActRef.current;

          const smoothNeg =
            smoothedNegRef.current;

          const smoothSpoof =
            smoothedSpoofRef.current;

          const score =
            Math.max(
              0,
              Math.min(
                1,
                smoothAct
              )
            );

          setLiveScore(
            Math.round(
              score * 100
            )
          );

          setLastLabel(
            results[0]?.label ||
              ''
          );

          const spoofed =
            smoothSpoof >=
            SPOOF_BLOCK_THRESHOLD;

          if (spoofed) {
            if (
              !spoofNoticeActiveRef.current
            ) {
              spoofNoticeActiveRef.current =
                true;

              setNotice(
                'This looks like it is coming from a screen or printed photo rather than you, live. Point the camera at yourself doing it in person.'
              );
            }

            passStreakRef.current =
              0;

            setPassStreak(0);

            scheduleNext(700);
            return;
          }

          if (
            spoofNoticeActiveRef.current
          ) {
            spoofNoticeActiveRef.current =
              false;

            setNotice(null);
          }

          const pass =
            (
              smoothAct -
              smoothNeg
            ) >= PASS_MARGIN &&
            smoothAct >=
              PASS_THRESHOLD;

          if (pass) {
            passStreakRef.current +=
              1;

            setPassStreak(
              Math.min(
                REQUIRED_PASSES,
                passStreakRef.current
              )
            );

            if (
              passStreakRef.current >=
              REQUIRED_PASSES
            ) {
              await runFinalVerification();

              return;
            }
          } else {
            passStreakRef.current =
              Math.max(
                0,
                passStreakRef.current - 1
              );

            setPassStreak(
              passStreakRef.current
            );
          }
        } catch (err) {
          if (
            !disposed
          ) {
            console.error(
              'Live scan failed:',
              err
            );
          }
        } finally {
          isScanningRef.current =
            false;

          if (!disposed) {
            setScanning(false);
            scheduleNext(1200);
          }
        }
      };

    scanLoop();

    return () => {
      disposed = true;

      scanRunRef.current +=
        1;

      clearTimeout(
        scanTimerRef.current
      );

      isScanningRef.current =
        false;
    };
  }, [
    phase,
    modelReady,
    confirmed,
    labels,
    classifierLabels,
    quest.reps,
    activeNegatives,
    runFinalVerification,
    scheduleNext,
    isCurrentRun
  ]);

  useEffect(() => {
    if (
      !hasSkeletonTracking ||
      phase !== 'live' ||
      confirmed
    ) {
      return undefined;
    }

    let cancelled = false;

    (async () => {
      try {
        setPoseError(null);

        const landmarker =
          await getPoseLandmarker();

        if (cancelled) return;

        setPoseReady(true);

        const loop = () => {
          if (cancelled) return;

          const video =
            videoRef.current;

          const canvas =
            skeletonCanvasRef.current;

          if (
            video &&
            canvas &&
            video.readyState >= 2 &&
            video.currentTime !==
              poseLastVideoTimeRef.current
          ) {
            poseLastVideoTimeRef.current =
              video.currentTime;

            const now =
              performance.now();

            try {
              const result =
                landmarker.detectForVideo(
                  video,
                  now
                );

              const landmarks =
                result?.landmarks?.[0] ??
                null;

              if (
                landmarks &&
                quest.reps
              ) {
                const updated =
                  updatePoseRepState(
                    poseStateRef.current,
                    quest.id,
                    landmarks,
                    now
                  );

                setRepsDone(
                  updated.reps
                );

                setRepPhase(
                  updated.phase
                );

                setRepCue(
                  updated.cue
                );

                if (
                  updated.reps >=
                    quest.reps &&
                  !confirmedRef.current
                ) {
                  confirmedRef.current =
                    true;

                  setConfirmed(true);

                  haptic([
                    15,
                    30,
                    15
                  ]);
                }
              }

              if (canvas) {
                const ctx =
                  canvas.getContext(
                    '2d'
                  );

                const rect =
                  video.getBoundingClientRect();

                if (
                  canvas.width !==
                    rect.width ||
                  canvas.height !==
                    rect.height
                ) {
                  canvas.width =
                    rect.width;
                  canvas.height =
                    rect.height;
                }

                ctx.clearRect(
                  0,
                  0,
                  canvas.width,
                  canvas.height
                );

                if (
                  landmarks
                ) {
                  ctx.lineWidth = 2;
                  ctx.strokeStyle =
                    'rgba(255,255,255,.76)';
                  ctx.fillStyle =
                    'rgba(10,132,255,.92)';

                  POSE_CONNECTIONS.forEach(
                    ([a, b]) => {
                      const p1 =
                        landmarks[a];

                      const p2 =
                        landmarks[b];

                      if (
                        !p1 ||
                        !p2 ||
                        !visiblePt(
                          p1,
                          .35
                        ) ||
                        !visiblePt(
                          p2,
                          .35
                        )
                      ) {
                        return;
                      }

                      ctx.beginPath();

                      ctx.moveTo(
                        p1.x *
                          canvas.width,
                        p1.y *
                          canvas.height
                      );

                      ctx.lineTo(
                        p2.x *
                          canvas.width,
                        p2.y *
                          canvas.height
                      );

                      ctx.stroke();
                    }
                  );

                  landmarks.forEach(
                    p => {
                      if (
                        !visiblePt(
                          p,
                          .35
                        )
                      ) {
                        return;
                      }

                      ctx.beginPath();

                      ctx.arc(
                        p.x *
                          canvas.width,
                        p.y *
                          canvas.height,
                        3.1,
                        0,
                        Math.PI * 2
                      );

                      ctx.fill();
                    }
                  );
                }
              }
            } catch (err) {
              console.error(
                'Pose tracking failed:',
                err
              );
            }
          }

          poseRafRef.current =
            requestAnimationFrame(
              loop
            );
        };

        loop();
      } catch (err) {
        if (!cancelled) {
          console.error(
            'Pose landmarker failed:',
            err
          );

          setPoseError(
            'Pose tracking could not start. You can still use the normal camera verification.'
          );
        }
      }
    })();

    return () => {
      cancelled = true;

      if (
        poseRafRef.current
      ) {
        cancelAnimationFrame(
          poseRafRef.current
        );
      }

      poseRafRef.current =
        null;
    };
  }, [
    hasSkeletonTracking,
    phase,
    confirmed,
    quest.id,
    quest.reps
  ]);

  useEffect(() => {
    if (
      phase !== 'live' ||
      !quest.reps ||
      confirmed
    ) {
      return undefined;
    }

    let timer = null;
    let cancelled = false;

    const check = async () => {
      if (
        cancelled ||
        !videoRef.current ||
        !videoRef.current.videoWidth
      ) {
        timer =
          window.setTimeout(
            check,
            1800
          );

        return;
      }

      try {
        const video =
          videoRef.current;

        const canvas =
          document.createElement(
            'canvas'
          );

        canvas.width = 224;
        canvas.height = 224;

        const ctx =
          canvas.getContext(
            '2d'
          );

        const scale =
          Math.min(
            224 /
              video.videoWidth,
            224 /
              video.videoHeight
          );

        const w =
          video.videoWidth *
          scale;

        const h =
          video.videoHeight *
          scale;

        ctx.fillStyle =
          '#000';

        ctx.fillRect(
          0,
          0,
          224,
          224
        );

        ctx.drawImage(
          video,
          0,
          0,
          video.videoWidth,
          video.videoHeight,
          (224 - w) / 2,
          (224 - h) / 2,
          w,
          h
        );

        const dataUrl =
          canvas.toDataURL(
            'image/jpeg',
            0.5
          );

        const classifier =
          await getClassifier();

        const results =
          await classifier(
            dataUrl,
            [
              ...ANTI_SPOOF_LABELS,
              'a real person exercising in a room'
            ]
          );

        if (!cancelled) {
          const spoofScore =
            getMaxLabelScore(
              results,
              ANTI_SPOOF_LABELS
            );

          smoothedRepSpoofRef.current =
            smoothedRepSpoofRef.current ===
            null
              ? spoofScore
              : smoothedRepSpoofRef.current *
                  0.5 +
                spoofScore *
                  0.5;

          const suspected =
            smoothedRepSpoofRef.current >=
            SPOOF_BLOCK_THRESHOLD;

          repSpoofSuspectedRef.current =
            suspected;

          if (suspected) {
            repSpoofNoticeActiveRef.current =
              true;

            setNotice(
              "This looks like it's coming from a screen or printed photo rather than you, live. Point the camera at yourself doing it in person."
            );
          } else if (
            repSpoofNoticeActiveRef.current
          ) {
            repSpoofNoticeActiveRef.current =
              false;

            setNotice(null);
          }
        }
      } catch {}

      if (!cancelled) {
        timer =
          window.setTimeout(
            check,
            2500
          );
      }
    };

    timer =
      window.setTimeout(
        check,
        1500
      );

    return () => {
      cancelled = true;

      clearTimeout(timer);
    };
  }, [
    quest.reps,
    phase,
    confirmed
  ]);

  useEffect(() => {
    if (
      modelReady ||
      modelError
    ) {
      return;
    }

    let cancelled = false;

    getClassifier(progress => {
      if (cancelled) return;

      const value =
        typeof progress ===
        'object'
          ? progress.progress
          : progress;

      if (
        Number.isFinite(
          value
        )
      ) {
        setModelProgress(
          Math.round(value)
        );
      }
    })
      .then(() => {
        if (
          cancelled
        ) return;

        setModelReady(true);
        setModelProgress(
          100
        );
      })
      .catch(err => {
        if (
          cancelled
        ) return;

        console.error(
          'Model load failed:',
          err
        );

        setModelError(
          'AI model could not load.'
        );
      });

    return () => {
      cancelled = true;
    };
  }, [
    modelReady,
    modelError
  ]);

  const captureAndConfirm =
    () => {
      if (
        confirmingRef.current
      ) {
        return;
      }

      if (!confirmed) {
        return;
      }

      onConfirm({
        progress:
          quest.reps
            ? repsDone
            : quest.duration
              ? secondsLeft
              : 1,

        proof:
          uploadedProof
      });
    };

  const meterColor =
    liveScore >= 75
      ? c.green
      : liveScore >= 45
        ? c.orange
        : c.red;

  return (
    <div
      className="fixed inset-0 z-[60] flex items-end justify-center sq-modal-in"
      style={{
        background:
          'rgba(0,0,0,0.52)',

        backdropFilter:
          'blur(10px)',

        WebkitBackdropFilter:
          'blur(10px)'
      }}
    >
      <div
        className="sq-glass"
        style={{
          ...glassStyle(
            c,
            true
          ),

          background:
            c.bg,

          width: '100%',
          height: '95vh',

          borderTopLeftRadius:
            26,

          borderTopRightRadius:
            26,

          overflow:
            'hidden',

          display:
            'flex',

          flexDirection:
            'column',

          paddingBottom:
            'max(env(safe-area-inset-bottom), 14px)'
        }}
      >
        <div
          className="relative flex items-center justify-center px-6 pb-3"
        >
          <div className="text-center">
            <p
              style={{
                fontSize: 15,
                fontWeight: 650,
                color: c.label
              }}
            >
              Quest check-in
            </p>

            <p
              style={{
                fontSize: 11,
                color:
                  c.labelSecondary,
                marginTop: 2,
                maxWidth: 250,
                whiteSpace:
                  'nowrap',
                overflow:
                  'hidden',
                textOverflow:
                  'ellipsis'
              }}
            >
              {uiSubtext}
            </p>
          </div>

          <button
            type="button"
            onClick={
              handleCancel
            }
            aria-label="Close camera"
            className="sq-control"
            style={{
              position:
                'absolute',

              right: 16,
              top: -4,

              width: 38,
              height: 38,
              minHeight: 38,

              borderRadius:
                '50%',

              background:
                c.fill,

              color:
                c.label,

              display:
                'flex',

              alignItems:
                'center',

              justifyContent:
                'center',

              zIndex: 40,

              boxShadow:
                '0 6px 22px -16px rgba(0,0,0,.65)'
            }}
          >
            <span
              style={{
                fontSize: 22,
                lineHeight: 1,
                transform:
                  'translateY(-1px)'
              }}
            >
              ×
            </span>
          </button>
        </div>

        <div
          className="relative flex-1 mx-4 rounded-3xl overflow-hidden sq-cam-shell"
          style={{
            background:
              '#000'
          }}
        >
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            style={{
              width: '100%',
              height: '100%',
              objectFit:
                'cover'
            }}
          />

          <canvas
            ref={
              skeletonCanvasRef
            }
            className="absolute inset-0 w-full h-full pointer-events-none"
          />

          {phase ===
            'starting' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
              <ThinkingOrbs label="Opening camera…" />
            </div>
          )}

          {phase ===
            'error' && (
            <div
              className="absolute inset-0 flex flex-col items-center justify-center px-7 text-center"
            >
              <Camera
                size={36}
                strokeWidth={
                  1.5
                }
                color="#fff"
              />

              <p
                style={{
                  color: '#fff',
                  fontSize: 15,
                  fontWeight: 600,
                  marginTop: 14
                }}
              >
                Camera unavailable
              </p>

              <p
                style={{
                  color:
                    'rgba(255,255,255,.62)',
                  fontSize: 12,
                  lineHeight: 1.5,
                  marginTop: 6,
                  maxWidth: 300
                }}
              >
                {camError}
              </p>
            </div>
          )}

          {notice && (
            <div
              className="absolute inset-x-3 top-3 z-30"
              style={{
                borderRadius: 14,
                background:
                  'rgba(255,59,48,0.90)',
                backdropFilter:
                  'blur(14px)',
                WebkitBackdropFilter:
                  'blur(14px)',
                padding:
                  '10px 14px'
              }}
            >
              <p
                style={{
                  fontSize: 12,
                  fontWeight: 500,
                  lineHeight: 1.45,
                  color: '#fff'
                }}
              >
                {notice}
              </p>

              <button
                type="button"
                onClick={() =>
                  setNotice(null)
                }
                style={{
                  fontSize: 10,
                  fontWeight: 700,
                  color:
                    'rgba(255,255,255,.72)',
                  marginTop: 6,
                  textTransform:
                    'uppercase',
                  letterSpacing:
                    .3
                }}
              >
                Dismiss
              </button>
            </div>
          )}

          {sourceWarning && (
            <div
              className="absolute inset-x-3 top-3 z-30"
              style={{
                borderRadius: 12,
                background:
                  'rgba(255,159,10,0.92)',
                backdropFilter:
                  'blur(10px)',
                padding:
                  '10px 14px'
              }}
            >
              <p
                style={{
                  fontSize: 13,
                  fontWeight: 500,
                  lineHeight: 1.4,
                  color: '#fff'
                }}
              >
                {sourceWarning}
              </p>

              <button
                type="button"
                onClick={() =>
                  setSourceWarning(
                    null
                  )
                }
                style={{
                  fontSize: 11,
                  fontWeight: 700,
                  color:
                    'rgba(255,255,255,.75)',
                  marginTop: 6,
                  textTransform:
                    'uppercase',
                  letterSpacing:
                    .3
                }}
              >
                Dismiss
              </button>
            </div>
          )}

          {phase ===
            'live' &&
            quest.reps && (
              <div
                className="absolute inset-x-0 bottom-0 px-4 pb-4 pt-10 z-20"
                style={{
                  background:
                    'linear-gradient(to top, rgba(0,0,0,0.75), transparent)'
                }}
              >
                {confirmed ? (
                  <div
                    className="flex items-center gap-2 w-max px-3 py-1.5 rounded-full"
                    style={{
                      background:
                        'rgba(48,209,88,.2)'
                    }}
                  >
                    <div
                      style={{
                        width: 18,
                        height: 18,
                        borderRadius:
                          '50%',
                        background:
                          c.green,
                        display:
                          'flex',
                        alignItems:
                          'center',
                        justifyContent:
                          'center'
                      }}
                    >
                      <CheckIcon
                        size={10}
                      />
                    </div>

                    <p
                      style={{
                        color:
                          '#DFFBE8',
                        fontSize: 12,
                        fontWeight: 600
                      }}
                    >
                      Verified
                    </p>
                  </div>
                ) : poseError ? (
                  <p
                    style={{
                      color:
                        '#FF9F9A',
                      fontSize: 12,
                      fontWeight: 500
                    }}
                  >
                    {poseError}
                  </p>
                ) : !poseReady ? (
                  <ThinkingOrbs label="Getting ready…" />
                ) : (
                  <>
                    <div className="flex items-end justify-between mb-2">
                      <p
                        style={{
                          color:
                            'rgba(255,255,255,.9)',
                          fontSize: 13,
                          fontWeight: 500
                        }}
                      >
                        {repCue}
                      </p>

                      <p
                        className="sq-mono"
                        style={{
                          color:
                            '#fff',
                          fontSize: 12,
                          fontWeight: 700,
                          background:
                            c.blue,
                          padding:
                            '4px 10px',
                          borderRadius:
                            8
                        }}
                      >
                        {repsDone} /{' '}
                        {quest.reps}{' '}
                        reps
                      </p>
                    </div>

                    <div
                      style={{
                        width: '100%',
                        height: 4,
                        borderRadius:
                          999,
                        background:
                          'rgba(255,255,255,.2)',
                        overflow:
                          'hidden'
                      }}
                    >
                      <div
                        style={{
                          height: '100%',
                          background:
                            c.green,
                          width: `${Math.min(
                            100,
                            (
                              repsDone /
                              quest.reps
                            ) * 100
                          )}%`,
                          transition:
                            'width .3s ease-out'
                        }}
                      />
                    </div>
                  </>
                )}
              </div>
            )}

          {phase ===
            'live' &&
            !quest.reps &&
            modelReady && (
              <div
                className="absolute inset-x-0 bottom-0 px-4 pb-4 pt-10 z-20"
                style={{
                  background:
                    'linear-gradient(to top, rgba(0,0,0,.75), transparent)'
                }}
              >
                {confirmed ? (
                  <div
                    className="flex items-center gap-2 w-max px-3 py-1.5 rounded-full"
                    style={{
                      background:
                        'rgba(48,209,88,.2)'
                    }}
                  >
                    <div
                      style={{
                        width: 18,
                        height: 18,
                        borderRadius:
                          '50%',
                        background:
                          c.green,
                        display:
                          'flex',
                        alignItems:
                          'center',
                        justifyContent:
                          'center'
                      }}
                    >
                      <CheckIcon
                        size={10}
                      />
                    </div>

                    <p
                      style={{
                        color:
                          '#DFFBE8',
                        fontSize: 12,
                        fontWeight: 600
                      }}
                    >
                      Verified
                    </p>
                  </div>
                ) : verifying ? (
                  <ThinkingOrbs label="Confirming…" />
                ) : (
                  <>
                    <div className="flex items-end justify-between mb-2">
                      <p
                        style={{
                          color:
                            'rgba(255,255,255,.9)',
                          fontSize: 13,
                          fontWeight: 500
                        }}
                      >
                        {scanning
                          ? 'AI is checking…'
                          : `${liveScore}% confidence`}
                      </p>

                      <div className="flex items-center gap-1">
                        {Array.from({
                          length:
                            REQUIRED_PASSES
                        }).map(
                          (_, i) => (
                            <div
                              key={i}
                              style={{
                                width: 6,
                                height: 6,
                                borderRadius:
                                  '50%',
                                background:
                                  i <
                                  passStreak
                                    ? c.green
                                    : 'rgba(255,255,255,.3)',
                                transition:
                                  'background .3s'
                              }}
                            />
                          )
                        )}
                      </div>
                    </div>

                    <div
                      style={{
                        width:
                          '100%',
                        height: 4,
                        borderRadius:
                          999,
                        background:
                          'rgba(255,255,255,.2)',
                        overflow:
                          'hidden'
                      }}
                    >
                      <div
                        style={{
                          height:
                            '100%',
                          background:
                            meterColor,
                          width: `${liveScore}%`,
                          transition:
                            'width .5s ease-out'
                        }}
                      />
                    </div>

                    {lastLabel && (
                      <p
                        className="sq-mono"
                        style={{
                          color:
                            'rgba(255,255,255,.4)',
                          fontSize: 9,
                          marginTop: 6,
                          textTransform:
                            'uppercase',
                          letterSpacing:
                            .3,
                          whiteSpace:
                            'nowrap',
                          overflow:
                            'hidden',
                          textOverflow:
                            'ellipsis'
                        }}
                      >
                        {lastLabel}
                      </p>
                    )}
                  </>
                )}
              </div>
            )}

          {phase ===
            'live' &&
            !quest.reps &&
            !modelReady &&
            !modelError && (
              <div
                className="absolute inset-x-0 bottom-0 px-4 pb-4 pt-10 z-20"
                style={{
                  background:
                    'linear-gradient(to top, rgba(0,0,0,.7), transparent)'
                }}
              >
                <div className="flex items-center justify-between mb-2">
                  <p
                    style={{
                      color:
                        'rgba(255,255,255,.8)',
                      fontSize: 11,
                      fontWeight: 600
                    }}
                  >
                    Getting ready…{' '}
                    {modelProgress ??
                      0}
                    %
                  </p>

                  <ThinkingOrbs label="Thinking…" />
                </div>

                <div
                  style={{
                    width:
                      '100%',
                    height: 4,
                    borderRadius:
                      999,
                    background:
                      'rgba(255,255,255,.2)',
                    overflow:
                      'hidden'
                  }}
                >
                  <div
                    style={{
                      height:
                        '100%',
                      background:
                        c.blue,
                      width: `${modelProgress ?? 0}%`,
                      transition:
                        'width .3s'
                    }}
                  />
                </div>
              </div>
            )}

          <canvas
            ref={aiCanvasRef}
            className="hidden"
          />
        </div>

        {quest.duration && (
          <div
            className="mx-4 mt-4"
            style={{
              ...glassStyle(
                c
              ),

              padding:
                '13px 16px',

              borderRadius:
                20,

              display:
                'flex',

              alignItems:
                'center',

              justifyContent:
                'space-between'
            }}
          >
            <div>
              <span
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  textTransform:
                    'uppercase',
                  letterSpacing:
                    .4,
                  color:
                    c.labelSecondary
                }}
              >
                Duration
              </span>

              <p
                className="sq-mono"
                style={{
                  fontSize: 20,
                  fontWeight: 700,
                  color:
                    c.label,
                  marginTop: 2
                }}
              >
                {formatTimerString(
                  secondsLeft
                )}
              </p>
            </div>

            <button
              type="button"
              onClick={() =>
                setTimerRunning(
                  !timerRunning
                )
              }
              disabled={
                secondsLeft === 0
              }
              className="sq-control"
              style={{
                padding:
                  '10px 20px',
                borderRadius:
                  12,
                fontSize: 13,
                fontWeight: 600,
                color: '#fff',
                background:
                  secondsLeft === 0
                    ? c.gray
                    : timerRunning
                      ? c.red
                      : c.green,
                opacity:
                  secondsLeft === 0
                    ? .5
                    : 1
              }}
            >
              {secondsLeft ===
              0
                ? 'Done'
                : timerRunning
                  ? 'Pause'
                  : 'Start'}
            </button>
          </div>
        )}

        <div className="px-4 pt-4 pb-1">
          {phase ===
            'live' && (
            <div className="space-y-2.5">
              <button
                type="button"
                onClick={
                  captureAndConfirm
                }
                disabled={
                  !confirmed
                }
                className="sq-control sq-glass"
                style={{
                  width:
                    '100%',
                  padding:
                    '15px',
                  borderRadius:
                    999,
                  fontSize: 16,
                  fontWeight: 600,
                  color:
                    confirmed
                      ? '#fff'
                      : c.labelTertiary,
                  background:
                    confirmed
                      ? c.blue
                      : c.fill,
                  opacity:
                    confirmed
                      ? 1
                      : .85
                }}
              >
                {confirmed
                  ? 'Complete quest'
                  : verifying
                    ? 'Confirming…'
                    : instructionText}
              </button>

              <div className="flex gap-2.5">
                <button
                  type="button"
                  onClick={
                    flipCamera
                  }
                  className="sq-control"
                  style={{
                    flex:
                      quest.reps
                        ? '1 1 100%'
                        : 1,
                    padding:
                      '13px',
                    borderRadius:
                      999,
                    fontSize: 14,
                    fontWeight: 600,
                    background:
                      c.fill,
                    color:
                      c.label
                  }}
                >
                  Flip camera
                </button>

                {!quest.reps && (
                  <label
                    className="sq-control"
                    style={{
                      flex: 1,
                      padding:
                        '13px',
                      borderRadius:
                        999,
                      fontSize: 14,
                      fontWeight: 600,
                      textAlign:
                        'center',
                      cursor:
                        'pointer',
                      background:
                        questType ===
                        'map'
                          ? c.bgElevated
                          : c.fill,
                      color:
                        questType ===
                        'map'
                          ? c.blue
                          : c.label,
                      border:
                        questType ===
                        'map'
                          ? `1.5px solid ${c.blue}`
                          : 'none'
                    }}
                  >
                    Upload photo

                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={
                        handleFileUpload
                      }
                    />
                  </label>
                )}
              </div>
            </div>
          )}

          {(phase ===
            'error' ||
            phase ===
              'starting') &&
            !quest.reps && (
              <label
                className="sq-control"
                style={{
                  display:
                    'flex',
                  alignItems:
                    'center',
                  justifyContent:
                    'center',
                  width:
                    '100%',
                  padding:
                    '15px',
                  borderRadius:
                    999,
                  fontSize: 16,
                  fontWeight: 600,
                  background:
                    c.fill,
                  color:
                    c.label,
                  cursor:
                    'pointer'
                }}
              >
                Upload a photo

                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={
                    handleFileUpload
                  }
                />
              </label>
            )}

          <button
            type="button"
            onClick={
              handleCancel
            }
            className="sq-control"
            style={{
              width:
                '100%',
              marginTop:
                10,
              padding:
                '13px',
              borderRadius:
                999,
              fontSize: 14,
              fontWeight: 600,
              background:
                c.fill,
              color:
                c.labelSecondary
            }}
          >
            Close camera
          </button>
        </div>
      </div>
    </div>
  );
}

function AuthModal({
  onSignedUp,
  onLoggedIn,
  c,
  initialError
}) {
  const [mode, setMode] =
    useState('login');

  const [username, setUsername] =
    useState('');

  const [password, setPassword] =
    useState('');

  const [confirmPassword, setConfirmPassword] =
    useState('');

  const [error, setError] =
    useState(null);

  const [submitting, setSubmitting] =
    useState(false);

  const [googleSubmitting, setGoogleSubmitting] =
    useState(false);

  useEffect(() => {
    if (initialError) {
      setError(initialError);
      setGoogleSubmitting(false);
    }
  }, [initialError]);

  const handleSubmit =
    async e => {
      e.preventDefault();

      setError(null);

      const userError =
        validateUsername(
          username
        );

      if (userError) {
        setError(userError);
        return;
      }

      const passwordError =
        validatePassword(
          password
        );

      if (passwordError) {
        setError(
          passwordError
        );
        return;
      }

      if (
        mode === 'signup' &&
        password !==
          confirmPassword
      ) {
        setError(
          'Passwords do not match.'
        );
        return;
      }

      setSubmitting(true);

      try {
        if (
          mode === 'signup'
        ) {
          const result =
            await signUpAccount(
              username,
              password
            );

          onSignedUp(
            result
          );
        } else {
          const result =
            await logInAccount(
              username,
              password
            );

          onLoggedIn(
            result
          );
        }
      } catch (err) {
        console.error(
          'Auth submit failed:',
          err
        );

        setError(
          friendlyAuthError(
            err
          )
        );
      } finally {
        setSubmitting(false);
      }
    };

  const handleGoogleClick =
    async () => {
      if (
        submitting ||
        googleSubmitting
      ) {
        return;
      }

      setError(null);
      setGoogleSubmitting(
        true
      );

      try {
        await signInWithGoogle();
      } catch (err) {
        console.error(
          'Google sign-in failed:',
          err
        );

        setError(
          friendlyAuthError(
            err
          )
        );

        setGoogleSubmitting(false);
      }
    };

  return (
    <div
      className="fixed inset-0 z-[80] flex items-center justify-center px-5 sq-modal-in"
      style={{
        background:
          'rgba(0,0,0,.46)',

        backdropFilter:
          'blur(18px) saturate(130%)',

        WebkitBackdropFilter:
          'blur(18px) saturate(130%)'
      }}
    >
      <div
        className="sq-glass"
        style={{
          ...glassStyle(
            c,
            true
          ),

          width:
            '100%',

          maxWidth:
            420,

          borderRadius:
            28,

          padding:
            24
        }}
      >
        <div
          style={{
            width: 58,
            height: 58,
            borderRadius: 18,
            background:
              `linear-gradient(145deg, ${c.blue}, ${c.indigo})`,
            display:
              'flex',
            alignItems:
              'center',
            justifyContent:
              'center',
            margin:
              '0 auto 16px',
            boxShadow:
              `0 18px 45px -22px ${c.blue}`
          }}
        >
          <CheckSquare
            size={28}
            color="#fff"
            strokeWidth={2}
          />
        </div>

        <h1
          className="sq-large-title"
          style={{
            fontSize: 28,
            fontWeight: 750,
            color:
              c.label,
            textAlign:
              'center'
          }}
        >
          QuestDaily
        </h1>

        <p
          style={{
            marginTop: 7,
            textAlign:
              'center',
            color:
              c.labelSecondary,
            fontSize: 13
          }}
        >
          Turn your day into quests.
        </p>

        <div
          className="flex mt-6 p-1"
          style={{
            borderRadius: 15,
            background:
              c.fill
          }}
        >
          {[
            ['login', 'Log in'],
            ['signup', 'Sign up']
          ].map(
            ([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => {
                  setMode(
                    value
                  );
                  setError(null);
                }}
                className="sq-control"
                style={{
                  flex: 1,
                  padding:
                    '10px 8px',
                  minHeight:
                    40,
                  borderRadius:
                    12,
                  background:
                    mode ===
                    value
                      ? c.bgElevated
                      : 'transparent',
                  color:
                    c.label,
                  fontSize:
                    13,
                  fontWeight: 650,
                  boxShadow:
                    mode ===
                    value
                      ? '0 4px 15px -12px rgba(0,0,0,.4)'
                      : 'none'
                }}
              >
                {label}
              </button>
            )
          )}
        </div>

        <form
          onSubmit={
            handleSubmit
          }
          className="space-y-3 mt-5"
        >
          <div>
            <label
              style={{
                display:
                  'block',
                fontSize:
                  11,
                fontWeight:
                  650,
                color:
                  c.labelSecondary,
                marginBottom:
                  7
              }}
            >
              Username
            </label>

            <div
              style={{
                display:
                  'flex',
                alignItems:
                  'center',
                gap: 9,
                padding:
                  '12px 13px',
                borderRadius:
                  15,
                background:
                  c.fill,
                border:
                  `1px solid ${c.separator}`
              }}
            >
              <AtSign
                size={17}
                color={
                  c.labelSecondary
                }
              />

              <input
                value={
                  username
                }
                onChange={e =>
                  setUsername(
                    e.target.value
                  )
                }
                autoComplete="username"
                placeholder="yourname"
                style={{
                  flex: 1,
                  background:
                    'transparent',
                  outline:
                    'none',
                  border:
                    'none',
                  color:
                    c.label,
                  fontSize:
                    14
                }}
              />
            </div>
          </div>

          <div>
            <label
              style={{
                display:
                  'block',
                fontSize:
                  11,
                fontWeight:
                  650,
                color:
                  c.labelSecondary,
                marginBottom:
                  7
              }}
            >
              Password
            </label>

            <div
              style={{
                display:
                  'flex',
                alignItems:
                  'center',
                gap: 9,
                padding:
                  '12px 13px',
                borderRadius:
                  15,
                background:
                  c.fill,
                border:
                  `1px solid ${c.separator}`
              }}
            >
              <Lock
                size={17}
                color={
                  c.labelSecondary
                }
              />

              <input
                type="password"
                value={
                  password
                }
                onChange={e =>
                  setPassword(
                    e.target.value
                  )
                }
                autoComplete={
                  mode ===
                  'signup'
                    ? 'new-password'
                    : 'current-password'
                }
                placeholder="Password"
                style={{
                  flex: 1,
                  background:
                    'transparent',
                  outline:
                    'none',
                  border:
                    'none',
                  color:
                    c.label,
                  fontSize:
                    14
                }}
              />
            </div>
          </div>

          {mode ===
            'signup' && (
            <div>
              <label
                style={{
                  display:
                    'block',
                  fontSize:
                    11,
                  fontWeight:
                    650,
                  color:
                    c.labelSecondary,
                  marginBottom:
                    7
                }}
              >
                Confirm password
              </label>

              <div
                style={{
                  display:
                    'flex',
                  alignItems:
                    'center',
                  gap: 9,
                  padding:
                    '12px 13px',
                  borderRadius:
                    15,
                  background:
                    c.fill,
                  border:
                    `1px solid ${c.separator}`
                }}
              >
                <ShieldCheck
                  size={17}
                  color={
                    c.labelSecondary
                  }
                />

                <input
                  type="password"
                  value={
                    confirmPassword
                  }
                  onChange={e =>
                    setConfirmPassword(
                      e.target.value
                    )
                  }
                  autoComplete="new-password"
                  placeholder="Confirm password"
                  style={{
                    flex: 1,
                    background:
                      'transparent',
                    outline:
                      'none',
                    border:
                      'none',
                    color:
                      c.label,
                    fontSize:
                      14
                  }}
                />
              </div>
            </div>
          )}

          {error && (
            <div
              className="sq-anim-in"
              style={{
                borderRadius:
                  14,
                background:
                  'rgba(255,59,48,.12)',
                border:
                  '1px solid rgba(255,59,48,.24)',
                padding:
                  '11px 13px',
                color:
                  c.red,
                fontSize:
                  12,
                lineHeight:
                  1.45
              }}
            >
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={
              submitting ||
              googleSubmitting
            }
            className="sq-control sq-glass"
            style={{
              width:
                '100%',
              padding:
                '13px',
              marginTop:
                4,
              borderRadius:
                16,
              fontSize:
                14,
              fontWeight:
                650,
              color:
                '#fff',
              background:
                c.blue,
              opacity:
                submitting
                  ? .7
                  : 1,
              boxShadow:
                `0 14px 30px -20px ${c.blue}`
            }}
          >
            {submitting
              ? (
                <ThinkingOrbs label="Working…" />
              )
              : mode ===
                  'signup'
                ? 'Create account'
                : 'Log in'}
          </button>
        </form>

        <div
          className="flex items-center gap-3 my-5"
        >
          <div
            style={{
              height: 1,
              flex: 1,
              background:
                c.separator
            }}
          />

          <span
            style={{
              fontSize:
                10,
              fontWeight:
                600,
              color:
                c.labelTertiary,
              textTransform:
                'uppercase',
              letterSpacing:
                .5
            }}
          >
            or
          </span>

          <div
            style={{
              height: 1,
              flex: 1,
              background:
                c.separator
            }}
          />
        </div>

        <button
          type="button"
          onClick={
            handleGoogleClick
          }
          disabled={
            submitting ||
            googleSubmitting
          }
          className="sq-control sq-glass"
          style={{
            width:
              '100%',
            padding:
              '12px 14px',
            borderRadius:
              16,
            fontSize:
              14,
            fontWeight:
              650,
            color:
              c.label,
            background:
              c.bgSecondary,
            border:
              `1px solid ${c.separator}`,
            display:
              'flex',
            alignItems:
              'center',
            justifyContent:
              'center',
            gap:
              10,
            opacity:
              googleSubmitting
                ? .7
                : 1,
            boxShadow:
              googleSubmitting
                ? `0 8px 24px -18px ${c.blue}`
                : undefined
          }}
        >
          {googleSubmitting ? (
            <ThinkingOrbs
              label="Connecting…"
            />
          ) : (
            <>
              <GoogleIcon
                size={17}
              />

              <span>
                Continue with Google
              </span>
            </>
          )}
        </button>

        <p
          style={{
            marginTop:
              15,
            textAlign:
              'center',
            color:
              c.labelTertiary,
            fontSize:
              10,
            lineHeight:
              1.45
          }}
        >
          Your progress syncs securely
          across sessions.
        </p>
      </div>
    </div>
  );
}

function Confetti({
  big = false,
  c
}) {
  const pieces =
    useMemo(
      () =>
        Array.from(
          {
            length:
              big ? 42 : 24
          },
          (_, i) => ({
            id: i,
            left:
              Math.random() *
              100,
            delay:
              Math.random() *
              .8,
            duration:
              2.6 +
              Math.random() *
                2.2,
            drift:
              `${-80 + Math.random() * 160}px`,
            rot:
              `${-680 + Math.random() * 1360}deg`,
            size:
              5 +
              Math.random() * 6
          })
        ),
      [big]
    );

  const colors = [
    c.blue,
    c.green,
    c.orange,
    c.purple,
    c.yellow,
    c.pink
  ];

  return (
    <div
      className="fixed inset-0 pointer-events-none z-[100] overflow-hidden"
      aria-hidden="true"
    >
      {pieces.map(
        piece => (
          <span
            key={piece.id}
            className="sq-confetti-piece"
            style={{
              left:
                `${piece.left}%`,

              width:
                piece.size,

              height:
                piece.size * 1.8,

              background:
                colors[
                  piece.id %
                    colors.length
                ],

              animationDuration:
                `${piece.duration}s`,

              animationDelay:
                `${piece.delay}s`,

              '--sq-drift':
                piece.drift,

              '--sq-rot':
                piece.rot
            }}
          />
        )
      )}
    </div>
  );
}

function CompletionView({
  quest,
  onBack,
  c,
  leveledUp,
  accent,
  quote
}) {
  return (
    <div
      className="min-h-full flex flex-col px-4 py-8 sq-anim-in"
      style={{
        color:
          c.label
      }}
    >
      {leveledUp && (
        <Confetti
          big
          c={c}
        />
      )}

      {leveledUp && (
        <span
          className="sq-mono"
          style={{
            alignSelf:
              'center',

            marginBottom:
              12,

            fontSize:
              11,

            fontWeight:
              700,

            letterSpacing:
              .6,

            textTransform:
              'uppercase',

            color:
              '#fff',

            background:
              c.orange,

            padding:
              '5px 14px',

            borderRadius:
              999
          }}
        >
          Level up
        </span>
      )}

      <div
        className="relative mt-2 flex flex-col items-center text-center"
      >
        <div
          className="sq-anim-check sq-icon-fff"
          style={{
            width: 88,
            height: 88,
            borderRadius:
              '50%',
            display:
              'flex',
            alignItems:
              'center',
            justifyContent:
              'center',
            background:
              accent,
            boxShadow:
              `0 20px 45px -20px ${accent}`
          }}
        >
          <CheckIcon
            size={38}
          />
        </div>

        <h2
          style={{
            marginTop:
              18,
            fontSize:
              22,
            fontWeight:
              700,
            color:
              c.label
          }}
        >
          Quest complete
        </h2>

        <span
          className="sq-mono"
          style={{
            marginTop:
              8,
            fontSize:
              13,
            fontWeight:
              700,
            color:
              '#fff',
            background:
              accent,
            padding:
              '4px 12px',
            borderRadius:
              999
          }}
        >
          +{quest?.xp ??
            0}{' '}
          XP
        </span>
      </div>

      <div
        className="mt-8"
      >
        <div
          className="sq-glass"
          style={{
            ...glassStyle(
              c
            ),
            borderRadius:
              20,
            padding:
              20,
            textAlign:
              'center'
          }}
        >
          <p
            style={{
              fontSize:
                15,
              fontWeight:
                500,
              lineHeight:
                1.5,
              color:
                c.label
            }}
          >
            {quote}
          </p>
        </div>
      </div>

      <div
        className="mt-auto pt-6"
      >
        <button
          type="button"
          onClick={
            onBack
          }
          className="sq-control"
          style={{
            width:
              '100%',
            padding:
              '15px',
            borderRadius:
              999,
            fontSize:
              16,
            fontWeight:
              600,
            color:
              c.label,
            background:
              c.fill
          }}
        >
          Back to Quests
        </button>
      </div>
    </div>
  );
}

export default function QuestDailyApp() {
  const [isMounted, setIsMounted] =
    useState(false);

  const {
    user: authUser,
    loading: authLoading,
    authError
  } = useAuthUser();

  const uid =
    authUser?.uid ??
    null;

  const [dark, setDark] =
    useState(true);

  const [showInstallPrompt, setShowInstallPrompt] =
    useState(false);

  const [level, setLevel] =
    useState(1);

  const [xp, setXp] =
    useState(0);

  const [totalXpEarned, setTotalXpEarned] =
    useState(0);

  const [streak, setStreak] =
    useState(0);

  const [username, setUsername] =
    useState(null);

  const [showLeaderboard, setShowLeaderboard] =
    useState(false);

  const [leaderboardSyncError, setLeaderboardSyncError] =
    useState(null);

  const [quests, setQuests] =
    useState([]);

  const [lastReset, setLastReset] =
    useState(0);

  const [proofImages, setProofImages] =
    useState({});

  const [history, setHistory] =
    useState([]);

  const [activeTab, setActiveTab] =
    useState('quests');

  const [photoURL, setPhotoURL] =
    useState(null);

  const [photoUploading, setPhotoUploading] =
    useState(false);

  const [photoError, setPhotoError] =
    useState(null);

  const [hasSeenWelcome, setHasSeenWelcome] =
    useState(true);

  const [hapticsOn, setHapticsOnState] =
    useState(true);

  const [dailyReminderOn, setDailyReminderOn] =
    useState(false);

  const notificationsSupported =
    typeof window !==
      'undefined' &&
    'Notification' in
      window;

  useEffect(() => {
    setIsMounted(true);

    const savedDark =
      localStorage.getItem(
        'sq_dark'
      );

    setDark(
      savedDark !== null
        ? savedDark ===
          'true'
        : window.matchMedia(
            '(prefers-color-scheme: dark)'
          ).matches
    );

    if (
      !localStorage.getItem(
        'sq_has_seen_install_v2'
      )
    ) {
      setShowInstallPrompt(
        true
      );
    }

    setLevel(
      parseInt(
        localStorage.getItem(
          'sq_level'
        )
      ) || 1
    );

    setXp(
      parseInt(
        localStorage.getItem(
          'sq_xp'
        )
      ) || 0
    );

    setTotalXpEarned(
      parseInt(
        localStorage.getItem(
          'sq_totalXpEarned'
        )
      ) || 0
    );

    setStreak(
      parseInt(
        localStorage.getItem(
          'sq_streak'
        )
      ) || 0
    );

    setLastReset(
      parseInt(
        localStorage.getItem(
          'sq_lastReset'
        )
      ) || 0
    );

    setHasSeenWelcome(
      localStorage.getItem(
        'sq_has_seen_welcome'
      ) === 'true'
    );

    const savedHaptics =
      localStorage.getItem(
        'sq_haptics_enabled'
      );

    const hapticsInitial =
      savedHaptics === null
        ? true
        : savedHaptics ===
          'true';

    setHapticsOnState(
      hapticsInitial
    );

    setHapticsPref(
      hapticsInitial
    );
  }, []);

  const c =
    dark ? C.dark : C.light;

  const [cameraQuest, setCameraQuest] =
    useState(null);

  const [completionQuest, setCompletionQuest] =
    useState(null);

  const [leveledUp, setLeveledUp] =
    useState(false);

  const [showSettings, setShowSettings] =
    useState(false);

  const [showProfile, setShowProfile] =
    useState(false);

  const [quote, setQuote] =
    useState(
      'Small progress still counts.'
    );

  const {
    entries: leaderboardEntries,
    status: leaderboardStatus,
    errorDetail: leaderboardError
  } = useLeaderboard();

  useEffect(() => {
    if (!uid) {
      setUsername(null);
      setPhotoURL(null);
      return;
    }

    let cancelled = false;

    (async () => {
      try {
        const profile =
          await fetchRemoteProfile(
            uid
          );

        if (
          cancelled ||
          !profile
        ) {
          return;
        }

        if (
          profile.username &&
          !localStorage.getItem(
            'sq_username'
          )
        ) {
          setUsername(
            profile.username
          );

          localStorage.setItem(
            'sq_username',
            profile.username
          );
        }

        if (profile.photoURL) {
          setPhotoURL(
            profile.photoURL
          );
        }

        if (
          typeof profile.level ===
          'number'
        ) {
          setLevel(
            profile.level
          );
        }

        if (
          typeof profile.xp ===
          'number'
        ) {
          setXp(
            profile.xp
          );
        }

        if (
          typeof profile.totalXpEarned ===
          'number'
        ) {
          setTotalXpEarned(
            profile.totalXpEarned
          );
        }

        if (
          typeof profile.streak ===
          'number'
        ) {
          setStreak(
            profile.streak
          );
        }
      } catch (err) {
        console.error(
          'Account hydration failed:',
          err
        );
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [uid]);

  useEffect(() => {
    if (!uid) return;

    const storedUsername =
      localStorage.getItem(
        'sq_username'
      );

    if (
      storedUsername &&
      !username
    ) {
      setUsername(
        storedUsername
      );
    }
  }, [
    uid,
    username
  ]);

  useEffect(() => {
    if (!uid) return;

    const storedHistory =
      localStorage.getItem(
        'sq_history'
      );

    if (storedHistory) {
      try {
        setHistory(
          JSON.parse(
            storedHistory
          )
        );
      } catch {}
    }

    let cancelled = false;

    fetchRemoteHistory(uid)
      .then(remote => {
        if (
          cancelled ||
          !remote?.length
        ) {
          return;
        }

        setHistory(prev => {
          const merged = [
            ...remote,
            ...prev
          ];

          const map =
            new Map();

          merged.forEach(
            item =>
              map.set(
                item.id,
                item
              )
          );

          const next =
            Array.from(
              map.values()
            )
              .sort(
                (a, b) =>
                  (b.ts || 0) -
                  (a.ts || 0)
              )
              .slice(
                0,
                500
              );

          localStorage.setItem(
            'sq_history',
            JSON.stringify(
              next
            )
          );

          return next;
        });
      })
      .catch(err =>
        console.error(
          'Remote history hydration failed:',
          err
        )
      );

    return () => {
      cancelled = true;
    };
  }, [uid]);

  useEffect(() => {
    if (!isMounted) return;

    localStorage.setItem(
      'sq_dark',
      String(dark)
    );
  }, [
    dark,
    isMounted
  ]);

  useEffect(() => {
    if (!isMounted) return;

    localStorage.setItem(
      'sq_level',
      String(level)
    );
  }, [
    level,
    isMounted
  ]);

  useEffect(() => {
    if (!isMounted) return;

    localStorage.setItem(
      'sq_xp',
      String(xp)
    );
  }, [
    xp,
    isMounted
  ]);

  useEffect(() => {
    if (!isMounted) return;

    localStorage.setItem(
      'sq_totalXpEarned',
      String(totalXpEarned)
    );
  }, [
    totalXpEarned,
    isMounted
  ]);

  useEffect(() => {
    if (!isMounted) return;

    localStorage.setItem(
      'sq_streak',
      String(streak)
    );
  }, [
    streak,
    isMounted
  ]);

  useEffect(() => {
    if (!isMounted) return;

    localStorage.setItem(
      'sq_history',
      JSON.stringify(
        history
      )
    );
  }, [
    history,
    isMounted
  ]);

  useEffect(() => {
    if (!uid) return;
    if (!username) return;

    syncLeaderboardEntry(
      uid,
      {
        username,
        level,
        xp,
        totalXpEarned,
        streak
      }
    ).then(result => {
      if (!result.ok) {
        setLeaderboardSyncError(
          result.error
        );
      } else {
        setLeaderboardSyncError(
          null
        );
      }
    });
  }, [
    uid,
    username,
    level,
    xp,
    totalXpEarned,
    streak
  ]);

  useEffect(() => {
    const generate =
      () => {
        const shuffled =
          [...QUEST_POOL]
            .sort(
              () =>
                Math.random() -
                .5
            )
            .slice(
              0,
              5
            );

        return shuffled.map(
          randomizeQuest
        );
      };

    const saved =
      localStorage.getItem(
        'sq_quests'
      );

    const savedReset =
      parseInt(
        localStorage.getItem(
          'sq_lastReset'
        )
      ) || 0;

    const shouldReset =
      !saved ||
      Date.now() -
        savedReset >=
        ONE_DAY_MS;

    if (shouldReset) {
      const fresh =
        generate();

      setQuests(
        fresh
      );

      const now =
        Date.now();

      setLastReset(
        now
      );

      localStorage.setItem(
        'sq_quests',
        JSON.stringify(
          fresh
        )
      );

      localStorage.setItem(
        'sq_lastReset',
        String(now)
      );
    } else {
      try {
        setQuests(
          JSON.parse(
            saved
          )
        );
      } catch {
        setQuests(
          generate()
        );
      }
    }
  }, []);

  useEffect(() => {
    if (
      quests.length
    ) {
      localStorage.setItem(
        'sq_quests',
        JSON.stringify(
          quests
        )
      );
    }
  }, [quests]);

  const handleSignedUp =
    result => {
      setUsername(
        result.username
      );

      localStorage.setItem(
        'sq_username',
        result.username
      );
    };

  const handleLoggedIn =
    result => {
      setUsername(
        result.username
      );

      localStorage.setItem(
        'sq_username',
        result.username
      );
    };

  const awardXp =
    useCallback(
      amount => {
        setXp(prev => {
          let next =
            prev + amount;

          let nextLevel =
            level;

          let didLevel =
            false;

          while (
            next >=
            nextLevel * 100
          ) {
            next -=
              nextLevel *
              100;

            nextLevel +=
              1;

            didLevel = true;
          }

          if (
            didLevel
          ) {
            setLevel(
              nextLevel
            );

            setLeveledUp(
              true
            );

            setTimeout(
              () =>
                setLeveledUp(
                  false
                ),
              1800
            );
          }

          return next;
        });

        setTotalXpEarned(
          prev =>
            prev +
            amount
        );
      },
      [level]
    );

  const completeQuest =
    useCallback(
      (
        questId,
        progressValue
      ) => {
        setQuests(
          prev =>
            prev.map(
              quest =>
                quest.id ===
                questId
                  ? {
                      ...quest,
                      completed:
                        true,
                      progress:
                        progressValue
                    }
                  : quest
            )
        );

        const quest =
          quests.find(
            q =>
              q.id ===
              questId
          );

        if (!quest) {
          return;
        }

        awardXp(
          quest.xp
        );

        const entry = {
          id:
            `${quest.id}-${Date.now()}-${Math.random()
              .toString(36)
              .slice(2, 8)}`,

          questId:
            quest.id,

          text:
            quest.text,

          xp:
            quest.xp,

          progress:
            progressValue,

          ts:
            Date.now()
        };

        setHistory(
          prev => [
            entry,
            ...prev
          ].slice(
            0,
            500
          )
        );

        if (uid) {
          syncHistoryEntry(
            uid,
            entry
          );
        }

        setCompletionQuest(
          quest
        );

        const quotes = [
          'Consistency compounds.',
          'You showed up. That is the hardest part.',
          'One more step forward.',
          'Progress loves repetition.',
          'Done is better than perfect.',
          'Keep the streak alive.',
          'Future you will thank you.',
        ];

        setQuote(
          quotes[
            Math.floor(
              Math.random() *
                quotes.length
            )
          ]
        );
      },
      [
        quests,
        awardXp,
        uid
      ]
    );

  const openQuest =
    quest => {
      if (
        quest.completed
      ) {
        return;
      }

      setCameraQuest(
        quest
      );
    };

  const handleCameraConfirm =
    result => {
      if (!cameraQuest) {
        return;
      }

      const questId =
        cameraQuest.id;

      setCameraQuest(
        null
      );

      completeQuest(
        questId,
        result?.progress ??
          1
      );
    };

  const handleCameraCancel =
    progress => {
      if (!cameraQuest) {
        return;
      }

      if (
        progress !== undefined &&
        progress > 0
      ) {
        setQuests(
          prev =>
            prev.map(
              q =>
                q.id ===
                cameraQuest.id
                  ? {
                      ...q,
                      progress
                    }
                  : q
            )
        );
      }

      setCameraQuest(
        null
      );
    };

  const currentLevelXp =
    level * 100;

  const levelProgress =
    Math.min(
      100,
      Math.round(
        (xp /
          currentLevelXp) *
          100
      )
    );

  const handleLogout =
    async () => {
      try {
        await logOutAccount();

        setUsername(
          null
        );

        setPhotoURL(
          null
        );

        localStorage.removeItem(
          'sq_username'
        );
      } catch (err) {
        console.error(
          'Logout failed:',
          err
        );
      }
    };

  const handlePhotoUpload =
    async file => {
      if (!uid || !file) {
        return;
      }

      setPhotoError(null);
      setPhotoUploading(true);

      try {
        const dataUrl =
          await resizeImageToSquareDataUrl(
            file
          );

        await updateProfilePhoto(
          uid,
          dataUrl
        );

        setPhotoURL(
          dataUrl
        );
      } catch (err) {
        console.error(
          'Profile photo update failed:',
          err
        );

        setPhotoError(
          err?.message ||
            'Could not update profile photo.'
        );
      } finally {
        setPhotoUploading(false);
      }
    };

  const toggleHaptics =
    enabled => {
      setHapticsOnState(
        enabled
      );

      setHapticsPref(
        enabled
      );
    };

  const handleReminderToggle =
    async enabled => {
      if (
        enabled &&
        notificationsSupported &&
        Notification.permission !==
          'granted'
      ) {
        const permission =
          await Notification.requestPermission();

        if (
          permission !==
          'granted'
        ) {
          setDailyReminderOn(
            false
          );

          return;
        }
      }

      setDailyReminderOn(
        enabled
      );

      localStorage.setItem(
        'sq_daily_reminder',
        enabled
          ? 'true'
          : 'false'
      );
    };

  useEffect(() => {
    const saved =
      localStorage.getItem(
        'sq_daily_reminder'
      );

    setDailyReminderOn(
      saved ===
        'true'
    );
  }, []);

  const resetAllQuests =
    () => {
      const fresh =
        [...QUEST_POOL]
          .sort(
            () =>
              Math.random() -
              .5
          )
          .slice(
            0,
            5
          )
          .map(
            randomizeQuest
          );

      setQuests(
        fresh
      );

      const now =
        Date.now();

      setLastReset(
        now
      );

      localStorage.setItem(
        'sq_quests',
        JSON.stringify(
          fresh
        )
      );

      localStorage.setItem(
        'sq_lastReset',
        String(now)
      );
    };

  if (!isMounted) {
    return null;
  }

  if (authLoading) {
    return (
      <div
        className="sq-root fixed inset-0 flex items-center justify-center"
        style={{
          background:
            C.dark.bg,
          color:
            '#fff'
        }}
      >
        <ThinkingOrbs
          label="Loading QuestDaily…"
        />
      </div>
    );
  }

  if (
    cameraQuest
  ) {
    return (
      <div
        className="sq-root min-h-screen"
        style={{
          background:
            c.bg,
          color:
            c.label
        }}
      >
        <SystemType />

        <CameraModal
          quest={
            cameraQuest
          }
          onConfirm={
            handleCameraConfirm
          }
          onCancel={
            handleCameraCancel
          }
          c={c}
        />
      </div>
    );
  }

  if (
    completionQuest
  ) {
    return (
      <div
        className="sq-root min-h-screen"
        style={{
          background:
            c.bg,
          color:
            c.label
        }}
      >
        <SystemType />

        <AmbientBackground
          c={c}
        />

        <div
          className="relative z-10 min-h-screen"
        >
          <CompletionView
            quest={
              completionQuest
            }
            onBack={() =>
              setCompletionQuest(
                null
              )
            }
            c={c}
            leveledUp={
              leveledUp
            }
            accent={
              c.green
            }
            quote={
              quote
            }
          />
        </div>
      </div>
    );
  }

  return (
    <div
      className="sq-root min-h-screen"
      style={{
        background:
          c.bg,
        color:
          c.label
      }}
    >
      <SystemType />

      <AmbientBackground
        c={c}
      />

      <div
        className="relative z-10 min-h-screen pb-24"
      >
        <header
          className="sticky top-0 z-30"
          style={{
            paddingTop:
              'max(env(safe-area-inset-top), 12px)'
          }}
        >
          <div
            className="mx-3 sq-glass"
            style={{
              ...glassStyle(
                c,
                true
              ),

              borderRadius:
                20,

              padding:
                '11px 13px',

              display:
                'flex',

              alignItems:
                'center',

              justifyContent:
                'space-between'
            }}
          >
            <div
              style={{
                minWidth:
                  0
              }}
            >
              <p
                style={{
                  fontSize:
                    11,
                  color:
                    c.labelSecondary,
                  fontWeight:
                    650,
                  marginBottom:
                    2
                }}
              >
                {activeTab ===
                'quests'
                  ? 'TODAY'
                  : activeTab ===
                      'history'
                    ? 'HISTORY'
                    : activeTab ===
                        'settings'
                      ? 'SETTINGS'
                      : 'PROFILE'}
              </p>

              <h1
                className="sq-large-title"
                style={{
                  fontSize:
                    24,
                  lineHeight:
                    1.1,
                  fontWeight:
                    750
                }}
              >
                QuestDaily
              </h1>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                className="sq-control"
                onClick={() =>
                  setDark(
                    d => !d
                  )
                }
                aria-label="Toggle theme"
                style={{
                  width:
                    42,
                  height:
                    42,
                  minHeight:
                    42,
                  borderRadius:
                    14,
                  background:
                    c.fill,
                  color:
                    c.label,
                  display:
                    'flex',
                  alignItems:
                    'center',
                  justifyContent:
                    'center'
                }}
              >
                <span className="sq-icon-tap">
                  {dark ? (
                    <Sun
                      size={
                        17
                      }
                    />
                  ) : (
                    <Moon
                      size={
                        17
                      }
                    />
                  )}
                </span>
              </button>

              {username && (
                <button
                  type="button"
                  className="sq-control"
                  onClick={() =>
                    setShowProfile(
                      true
                    )
                  }
                  aria-label="Profile"
                  style={{
                    width:
                      42,
                    height:
                      42,
                    minHeight:
                      42,
                    borderRadius:
                      '50%',
                    overflow:
                      'hidden',
                    background:
                      c.fill,
                    color:
                      c.label,
                    display:
                      'flex',
                    alignItems:
                      'center',
                    justifyContent:
                      'center'
                  }}
                >
                  {photoURL ? (
                    <img
                      src={
                        photoURL
                      }
                      alt=""
                      style={{
                        width:
                          '100%',
                        height:
                          '100%',
                        objectFit:
                          'cover'
                      }}
                    />
                  ) : (
                    <CircleUserRound
                      size={
                        19
                      }
                    />
                  )}
                </button>
              )}
            </div>
          </div>
        </header>

        <main className="px-4 pt-5">
          {activeTab ===
            'quests' && (
            <>
              <section
                className="sq-anim-in"
                style={{
                  ...glassStyle(
                    c,
                    true
                  ),

                  borderRadius:
                    24,

                  padding:
                    18
                }}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p
                      style={{
                        fontSize:
                          11,
                        fontWeight:
                          650,
                        color:
                          c.labelSecondary,
                        textTransform:
                          'uppercase',
                        letterSpacing:
                          .5
                      }}
                    >
                      Level {level}
                    </p>

                    <h2
                      className="sq-title"
                      style={{
                        fontSize:
                          27,
                        fontWeight:
                          750,
                        marginTop:
                          4
                      }}
                    >
                      {getLevelTitle(
                        level
                      )}
                    </h2>
                  </div>

                  <div
                    className="sq-mono"
                    style={{
                      fontSize:
                        12,
                      fontWeight:
                        700,
                      color:
                        c.blue,
                      padding:
                        '7px 10px',
                      borderRadius:
                        12,
                      background:
                        c.fill
                    }}
                  >
                    {totalXpEarned}{' '}
                    XP
                  </div>
                </div>

                <div
                  className="mt-5"
                  style={{
                    height:
                      7,
                    borderRadius:
                      999,
                    background:
                      c.fill,
                    overflow:
                      'hidden'
                  }}
                >
                  <div
                    style={{
                      height:
                        '100%',
                      width: `${levelProgress}%`,
                      borderRadius:
                        999,
                      background:
                        `linear-gradient(90deg, ${c.blue}, ${c.indigo})`,
                      transition:
                        'width .7s cubic-bezier(.22,1,.36,1)'
                    }}
                  />
                </div>

                <div className="flex items-center justify-between mt-2">
                  <p
                    className="sq-mono"
                    style={{
                      fontSize:
                        10,
                      color:
                        c.labelSecondary
                    }}
                  >
                    {xp} /{' '}
                    {currentLevelXp}{' '}
                    XP
                  </p>

                  <p
                    style={{
                      fontSize:
                        10,
                      color:
                        c.labelSecondary
                    }}
                  >
                    {streak > 0
                      ? `${streak} day streak`
                      : 'Start your streak'}
                  </p>
                </div>
              </section>

              <div className="flex items-center justify-between mt-7 mb-3">
                <div>
                  <h2
                    className="sq-title"
                    style={{
                      fontSize:
                        21,
                      fontWeight:
                        750
                    }}
                  >
                    Today's quests
                  </h2>

                  <p
                    style={{
                      marginTop:
                        3,
                      fontSize:
                        11,
                      color:
                        c.labelSecondary
                    }}
                  >
                    Complete them to
                    earn XP.
                  </p>
                </div>

                <button
                  type="button"
                  onClick={
                    resetAllQuests
                  }
                  className="sq-control"
                  style={{
                    padding:
                      '8px 11px',
                    borderRadius:
                      12,
                    background:
                      c.fill,
                    color:
                      c.labelSecondary,
                    fontSize:
                      11,
                    fontWeight:
                      650
                  }}
                >
                  Refresh
                </button>
              </div>

              <div className="space-y-3">
                {quests.map(
                  (
                    quest,
                    index
                  ) => {
                    const q =
                      QUEST_LABELS[
                        quest.id
                      ];

                    return (
                      <button
                        key={
                          `${quest.id}-${index}`
                        }
                        type="button"
                        disabled={
                          quest.completed
                        }
                        onClick={() =>
                          openQuest(
                            quest
                          )
                        }
                        className="sq-control sq-glass w-full text-left sq-anim-in"
                        style={{
                          ...glassStyle(
                            c
                          ),

                          borderRadius:
                            22,

                          padding:
                            15,

                          opacity:
                            quest.completed
                              ? .62
                              : 1,

                          animationDelay:
                            `${index * 35}ms`
                        }}
                      >
                        <div className="flex items-center gap-3">
                          <div
                            style={{
                              width:
                                46,
                              height:
                                46,
                              flexShrink:
                                0,
                              borderRadius:
                                15,
                              background:
                                quest.completed
                                  ? c.green
                                  : c.fill,
                              display:
                                'flex',
                              alignItems:
                                'center',
                              justifyContent:
                                'center',
                              color:
                                quest.completed
                                  ? '#fff'
                                  : c.blue
                            }}
                          >
                            {quest.completed ? (
                              <CheckIcon
                                size={
                                  20
                                }
                              />
                            ) : q?.type ===
                                'reps' ? (
                              <Dumbbell
                                size={
                                  20
                                }
                              />
                            ) : q?.type ===
                                'food' ? (
                              <Utensils
                                size={
                                  20
                                }
                              />
                            ) : q?.type ===
                                'map' ? (
                              <Target
                                size={
                                  20
                                }
                              />
                            ) : (
                              <Sparkles
                                size={
                                  20
                                }
                              />
                            )}
                          </div>

                          <div
                            className="min-w-0 flex-1"
                          >
                            <p
                              style={{
                                fontSize:
                                  14,
                                fontWeight:
                                  650,
                                color:
                                  c.label,
                                lineHeight:
                                  1.3
                              }}
                            >
                              {quest.text}
                            </p>

                            <div className="flex items-center gap-2 mt-1.5">
                              <span
                                className="sq-mono"
                                style={{
                                  fontSize:
                                    10,
                                  fontWeight:
                                    700,
                                  color:
                                    c.blue
                                }}
                              >
                                +{quest.xp}{' '}
                                XP
                              </span>

                              {quest.reps && (
                                <span
                                  style={{
                                    fontSize:
                                      10,
                                    color:
                                      c.labelSecondary
                                  }}
                                >
                                  {quest.progress ||
                                    0}{' '}
                                  /{' '}
                                  {quest.reps}
                                </span>
                              )}
                            </div>
                          </div>

                          <ChevronRight
                            size={
                              18
                            }
                            color={
                              quest.completed
                                ? c.labelTertiary
                                : c.labelSecondary
                            }
                          />
                        </div>

                        {quest.reps &&
                          !quest.completed && (
                            <div
                              className="mt-3"
                              style={{
                                height:
                                  4,
                                borderRadius:
                                  999,
                                background:
                                  c.fill,
                                overflow:
                                  'hidden'
                              }}
                            >
                              <div
                                style={{
                                  width: `${Math.min(
                                    100,
                                    ((quest.progress ||
                                      0) /
                                      quest.reps) *
                                      100
                                  )}%`,
                                  height:
                                    '100%',
                                  background:
                                    c.blue,
                                  borderRadius:
                                    999,
                                  transition:
                                    'width .4s ease'
                                }}
                              />
                            </div>
                          )}
                      </button>
                    );
                  }
                )}
              </div>
            </>
          )}

          {activeTab ===
            'history' && (
            <section
              className="sq-anim-in"
            >
              <h2
                className="sq-title"
                style={{
                  fontSize:
                    24,
                  fontWeight:
                    750
                }}
              >
                History
              </h2>

              <p
                style={{
                  marginTop:
                    4,
                  fontSize:
                    12,
                  color:
                    c.labelSecondary
                }}
              >
                Your completed quests.
              </p>

              <div className="space-y-3 mt-5">
                {history.length ===
                0 ? (
                  <div
                    className="sq-glass"
                    style={{
                      ...glassStyle(
                        c
                      ),
                      borderRadius:
                        22,
                      padding:
                        24,
                      textAlign:
                        'center'
                    }}
                  >
                    <HistoryIcon
                      size={30}
                      color={
                        c.labelSecondary
                      }
                    />

                    <p
                      style={{
                        marginTop:
                          10,
                        fontSize:
                          14,
                        fontWeight:
                          600
                      }}
                    >
                      Nothing here yet
                    </p>

                    <p
                      style={{
                        marginTop:
                          4,
                        fontSize:
                          11,
                        color:
                          c.labelSecondary
                      }}
                    >
                      Complete your
                      first quest.
                    </p>
                  </div>
                ) : (
                  history.map(
                    (
                      entry,
                      index
                    ) => (
                      <div
                        key={
                          entry.id
                        }
                        className="sq-glass sq-anim-in"
                        style={{
                          ...glassStyle(
                            c
                          ),
                          borderRadius:
                            20,
                          padding:
                            15,
                          animationDelay:
                            `${index * 25}ms`
                        }}
                      >
                        <div className="flex items-center gap-3">
                          <div
                            style={{
                              width:
                                40,
                              height:
                                40,
                              borderRadius:
                                13,
                              display:
                                'flex',
                              alignItems:
                                'center',
                              justifyContent:
                                'center',
                              background:
                                c.fill,
                              color:
                                c.green
                            }}
                          >
                            <CheckIcon
                              size={
                                17
                              }
                            />
                          </div>

                          <div className="flex-1 min-w-0">
                            <p
                              style={{
                                fontSize:
                                  13,
                                fontWeight:
                                  650
                              }}
                            >
                              {entry.text}
                            </p>

                            <p
                              style={{
                                marginTop:
                                  3,
                                fontSize:
                                  10,
                                color:
                                  c.labelSecondary
                              }}
                            >
                              {new Date(
                                entry.ts
                              ).toLocaleString()}
                            </p>
                          </div>

                          <span
                            className="sq-mono"
                            style={{
                              fontSize:
                                11,
                              fontWeight:
                                700,
                              color:
                                c.green
                            }}
                          >
                            +{entry.xp}
                          </span>
                        </div>
                      </div>
                    )
                  )
                )}
              </div>
            </section>
          )}

          {activeTab ===
            'settings' && (
            <section className="sq-anim-in">
              <h2
                className="sq-title"
                style={{
                  fontSize:
                    24,
                  fontWeight:
                    750
                }}
              >
                Settings
              </h2>

              <div className="space-y-3 mt-5">
                <div
                  className="sq-glass"
                  style={{
                    ...glassStyle(
                      c
                    ),
                    borderRadius:
                      20,
                    overflow:
                      'hidden'
                  }}
                >
                  <div
                    className="flex items-center justify-between px-4 py-4"
                  >
                    <div>
                      <p
                        style={{
                          fontSize:
                            14,
                          fontWeight:
                            650
                        }}
                      >
                        Haptics
                      </p>

                      <p
                        style={{
                          marginTop:
                            3,
                          fontSize:
                            10,
                          color:
                            c.labelSecondary
                        }}
                      >
                        Vibrate on important actions.
                      </p>
                    </div>

                    <button
                      type="button"
                      onClick={() =>
                        toggleHaptics(
                          !hapticsOn
                        )
                      }
                      style={{
                        width:
                          49,
                        height:
                          30,
                        borderRadius:
                          999,
                        padding:
                          3,
                        background:
                          hapticsOn
                            ? c.green
                            : c.fill,
                        transition:
                          'background .25s ease'
                      }}
                    >
                      <span
                        style={{
                          display:
                            'block',
                          width:
                            24,
                          height:
                            24,
                          borderRadius:
                            '50%',
                          background:
                            '#fff',
                          transform:
                            `translateX(${hapticsOn ? 19 : 0}px)`,
                          transition:
                            'transform .28s cubic-bezier(.22,1,.36,1)',
                          boxShadow:
                            '0 3px 10px -5px rgba(0,0,0,.55)'
                        }}
                      />
                    </button>
                  </div>

                  <div
                    style={{
                      height:
                        1,
                      background:
                        c.separator
                    }}
                  />

                  <div
                    className="flex items-center justify-between px-4 py-4"
                  >
                    <div>
                      <p
                        style={{
                          fontSize:
                            14,
                          fontWeight:
                            650
                        }}
                      >
                        Daily reminders
                      </p>

                      <p
                        style={{
                          marginTop:
                            3,
                          fontSize:
                            10,
                          color:
                            c.labelSecondary
                        }}
                      >
                        Get a browser reminder to keep your streak.
                      </p>
                    </div>

                    <button
                      type="button"
                      onClick={() =>
                        handleReminderToggle(
                          !dailyReminderOn
                        )
                      }
                      style={{
                        width:
                          49,
                        height:
                          30,
                        borderRadius:
                          999,
                        padding:
                          3,
                        background:
                          dailyReminderOn
                            ? c.green
                            : c.fill
                      }}
                    >
                      <span
                        style={{
                          display:
                            'block',
                          width:
                            24,
                          height:
                            24,
                          borderRadius:
                            '50%',
                          background:
                            '#fff',
                          transform:
                            `translateX(${dailyReminderOn ? 19 : 0}px)`,
                          transition:
                            'transform .28s cubic-bezier(.22,1,.36,1)',
                          boxShadow:
                            '0 3px 10px -5px rgba(0,0,0,.55)'
                        }}
                      />
                    </button>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={
                    handleLogout
                  }
                  className="sq-control sq-glass"
                  style={{
                    ...glassStyle(
                      c
                    ),

                    width:
                      '100%',

                    borderRadius:
                      20,

                    padding:
                      16,

                    display:
                      'flex',

                    alignItems:
                      'center',

                    gap:
                      12,

                    color:
                      c.red,

                    fontSize:
                      14,

                    fontWeight:
                      650
                  }}
                >
                  <LogOut
                    size={
                      18
                    }
                  />

                  Log out
                </button>

                {leaderboardSyncError && (
                  <div
                    style={{
                      borderRadius:
                        15,
                      background:
                        'rgba(255,159,10,.12)',
                      border:
                        '1px solid rgba(255,159,10,.2)',
                      padding:
                        '11px 13px',
                      fontSize:
                        11,
                      color:
                        c.orange,
                      lineHeight:
                        1.45
                    }}
                  >
                    {leaderboardSyncError}
                  </div>
                )}
              </div>
            </section>
          )}

          {activeTab ===
            'profile' && (
            <section className="sq-anim-in">
              <h2
                className="sq-title"
                style={{
                  fontSize:
                    24,
                  fontWeight:
                    750
                }}
              >
                Profile
              </h2>

              <div
                className="sq-glass mt-5"
                style={{
                  ...glassStyle(
                    c,
                    true
                  ),
                  borderRadius:
                    24,
                  padding:
                    20
                }}
              >
                <div className="flex flex-col items-center text-center">
                  <label
                    style={{
                      position:
                        'relative',
                      width:
                        88,
                      height:
                        88,
                      borderRadius:
                        '50%',
                      overflow:
                        'hidden',
                      cursor:
                        'pointer',
                      background:
                        c.fill,
                      display:
                        'flex',
                      alignItems:
                        'center',
                      justifyContent:
                        'center'
                    }}
                  >
                    {photoURL ? (
                      <img
                        src={
                          photoURL
                        }
                        alt=""
                        style={{
                          width:
                            '100%',
                          height:
                            '100%',
                          objectFit:
                            'cover'
                        }}
                      />
                    ) : (
                      <CircleUserRound
                        size={
                          38
                        }
                        color={
                          c.labelSecondary
                        }
                      />
                    )}

                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={e =>
                        handlePhotoUpload(
                          e.target.files?.[0]
                        )
                      }
                    />

                    {photoUploading && (
                      <div
                        className="absolute inset-0 flex items-center justify-center"
                        style={{
                          background:
                            'rgba(0,0,0,.45)'
                        }}
                      >
                        <ThinkingOrbs
                          label=""
                        />
                      </div>
                    )}
                  </label>

                  <h3
                    style={{
                      fontSize:
                        20,
                      fontWeight:
                        700,
                      marginTop:
                        14
                    }}
                  >
                    {username ||
                      'Player'}
                  </h3>

                  <p
                    style={{
                      marginTop:
                        4,
                      fontSize:
                        11,
                      color:
                        c.labelSecondary
                    }}
                  >
                    {getLevelTitle(
                      level
                    )}{' '}
                    · Level {level}
                  </p>

                  {photoError && (
                    <p
                      style={{
                        marginTop:
                          8,
                        color:
                          c.red,
                        fontSize:
                          11
                      }}
                    >
                      {photoError}
                    </p>
                  )}
                </div>

                <div className="grid grid-cols-3 gap-2 mt-6">
                  {[
                    [
                      'XP',
                      totalXpEarned
                    ],
                    [
                      'Streak',
                      streak
                    ],
                    [
                      'Done',
                      history.length
                    ]
                  ].map(
                    ([label, value]) => (
                      <div
                        key={
                          label
                        }
                        style={{
                          borderRadius:
                            16,
                          background:
                            c.fill,
                          padding:
                            '11px 8px',
                          textAlign:
                            'center'
                        }}
                      >
                        <p
                          className="sq-mono"
                          style={{
                            fontSize:
                              17,
                            fontWeight:
                              750
                          }}
                        >
                          {value}
                        </p>

                        <p
                          style={{
                            marginTop:
                              3,
                            fontSize:
                              9,
                            color:
                              c.labelSecondary
                          }}
                        >
                          {label}
                        </p>
                      </div>
                    )
                  )}
                </div>
              </div>
            </section>
          )}

          <section className="mt-7">
            <div className="flex items-center justify-between mb-3">
              <div>
                <p
                  style={{
                    fontSize:
                      10,
                    fontWeight:
                      700,
                    color:
                      c.labelSecondary,
                    textTransform:
                      'uppercase',
                    letterSpacing:
                      .5
                  }}
                >
                  Leaderboard
                </p>

                <h2
                  className="sq-title"
                  style={{
                    fontSize:
                      20,
                    fontWeight:
                      750
                  }}
                >
                  Top players
                </h2>
              </div>

              <button
                type="button"
                onClick={() =>
                  setShowLeaderboard(
                    true
                  )
                }
                className="sq-control"
                style={{
                  padding:
                    '8px 11px',
                  borderRadius:
                    12,
                  background:
                    c.fill,
                  color:
                    c.labelSecondary,
                  fontSize:
                    11,
                  fontWeight:
                    650
                }}
              >
                View all
              </button>
            </div>

            <div
              className="sq-glass"
              style={{
                ...glassStyle(
                  c
                ),
                borderRadius:
                  20,
                overflow:
                  'hidden'
              }}
            >
              {leaderboardStatus ===
              'loading' ? (
                <div
                  className="flex items-center justify-center py-7"
                >
                  <ThinkingOrbs
                    label="Loading…"
                  />
                </div>
              ) : leaderboardStatus ===
                  'error' ? (
                <div
                  style={{
                    padding:
                      18,
                    fontSize:
                      11,
                    lineHeight:
                      1.45,
                    color:
                      c.labelSecondary
                  }}
                >
                  {leaderboardError}
                </div>
              ) : (
                leaderboardEntries
                  .slice(
                    0,
                    5
                  )
                  .map(
                    (
                      entry,
                      index
                    ) => (
                      <div
                        key={
                          entry.id
                        }
                        className="flex items-center gap-3 px-4 py-3"
                        style={{
                          borderBottom:
                            index <
                            Math.min(
                              4,
                              leaderboardEntries.length -
                                1
                            )
                              ? `1px solid ${c.separator}`
                              : 'none'
                        }}
                      >
                        <span
                          className="sq-mono"
                          style={{
                            width:
                              22,
                            fontSize:
                              11,
                            fontWeight:
                              750,
                            color:
                              index ===
                              0
                                ? c.orange
                                : c.labelSecondary,
                            textAlign:
                              'center'
                          }}
                        >
                          {index +
                            1}
                        </span>

                        <div
                          style={{
                            width:
                              32,
                            height:
                              32,
                            borderRadius:
                              '50%',
                            overflow:
                              'hidden',
                            background:
                              c.fill,
                            display:
                              'flex',
                            alignItems:
                              'center',
                            justifyContent:
                              'center',
                            flexShrink:
                              0
                          }}
                        >
                          {entry.photoURL ? (
                            <img
                              src={
                                entry.photoURL
                              }
                              alt=""
                              style={{
                                width:
                                  '100%',
                                height:
                                  '100%',
                                objectFit:
                                  'cover'
                              }}
                            />
                          ) : (
                            <CircleUserRound
                              size={
                                16
                              }
                              color={
                                c.labelSecondary
                              }
                            />
                          )}
                        </div>

                        <div className="flex-1 min-w-0">
                          <p
                            style={{
                              fontSize:
                                12,
                              fontWeight:
                                650,
                              whiteSpace:
                                'nowrap',
                              overflow:
                                'hidden',
                              textOverflow:
                                'ellipsis'
                            }}
                          >
                            {entry.username ||
                              'Player'}
                          </p>

                          <p
                            style={{
                              marginTop:
                                2,
                              fontSize:
                                9,
                              color:
                                c.labelSecondary
                            }}
                          >
                            Level{' '}
                            {entry.level ??
                              1}
                          </p>
                        </div>

                        <span
                          className="sq-mono"
                          style={{
                            fontSize:
                              11,
                            fontWeight:
                              700,
                            color:
                              c.blue
                          }}
                        >
                          {entry.totalXpEarned ??
                            0}
                        </span>
                      </div>
                    )
                  )
              )}
            </div>
          </section>
        </main>

        <nav
          className="fixed bottom-0 inset-x-0 z-40 px-3 pb-3"
          style={{
            paddingBottom:
              'max(env(safe-area-inset-bottom), 12px)'
          }}
        >
          <div
            className="sq-glass"
            style={{
              ...glassStyle(
                c,
                true
              ),

              borderRadius:
                22,

              padding:
                '7px 7px',

              display:
                'grid',

              gridTemplateColumns:
                'repeat(4, 1fr)',

              gap:
                4,

              maxWidth:
                560,

              margin:
                '0 auto'
            }}
          >
            {[
              [
                'quests',
                CheckSquare,
                'Quests'
              ],
              [
                'history',
                HistoryIcon,
                'History'
              ],
              [
                'settings',
                SettingsIcon,
                'Settings'
              ],
              [
                'profile',
                CircleUserRound,
                'Profile'
              ]
            ].map(
              ([
                tab,
                Icon,
                label
              ]) => (
                <button
                  key={tab}
                  type="button"
                  onClick={() =>
                    setActiveTab(
                      tab
                    )
                  }
                  className="sq-control"
                  style={{
                    minHeight:
                      52,
                    borderRadius:
                      16,
                    background:
                      activeTab ===
                      tab
                        ? c.fill
                        : 'transparent',
                    color:
                      activeTab ===
                      tab
                        ? c.label
                        : c.labelSecondary,
                    display:
                      'flex',
                    flexDirection:
                      'column',
                    alignItems:
                      'center',
                    justifyContent:
                      'center',
                    gap:
                      4,
                    fontSize:
                      9,
                    fontWeight:
                      650
                  }}
                >
                  <Icon
                    size={
                      18
                    }
                    strokeWidth={
                      activeTab ===
                      tab
                        ? 2.2
                        : 1.7
                    }
                    className={`sq-tab-icon ${
                      activeTab ===
                      tab
                        ? 'sq-tab-icon-active'
                        : ''
                    }`}
                  />

                  <span>
                    {label}
                  </span>
                </button>
              )
            )}
          </div>
        </nav>

        {showLeaderboard && (
          <div
            className="fixed inset-0 z-[70] flex items-end justify-center sq-modal-in"
            style={{
              background:
                'rgba(0,0,0,.52)',
              backdropFilter:
                'blur(12px)',
              WebkitBackdropFilter:
                'blur(12px)'
            }}
          >
            <div
              className="sq-glass"
              style={{
                ...glassStyle(
                  c,
                  true
                ),
                background:
                  c.bg,
                width:
                  '100%',
                maxWidth:
                  620,
                maxHeight:
                  '86vh',
                borderTopLeftRadius:
                  26,
                borderTopRightRadius:
                  26,
                overflow:
                  'hidden'
              }}
            >
              <div
                className="flex items-center justify-between px-5 pt-5 pb-4"
                style={{
                  borderBottom:
                    `1px solid ${c.separator}`
                }}
              >
                <div>
                  <p
                    style={{
                      fontSize:
                        11,
                      color:
                        c.labelSecondary,
                      fontWeight:
                        650
                    }}
                  >
                    GLOBAL
                  </p>

                  <h2
                    style={{
                      marginTop:
                        3,
                      fontSize:
                        24,
                      fontWeight:
                        750
                    }}
                  >
                    Leaderboard
                  </h2>
                </div>

                <button
                  type="button"
                  className="sq-control"
                  onClick={() =>
                    setShowLeaderboard(
                      false
                    )
                  }
                  style={{
                    width:
                      38,
                    height:
                      38,
                    minHeight:
                      38,
                    borderRadius:
                      '50%',
                    background:
                      c.fill,
                    color:
                      c.label,
                    display:
                      'flex',
                    alignItems:
                      'center',
                    justifyContent:
                      'center'
                  }}
                >
                  ×
                </button>
              </div>

              <div
                className="sq-scroll overflow-y-auto"
                style={{
                  maxHeight:
                    'calc(86vh - 95px)'
                }}
              >
                {leaderboardEntries.map(
                  (
                    entry,
                    index
                  ) => (
                    <div
                      key={
                        entry.id
                      }
                      className="flex items-center gap-3 px-5 py-4"
                      style={{
                        borderBottom:
                          `1px solid ${c.separator}`
                      }}
                    >
                      <div
                        style={{
                          width:
                            34,
                          textAlign:
                            'center'
                        }}
                      >
                        <span
                          className="sq-mono"
                          style={{
                            fontSize:
                              13,
                            fontWeight:
                              750,
                            color:
                              index ===
                              0
                                ? c.orange
                                : index ===
                                    1
                                  ? c.labelSecondary
                                  : index ===
                                      2
                                    ? c.orange
                                    : c.labelSecondary
                          }}
                        >
                          {index +
                            1}
                        </span>
                      </div>

                      <div
                        style={{
                          width:
                            40,
                          height:
                            40,
                          borderRadius:
                            '50%',
                          overflow:
                            'hidden',
                          background:
                            c.fill,
                          display:
                            'flex',
                          alignItems:
                            'center',
                          justifyContent:
                            'center',
                          flexShrink:
                            0
                        }}
                      >
                        {entry.photoURL ? (
                          <img
                            src={
                              entry.photoURL
                            }
                            alt=""
                            style={{
                              width:
                                '100%',
                              height:
                                '100%',
                              objectFit:
                                'cover'
                            }}
                          />
                        ) : (
                          <CircleUserRound
                            size={
                              19
                            }
                            color={
                              c.labelSecondary
                            }
                          />
                        )}
                      </div>

                      <div className="flex-1 min-w-0">
                        <p
                          style={{
                            fontSize:
                              13,
                            fontWeight:
                              650
                          }}
                        >
                          {entry.username ||
                            'Player'}
                        </p>

                        <p
                          style={{
                            marginTop:
                              3,
                            fontSize:
                              10,
                            color:
                              c.labelSecondary
                          }}
                        >
                          Level{' '}
                          {entry.level ??
                            1}{' '}
                          ·{' '}
                          {getLevelTitle(
                            entry.level ??
                              1
                          )}
                        </p>
                      </div>

                      <div className="text-right">
                        <p
                          className="sq-mono"
                          style={{
                            fontSize:
                              12,
                            fontWeight:
                              750,
                            color:
                              c.blue
                          }}
                        >
                          {entry.totalXpEarned ??
                            0}
                        </p>

                        <p
                          style={{
                            marginTop:
                              2,
                            fontSize:
                              9,
                            color:
                              c.labelSecondary
                          }}
                        >
                          XP
                        </p>
                      </div>
                    </div>
                  )
                )}
              </div>
            </div>
          </div>
        )}

        {showProfile && (
          <div
            className="fixed inset-0 z-[75] flex items-center justify-center px-4 sq-modal-in"
            style={{
              background:
                'rgba(0,0,0,.50)',
              backdropFilter:
                'blur(14px)',
              WebkitBackdropFilter:
                'blur(14px)'
            }}
          >
            <div
              className="sq-glass"
              style={{
                ...glassStyle(
                  c,
                  true
                ),
                width:
                  '100%',
                maxWidth:
                  420,
                borderRadius:
                  25,
                padding:
                  20
              }}
            >
              <div className="flex items-center justify-between mb-4">
                <h2
                  style={{
                    fontSize:
                      20,
                    fontWeight:
                      750
                  }}
                >
                  Profile
                </h2>

                <button
                  type="button"
                  onClick={() =>
                    setShowProfile(
                      false
                    )
                  }
                  className="sq-control"
                  style={{
                    width:
                      38,
                    height:
                      38,
                    minHeight:
                      38,
                    borderRadius:
                      '50%',
                    background:
                      c.fill,
                    color:
                      c.label,
                    display:
                      'flex',
                    alignItems:
                      'center',
                    justifyContent:
                      'center'
                  }}
                >
                  ×
                </button>
              </div>

              <div className="flex flex-col items-center">
                <div
                  style={{
                    width:
                      82,
                    height:
                      82,
                    borderRadius:
                      '50%',
                    overflow:
                      'hidden',
                    background:
                      c.fill,
                    display:
                      'flex',
                    alignItems:
                      'center',
                    justifyContent:
                      'center'
                  }}
                >
                  {photoURL ? (
                    <img
                      src={
                        photoURL
                      }
                      alt=""
                      style={{
                        width:
                          '100%',
                        height:
                          '100%',
                        objectFit:
                          'cover'
                      }}
                    />
                  ) : (
                    <CircleUserRound
                      size={
                        36
                      }
                      color={
                        c.labelSecondary
                      }
                    />
                  )}
                </div>

                <p
                  style={{
                    marginTop:
                      12,
                    fontSize:
                      18,
                    fontWeight:
                      700
                  }}
                >
                  {username ||
                    'Player'}
                </p>

                <p
                  style={{
                    marginTop:
                      4,
                    fontSize:
                      11,
                    color:
                      c.labelSecondary
                  }}
                >
                  Level{' '}
                  {level}
                  {' · '}
                  {getLevelTitle(
                    level
                  )}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3 mt-6">
                <div
                  style={{
                    borderRadius:
                      17,
                    background:
                      c.fill,
                    padding:
                      15
                  }}
                >
                  <p
                    style={{
                      fontSize:
                        10,
                      color:
                        c.labelSecondary
                    }}
                  >
                    XP
                  </p>

                  <p
                    className="sq-mono"
                    style={{
                      marginTop:
                        4,
                      fontSize:
                        20,
                      fontWeight:
                        750
                    }}
                  >
                    {totalXpEarned}
                  </p>
                </div>

                <div
                  style={{
                    borderRadius:
                      17,
                    background:
                      c.fill,
                    padding:
                      15
                  }}
                >
                  <p
                    style={{
                      fontSize:
                        10,
                      color:
                        c.labelSecondary
                    }}
                  >
                    Streak
                  </p>

                  <p
                    className="sq-mono"
                    style={{
                      marginTop:
                        4,
                      fontSize:
                        20,
                      fontWeight:
                        750
                    }}
                  >
                    {streak}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {!authUser &&
          !cameraQuest &&
          !completionQuest &&
          !showInstallPrompt && (
            <AuthModal
              onSignedUp={
                handleSignedUp
              }
              onLoggedIn={
                handleLoggedIn
              }
              c={c}
              initialError={
                authError
              }
            />
          )}
      </div>
    </div>
  );
}
