// firebase.js
//
// This connects QuestDaily to a shared Firestore database so the leaderboard
// is the same on every phone, not just stored locally on one device.
//
// SETUP (takes about 5 minutes, free):
// 1. Go to https://console.firebase.google.com and click "Add project".
//    Give it any name (e.g. "quest-daily") and finish creation.
// 2. In the left sidebar, go to Build → Firestore Database → "Create database".
//    Choose "Start in test mode" (fine for a small app between you and your brother).
// 3. In the left sidebar, click the gear icon → Project settings.
//    Under "Your apps", click the "</>" (Web) icon to register a web app.
// 4. Copy the firebaseConfig object it gives you and paste the values below.
// 5. In your project folder, run: npm install firebase
//
// That's it — both phones will now read/write the same leaderboard data.

import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: "AIzaSyAWw9LhZtWoC4E8TjgjGtYXOFy91ypGEF4",
  authDomain: "questdaily-35d87.firebaseapp.com",
  projectId: "questdaily-35d87",
  storageBucket: "questdaily-35d87.firebasestorage.app",
  messagingSenderId: "1009338940863",
  appId: "1:1009338940863:web:acb1ca1e7f84ffc89cb50a",
  measurementId: "G-NRXPC397K7",
};

export const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
