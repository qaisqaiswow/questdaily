# Side Quests ⚡️

Side Quests is a gamified, mobile-first daily habit and fitness tracker built with **Vite + React** and **Hugging Face Transformers.js**. It turns your daily routines, workouts, and healthy habits into real RPG-style objectives — verified entirely on-device by computer vision. 🎮

All AI verification runs entirely on-device — no images, video, or health metrics ever leave your phone. The only thing that goes to a server is your chosen username, level, and XP, for the worldwide leaderboard. 🔒

---

## ✨ Features

- 🧠 **AI-Powered Verification** — Uses `@huggingface/transformers` with a **two-model setup**: a quantized `Xenova/siglip-base-patch16-224` for fast, real-time zero-shot classification while the live camera loop is scanning, and a larger, higher-precision `Xenova/siglip-large-patch16-256` for one-shot checks — uploaded proof photos, photo authenticity screening, and the final confirmation frame that locks in a live quest. The fast model builds up a confidence streak in real time; the accurate model gets the final say before anything counts as verified. Plus live biometric rep-tracking via pose detection.
- 🔬 **Smoothed, Streak-Based Confidence** — Live scores are smoothed across frames (EMA) so a single blurry or occluded frame doesn't reset your progress — the streak decays gently instead of snapping back to zero, then hands off to the high-accuracy model for final confirmation.
- 💀 **Live Skeleton HUD** — A real-time neon skeleton overlay tracks your body during reps *and* timed holds (planks, wall sits, burpees, mountain climbers, and more) — not just counted exercises.
- 🕵️ **Photo Authenticity Check** — Uploaded proof photos are checked for camera metadata (EXIF) and scored by the high-accuracy model against "real photo" vs. "stock/screenshot/downloaded image" — so a picture grabbed off Google won't slide through as proof. A confidence margin (not a coin-flip edge) is required before a photo is flagged, so genuine photos aren't wrongly rejected.
- 🎲 **Dynamic Quest Pools** — Automatically rolls randomized daily objectives ranging from pushups and runs to journaling and hydration.
- 📈 **RPG Progression** — Level up your adventurer profile by earning XP as you complete your daily checklist, with an animated XP bar and a dedicated confetti "Level Up" celebration when you cross a threshold.
- 🌍 **Worldwide Leaderboard** — Pick a username once, then your level and XP sync to a live, real-time leaderboard powered by Firebase. Rankings update instantly for everyone the moment anyone completes a quest — no refresh needed.
- 📳 **Haptic Feedback** — Subtle vibration cues on rep counts, verification passes, and quest completions for a more responsive feel on mobile.
- 📸 **Live Camera & Upload Proofs** — Real-time feedback loops for exercise reps, GPS workout map screenshots, and meal tracking.
- 📱 **PWA Ready** — Fully optimized mobile viewport layout with an interactive iOS "Add to Home Screen" guide for new users.
- 🌗 **Dark/Light Mode** — Seamlessly switch themes anytime across the main dashboard, detail views, and completion screens.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have **Node.js** (v18+) and npm installed on your machine.

### Installation

1. Clone the repository and install dependencies:
```bash
   git clone https://github.com/your-username/side-quests.git
   cd side-quests
   npm install
```

2. Run the development server:
```bash
   npm run dev
```

3. Open the local URL Vite prints (usually `http://localhost:5173`) in your browser.
   > ⚠️ Camera-based quests require HTTPS or `localhost` access for browser media permissions.

### Build for production

```bash
npm run build
npm run preview
```

---

## 📱 How It Works

1. ✅ **Check Your Quests** — Review your randomized objectives for the day.
2. 🎥 **Launch Verification** — Tap any quest to view details, then hit **Mark as Completed** to open the AI viewfinder.
3. 🤖 **AI Scan:**
   - 💪 **For Reps** (pushups, squats, etc.): Position your device for full-body frame visibility and complete your reps — the on-device model tracks movement phases and draws a live skeleton overlay.
   - 🧘 **For Timed Holds** (planks, wall sits, burpees, etc.): Same live skeleton tracking, just against the clock instead of a rep count.
   - 🏃 **For Actions** (meditating, stretching, etc.): A fast on-device model scans in real time and builds a confidence streak; once it's sustained, a larger high-accuracy model checks one final frame before locking in "Verified & Locked."
   - 🗺️ **For Maps/Food:** Upload a clear screenshot of your fitness tracker map or a photo of your meal — checked for both content *and* authenticity by the high-accuracy model.
4. 🏆 **Level Up** — Earn XP upon successful verification, watch your progress ring and XP bar grow, and enjoy a confetti celebration whenever you level up.

---

## 🌍 Leaderboard Backend Setup (Firebase)

The worldwide leaderboard runs on [Firebase](https://console.firebase.google.com) (Firestore + Anonymous Auth). Since this uses your own Firebase project, you'll need to connect it once:

1. **Create a Firebase project** at [console.firebase.google.com](https://console.firebase.google.com) (free "Spark" tier is enough for this).
2. **Enable Firestore** — Build → Firestore Database → Create database → start in *production mode*.
3. **Enable Anonymous sign-in** — Build → Authentication → Sign-in method → enable **Anonymous**. This gives each device a stable ID to own its leaderboard entry, without asking anyone to make an account or password.
4. **Get your web app config** — Project Settings → General → "Your apps" → Add app → Web. Copy the config object it gives you.
5. **Paste it into `SideQuestsApp.jsx`** — near the top of the file, replace the placeholder values in `firebaseConfig`:
   ```js
   const firebaseConfig = {
     apiKey: "...",
     authDomain: "...",
     projectId: "...",
     storageBucket: "...",
     messagingSenderId: "...",
     appId: "...",
   };
   ```
   This config is safe to ship in client code — it's not a secret. Access control is handled by Firestore's security rules, not by hiding this object.
6. **Set Firestore security rules** — Firestore Database → Rules — so anyone can *read* the leaderboard, but a device can only *write* its own entry:
   ```
   rules_version = '2';
   service cloud.firestore {
     match /databases/{database}/documents {
       match /leaderboard/{uid} {
         allow read: if true;
         allow write: if request.auth != null && request.auth.uid == uid;
       }
     }
   }
   ```
7. **Install the dependency** (already in `package.json`) and run the app:
   ```bash
   npm install
   npm run dev
   ```

That's it — the first time someone opens the app, they'll be asked to pick a username, which is sent to Firestore alongside their level and XP. Every quest completion pushes a fresh update, and everyone's leaderboard view updates live via a real-time listener — no polling, no refresh button.

> Ranking is by **lifetime XP earned** (a number that only ever goes up), not "current XP toward next level" — so leveling up never makes anyone drop in rank.

---

## 🌐 Deployment

Deploys cleanly to **Vercel** — push to your connected GitHub repo and it builds automatically via `npm run build`.

---

## 🔒 Privacy & Licensing

Side Quests processes all computer vision models locally inside your browser via WebAssembly/WebGPU. No images, video streams, or personal health metrics ever leave your phone. 🛡️

> ℹ️ On first use, the larger high-accuracy model (`siglip-large-patch16-256`, ~650MB) downloads and caches in the browser alongside the smaller fast model — this only happens once per device, but is worth knowing on a slow connection.

rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /leaderboard/{uid} {
      allow read: if true;
      allow write: if request.auth != null && request.auth.uid == uid;
    }
  }
}

---

Made by [qaiskurdieh](https://github.com/qaisqaiswow) 🚀
