# Side Quests ⚡️

Side Quests is a gamified, mobile-first daily habit and fitness tracker built with **Vite + React** and **Hugging Face Transformers.js**. It turns your daily routines, workouts, and healthy habits into real RPG-style objectives — verified entirely on-device by computer vision. 🎮

No backend servers, no cloud tracking — your data stays 100% private on your device. 🔒

---

## ✨ Features

- 🧠 **AI-Powered Verification** — Uses `@huggingface/transformers` (`Xenova/siglip-base-patch16-224`) for zero-shot image classification, plus live biometric rep-tracking via pose detection.
- 💀 **Live Skeleton HUD** — A real-time neon skeleton overlay tracks your body during reps *and* timed holds (planks, wall sits, burpees, mountain climbers, and more) — not just counted exercises.
- 🕵️ **Photo Authenticity Check** — Uploaded proof photos are checked for camera metadata (EXIF) and scored against "real photo" vs. "stock/screenshot/downloaded image" — so a picture grabbed off Google won't slide through as proof.
- 🎲 **Dynamic Quest Pools** — Automatically rolls randomized daily objectives ranging from pushups and runs to journaling and hydration.
- 📈 **RPG Progression** — Level up your adventurer profile by earning XP as you complete your daily checklist.
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
   - 🗺️ **For Maps/Food:** Upload a clear screenshot of your fitness tracker map or a photo of your meal — checked for both content *and* authenticity.
4. 🏆 **Level Up** — Earn XP upon successful verification, watch your progress ring grow, and maintain your daily streak!

---

## 🌐 Deployment

Deploys cleanly to **Netlify** — push to your connected GitHub repo and it builds automatically via `npm run build`.

---

## 🔒 Privacy & Licensing

Side Quests processes all computer vision models locally inside your browser via WebAssembly/WebGPU. No images, video streams, or personal health metrics ever leave your phone. 🛡️

---

Made by [qaiskurdieh](https://github.com/qaisqaiswow) 🚀
