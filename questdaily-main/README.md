# ⚔️ Side Quests

**Side Quests** transforms your everyday life, fitness routine, and health habits into an immersive, real-world RPG. Complete your daily objectives, gain experience points (XP), and level up your character. 

Unlike typical habit trackers that rely purely on the honor system, **Side Quests** uses cutting-edge, **on-device computer vision AI** to dynamically track and verify your physical movements, meals, and workout logs in real time.

---

## 🚀 Key Features

*   **⚡ Real Joint-Angle Rep Counter**
    Built on `@mediapipe/tasks-vision` PoseLandmarker running in a dedicated Web Worker. The app computes actual elbow/knee/hip joint angles (and a vertical-oscillation detector for jump rope) via a debounced hysteresis state machine, tracking the exact contraction and extension phases of each rep — no main-thread jank, no stuck rep loops.
*   **📊 Strict Telemetry Map Validation**
    No more spoofing maps. For running and cycling objectives, the built-in zero-shot vision model explicitly validates application dashboard telemetry (e.g., Strava, Nike Run Club, Garmin summaries) by analyzing route maps alongside actual distance metrics, pace records, and active duration logs.
*   **🤖 Biometric Matrix HUD Overlay**
    Experience a high-tech, sci-fi workout interface. The app overlays a dynamic tracking frame on your live video feed, visually mapping out and displaying the exact biometric muscle groups and skeletal focus points the AI is analyzing.
*   **⏱ Integrated Session Timers**
    Built-in countdown and count-up stopwatches are integrated right inside the verification modal for duration-heavy actions (e.g., planks, meditation, cycling, deep breathing).
*   **🔒 100% Privacy-First Architecture**
    Powered by `@huggingface/transformers` running a localized `SigLIP` zero-shot vision pipeline, and `@mediapipe/tasks-vision` for pose landmarking — both entirely in-browser, each in its own Web Worker. All image processing and classification happen directly in your browser's memory sandbox. **No camera feeds, photos, or personal data ever leave your device.**
*   **🌙 Seamless Native UI & Dark Mode**
    A fluid, iOS-inspired responsive client engine featuring automatic system theme syncing, debounced local-storage persistence, and automated 24-hour daily objective pools.

---

## 🛠️ Tech Stack

*   **Frontend Library:** React (Hooks, Web Workers, Web Canvas API Integration)
*   **Styling Engine:** Tailwind CSS (Fluid responsive utility design)
*   **AI Inference Engines:** `@huggingface/transformers` (Xenova/siglip-base-patch16-224) for zero-shot verification, `@mediapipe/tasks-vision` (PoseLandmarker) for rep counting — both off the main thread
*   **State & Storage:** LocalStorage API & Custom Application Persistent Refs

---

## 📦 Installation & Local Setup

Get your local copy of the application up and running inside a standard React environment:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/side-quests.git](https://github.com/yourusername/side-quests.git)
   cd side-quests
Install dependencies:

Bash
npm install
Install the HuggingFace Transformers package (if not bundled):

Bash
npm install @huggingface/transformers
Boot up the local Vite development server:

Bash
npm run dev
🛡️ Privacy & Security Disclaimer
Side Quests is completely decentralized. The application opens a fully sandbox-contained media capture context inside your browser window. No media assets, images, tracking maps, or metadata logs are transferred over the internet. Your workout records belong to you alone.

-Made By Qais Kurdieh 
