# Side Quests ⚡️

Side Quests is a gamified, mobile-first daily habit and fitness tracker built with **Next.js ("use client")** and **Hugging Face Transformers.js**. It turns your daily routines, workouts, and healthy habits into real RPG-style objectives verified entirely on-device by computer vision (CLIP). 

No backend servers, no cloud tracking—your data stays 100% private on your device.

---

## ✨ Features

- **AI-Powered Verification:** Uses `@huggingface/transformers` (`Xenova/clip-vit-base-patch32`) for zero-shot image classification and live biometric rep-tracking.
- **Dynamic Quest Pools:** Automatically rolls randomized daily objectives ranging from pushups and runs to journaling and hydration.
- **RPG Progression:** Level up your adventurer profile by earning XP as you complete your daily checklist.
- **Live Camera & Upload Proofs:** Real-time feedback loops for exercise reps, GPS workout map screenshots, and meal tracking.
- **PWA Ready:** Fully optimized mobile viewport layout with an interactive iOS "Add to Home Screen" guide for new users.
- **Dark/Light Mode:** Seamlessly switch themes anytime across the main dashboard, detail views, and completion screens.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have **Node.js** (v18+) and npm installed on your machine.

### Installation

1. Clone the repository and install dependencies:
   ```bash
   git clone [https://github.com/your-username/side-quests.git](https://github.com/your-username/side-quests.git)
   cd side-quests
   npm install

 * Run the development server:
   npm run dev

 * Open http://localhost:3000 in your browser. (Note: Camera-based quests require HTTPS or localhost access for browser media permissions).
📱 How It Works
 * Check Your Quests: Review your randomized objectives for the day.
 * Launch Verification: Tap any quest to view details, then click Mark as Completed to open the AI viewfinder.
 * AI Scan:
   * For Reps (e.g., Pushups, Squats): Position your device for full-body frame visibility and complete your reps. The on-device model tracks movement phases.
   * For Maps/Food: Upload a clear screenshot of your fitness tracker map or a photo of your meal.
 * Level Up: Earn XP upon successful verification, watch your progress ring grow, and maintain your daily streak!
🔒 Privacy & Licensing
Side Quests processes all computer vision models locally inside your browser via WebAssembly and WebGL. No images, video streams, or personal health metrics ever leave your phone.

