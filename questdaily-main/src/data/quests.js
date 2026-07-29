// ─── QUEST POOL WITH TIMERS ───────────────────────────────────────────────────
export const QUEST_POOL = [
  { id: 'q1',  text: 'Do 20 pushups',                            xp: 50, reps: 20 },
  { id: 'q2',  text: 'Do 30 squats',                             xp: 45, reps: 30 },
  { id: 'q3',  text: 'Go for a 15-minute run',                   xp: 75, duration: 900 },
  { id: 'q4',  text: 'Do 10 pullups',                            xp: 60, reps: 10 },
  { id: 'q16', text: 'Do 15 minutes of cycling',                 xp: 55, duration: 900 },
  { id: 'q17', text: 'Do 50 jumping rope reps',                  xp: 40, reps: 50 },
  { id: 'q18', text: 'Take a cold shower',                       xp: 50 },
  { id: 'q19', text: 'Eat no sugar today',                       xp: 65 },
  { id: 'q20', text: 'Cook a meal from scratch',                 xp: 45 },
  { id: 'q21', text: 'Do 10 minutes of deep breathing',          xp: 30, duration: 600 },
  { id: 'q22', text: 'Take 10,000 steps',                        xp: 70 },
  { id: 'q23', text: 'Do 3 sets of lunges',                      xp: 40, reps: 30 },
  { id: 'q24', text: 'Go to bed before 11pm',                    xp: 35 },
  { id: 'q25', text: 'Do a 5-minute ice bath or cold plunge',    xp: 80, duration: 300 },
  { id: 'q5',  text: 'Walk outside for 20 minutes',              xp: 35, duration: 1200 },
  { id: 'q6',  text: 'Drink 2 liters of water today',            xp: 25 },
  { id: 'q7',  text: 'Meditate for 10 minutes',                  xp: 45, duration: 600 },
  { id: 'q8',  text: 'Stretch for 10 minutes',                   xp: 35, duration: 600 },
  { id: 'q9',  text: 'Eat a healthy meal',                       xp: 40 },
  { id: 'q10', text: 'Get 8 hours of sleep',                     xp: 55 },
  { id: 'q11', text: 'Do 3 minutes of jumping jacks',            xp: 30, duration: 180 },
  { id: 'q12', text: 'Do 20 situps',                             xp: 40, reps: 20 },
  { id: 'q13', text: 'Write in your journal',                    xp: 20 },
  { id: 'q14', text: 'Drink a green smoothie',                   xp: 30 },
  { id: 'q15', text: 'Hold a plank for 60 seconds',              xp: 50, duration: 60 },
];

// Rep quests are verified by the on-device pose-landmarker + joint-angle engine
// (see src/data/repProfiles.js) — no text-prompt classification needed for them.
// Everything else still goes through the zero-shot image classifier.
export const QUEST_LABELS = {
  q1:  { type: 'reps', label: 'doing pushups', bodyParts: ['Chest', 'Triceps', 'Shoulders', 'Core'] },
  q2:  { type: 'reps', label: 'doing squats', bodyParts: ['Quads', 'Hamstrings', 'Glutes', 'Core'] },
  q4:  { type: 'reps', label: 'doing pullups', bodyParts: ['Lats', 'Upper Back', 'Biceps', 'Forearms'] },
  q12: { type: 'reps', label: 'doing situps', bodyParts: ['Abs', 'Obliques', 'Hip Flexors'] },
  q17: { type: 'reps', label: 'jumping rope', bodyParts: ['Calves', 'Quads', 'Shoulders', 'Cardio'] },
  q23: { type: 'reps', label: 'doing lunges', bodyParts: ['Quads', 'Glutes', 'Hamstrings'] },

  // Telemetry dashboard text verification for cardio quests
  q3:  { type: 'map', activity: ['strava running workout map dashboard with duration and speed metrics', 'running pace distance tracking workout summary screen'], label: 'running metrics map' },
  q16: { type: 'map', activity: ['cycling route ride summary dashboard with speed and time logs', 'bicycle fitness tracking dashboard workout summary'], label: 'cycling metrics map' },
  q5:  { type: 'map', activity: ['gps tracking map route screenshot', 'walking route map tracker'], label: 'walking map screenshot' },
  q22: { type: 'map', activity: ['gps tracking map route screenshot', 'step counter fitness app screenshot'], label: 'step tracking map' },

  q9:  { type: 'food', activity: ['healthy food meal salad vegetables on a plate', 'nutritious meal in a bowl'], label: 'plate of healthy food' },
  q19: { type: 'food', activity: ['clean healthy meal on plate', 'plate of vegetables and whole foods'], label: 'plate of clean food' },
  q20: { type: 'food', activity: ['cooked food on a plate', 'homemade meal in a bowl or plate'], label: 'cooked meal' },
  q14: { type: 'food', activity: ['glass of green smoothie', 'blended green juice drink'], label: 'green smoothie' },
  q6:  { type: 'food', activity: ['glass of water', 'reusable water bottle filled'], label: 'water bottle' },

  q7:  { type: 'action', activity: ['person meditating cross-legged', 'mindfulness exercise'], label: 'meditating' },
  q8:  { type: 'action', activity: ['person stretching muscles', 'yoga stretch pose'], label: 'stretching' },
  q10: { type: 'action', activity: ['person sleeping in bed', 'person resting in bed eyes closed'], label: 'getting good sleep' },
  q11: { type: 'action', activity: ['person doing jumping jacks or burpees'], label: 'doing cardio' },
  q13: { type: 'action', activity: ['handwriting in notebook or journal'], label: 'journaling' },
  q15: { type: 'action', activity: ['person doing plank exercise', 'plank position core exercise'], label: 'holding a plank' },
  q18: { type: 'action', activity: ['shower running water', 'bathroom shower head with water'], label: 'in the shower' },
  q21: { type: 'action', activity: ['person breathing deeply eyes closed'], label: 'deep breathing' },
  q24: { type: 'action', activity: ['person sleeping in bed at night', 'sleeping in dark bedroom'], label: 'sleeping early' },
  q25: { type: 'action', activity: ['person in ice bath tub', 'cold plunge tub with ice'], label: 'in a cold plunge' },
};

export const getNegativeLabels = (type) => {
  const base = ['person sitting doing nothing', 'person standing still straight', 'random everyday object'];
  if (type === 'map') return [...base, 'google maps navigation screen', 'empty city street map with no data', 'world map atlas website', 'sweaty selfie face', 'picture of running shoes', 'treadmill machine indoors'];
  if (type === 'food') return [...base, 'empty plate or bowl', 'restaurant paper menu', 'store product barcode', 'person eating face'];
  return [...base, 'phone or computer screen'];
};

export const ONE_DAY_MS = 24 * 60 * 60 * 1000;

// SigLIP's sigmoid head scores each label independently (no forced softmax
// competition against the negative labels), so a real match reliably clears
// this bar on its own instead of getting diluted by how many negatives we list.
export const REQUIRED_PASSES = 2;
export const PASS_THRESHOLD = 0.45;
export const CLASSIFIER_SCAN_INTERVAL_MS = 1100;

export function pickDailyQuests() {
  const n = Math.floor(Math.random() * 3) + 5;
  return [...QUEST_POOL].sort(() => 0.5 - Math.random()).slice(0, n).map(q => ({ ...q, completed: false }));
}
