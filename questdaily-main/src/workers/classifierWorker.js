// Runs the Transformers.js zero-shot image classifier off the main thread.
// SigLIP scores each label independently via a sigmoid head (unlike CLIP's
// softmax, where scores are relative to every other label in the call), so
// pass/fail thresholds behave consistently regardless of how many negative
// labels we throw in alongside the real one.
import { pipeline, env } from '@huggingface/transformers';

env.allowLocalModels = false;

const MODEL_ID = 'Xenova/siglip-base-patch16-224';

let classifierPromise = null;

function getClassifier(onProgress) {
  if (!classifierPromise) {
    classifierPromise = pipeline(
      'zero-shot-image-classification',
      MODEL_ID,
      onProgress ? { progress_callback: onProgress } : undefined
    ).catch(err => {
      classifierPromise = null;
      throw err;
    });
  }
  return classifierPromise;
}

self.onmessage = async (event) => {
  const msg = event.data;

  if (msg.type === 'init') {
    try {
      await getClassifier((evt) => {
        if (evt.status === 'progress' && evt.total) {
          self.postMessage({ type: 'progress', percent: Math.round((evt.loaded / evt.total) * 100) });
        }
      });
      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'error', message: err?.message || 'The AI model could not be loaded.' });
    }
    return;
  }

  if (msg.type === 'classify') {
    const { requestId, dataUrl, labels } = msg;
    try {
      const classifier = await getClassifier();
      const results = await classifier(dataUrl, labels);
      self.postMessage({ type: 'result', requestId, results });
    } catch (err) {
      self.postMessage({ type: 'error', requestId, message: err?.message || 'AI analysis failed.' });
    }
  }
};
