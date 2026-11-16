// Web Worker: локальное распознавание Whisper через transformers.js (ONNX Runtime Web, Wasm).
// Модель и wasm лежат в образе (/models, /vendor) — сеть для загрузки не нужна.
import { pipeline, env } from '/vendor/transformers.min.js';

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = '/models/';
env.useBrowserCache = false;
env.backends.onnx.wasm.wasmPaths = '/vendor/';

let asr = null;

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'load') {
      env.backends.onnx.wasm.numThreads = msg.threads ?? 1;
      const t0 = performance.now();
      asr = await pipeline('automatic-speech-recognition', msg.model, {
        device: 'wasm',
        dtype: { encoder_model: 'q8', decoder_model_merged: 'q8' },
      });
      self.postMessage({ type: 'loaded', loadMs: performance.now() - t0, version: env.version });
    } else if (msg.type === 'run') {
      const t0 = performance.now();
      const out = await asr(msg.pcm, { language: 'russian', task: 'transcribe', return_timestamps: false, max_new_tokens: msg.maxNewTokens ?? 64 });
      self.postMessage({ type: 'done', id: msg.id, text: out.text.trim(), inferMs: performance.now() - t0 });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, error: String(err && err.stack || err) });
  }
};
