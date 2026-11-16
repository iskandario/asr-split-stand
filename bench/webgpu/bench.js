// Замер локального распознавания в настоящем браузере с GPU: Wasm (1 и 4 потока) против WebGPU.
// Та же модель Whisper base и те же фрагменты Golos, что в стенде. Запускается из run_bench.py.
import { pipeline, env } from './vendor/transformers.min.js';

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = '/models/';
env.useBrowserCache = false;
env.backends.onnx.wasm.wasmPaths = '/vendor/';

function parseWav(buf) {
  const v = new DataView(buf); let p = 12, off = -1, len = 0;
  while (p + 8 <= v.byteLength) {
    const id = String.fromCharCode(...[0, 1, 2, 3].map((k) => v.getUint8(p + k))), l = v.getUint32(p + 4, true);
    if (id === 'data') { off = p + 8; len = l; break; }
    p += 8 + l + (l % 2);
  }
  const pcm = new Float32Array(len / 2);
  for (let i = 0; i < pcm.length; i++) pcm[i] = v.getInt16(off + 2 * i, true) / 32768;
  return pcm;
}

const MODES = {
  'wasm-1t': { device: 'wasm', threads: 1, dtype: { encoder_model: 'q8', decoder_model_merged: 'q8' } },
  'wasm-4t': { device: 'wasm', threads: 4, dtype: { encoder_model: 'q8', decoder_model_merged: 'q8' } },
  'webgpu-fp32q4': { device: 'webgpu', dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' } },
  'webgpu-fp16': { device: 'webgpu', dtype: { encoder_model: 'fp16', decoder_model_merged: 'fp16' } },
};

window.bench = {
  async info() {
    const a = navigator.gpu ? await navigator.gpu.requestAdapter() : null;
    let gpu = null;
    if (a) {
      const i = a.info || (a.requestAdapterInfo ? await a.requestAdapterInfo() : {});
      gpu = { vendor: i.vendor, architecture: i.architecture, description: i.description, f16: a.features.has('shader-f16') };
    }
    return { ua: navigator.userAgent, cores: navigator.hardwareConcurrency, isolated: crossOriginIsolated, webgpu: !!a, gpu, transformers: env.version };
  },
  async run(mode, warmup, frags, reps) {
    const m = MODES[mode];
    if (m.threads) env.backends.onnx.wasm.numThreads = m.threads;
    const t0 = performance.now();
    const asr = await pipeline('automatic-speech-recognition', 'whisper-base', { device: m.device, dtype: m.dtype });
    const loadMs = performance.now() - t0;
    const audio = {};
    for (const f of [...warmup, ...frags]) audio[f] = parseWav(await (await fetch(`./audio/${f}.wav`)).arrayBuffer());
    const opts = { language: 'russian', task: 'transcribe', return_timestamps: false, max_new_tokens: 64 };
    for (const f of warmup) await asr(audio[f], opts);
    const out = [];
    for (let r = 0; r < reps; r++) {
      for (const f of frags) {
        const t = performance.now();
        const res = await asr(audio[f], opts);
        out.push({ rep: r, id: f, infer_ms: performance.now() - t, text: res.text.trim() });
      }
    }
    await asr.dispose?.();
    return { mode, loadMs, results: out };
  },
};
document.title = 'ready';
