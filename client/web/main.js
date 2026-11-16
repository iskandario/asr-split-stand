// Клиент стенда: подаёт фрагменты речи в реальном времени и распределяет их по стратегии S1–S4.
// Драйвер (Playwright) вызывает window.stand.run(cfg) и забирает window.stand.result.
import { NetEstimator, ComputeEstimator, decide, serverDeadlineMs } from './scheduler.js';

const SERVER = new URLSearchParams(location.search).get('server') ?? 'http://server:8000';
const now = () => performance.now();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.log('[stand]', ...a);

function parseWav(buf) {
  const v = new DataView(buf);
  let p = 12, sr = 16000, dataOff = -1, dataLen = 0;
  while (p + 8 <= v.byteLength) {
    const id = String.fromCharCode(v.getUint8(p), v.getUint8(p + 1), v.getUint8(p + 2), v.getUint8(p + 3));
    const len = v.getUint32(p + 4, true);
    if (id === 'fmt ') sr = v.getUint32(p + 12, true);
    if (id === 'data') { dataOff = p + 8; dataLen = len; break; }
    p += 8 + len + (len % 2);
  }
  const n = dataLen / 2, pcm = new Float32Array(n);
  for (let i = 0; i < n; i++) pcm[i] = v.getInt16(dataOff + 2 * i, true) / 32768;
  return { pcm, sr };
}

/** Очередь локального распознавания: в воркер уходит один фрагмент за раз. */
class LocalASR {
  constructor() {
    this.worker = new Worker('./asr-worker.js', { type: 'module' });
    this.queue = []; this.busy = null; this.waiters = new Map();
    this.worker.onmessage = (e) => this.#onMessage(e.data);
  }
  load(model, threads, maxNewTokens = 64) {
    this.maxNewTokens = maxNewTokens;
    return new Promise((resolve, reject) => {
      this.waiters.set('load', { resolve, reject });
      this.worker.postMessage({ type: 'load', model, threads });
    });
  }
  get pending() { return this.queue.length + (this.busy ? 1 : 0); }
  get headElapsed() { return this.busy ? now() - this.busy.started : 0; }
  submit(id, pcm) {
    return new Promise((resolve, reject) => { this.queue.push({ id, pcm, resolve, reject }); this.#pump(); });
  }
  #pump() {
    if (this.busy || !this.queue.length) return;
    this.busy = { ...this.queue.shift(), started: now() };
    this.worker.postMessage({ type: 'run', id: this.busy.id, pcm: this.busy.pcm, maxNewTokens: this.maxNewTokens });
  }
  #onMessage(m) {
    if (m.type === 'loaded') { this.waiters.get('load').resolve(m); return; }
    if (m.type === 'error' && !m.id) { this.waiters.get('load')?.reject(new Error(m.error)); return; }
    const job = this.busy; this.busy = null;
    if (m.type === 'done') job.resolve({ text: m.text, inferMs: m.inferMs });
    else job.reject(new Error(m.error));
    this.#pump();
  }
}

class Traffic { constructor() { this.audio = 0; this.probe = 0; } }

async function timedFetch(url, opts, timeoutMs) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try { return await fetch(url, { ...opts, signal: ctl.signal, cache: 'no-store' }); }
  finally { clearTimeout(timer); }
}

async function run(cfg) {
  const net = new NetEstimator({ alpha: cfg.ewma_alpha_net ?? 0.4 });
  const localEst = new ComputeEstimator({ alpha: cfg.ewma_alpha_compute ?? 0.3 });
  const serverEst = new ComputeEstimator({ alpha: cfg.ewma_alpha_compute ?? 0.3 });
  const traffic = new Traffic();
  const local = new LocalASR();
  const meta = { server: SERVER, ua: navigator.userAgent, cores: navigator.hardwareConcurrency };

  const tl0 = now();
  const loaded = await local.load(cfg.local_model, cfg.threads ?? 1, cfg.max_new_tokens ?? 64);
  meta.localLoadMs = now() - tl0; meta.transformersVersion = loaded.version;
  log('model loaded', Math.round(meta.localLoadMs), 'ms');

  const fetchAudio = async (f) => {
    const buf = await (await fetch(`/audio/${f.id}.wav`, { cache: 'no-store' })).arrayBuffer();
    return { ...f, bytes: buf.byteLength, wav: buf, pcm: parseWav(buf).pcm };
  };
  const warm = await Promise.all(cfg.warmup.map(fetchAudio));
  const frags = await Promise.all(cfg.fragments.map(fetchAudio));

  // зонды сети: только у стратегий, которые на них опираются
  let probing = ['S3', 'S4'].includes(cfg.strategy);
  const probeBody = new Uint8Array(cfg.probe_bytes ?? 16384);
  const probeLoop = async () => {
    let k = 0;
    while (probing) {
      const t = now();
      try {
        await timedFetch(`${SERVER}/ping`, {}, cfg.ping_timeout_ms ?? 2000);
        net.onPing(now() - t, now());
        if (k % (cfg.probe_every ?? 3) === 0) {
          const t2 = now();
          await timedFetch(`${SERVER}/probe`, { method: 'POST', body: probeBody }, cfg.ping_timeout_ms ?? 2000);
          traffic.probe += probeBody.byteLength;
          net.onTransfer(probeBody.byteLength, now() - t2, 0, now());
        }
      } catch { net.onFailure(now()); }
      k += 1;
      await sleep(cfg.probe_interval_ms ?? 1000);
    }
  };
  const probeDone = probeLoop();

  const serverCall = async (f, timeoutMs) => {
    const t = now();
    const r = await timedFetch(`${SERVER}/transcribe`, {
      method: 'POST', body: f.wav, headers: { 'content-type': 'audio/wav' },
    }, timeoutMs);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    return { ...j, totalMs: now() - t };
  };

  // прогрев: каждый фрагмент и локально, и на сервере; в замеры не идёт, но задаёт стартовые оценки
  for (const f of warm) {
    try { const l = await local.submit(f.id, f.pcm); localEst.onDone(l.inferMs); } catch (e) { log('warm local', e.message); }
    try {
      const s = await serverCall(f, cfg.server_timeout_ms ?? 15000);
      traffic.audio += f.bytes;
      serverEst.onDone(s.infer_ms + s.queue_ms);
      net.onTransfer(f.bytes, s.totalMs, s.infer_ms + s.queue_ms, now());
    } catch (e) { net.onFailure(now()); log('warm server', e.message); }
  }
  meta.afterWarmup = { localMs: localEst.infer.value, serverMs: serverEst.infer.value, mbps: net.mbps(), rtt: net.rtt.value };
  log('warmup', JSON.stringify(meta.afterWarmup));

  let serverPending = 0, queuedBytes = 0;
  const records = [];
  const t0 = now() + 300;
  meta.t0_epoch_ms = performance.timeOrigin + t0;

  const dispatch = async (f, schedMs) => {
    const state = { net, local: localEst, server: serverEst, localPending: local.pending,
      localElapsed: local.headElapsed, serverPending, queuedBytes };
    const d = decide(cfg.strategy, f.bytes, state, { thresholdMbps: cfg.threshold_mbps, margin: cfg.margin });
    const rec = { id: f.id, duration_s: f.duration_s, sched_ms: schedMs, ready_ms: now() - t0, where: d.where, reason: d.reason,
      pred_local_ms: d.predLocal, pred_server_ms: d.predServer, est_mbps: net.mbps(), est_rtt_ms: net.rtt.value,
      online: net.online, fallback: false, failed: false };
    const ready = now();
    const runLocal = async () => {
      const l = await local.submit(f.id, f.pcm);
      localEst.onDone(l.inferMs);
      rec.text = l.text; rec.infer_ms = l.inferMs; rec.done_where = 'local';
    };
    try {
      if (d.where === 'local') await runLocal();
      else {
        const deadline = serverDeadlineMs(cfg.strategy, d.predServer, { hardTimeoutMs: cfg.server_timeout_ms });
        rec.deadline_ms = deadline;
        serverPending += 1; queuedBytes += f.bytes;
        try {
          const s = await serverCall(f, deadline);
          traffic.audio += f.bytes;
          serverEst.onDone(s.infer_ms + s.queue_ms);
          net.onTransfer(f.bytes, s.totalMs, s.infer_ms + s.queue_ms, now());
          rec.text = s.text; rec.infer_ms = s.infer_ms; rec.server_queue_ms = s.queue_ms; rec.done_where = 'server';
        } catch (e) {
          traffic.audio += f.bytes; // байты ушли в канал, даже если ответа не дождались
          rec.server_error = e.name === 'AbortError' ? 'timeout' : String(e.message);
          if (e.name === 'AbortError') net.thr.push((f.bytes * 8) / (deadline / 1000)); // канал хуже прогноза
          else net.onFailure(now());
          if (cfg.strategy === 'S4') { rec.fallback = true; await runLocal(); }
          else rec.failed = true;
        } finally { serverPending -= 1; queuedBytes -= f.bytes; }
      }
    } catch (e) { rec.failed = true; rec.error = String(e.message); }
    rec.latency_ms = rec.failed ? null : now() - ready;
    records.push(rec);
  };

  const jobs = [];
  let acc = 0;
  for (const f of frags) {
    acc += f.duration_s * 1000;
    const wait = t0 + acc - now();
    if (wait > 0) await sleep(wait);
    jobs.push(dispatch(f, acc));
  }
  await Promise.all(jobs);
  probing = false; await probeDone;
  meta.wallMs = now() - t0;
  meta.traffic = { audio_bytes: traffic.audio, probe_bytes: traffic.probe };
  log('done', records.length, 'fragments');
  return { meta, records: records.sort((a, b) => a.id.localeCompare(b.id)) };
}

window.stand = {
  status: 'idle',
  result: null,
  async run(cfg) {
    this.status = 'running';
    try { this.result = await run(cfg); this.status = 'done'; }
    catch (e) { this.result = { error: String(e && e.stack || e) }; this.status = 'error'; }
  },
};
document.getElementById('status').textContent = 'ready';
