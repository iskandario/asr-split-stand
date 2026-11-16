// Планировщик: где распознавать фрагмент речи — в браузере или на сервере.
// Модуль без зависимостей от DOM: его же гоняют юнит-тесты (node --test tests/scheduler.test.mjs).

export const STRATEGIES = ['S1', 'S2', 'S3', 'S4'];

/** Экспоненциальное сглаживание; первое значение принимается как есть. */
export class Ewma {
  constructor(alpha, initial = null) { this.alpha = alpha; this.value = initial; this.n = 0; }
  push(x) {
    if (!Number.isFinite(x)) return this.value;
    this.value = this.value === null ? x : this.alpha * x + (1 - this.alpha) * this.value;
    this.n += 1;
    return this.value;
  }
}

/**
 * Оценка состояния сети по зондам (/ping, /probe) и по реальным загрузкам звука.
 * rtt — мс, thr — бит/с. online=false после неудачного зонда.
 */
export class NetEstimator {
  constructor({ alpha = 0.4 } = {}) {
    this.rtt = new Ewma(alpha);
    this.thr = new Ewma(alpha);
    this.online = true;
    this.lastUpdate = -Infinity;
  }
  onPing(ms, now) { this.online = true; this.rtt.push(ms); this.lastUpdate = now; }
  onFailure(now) { this.online = false; this.lastUpdate = now; }
  /** Загрузка bytes байт заняла totalMs, из них serverMs — работа сервера. */
  onTransfer(bytes, totalMs, serverMs, now) {
    const rtt = this.rtt.value ?? 0;
    const transferMs = totalMs - serverMs - rtt;
    if (transferMs <= 1) return; // слишком быстро, чтобы оценить канал
    this.online = true;
    this.thr.push((bytes * 8) / (transferMs / 1000));
    this.lastUpdate = now;
  }
  /** Сколько займёт отправка bytes байт при уже стоящих в очереди queuedBytes. */
  uploadMs(bytes, queuedBytes = 0) {
    const thr = this.thr.value;
    if (!this.online || !thr) return Infinity;
    return ((bytes + queuedBytes) * 8 / thr) * 1000 + (this.rtt.value ?? 0);
  }
  mbps() { return this.online && this.thr.value ? this.thr.value / 1e6 : 0; }
}

/** Оценка времени работы модели (мс на фрагмент) на одной стороне + очередь. */
export class ComputeEstimator {
  constructor({ alpha = 0.3 } = {}) { this.infer = new Ewma(alpha); }
  onDone(inferMs) { this.infer.push(inferMs); }
  /** pendingCount — сколько фрагментов уже ждут или считаются, elapsedHead — сколько уже идёт текущий. */
  waitMs(pendingCount, elapsedHead = 0) {
    const t = this.infer.value;
    if (t === null) return Infinity;
    if (pendingCount <= 0) return t;
    return Math.max(0, t - elapsedHead) + (pendingCount - 1) * t + t;
  }
}

/**
 * Решение для одного фрагмента.
 * state: { net: NetEstimator, local: ComputeEstimator, server: ComputeEstimator,
 *          localPending, localElapsed, serverPending, queuedBytes }
 * Возвращает { where: 'local'|'server', predLocal, predServer, reason }.
 */
export function decide(strategy, bytes, state, opts = {}) {
  const thresholdMbps = opts.thresholdMbps ?? 5;
  const margin = opts.margin ?? 0.1;
  const predLocal = state.local.waitMs(state.localPending, state.localElapsed);
  const up = state.net.uploadMs(bytes, state.queuedBytes);
  const predServer = up + state.server.waitMs(state.serverPending, 0);
  const base = { predLocal, predServer };
  switch (strategy) {
    case 'S1': return { ...base, where: 'local', reason: 'всегда локально' };
    case 'S2': return { ...base, where: 'server', reason: 'всегда сервер' };
    case 'S3': {
      const fast = state.net.mbps() > thresholdMbps;
      return { ...base, where: fast ? 'server' : 'local', reason: fast ? 'канал выше порога' : 'канал ниже порога' };
    }
    case 'S4': {
      if (!state.net.online || !Number.isFinite(predServer)) return { ...base, where: 'local', reason: 'сеть недоступна' };
      // сервер выбирается, только если он заметно быстрее: иначе звук не покидает устройство
      const server = predServer < predLocal * (1 - margin);
      return { ...base, where: server ? 'server' : 'local', reason: server ? 'прогноз: сервер быстрее' : 'прогноз: локально не хуже' };
    }
    default: throw new Error(`неизвестная стратегия ${strategy}`);
  }
}

/** Срок ожидания ответа сервера, после которого S4 переключает фрагмент на локальное распознавание. */
export function serverDeadlineMs(strategy, predServer, opts = {}) {
  const hard = opts.hardTimeoutMs ?? 15000;
  if (strategy !== 'S4') return hard;
  if (!Number.isFinite(predServer)) return 1500;
  return Math.min(hard, Math.max(1500, 2.5 * predServer));
}
