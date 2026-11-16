// Юнит-тесты планировщика: node --test tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Ewma, NetEstimator, ComputeEstimator, decide, serverDeadlineMs } from '../client/web/scheduler.js';

const state = ({ mbps = 50, rtt = 20, local = 1500, server = 1000, online = true, localPending = 0 } = {}) => {
  const net = new NetEstimator();
  net.rtt.push(rtt);
  net.thr.push(mbps * 1e6);
  net.online = online;
  const l = new ComputeEstimator(); l.onDone(local);
  const s = new ComputeEstimator(); s.onDone(server);
  return { net, local: l, server: s, localPending, localElapsed: 0, serverPending: 0, queuedBytes: 0 };
};
const BYTES = 120_000; // ≈ 3,7 с WAV 16 кГц

test('EWMA: первое значение как есть, дальше сглаживание, NaN игнорируется', () => {
  const e = new Ewma(0.5);
  assert.equal(e.push(10), 10);
  assert.equal(e.push(20), 15);
  assert.equal(e.push(NaN), 15);
});

test('оценка канала по передаче: вычитается RTT и работа сервера', () => {
  const n = new NetEstimator({ alpha: 1 });
  n.onPing(100, 0);
  n.onTransfer(125_000, 1100 + 100, 100, 1); // 1 Мбит за 1 с
  assert.ok(Math.abs(n.thr.value - 1e6) < 1);
});

test('S1 и S2 не зависят от состояния', () => {
  assert.equal(decide('S1', BYTES, state({ mbps: 100 })).where, 'local');
  assert.equal(decide('S2', BYTES, state({ online: false })).where, 'server');
});

test('S3: порог 5 Мбит/с', () => {
  assert.equal(decide('S3', BYTES, state({ mbps: 8 })).where, 'server');
  assert.equal(decide('S3', BYTES, state({ mbps: 1 })).where, 'local');
});

test('S4: хорошая сеть и слабое устройство -> сервер', () => {
  assert.equal(decide('S4', BYTES, state({ mbps: 50, local: 6000 })).where, 'server');
});

test('S4: плохая сеть -> локально', () => {
  assert.equal(decide('S4', BYTES, state({ mbps: 1, rtt: 250, local: 1500 })).where, 'local');
});

test('S4: без сети -> локально', () => {
  assert.equal(decide('S4', BYTES, state({ online: false })).where, 'local');
});

test('S4: очередь локальной модели переводит фрагмент на сервер', () => {
  const st = state({ mbps: 8, rtt: 80, local: 1500, localPending: 3 });
  assert.equal(decide('S4', BYTES, st).where, 'server');
});

test('S4: при равных прогнозах остаётся локально (запас margin)', () => {
  const st = state({ mbps: 1000, rtt: 0, local: 1050, server: 1000 });
  assert.equal(decide('S4', BYTES, st, { margin: 0.1 }).where, 'local');
});

test('срок ожидания сервера: S4 — 2,5 прогноза в пределах [1,5; 15] с, остальные — жёсткий', () => {
  assert.equal(serverDeadlineMs('S4', 1000), 2500);
  assert.equal(serverDeadlineMs('S4', 100), 1500);
  assert.equal(serverDeadlineMs('S4', 1e6), 15000);
  assert.equal(serverDeadlineMs('S2', 100), 15000);
});
