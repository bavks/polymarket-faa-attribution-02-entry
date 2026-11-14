// polymarket-faa-attribution-02 entry: an empirical forecaster for sticky Polymarket Yes prices.
//
// Idea: these prices sit still for long stretches, then tick or jump. Instead of assuming a normal
// random walk (the house baseline spreads every market by the same fixed sigma), each market's own
// history says how far it actually moves over h minutes: for every horizon h = 1..horizon we collect
// the h-minute changes p[t+h] - p[t] over the recent history, take their empirical quantiles at the
// scored levels, and centre them on the checkpoint price. Quiet markets get narrow bands, jumpy
// markets get wide ones, and the tails are the tails the market really had.
//
// The empirical quantiles are blended a little toward the house baseline's normal shape
// (z_tau * 0.0025 * sqrt(h)) with a weight that grows with the number of sampled changes, so a thin
// sample cannot produce wild tails. Everything is deterministic: fixed sampling stride, no clocks,
// no randomness, so two runs write byte-identical files.
//
// Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and writes <id>.out to
// $MAKEFUN_OUTPUT_DIR (else /output): {"forecasts":[{"id","q"}]} with horizon rows of the five
// quantiles at [0.05, 0.25, 0.5, 0.75, 0.95], each in [0,1], non-decreasing, rounded to 6 decimals.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, to 6 decimals. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** The house baseline's spread, in price units per square-root minute. */
const BASE_SIGMA = 0.0025;

/** Minutes of recent history the change distribution is estimated from. */
const WINDOW = 2880;
/** Most sampled changes per horizon; larger samples are strided down to about this many. */
const MAX_SAMPLES = 2048;
/**
 * Shrinkage strength: the baseline's weight is SHRINK_K / (samples + SHRINK_K). Tuned on a
 * truncated-history backtest over the six practice cases, where it beat both weaker and stronger
 * shrinkage: enough trust in the market's own moves to price its real tails, enough pull toward the
 * baseline's shape that one violent hour in the window cannot blow the bands wide open.
 */
const SHRINK_K = 600;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** The tau-quantile of a sorted sample, by linear interpolation between closest ranks. */
function sampleQuantile(sorted: Float64Array, tau: number): number {
  const n = sorted.length;
  if (n === 1) return sorted[0] as number;
  const pos = tau * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, n - 1);
  const frac = pos - lo;
  return (sorted[lo] as number) * (1 - frac) + (sorted[hi] as number) * frac;
}

/** One horizon row: the five quantiles for minute h after the checkpoint. */
function horizonRow(p: Float64Array, w: number, x: number, h: number): number[] {
  const m = w - h; // changes p[t+h] - p[t] exist for t in [0, w-1-h]
  const spread = BASE_SIGMA * Math.sqrt(h);
  if (m < 8) return Z.map((z) => price6(x + z * spread)); // too little history: the baseline row
  const stride = Math.max(1, Math.ceil(m / MAX_SAMPLES));
  const count = Math.floor((m - 1) / stride) + 1;
  const changes = new Float64Array(count);
  for (let i = 0, t = 0; t < m; i++, t += stride) changes[i] = (p[t + h] as number) - (p[t] as number);
  changes.sort();
  const wEmp = count / (count + SHRINK_K);
  const row = QUANTILES.map((tau, j) => x + wEmp * sampleQuantile(changes, tau) + (1 - wEmp) * (Z[j] as number) * spread);
  // Enforce non-decreasing, then clamp and round (rounding preserves the order).
  for (let j = 1; j < row.length; j++) if ((row[j] as number) < (row[j - 1] as number)) row[j] = row[j - 1] as number;
  return row.map(price6);
}

/** The forecast of one market: horizon rows of the five quantiles. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  const n = m.prices.length;
  if (n === 0) return Array.from({ length: horizon }, () => [0.5, 0.5, 0.5, 0.5, 0.5]);
  const x = m.prices[n - 1] as number;
  const w = Math.min(n, WINDOW);
  const p = new Float64Array(m.prices.slice(n - w));
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) rows.push(horizonRow(p, w, x, h));
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
