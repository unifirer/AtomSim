import { N_MAX, isValidState } from "./quantum";

/**
 * One state in a superposition, as the builder holds it.
 *
 * `weight` is a relative population, normalized by the server (it returns the
 * normalized c_k, which is what the clock uses). `phasePi` is the phase of c_k
 * in units of pi, so a link reads `0.5` rather than `1.5707963267948966`.
 */
export interface SupComponent {
  n: number;
  l: number;
  m: number;
  weight: number;
  phasePi: number;
}

export const MAX_COMPONENTS = 4;

/** 1s + 2p0, equal populations, in phase: the sloshing dipole. */
export const DEFAULT_SUP: SupComponent[] = [
  { n: 1, l: 0, m: 0, weight: 1, phasePi: 0 },
  { n: 2, l: 1, m: 0, weight: 1, phasePi: 0 },
];

/** Why a mix cannot be sent, or null when it can. Mirrors the server's checks. */
export function supProblem(comps: SupComponent[]): string | null {
  if (comps.length < 1 || comps.length > MAX_COMPONENTS) {
    return `pick 1 to ${MAX_COMPONENTS} states`;
  }
  const seen = new Set<string>();
  for (const c of comps) {
    if (!isValidState(c.n, c.l, c.m) || c.n > N_MAX) return "not a valid state";
    if (!(Number.isFinite(c.weight) && c.weight >= 0)) return "populations must be ≥ 0";
    if (!Number.isFinite(c.phasePi)) return "phases must be numbers";
    const k = `${c.n},${c.l},${c.m}`;
    if (seen.has(k)) return "the same state appears twice; merge them";
    seen.add(k);
  }
  if (!comps.some((c) => c.weight > 0)) return "at least one population must be above zero";
  return null;
}

/**
 * One `n_l_m_weight_phase` string per state, written as a repeated `sup` key.
 * Underscores because URLSearchParams leaves them alone, and a weight carries
 * a `.` and m a `-`, so neither of those can be the separator.
 */
export function serializeSup(comps: SupComponent[]): string[] {
  return comps.map((c) => [c.n, c.l, c.m, c.weight, c.phasePi].join("_"));
}

/** The parsed mix, or undefined for anything the server would refuse. */
export function parseSup(raws: string[]): SupComponent[] | undefined {
  if (raws.length === 0) return undefined;
  const comps: SupComponent[] = [];
  for (const part of raws) {
    const f = part.split("_");
    if (f.length !== 5 || f.some((x) => x.trim() === "")) return undefined;
    const [n, l, m, weight, phasePi] = f.map(Number);
    comps.push({ n, l, m: m + 0, weight, phasePi });
  }
  return supProblem(comps) === null ? comps : undefined;
}

export function sameSup(a: SupComponent[], b: SupComponent[]): boolean {
  return serializeSup(a).join(" ") === serializeSup(b).join(" ");
}

export interface Coefficient {
  re: number;
  im: number;
}

/**
 * |Psi(x, 0, z, t)|^2 into `out`, with t in atomic units of time.
 *
 * Psi = sum_k c_k psi_k exp(-i E_k t). Every psi_k is real on y = 0, so each
 * frame is K complex scalars and one pass over the grid. This is the same sum
 * as `evolve_density` in superposition.py, which the engine tests hold to the
 * full complex wavefunction.
 */
export function evolveDensity(
  psi: Float32Array[],
  coeffs: Coefficient[],
  energies: number[],
  t: number,
  out: Float32Array,
): Float32Array {
  const K = psi.length;
  const ar = new Float64Array(K);
  const ai = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    // c e^{-i theta} = (re cos + im sin) + i (im cos - re sin)
    const th = energies[k] * t;
    const cs = Math.cos(th);
    const sn = Math.sin(th);
    ar[k] = coeffs[k].re * cs + coeffs[k].im * sn;
    ai[k] = coeffs[k].im * cs - coeffs[k].re * sn;
  }
  for (let i = 0; i < out.length; i++) {
    let re = 0;
    let im = 0;
    for (let k = 0; k < K; k++) {
      const v = psi[k][i];
      re += ar[k] * v;
      im += ai[k] * v;
    }
    out[i] = re * re + im * im;
  }
  return out;
}

/**
 * max over the grid of (sum_k |c_k| |psi_k|)^2: no frame can exceed it, so a
 * colour scale fixed here lets the density visibly breathe, where a per-frame
 * maximum would quietly renormalize the motion away.
 */
export function densityBound(psi: Float32Array[], coeffs: Coefficient[]): number {
  const mags = coeffs.map((c) => Math.hypot(c.re, c.im));
  let best = 0;
  const N = psi[0]?.length ?? 0;
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let k = 0; k < psi.length; k++) s += mags[k] * Math.abs(psi[k][i]);
    best = Math.max(best, s * s);
  }
  return best;
}

/** Femtoseconds as a short readout: 0.405 fs, 12.3 fs, 1.24 ps. */
export function formatFs(fs: number): string {
  if (fs >= 1000) return `${(fs / 1000).toPrecision(3)} ps`;
  if (fs >= 100) return `${fs.toFixed(0)} fs`;
  return `${fs.toPrecision(3)} fs`;
}

/** Whether two mixes name the same states in the same order (the grids' only input). */
export function sameStates(a: SupComponent[], b: SupComponent[]): boolean {
  return a.length === b.length && a.every((c, i) => c.n === b[i].n && c.l === b[i].l && c.m === b[i].m);
}

/**
 * c_k = sqrt(w_k / sum w) e^{i pi phasePi_k}, the server's normalization.
 *
 * Done here as well so a population or phase slider redraws the movie live
 * instead of asking for the same grids again: the psi_k and E_k depend only on
 * which states are mixed, and the c_k are the reader's input, not a result.
 */
export function coefficients(comps: SupComponent[]): Coefficient[] {
  const total = comps.reduce((s, c) => s + c.weight, 0);
  return comps.map((c) => {
    const mag = total > 0 ? Math.sqrt(c.weight / total) : 0;
    return { re: mag * Math.cos(Math.PI * c.phasePi), im: mag * Math.sin(Math.PI * c.phasePi) };
  });
}

/**
 * The distinct beat periods 2 pi / |E_j - E_k| (atomic units of time) between
 * populated states, longest first. Empty means the mix is stationary.
 */
export function beatPeriods(energies: number[], comps: SupComponent[]): number[] {
  const live = energies.filter((_, k) => comps[k]?.weight > 0);
  const gaps: number[] = [];
  for (let i = 0; i < live.length; i++) {
    for (let j = i + 1; j < live.length; j++) {
      const d = Math.abs(live[i] - live[j]);
      // Same tolerance as DEGENERATE_TOL in superposition.py
      if (d > 1e-12 && gaps.every((g) => Math.abs(g - d) > 1e-12)) gaps.push(d);
    }
  }
  return gaps.sort((a, b) => a - b).map((d) => (2 * Math.PI) / d);
}
