/**
 * Physically based clear-sky model (single scattering + a cheap multiple
 * scattering term), shared by the GPU sky shader and CPU-side estimates
 * (sun color through the atmosphere, sky irradiance for exposure, fog tint).
 *
 * Coefficients: Bruneton / Hillaire Earth atmosphere (Rayleigh, Mie with a
 * haze multiplier, ozone absorption). Units: meters. Radiance is relative to
 * a top-of-atmosphere sun irradiance of 1.
 */

export const EARTH_R = 6360e3;
export const ATMO_R = 6460e3;
export const RAYLEIGH: [number, number, number] = [5.802e-6, 13.558e-6, 33.1e-6];
export const MIE_SCAT = 3.996e-6;
export const MIE_EXT = 4.44e-6;
export const OZONE: [number, number, number] = [0.65e-6, 1.881e-6, 0.085e-6];
export const H_RAYLEIGH = 8000;
export const H_MIE = 1200;
export const MIE_G = 0.8;

/** GLSL implementation (included by the sky cube shader). */
export const ATMOSPHERE_GLSL = /* glsl */ `
#define ATM_PI 3.141592653589793
const float ATM_RE = ${EARTH_R.toFixed(1)};
const float ATM_RA = ${ATMO_R.toFixed(1)};
const vec3 ATM_BR = vec3(${RAYLEIGH.map((v) => v.toExponential(4)).join(', ')});
const float ATM_BMS = ${MIE_SCAT.toExponential(4)};
const float ATM_BME = ${MIE_EXT.toExponential(4)};
const vec3 ATM_BO = vec3(${OZONE.map((v) => v.toExponential(4)).join(', ')});
const float ATM_HR = ${H_RAYLEIGH.toFixed(1)};
const float ATM_HM = ${H_MIE.toFixed(1)};

vec2 atmSphere(vec3 ro, vec3 rd, float r) {
  float b = dot(ro, rd);
  float c = dot(ro, ro) - r * r;
  float d = b * b - c;
  if (d < 0.0) return vec2(-1.0);
  d = sqrt(d);
  return vec2(-b - d, -b + d);
}
vec3 atmExt(float h, float mie) {
  h = max(h, 0.0);
  float dO = max(0.0, 1.0 - abs(h - 25000.0) / 15000.0);
  return ATM_BR * exp(-h / ATM_HR) + ATM_BME * mie * exp(-h / ATM_HM) + ATM_BO * dO;
}
/** transmittance from p to space toward the sun (0 below the horizon, soft over the sun disc) */
vec3 atmSunT(vec3 p, vec3 s, float mie) {
  float r = length(p);
  float mu = dot(p / r, s);
  float muH = -sqrt(max(0.0, 1.0 - (ATM_RE * ATM_RE) / (r * r)));
  float vis = smoothstep(muH - 0.0045, muH + 0.0045, mu);
  if (vis <= 0.0) return vec3(0.0);
  float tTop = atmSphere(p, s, ATM_RA).y;
  vec3 od = vec3(0.0);
  float prev = 0.0;
  for (int i = 1; i <= 14; i++) {
    float f = float(i) / 14.0;
    float t = tTop * f * f;
    float tm = 0.5 * (t + prev);
    od += atmExt(length(p + s * tm) - ATM_RE, mie) * (t - prev);
    prev = t;
  }
  return exp(-od) * vis;
}
float atmPhaseR(float mu) { return 3.0 / (16.0 * ATM_PI) * (1.0 + mu * mu); }
float atmPhaseM(float mu, float g) {
  float g2 = g * g;
  return 3.0 / (8.0 * ATM_PI) * ((1.0 - g2) * (1.0 + mu * mu)) / ((2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));
}
/** radiance toward the eye along rd from altitude alt (m), sun irradiance 1 */
vec3 atmSky(vec3 rd, vec3 s, float alt, float mie, vec3 groundAlbedo, float msK) {
  vec3 ro = vec3(0.0, ATM_RE + alt, 0.0);
  float tTop = atmSphere(ro, rd, ATM_RA).y;
  vec2 tg = atmSphere(ro, rd, ATM_RE);
  bool ground = tg.x > 0.0;
  float tMax = ground ? tg.x : tTop;
  float mu = dot(rd, s);
  float pR = atmPhaseR(mu);
  float pM = atmPhaseM(mu, ${MIE_G.toFixed(2)});
  vec3 L = vec3(0.0);
  vec3 T = vec3(1.0);
  float prev = 0.0;
  const int N = 36;
  for (int i = 1; i <= N; i++) {
    float f = float(i) / float(N);
    float t = tMax * f * f;
    float ds = t - prev;
    vec3 p = ro + rd * (0.5 * (t + prev));
    prev = t;
    float h = max(length(p) - ATM_RE, 0.0);
    vec3 sR = ATM_BR * exp(-h / ATM_HR);
    float sM = ATM_BMS * mie * exp(-h / ATM_HM);
    vec3 e = atmExt(h, mie);
    vec3 Ts = atmSunT(p, s, mie);
    // single scattering + isotropic multiple-scattering approximation
    float sunUp = clamp(dot(normalize(p), s) * 4.0 + 0.6, 0.0, 1.0);
    vec3 ms = (sR + sM) * msK * (Ts * 0.9 + 0.012 * sunUp);
    vec3 S = (sR * pR + sM * pM) * Ts + ms;
    vec3 Tstep = exp(-e * ds);
    // energy-conserving integration over the step
    L += T * (S - S * Tstep) / max(e, vec3(1e-12));
    T *= Tstep;
  }
  if (ground) {
    vec3 pg = ro + rd * tMax;
    vec3 n = normalize(pg);
    vec3 Tg = atmSunT(pg + n * 2.0, s, mie);
    float cosS = max(dot(n, s), 0.0);
    // ground lit by the sun plus a sky-ambient guess
    L += T * groundAlbedo / ATM_PI * (Tg * cosS + vec3(0.06, 0.08, 0.12) * clamp(s.y * 3.0 + 0.4, 0.0, 1.0));
  }
  return L;
}
`;

type V3 = [number, number, number];

function sphere(ro: V3, rd: V3, r: number): [number, number] {
  const b = ro[0] * rd[0] + ro[1] * rd[1] + ro[2] * rd[2];
  const c = ro[0] * ro[0] + ro[1] * ro[1] + ro[2] * ro[2] - r * r;
  const d = b * b - c;
  if (d < 0) return [-1, -1];
  const s = Math.sqrt(d);
  return [-b - s, -b + s];
}

function ext(h: number, mie: number, out: V3): V3 {
  const hh = Math.max(h, 0);
  const dR = Math.exp(-hh / H_RAYLEIGH);
  const dM = Math.exp(-hh / H_MIE) * mie;
  const dO = Math.max(0, 1 - Math.abs(hh - 25000) / 15000);
  out[0] = RAYLEIGH[0] * dR + MIE_EXT * dM + OZONE[0] * dO;
  out[1] = RAYLEIGH[1] * dR + MIE_EXT * dM + OZONE[1] * dO;
  out[2] = RAYLEIGH[2] * dR + MIE_EXT * dM + OZONE[2] * dO;
  return out;
}

/**
 * Sun transmittance (linear RGB, 0..1) from altitude `alt` (m) toward a sun at
 * elevation `elevDeg`. Pure; unit tested. Soft over the solar disc at the horizon.
 */
export function sunTransmittance(elevDeg: number, alt = 300, mie = 1): V3 {
  const el = (elevDeg * Math.PI) / 180;
  const s: V3 = [Math.cos(el), Math.sin(el), 0];
  const p: V3 = [0, EARTH_R + alt, 0];
  const r = EARTH_R + alt;
  const muH = -Math.sqrt(Math.max(0, 1 - (EARTH_R * EARTH_R) / (r * r)));
  const mu = s[1];
  const t0 = Math.min(1, Math.max(0, (mu - (muH - 0.0045)) / 0.009));
  const vis = t0 * t0 * (3 - 2 * t0);
  if (vis <= 0) return [0, 0, 0];
  const tTop = sphere(p, s, ATMO_R)[1];
  const od: V3 = [0, 0, 0];
  const e: V3 = [0, 0, 0];
  let prev = 0;
  const N = 48;
  for (let i = 1; i <= N; i++) {
    const f = i / N;
    const t = tTop * f * f;
    const tm = 0.5 * (t + prev);
    const q: V3 = [p[0] + s[0] * tm, p[1] + s[1] * tm, p[2] + s[2] * tm];
    ext(Math.hypot(q[0], q[1], q[2]) - EARTH_R, mie, e);
    const ds = t - prev;
    od[0] += e[0] * ds;
    od[1] += e[1] * ds;
    od[2] += e[2] * ds;
    prev = t;
  }
  return [Math.exp(-od[0]) * vis, Math.exp(-od[1]) * vis, Math.exp(-od[2]) * vis];
}

/** Rec. 709 luminance. */
export function luminance(c: V3): number {
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

function sunT(p: V3, s: V3, mie: number, steps: number): V3 {
  const r = Math.hypot(p[0], p[1], p[2]);
  const mu = (p[0] * s[0] + p[1] * s[1] + p[2] * s[2]) / r;
  const muH = -Math.sqrt(Math.max(0, 1 - (EARTH_R * EARTH_R) / (r * r)));
  const t0 = Math.min(1, Math.max(0, (mu - (muH - 0.0045)) / 0.009));
  const vis = t0 * t0 * (3 - 2 * t0);
  if (vis <= 0) return [0, 0, 0];
  const tTop = sphere(p, s, ATMO_R)[1];
  const od: V3 = [0, 0, 0];
  const e: V3 = [0, 0, 0];
  let prev = 0;
  for (let i = 1; i <= steps; i++) {
    const f = i / steps;
    const t = tTop * f * f;
    const tm = 0.5 * (t + prev);
    ext(Math.hypot(p[0] + s[0] * tm, p[1] + s[1] * tm, p[2] + s[2] * tm) - EARTH_R, mie, e);
    const ds = t - prev;
    od[0] += e[0] * ds;
    od[1] += e[1] * ds;
    od[2] += e[2] * ds;
    prev = t;
  }
  return [Math.exp(-od[0]) * vis, Math.exp(-od[1]) * vis, Math.exp(-od[2]) * vis];
}

/**
 * CPU port of the shader's sky radiance (fewer samples), sun irradiance 1.
 * `rd` and `s` are unit vectors (y up). Used for fog tint and exposure.
 */
export function skyRadiance(rd: V3, s: V3, alt = 400, mie = 1, msK = 0.55): V3 {
  const ro: V3 = [0, EARTH_R + alt, 0];
  const tTop = sphere(ro, rd, ATMO_R)[1];
  const tg = sphere(ro, rd, EARTH_R);
  const tMax = tg[0] > 0 ? tg[0] : tTop;
  const mu = rd[0] * s[0] + rd[1] * s[1] + rd[2] * s[2];
  const pR = (3 / (16 * Math.PI)) * (1 + mu * mu);
  const g = MIE_G;
  const g2 = g * g;
  const pM = ((3 / (8 * Math.PI)) * ((1 - g2) * (1 + mu * mu))) / ((2 + g2) * Math.pow(Math.max(1 + g2 - 2 * g * mu, 1e-4), 1.5));
  const L: V3 = [0, 0, 0];
  const T: V3 = [1, 1, 1];
  const e: V3 = [0, 0, 0];
  let prev = 0;
  const N = 20;
  for (let i = 1; i <= N; i++) {
    const f = i / N;
    const t = tMax * f * f;
    const ds = t - prev;
    const tm = 0.5 * (t + prev);
    prev = t;
    const p: V3 = [ro[0] + rd[0] * tm, ro[1] + rd[1] * tm, ro[2] + rd[2] * tm];
    const r = Math.hypot(p[0], p[1], p[2]);
    const h = Math.max(r - EARTH_R, 0);
    const dR = Math.exp(-h / H_RAYLEIGH);
    const sM = MIE_SCAT * mie * Math.exp(-h / H_MIE);
    ext(h, mie, e);
    const Ts = sunT(p, s, mie, 8);
    const sunUp = Math.min(1, Math.max(0, ((p[0] * s[0] + p[1] * s[1] + p[2] * s[2]) / r) * 4 + 0.6));
    for (let c = 0; c < 3; c++) {
      const sR = RAYLEIGH[c]! * dR;
      const ms = (sR + sM) * msK * (Ts[c]! * 0.9 + 0.012 * sunUp);
      const S = (sR * pR + sM * pM) * Ts[c]! + ms;
      const ts = Math.exp(-e[c]! * ds);
      L[c] += (T[c]! * (S - S * ts)) / Math.max(e[c]!, 1e-12);
      T[c] *= ts;
    }
  }
  return L;
}
