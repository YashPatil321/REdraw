/** Color ramps for congestion (v/c) and vehicle speed. Pure functions. */

export type RGB = [number, number, number];

interface Stop {
  at: number;
  c: RGB;
}

function hex(h: number): RGB {
  return [((h >> 16) & 255) / 255, ((h >> 8) & 255) / 255, (h & 255) / 255];
}

/** v/c ramp: free flow green -> yellow near 0.75 -> red at capacity -> dark red. */
export const VC_STOPS: readonly Stop[] = [
  { at: 0.0, c: hex(0x1fbf5a) },
  { at: 0.5, c: hex(0x9ad83a) },
  { at: 0.75, c: hex(0xffd23f) },
  { at: 0.9, c: hex(0xff8c1a) },
  { at: 1.0, c: hex(0xff3b2f) },
  { at: 1.3, c: hex(0xa8001f) },
];

/** Speed ramp (km/h): stopped red -> slow orange -> moving yellow -> free green/cyan. */
export const SPEED_STOPS: readonly Stop[] = [
  { at: 0, c: hex(0xff2a2a) },
  { at: 12, c: hex(0xff8a1a) },
  { at: 28, c: hex(0xffe14a) },
  { at: 45, c: hex(0x7ee36a) },
  { at: 70, c: hex(0x5fd8ff) },
];

export function rampColor(stops: readonly Stop[], v: number, out: RGB = [0, 0, 0]): RGB {
  const first = stops[0]!;
  const last = stops[stops.length - 1]!;
  if (!(v > first.at)) {
    // also catches NaN
    out[0] = first.c[0];
    out[1] = first.c[1];
    out[2] = first.c[2];
    return out;
  }
  if (v >= last.at) {
    out[0] = last.c[0];
    out[1] = last.c[1];
    out[2] = last.c[2];
    return out;
  }
  for (let i = 1; i < stops.length; i++) {
    const b = stops[i]!;
    if (v <= b.at) {
      const a = stops[i - 1]!;
      const w = (v - a.at) / (b.at - a.at);
      out[0] = a.c[0] + (b.c[0] - a.c[0]) * w;
      out[1] = a.c[1] + (b.c[1] - a.c[1]) * w;
      out[2] = a.c[2] + (b.c[2] - a.c[2]) * w;
      return out;
    }
  }
  return out;
}

export function vcColor(vc: number, out?: RGB): RGB {
  return rampColor(VC_STOPS, vc, out);
}

export function speedColor(kph: number, out?: RGB): RGB {
  return rampColor(SPEED_STOPS, kph, out);
}

export function rgbToCss(c: RGB): string {
  return `rgb(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)})`;
}

/** Queue bar color from queue length relative to curb spots. */
export function queueColor(queueCars: number, curbSpots: number, out?: RGB): RGB {
  const ratio = curbSpots > 0 ? queueCars / (curbSpots * 3) : queueCars / 30;
  return rampColor(VC_STOPS, ratio, out);
}
