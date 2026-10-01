/** Small solar position function (NOAA spreadsheet algorithm, ~0.1 deg accuracy). */

const RAD = Math.PI / 180;

export interface SunPosition {
  /** degrees above the horizon (negative below) */
  elevation: number;
  /** degrees clockwise from north */
  azimuth: number;
}

export function solarPosition(date: Date, lat: number, lon: number): SunPosition {
  const jd = date.getTime() / 86400000 + 2440587.5;
  const T = (jd - 2451545) / 36525;
  const L0 = (((280.46646 + T * (36000.76983 + T * 0.0003032)) % 360) + 360) % 360;
  const M = 357.52911 + T * (35999.05029 - 0.0001537 * T);
  const e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T);
  const C =
    Math.sin(M * RAD) * (1.914602 - T * (0.004817 + 0.000014 * T)) +
    Math.sin(2 * M * RAD) * (0.019993 - 0.000101 * T) +
    Math.sin(3 * M * RAD) * 0.000289;
  const trueLong = L0 + C;
  const omega = 125.04 - 1934.136 * T;
  const lambda = trueLong - 0.00569 - 0.00478 * Math.sin(omega * RAD);
  const eps0 = 23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60;
  const eps = eps0 + 0.00256 * Math.cos(omega * RAD);
  const decl = Math.asin(Math.sin(eps * RAD) * Math.sin(lambda * RAD));
  const y = Math.tan((eps / 2) * RAD) ** 2;
  const eot =
    4 *
    (y * Math.sin(2 * L0 * RAD) -
      2 * e * Math.sin(M * RAD) +
      4 * e * y * Math.sin(M * RAD) * Math.cos(2 * L0 * RAD) -
      0.5 * y * y * Math.sin(4 * L0 * RAD) -
      1.25 * e * e * Math.sin(2 * M * RAD)) /
    RAD;
  const utcMin = date.getUTCHours() * 60 + date.getUTCMinutes() + date.getUTCSeconds() / 60;
  const tst = (((utcMin + eot + 4 * lon) % 1440) + 1440) % 1440;
  const ha = (tst / 4 - 180) * RAD;
  const latR = lat * RAD;
  const cosZ = Math.sin(latR) * Math.sin(decl) + Math.cos(latR) * Math.cos(decl) * Math.cos(ha);
  const zenith = Math.acos(Math.min(1, Math.max(-1, cosZ)));
  const az = Math.atan2(Math.sin(ha), Math.cos(ha) * Math.sin(latR) - Math.tan(decl) * Math.cos(latR)) / RAD + 180;
  return { elevation: 90 - zenith / RAD, azimuth: ((az % 360) + 360) % 360 };
}

/** Unit vector toward the sun in scene space (x east, y up, z south). */
export function sunDirection(p: SunPosition): [number, number, number] {
  const el = p.elevation * RAD;
  const az = p.azimuth * RAD;
  return [Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el)];
}

/** Offset of `timeZone` from UTC in minutes at `date` (e.g. -420 for PDT). */
export function tzOffsetMinutes(timeZone: string, date: Date): number {
  try {
    const parts = new Intl.DateTimeFormat('en-US', {
      timeZone,
      hourCycle: 'h23',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    }).formatToParts(date);
    const get = (t: string): number => Number(parts.find((p) => p.type === t)?.value ?? 0);
    const asUtc = Date.UTC(get('year'), get('month') - 1, get('day'), get('hour') % 24, get('minute'), get('second'));
    return Math.round((asUtc - date.getTime()) / 60000);
  } catch {
    return -480; // unknown zone: assume Pacific standard time
  }
}

/** Seconds since local midnight on `day` (in `timeZone`) -> absolute Date. */
export function localSecondsToDate(seconds: number, day: Date, timeZone: string): Date {
  const off = tzOffsetMinutes(timeZone, day);
  const local = new Date(day.getTime() + off * 60000);
  const midnightUtc = Date.UTC(local.getUTCFullYear(), local.getUTCMonth(), local.getUTCDate());
  return new Date(midnightUtc + seconds * 1000 - off * 60000);
}
