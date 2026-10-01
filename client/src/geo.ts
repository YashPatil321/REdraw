/**
 * Coordinate contract (docs/coordinates.md), TypeScript side.
 *
 * Self-contained UTM zone 11N (EPSG:32611) transverse Mercator using the
 * Krueger n-series to 6th order (sub-millimetre inside a UTM zone). No proj4.
 *
 * Scene space: meters, x east, y up, z south, relative to the scene origin
 * (center of the region bbox, projected to UTM 11N).
 */

export interface BBox {
  south: number;
  north: number;
  west: number;
  east: number;
}

export interface Origin {
  lat: number;
  lon: number;
  easting: number;
  northing: number;
}

// WGS84 ellipsoid and UTM 11N parameters.
const A_WGS84 = 6378137.0;
const F_WGS84 = 1 / 298.257223563;
const K0 = 0.9996;
const LON0_DEG = -117; // central meridian of zone 11: -183 + 6 * 11
const FALSE_EASTING = 500000;
const FALSE_NORTHING = 0;

const n = F_WGS84 / (2 - F_WGS84);
const n2 = n * n;
const n3 = n2 * n;
const n4 = n3 * n;
const n5 = n4 * n;
const n6 = n5 * n;
const RECT_A = (A_WGS84 / (1 + n)) * (1 + n2 / 4 + n4 / 64 + n6 / 256);
const ECC = (2 * Math.sqrt(n)) / (1 + n);

const ALPHA = [
  n / 2 - (2 * n2) / 3 + (5 * n3) / 16 + (41 * n4) / 180 - (127 * n5) / 288 + (7891 * n6) / 37800,
  (13 * n2) / 48 - (3 * n3) / 5 + (557 * n4) / 1440 + (281 * n5) / 630 - (1983433 * n6) / 1935360,
  (61 * n3) / 240 - (103 * n4) / 140 + (15061 * n5) / 26880 + (167603 * n6) / 181440,
  (49561 * n4) / 161280 - (179 * n5) / 168 + (6601661 * n6) / 7257600,
  (34729 * n5) / 80640 - (3418889 * n6) / 1995840,
  (212378941 * n6) / 319334400,
];

const BETA = [
  n / 2 - (2 * n2) / 3 + (37 * n3) / 96 - n4 / 360 - (81 * n5) / 512 + (96199 * n6) / 604800,
  n2 / 48 + n3 / 15 - (437 * n4) / 1440 + (46 * n5) / 105 - (1118711 * n6) / 3870720,
  (17 * n3) / 480 - (37 * n4) / 840 - (209 * n5) / 4480 + (5569 * n6) / 90720,
  (4397 * n4) / 161280 - (11 * n5) / 504 - (830251 * n6) / 7257600,
  (4583 * n5) / 161280 - (108847 * n6) / 3991680,
  (20648693 * n6) / 638668800,
];

const DELTA = [
  2 * n - (2 * n2) / 3 - 2 * n3 + (116 * n4) / 45 + (26 * n5) / 45 - (2854 * n6) / 675,
  (7 * n2) / 3 - (8 * n3) / 5 - (227 * n4) / 45 + (2704 * n5) / 315 + (2323 * n6) / 945,
  (56 * n3) / 15 - (136 * n4) / 35 - (1262 * n5) / 105 + (73814 * n6) / 2835,
  (4279 * n4) / 630 - (332 * n5) / 35 - (399572 * n6) / 14175,
  (4174 * n5) / 315 - (144838 * n6) / 6237,
  (601676 * n6) / 22275,
];

const DEG = Math.PI / 180;

/** WGS84 lat/lon (degrees) -> UTM 11N easting/northing (meters). */
export function latLonToUtm(lat: number, lon: number): { easting: number; northing: number } {
  const phi = lat * DEG;
  const dLam = (lon - LON0_DEG) * DEG;
  const sinPhi = Math.sin(phi);
  const t = Math.sinh(Math.atanh(sinPhi) - ECC * Math.atanh(ECC * sinPhi));
  const xiP = Math.atan2(t, Math.cos(dLam));
  const etaP = Math.atanh(Math.sin(dLam) / Math.sqrt(1 + t * t));
  let xi = xiP;
  let eta = etaP;
  for (let j = 1; j <= 6; j++) {
    const a = ALPHA[j - 1]!;
    xi += a * Math.sin(2 * j * xiP) * Math.cosh(2 * j * etaP);
    eta += a * Math.cos(2 * j * xiP) * Math.sinh(2 * j * etaP);
  }
  return {
    easting: FALSE_EASTING + K0 * RECT_A * eta,
    northing: FALSE_NORTHING + K0 * RECT_A * xi,
  };
}

/** UTM 11N easting/northing (meters) -> WGS84 lat/lon (degrees). */
export function utmToLatLon(easting: number, northing: number): { lat: number; lon: number } {
  const xi = (northing - FALSE_NORTHING) / (K0 * RECT_A);
  const eta = (easting - FALSE_EASTING) / (K0 * RECT_A);
  let xiP = xi;
  let etaP = eta;
  for (let j = 1; j <= 6; j++) {
    const b = BETA[j - 1]!;
    xiP -= b * Math.sin(2 * j * xi) * Math.cosh(2 * j * eta);
    etaP -= b * Math.cos(2 * j * xi) * Math.sinh(2 * j * eta);
  }
  const chi = Math.asin(Math.sin(xiP) / Math.cosh(etaP));
  let phi = chi;
  for (let j = 1; j <= 6; j++) phi += DELTA[j - 1]! * Math.sin(2 * j * chi);
  const lam = Math.atan2(Math.sinh(etaP), Math.cos(xiP));
  return { lat: phi / DEG, lon: LON0_DEG + lam / DEG };
}

/** Scene origin from lat/lon of the origin point. */
export function originFromLatLon(lat: number, lon: number): Origin {
  const { easting, northing } = latLonToUtm(lat, lon);
  return { lat, lon, easting, northing };
}

/** Scene origin = center of the region bbox (region.yaml scene_origin: center_of_bbox). */
export function originFromBbox(bbox: BBox): Origin {
  return originFromLatLon((bbox.south + bbox.north) / 2, (bbox.west + bbox.east) / 2);
}

/** WGS84 -> scene meters (x east, z south). */
export function latLonToScene(lat: number, lon: number, origin: Origin): { x: number; z: number } {
  const { easting, northing } = latLonToUtm(lat, lon);
  return { x: easting - origin.easting, z: -(northing - origin.northing) };
}

/** Scene meters -> WGS84. */
export function sceneToLatLon(x: number, z: number, origin: Origin): { lat: number; lon: number } {
  return utmToLatLon(x + origin.easting, origin.northing - z);
}

/** Scene meters -> Unreal centimeters (X east, Y south, Z up). */
export function sceneToUnreal(x: number, y: number, z: number): { X: number; Y: number; Z: number } {
  return { X: x * 100, Y: z * 100, Z: y * 100 };
}
