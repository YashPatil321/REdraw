# Coordinate contract

One contract for the pipeline, sim, web client, and the future Unreal client
(spec 4 and 13A.3).

| Space | Units | Axes | Used by |
| --- | --- | --- | --- |
| WGS84 | degrees | lat, lon | config files, plan JSON (map inputs), API |
| UTM 11N (EPSG:32611) | meters | easting, northing | all pipeline processing |
| Scene | meters | x east, y up, z south | glTF assets, playback, three.js |
| Unreal | centimeters | X east, Y south, Z up (left handed) | RedrawUE (post MVP) |

## Scene origin

The scene origin is the center of the region bbox in `data/config/region.yaml`
(`scene_origin: center_of_bbox`), projected to UTM 11N:

```
origin.lat = (south + north) / 2
origin.lon = (west + east) / 2
(origin.easting, origin.northing) = UTM11N(origin.lon, origin.lat)
```

## Conversions

```
x = easting  - origin.easting
z = -(northing - origin.northing)
y = elevation_m

UE.X = x * 100
UE.Y = z * 100
UE.Z = y * 100
```

Implementations:
- Python: `pipeline/geo.py` (`latlon_to_scene`, `scene_to_latlon`, `scene_to_unreal`)
- TypeScript: `client/src/geo.ts` (`latLonToScene`, `sceneToLatLon`, `sceneToUnreal`), a
  self-contained transverse Mercator implementation (no network, no proj4 needed).

## Test points

`docs/geo_test_points.json` holds lat/lon -> UTM -> scene values generated
with pyproj. `pipeline/tests/test_geo.py` and `client/src/geo.test.ts` both
assert against it (tolerance 1 cm). Example:

| lat | lon | x (m) | z (m) | UE.X (cm) | UE.Y (cm) |
| --- | --- | --- | --- | --- | --- |
| 33.0165 | -117.1225 | 0.0 | -0.0 | 0.0 | -0.0 |
| 33.01402 | -117.12246 | 3.4158 | 274.9383 | 341.58 | 27493.83 |
| 32.993 | -117.165 | -3973.5256 | 2599.7922 | -397352.56 | 259979.22 |
