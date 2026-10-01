# Processed data contract (pipeline -> sim / api / client)

Version: 1

`pipeline/build_all.py` writes everything below. The sim, API and residents
packages read ONLY these files (never `data/raw/`). The client never reads files
directly; it gets data through the API (spec 13A.5). Static meshes are served by
the API under `/assets/...` from `REDRAW_ASSETS_DIR` (default
`client/public/assets/`).

All scene coordinates follow `docs/coordinates.md`: meters, x east, z south,
y up, relative to the scene origin (center of the region bbox).

## Data modes

`build_all.py` has two modes:

| Mode | Command | When |
| --- | --- | --- |
| real (default) | `python pipeline/build_all.py` | Fetches OSM, USGS 3DEP, NAIP, ACS, LODES. Fails loudly with URL + manual download instructions if a source is unreachable (spec rule 14.6). |
| synthetic | `python pipeline/build_all.py --synthetic` | Procedurally generated stand-in world with the same file formats. For offline development and CI only. NOT real geography. |

Every output records which mode produced it (`region_meta.json -> synthetic`).
The API passes this through `/world/meta`, and the client shows a persistent
"SYNTHETIC DEV DATA" banner and the report card says so. Nothing in the
synthetic mode may be presented as real.

## `data/processed/` files

### `region_meta.json`
```json
{
  "contract_version": 1,
  "name": "4s_ranch_del_sur",
  "display_name": "4S Ranch and Del Sur, San Diego",
  "synthetic": false,
  "generated_at": "2026-10-01T12:00:00Z",
  "projection": "EPSG:32611",
  "bbox": {"south": 32.965, "north": 33.045, "west": -117.175, "east": -117.075},
  "origin": {"lat": 33.005, "lon": -117.125, "easting": 488323.68, "northing": 3651848.18},
  "extent_scene": {"min_x": -4700.0, "max_x": 4700.0, "min_z": -4450.0, "max_z": 4450.0},
  "counts": {"buildings": 0, "road_edges": 0, "road_nodes": 0, "households": 0, "persons": 0,
             "students_by_school": {"del_norte_hs": 0}},
  "sources": [{"name": "OpenStreetMap", "license": "ODbL", "retrieved": "2026-10-01"}]
}
```

### `network_nodes.parquet` (drive graph nodes)
| column | type | notes |
| --- | --- | --- |
| node_id | int64 | stable id (OSM node id in real mode) |
| x, z | float64 | scene meters |
| y | float32 | terrain elevation (m) |
| lat, lon | float64 | WGS84 |
| signalized | bool | OSM `highway=traffic_signals` at or adjacent to node |
| boundary_exit | string | exit id from region.yaml if this node is an exit node, else "" |

### `network_edges.parquet` (drive graph directed edges)
Row order IS `edge_idx` (0..E-1). Playback, plans (`turn_lane.edge`) and the
client all refer to edges by `edge_idx`. Two-way streets are two rows.
| column | type | notes |
| --- | --- | --- |
| edge_idx | int32 | 0..E-1, equals row number |
| u, v | int64 | from / to node_id |
| osmid | string | OSM way id(s), comma separated ("" synthetic) |
| name | string | street name ("" if none) |
| ref | string | e.g. "I 15" |
| highway | string | OSM class, `_link` suffix kept |
| lanes | int16 | lanes in this direction (>=1) |
| maxspeed_kph | float32 | |
| oneway | bool | |
| length_m | float32 | |
| capacity_vph | float32 | lanes * capacity_vphpl (class default) |
| free_flow_s | float32 | length_m / (maxspeed_kph/3.6) |
| geometry | list<float32> | flattened polyline `[x0,y0,z0, x1,y1,z1, ...]`, scene meters, draped on terrain, from u to v |
| label | string | arterial label from region.yaml if this edge belongs to a labeled arterial, else "" |

Also written for interoperability: `roads_drive.graphml` (OSMnx MultiDiGraph,
projected EPSG:32611) and `roads.geojson` (LineStrings in WGS84 with the same
properties plus `edge_idx`).

### `buildings.geojson`
FeatureCollection, Polygon geometries in WGS84. Properties:
| prop | type | notes |
| --- | --- | --- |
| id | int | building_id, stable, >= 1 (0 means "no building" in mesh attributes) |
| type | string | one of `house`, `apartments`, `commercial`, `school`, `other` |
| height_m | float | |
| base_elev_m | float | DEM at centroid |
| levels | int or null | |
| address | string or null | OSM addr:* if present |
| name | string or null | OSM name (schools, shops) |
| area_m2 | float | footprint area |
| centroid_x, centroid_z | float | scene meters |
| block_group | string | census block group GEOID ("" synthetic) |
| school_id | string or null | schools.yaml id if this footprint is a school building |
| tile | string | `r{row}_c{col}` |
| parcel_apn, parcel_land_use, parcel_year_built | null | reserved, empty in v1 |

### `households.parquet`
| column | type | notes |
| --- | --- | --- |
| household_id | int64 | |
| building_id | int64 | home building |
| block_group | string | |
| x, z | float64 | home location (scene meters, building centroid) |
| home_node | int64 | nearest drive node_id |
| size | int16 | |
| vehicles | int16 | |
| income_band | string | `lt50k`, `50_100k`, `100_150k`, `150_200k`, `gt200k` |
| n_kids | int16 | persons aged 5-18 |

### `persons.parquet`
| column | type | notes |
| --- | --- | --- |
| person_id | int64 | |
| household_id | int64 | |
| age | int16 | |
| is_worker | bool | |
| works_from_home | bool | |
| work_node | int64 | drive node_id of job location (inside bbox) or the exit node; -1 if not a worker |
| work_exit | string | exit id if job is outside the bbox, else "" |
| work_x, work_z | float64 | job location in scene meters (exit node location for external jobs); NaN if none |
| school_id | string | schools.yaml id, "" if not a student |
| grade | int16 | 0 (K) .. 12, -1 if not a student |
| has_license | bool | |
| baseline_mode_hint | string | "" (sim decides); reserved for real-household overrides |

### `schools_resolved.json`
Schools from `schools.yaml` merged with OSM-found schools, with entrances snapped
to the network.
```json
{"schools": [{
  "id": "del_norte_hs", "name": "Del Norte High School", "grades": [9, 12],
  "bell_start": "08:30", "lat": 33.0215, "lon": -117.101, "x": 2243.6, "z": -1826.8,
  "verified": false, "source": "schools.yaml",
  "building_ids": [123, 124],
  "entrances": [{
    "id": "main_dropoff", "lat": 33.0205, "lon": -117.1022, "x": 2130.0, "z": -1716.0,
    "node_id": 456, "approach_edge_idx": 789, "curb_spots": 10, "unload_seconds": 45,
    "verified": false
  }],
  "students": 2500
}]}
```
`node_id` is the drive node nearest the entrance; `approach_edge_idx` is an edge
whose `v == node_id` (the edge cars queue on).

### `exits.json`
```json
{"exits": [{"id": "i15_north", "label": "I 15 north", "node_id": 1, "x": 0.0, "z": 0.0, "bearing_deg": 0}]}
```

### `terrain_meta.json` (also copied to assets)
```json
{
  "heightmap": "terrain/heightmap.png",
  "width_px": 1009, "height_px": 1009,
  "min_x": -5200.0, "max_x": 5200.0, "min_z": -4950.0, "max_z": 4950.0,
  "min_elev_m": 120.0, "max_elev_m": 520.0,
  "elev_scale": 0.0061, "elev_offset": 120.0,
  "note": "elevation_m = elev_offset + pixel_value_uint16 * elev_scale; pixel (0,0) is at (min_x, min_z) i.e. north-west; +col is +x (east), +row is +z (south)"
}
```

## `client/public/assets/` (served at `/assets/` by the API)

```
assets/
  manifest.json
  terrain/terrain_r{r}_c{c}.glb      4x4 terrain tiles (Draco when available)
  terrain/albedo_r{r}_c{c}.jpg       (embedded in the glb; loose copies optional)
  terrain/heightmap.png              16-bit grayscale (spec 13A.2)
  terrain/terrain_meta.json
  buildings/buildings_r{r}_c{c}.glb  4x4 building tiles
  roads/roads.glb                    road ribbons (one file, or roads_r{r}_c{c}.glb)
```

### `manifest.json`
```json
{
  "contract_version": 1,
  "synthetic": false,
  "draco": true,
  "tiles": [{"id": "r0_c0", "row": 0, "col": 0,
             "bounds": {"min_x": -5200, "max_x": -2600, "min_z": -4950, "max_z": -2475, "min_y": 100, "max_y": 560},
             "terrain": "terrain/terrain_r0_c0.glb", "buildings": "buildings/buildings_r0_c0.glb"}],
  "roads": ["roads/roads.glb"],
  "terrain_meta": "terrain/terrain_meta.json",
  "triangles": {"terrain": 0, "buildings": 0, "roads": 0}
}
```

### Mesh conventions
- glTF 2.0 binary, positions in scene meters (x east, y up, z south), relative
  to the scene origin. No node transforms needed (identity), so tiles drop in.
- Building tiles: one mesh per tile, with a custom vertex attribute
  `_BUILDING_ID` (FLOAT, scalar; integral values; three.js exposes it as
  `geometry.attributes._building_id`). Building colors as vertex colors
  (`COLOR_0`) by type.
- Terrain tiles: positions + normals + `TEXCOORD_0`, base color texture = NAIP
  (real) or a procedural albedo (synthetic).
- Roads: ribbons lifted 0.3 m above terrain; vertex colors by class; client
  also applies polygonOffset.
- Draco: compress with `npx @gltf-transform/cli draco` when Node is available;
  if not, write uncompressed glb and set `manifest.draco = false`. The client
  must load both (GLTFLoader + DRACOLoader).

## Contract additions (v1, real-data build)

- `region_meta.json`: `population_source` (`acs_lodes` | `footprint_estimate`), `population_note`,
  `signal_source` (`osm` | `inferred`), `attribution`, `sources_not_available`.
- `exits.json`: each exit adds `target_distance_m` (distance from the region.yaml point to the
  snapped bbox-edge crossing). Exit split node ids start at 9,000,000,000.
- `schools_resolved.json` entrances add `snap_method`, `configured_lat`, `configured_lon`.
- `terrain_meta.json` adds `sources` and `texture_px`.
- `network_edges.osmid` may be `virtual_connector` (short links joining gated-community islands)
  or `virtual_exit_turnaround`.
