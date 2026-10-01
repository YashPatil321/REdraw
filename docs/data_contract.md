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

## HD world build (v1 additions, 2026-10-01) — tile grid

Everything below is written by `pipeline/build_all.py` (real and `--synthetic`). Old 4x4
paths in the sections above are superseded by these per-tile layers; `manifest.json` is the
index the client reads (through the API) and the only place file names come from.

**Tile grid (all tiled layers share it: terrain LODs, splat masks, buildings, roads, ground):**
- Grid extent = region bbox extent (`region_meta.extent_scene`, rounded out to 50 m) buffered by
  `region.yaml terrain_buffer_m` (500 m) = `region_meta.terrain_extent_scene`.
  For the current bbox: `min_x -4500, max_x 4500, min_z -3150, max_z 3150` (scene meters).
- `region.yaml tiles: [8, 8]` (rows, cols) -> each tile is 1125 m (x) by 787.5 m (z).
- Tile id `r{row}_c{col}`; row 0 is the NORTH edge (min_z), col 0 the WEST edge (min_x).
  Tile (r, c) covers `x in [min_x + c*w, min_x + (c+1)*w]`, `z in [min_z + r*d, min_z + (r+1)*d]`.
- A feature belongs to the tile containing its centroid (buildings: footprint centroid =
  `buildings.geojson centroid_x/centroid_z`, `tile` property). Python: `pipeline.common.tile_grid()`,
  `TileGrid.tile_of(x, z)`.
- `manifest.json tiles[]` lists every tile with `bounds` (incl. min_y/max_y over all layers).

### HD files (`client/public/assets/`)

```
manifest.json                       index (below)
terrain/terrain_r{r}_c{c}.glb       LOD0 terrain tile (adaptive RTIN from the 2 m lidar DEM)
terrain/terrain_r{r}_c{c}_lod1.glb  LOD1 (RTIN on a 129^2 sample grid, finest ~8.8 m)
terrain/terrain_r{r}_c{c}_lod2.glb  LOD2 (RTIN on a 65^2 sample grid, finest ~17.6 m)
terrain/albedo_r{r}_c{c}.jpg        full-size imagery for the tile (also embedded in each LOD)
terrain/splat_r{r}_c{c}_a.png       land-cover weights R lawn, G chaparral, B dirt
terrain/splat_r{r}_c{c}_b.png       land-cover weights R paved, G water, B canopy
terrain/heightmap.png, terrain/terrain_meta.json   (spec 13A.2, unchanged)
roads/roads_r{r}_c{c}.glb           road surfaces + paint, clipped at the tile edges
ground/ground_r{r}_c{c}.glb         sidewalks + curbs, medians, driveways, paths, pools
buildings/buildings_r{r}_c{c}.glb   procedural building tile (fallback when buildings_hd is absent)
buildings_hd/manifest_buildings.json  Blender-built buildings (separate agent), referenced when present
```

**Terrain LODs.** Every LOD is one primitive with POSITION, NORMAL, TEXCOORD_0 (u = (x - min_x) /
width, v = (z - min_z) / depth, i.e. north-up like the albedo and splat images) and an embedded
JPEG base color (LOD0 at the full `texture_px`, LOD1 half, LOD2 quarter). The mesh is a
right-triangulated irregular network (`pipeline/rtin.py`, Martini algorithm): one global max
vertical error per LOD chosen so LOD0 over all 64 tiles stays <= 2,000,000 triangles, LOD1 <= 20 %
and LOD2 <= 6 % of that. Each tile hangs a vertical skirt below its boundary edges (LOD0 3 m,
LOD1 6 m, LOD2 12 m) so mixed LODs and T-junctions never show cracks. Every draped layer (roads,
ground, building bases, `network_edges.geometry`) follows the LOD0 surface (`TerrainBuild.render`).
`manifest.terrain_lod.levels[]` gives `finest_spacing_m`, `max_error_m`, `triangles`,
`texture_px` per LOD; `tiles[].terrain_lods[]` the same per tile plus `path`.
`suggested_switch_distance_m` (camera distance to the tile bounds) is a hint only.

**Imagery.** The albedo is the best imagery reachable from the build machine at its native
resolution per tile (`texture_px` = next power of two >= tile size / pixel size, max 4096). In the
current real build that is Copernicus Sentinel-2 (10 m, resampled to 2.5 m in the mosaic) because
NAIP is not reachable (see `region_meta.sources_not_available`), so tiles are 512 px; the splat
masks and ground materials carry the close-range detail.

**Splat masks.** Two RGB PNGs per tile, `manifest.splat.px` square (512, or 1024 when the lidar
rasters exist), north up, same footprint as the albedo; the six 8-bit weights sum to 255 per
pixel. Inputs in priority order (`manifest.splat.inputs` lists what the build had): a soft color
classification of the imagery; Overture `land_cover` (ESA WorldCover classes, weak 10 m prior;
cached in `data/raw/overture/land_cover.parquet`); lidar 2014 canopy height / nDSM
(`data/raw/lidar/chm_0p5m.tif`, `ndsm_0p5m.tif`: tall -> canopy, knee-to-head high -> chaparral,
flat -> lawn/dirt; thresholds `assumptions.yaml hd_world.ndsm_*`); Overture `land_use`
(golf/park/pitch -> lawn, construction/bunker -> dirt, nature reserve -> no lawn); dirt trails;
then authoritative overrides: roads, sidewalks, driveways, paseos and roofs -> paved, water and
pools -> water (bright grey "water" in the imagery = covered reservoir -> paved).
`manifest.splat.suggested_ground_cells` maps channels to `materials_manifest.json` ground cells.

**Roads tile** (`roads_r{r}_c{c}.glb`), two primitives, all with POSITION, NORMAL, TEXCOORD_0,
COLOR_0 (color-only fallback) and the material convention attributes `_MAT` / `_VARIANT`
(uint8, see `materials_manifest.json conventions`):
- `asphalt` (`_MAT` 6 = ground atlas, `_VARIANT` = ground cell): carriageway ribbons (cell
  `asphalt_worn`; service roads / parking aisles / Overture driveways `asphalt_parking`) and
  junction patches. Ribbon UVs: u = meters across from the right edge / cell world size, v =
  meters along / cell world size; patches use planar UVs u = x / size, v = -z / size.
- `markings` (`_MAT` 8 = pipeline extension, `_VARIANT` = `ground_markings.png` column): lane
  lines, center lines, edge lines (u 0..1 across the column's `world_width_m`, v = meters along /
  `period_m`), continental crosswalks on every approach of a signalized junction and stop bars
  (quads exactly one column wide).
- Lifts above the LOD0 surface (m): service 0.04, road 0.06, junction patch 0.08, paint 0.10,
  crosswalk/stop bar 0.11, driveway 0.12, sidewalk top 0.06 + curb height (0.15) ; the client adds
  polygonOffset on top. Bridge decks (Overture `is_bridge`) are interpolated between the
  abutments instead of following the bare-earth DEM.
- Render roads = the unclipped drive graph PLUS real streets the routable graph leaves out
  (OSM access=private: gated communities and private estates, from Overture segments; never
  routable) PLUS Overture service roads.

**Ground tile** (`ground_r{r}_c{c}.glb`), primitives (same attributes, `_MAT` 6):
- `sidewalks`: concrete walk along every residential / unclassified / tertiary-and-up street
  (`streets.sidewalk_width_m`), raised by the curb height, with curb faces + gutter pans along
  the road side (cell `curb_gutter`, normals horizontal) and a short skirt elsewhere.
- `medians`: raised planted medians between paired one-way arterial carriageways (cell
  `coastal_sage`) with curbs.
- `driveways`: one per house whose garage wall faces a street within
  `streets.house_driveway_max_m` (garage side chosen from the wall nearest the street; cell
  `concrete_driveway`, u across / v along the driveway). Garage doors on the building tiles sit at
  the same spot.
- `paths`: Overture footway / pedestrian / cycleway segments away from the streets (paseos, park
  paths; cell `concrete_sidewalk`, `hd_world.paseo_width_m`) and path / track / bridleway (dirt
  trails, cell `decomposed_granite`, `hd_world.trail_width_m`). Sidewalk-type footways along
  streets are dropped (the generated sidewalks replace them).
- `pools`: only real mapped swimming pools (Overture water class `swimming_pool`): flat water
  surface (cell `pool_water`) at the highest ground of the outline + 0.05 m, and a coping ring.

**Building tile** (`buildings_r{r}_c{c}.glb`): one HD primitive (per-house hipped / gabled roofs on
orthogonalized footprints, parapets, garage doors on the street side) with `_BUILDING_ID` (uint16),
`_MAT`, `_VARIANT`, `_FRONT` (1 on street-facing walls) and COLOR_0, plus hero primitives
(`pipeline/hero_overrides/`, `_MAT`/`_VARIANT` copied from the hero glb). Kept as the fallback /
far LOD; when `manifest.buildings_hd` is set the client should prefer the Blender buildings and
use these tiles only for buildings the HD set does not cover.

### `manifest.json` (HD keys)
`hd_version` (1), `grid` {rows, cols, min_x, max_x, min_z, max_z, tile_width_m, tile_depth_m},
`tiles[]` {id, row, col, bounds (incl. min_y/max_y over all layers), terrain (LOD0 path),
terrain_lods[], albedo, splat [a, b], buildings, roads, ground, triangles {buildings, roads,
ground}}, `terrain_lod`, `splat`, `materials` (materials manifest path or null), `heroes[]`,
`triangles` {terrain (LOD0), terrain_lods {lod0, lod1, lod2}, buildings, roads, ground},
`street_stats` (counts of crosswalks, driveways, sidewalk area, paths, ...), `sizes_mb`,
`draco` / `draco_layers`, optional `buildings_hd` ("buildings_hd/manifest_buildings.json", only
when that file exists at build time).

### `buildings.geojson` additions (HD build)
- `source` (footprint dataset: OpenStreetMap, Microsoft ML Buildings, Esri Community Maps, lidar,
  ...) and `source_id` (that dataset's id: Overture GERS id for map footprints, `lidar_NNNNN` for
  lidar-only buildings). `buildings_roofs.parquet building_id` joins on `source_id`.
- When `data/raw/lidar/buildings_roofs.parquet` exists: `eave_height_m`, `ridge_height_m`
  (m above the lidar bare earth), `roof_type` (flat | gable | hip | complex | shed | unknown),
  `roof_pitch_deg`, `ridge_azimuth_deg`, `lidar_quality` (good | fair | poor | none),
  `lidar_status` (present | absent | absent_regraded_after_2014), `height_source`
  (lidar | height | levels | default | hero). For quality good/fair roofs present in the 2014
  flight, `height_m` is the lidar ridge height and the roof shape follows `roof_type`.
- Lidar-only buildings (`data/raw/lidar/missing_buildings.geojson`) are appended as
  `type: house`, `source: lidar`, after every mapped footprint, so mapped building ids do not
  change when they appear.

### `exits.json` additions
- `via` + `snap_method: "via_exit"` + `note`: an exit whose own road only clips a bbox corner
  and has no junction with the network inside the bbox (currently `deldios_north`, Del Dios
  Highway at the NW corner) keeps its id, label and bearing (so external jobs in that direction
  still exist) but uses the node of the nearest real exit (`via`, currently
  `sandieguito_west`); trips continue outside the bbox. Two exits may therefore share a
  `node_id`; `network_nodes.boundary_exit` names only the exit that owns the node.


## Lidar features (`data/raw/lidar/`, v1 2026-10-01)

Written by `pipeline/fetch_lidar.py` then `pipeline/lidar_features.py`; raw-data files (not committed),
read by `build_buildings.py`, `build_buildings_blender.py` and `build_props.py`.

    .venv/bin/python -m pipeline.fetch_lidar        # EPT nodes -> 1 km UTM LAZ tiles (cached, resumable)
    .venv/bin/python -m pipeline.lidar_features     # rasters, roofs, missing buildings, trees (cached per tile)
        --force           recompute everything       --force-roofs   recompute roofs only
        --force-trees     recompute tree QA only     --tiles-only    stop before roofs
        --no-shift        ignore the registration shift

Stages (each cached per 1 km tile in `data/raw/lidar/work/`, so a killed run resumes): tiles
(rasters, tree candidates, lidar-only building candidates) -> mosaic -> tree QA (per-tree vertical
return profile) -> trees.parquet + missing_buildings.geojson -> roofs -> buildings_roofs.parquet ->
lidar_validation.json. A fully cached re-run (assembly only) takes ~30 s.

Source: USGS 3DEP **CA_SanDiegoQL2_2014** EPT (`s3://usgs-lidar-public`, public domain), flown 2014,
~4.6 pts/m2 over the bbox. The brief's CA_SanDiego_2015_C17_1 has no points over the bbox. Class codes:
1 unclassified, 2 ground, 7 noise, 21/22 unclassified/ground in swath overlap, with no building or
vegetation class, so buildings and trees are separated geometrically (plane-fit residual + multi-return
penetration). **Anything built or planted after 2014 is missing**, and land regraded since then is
flagged by comparing with the 2024 3DEP DEM (`ground_change_0p5m.tif`, `lidar.ground_change_m`).
Lidar x/y are shifted by the measured lidar -> footprint registration (`work/registration.json`,
about (-1.0, +1.0) m) so roofs line up with the map footprints. Heights are NAVD88 m.

Rasters: GeoTIFF, EPSG:32611, 0.5 m, north-up, extent = bbox + `lidar.fetch_buffer_m` (snapped to 10 m),
float32 (NaN nodata), deflate, 512x512 tiles.

| file | content |
|---|---|
| `dsm_0p5m.tif` | max of all non-noise returns per cell, gaps filled, >= DTM |
| `dtm_0p5m.tif` | lidar bare earth (classes 2, 22): 1 m cell means + Delaunay interpolation under buildings / canopy |
| `ndsm_0p5m.tif` | DSM - DTM (height above bare earth, m, >= 0) |
| `chm_0p5m.tif` | canopy height model: nDSM on vegetation cells, else 0 |
| `ground_change_0p5m.tif` | 3DEP 2024 DEM (`dem_3dep_10m.tif`) - lidar 2014 DTM (m) |
| `surface_class_0p5m.tif` | uint8: 0 ground/low, 1 building-like (planar, opaque, nDSM > 2.5 m), 2 vegetation, 3 other elevated |

### `buildings_roofs.parquet` (one row per footprint)

Every `osm_buildings.geojson` footprint whose representative point is inside the region bbox.
Key: `building_id` = the footprint's `id` (Overture GERS id string) + centroid. Heights are above
ground (m). Roof points are the returns inside the footprint shrunk by `lidar.footprint_shrink_m`
(0.5 m) and higher than `lidar.roof_min_height_m`.

| column | meaning |
|---|---|
| `building_id`, `osm_way_id`, `footprint_source` | footprint ids / source (OpenStreetMap, Microsoft ML Buildings, Esri) |
| `centroid_lat`, `centroid_lon`, `scene_x`, `scene_z`, `easting`, `northing` | footprint centroid (WGS84, scene m, UTM 11N) |
| `lat`, `lon` | a point guaranteed inside the footprint (for spatial joins) |
| `footprint_area_m2`, `overture_height_m` | footprint area; parsed Overture/OSM `height` tag (validation only) |
| `ground_elev_m` (= `ground_elev`) | footprint median ground, 3DEP 2024 DEM (same surface as the scene terrain) |
| `ground_elev_lidar_m`, `ground_change_m` | 2014 lidar ground and 2024 - 2014 difference |
| `eave_height_m` (= `eave_h`) | 5th percentile of roof-point heights (flat roofs: 90th percentile) |
| `ridge_height_m` (= `ridge_h`) | 99th percentile of roof-point heights (flat roofs: 90th percentile) |
| `height_p50_m`, `height_max_m`, `ndsm_p90_m` | median / max roof-point height; 90th pct nDSM over the footprint |
| `roof_type` | `flat` / `gable` / `hip` / `complex` / `shed` / `unknown` (see below) |
| `roof_pitch_deg` (= `pitch_deg`) | inlier-weighted mean slope of sloped planes (flat: mean slope) |
| `ridge_azimuth_deg` | compass azimuth [0, 180) of the main ridge (intersection of the strongest opposite plane pair); NaN for flat |
| `n_planes`, `inlier_frac`, `plane_rmse_m` | RANSAC planes, share of roof points on a plane, plane-fit RMSE |
| `eave_perimeter_frac` | share of the (shrunk) outline next to a sloped plane draining outward across it (hip ~1, gable ~L/(L+W)) |
| `planes` | list of `{normal: [e, n, up], offset, area_m2, slope_deg, aspect_deg}`; `planes_json` holds the same plus `n_points`, `rmse_m` as JSON |
| `n_roof_points`, `n_points_footprint`, `roof_cover_frac`, `point_density_m2` | point counts inside the shrunk footprint |
| `lidar_status` | `present` / `absent` (no roof in 2014: built later or footprint error) / `absent_regraded_after_2014` |
| `quality` | `good` (>= 40 roof points, >= 70 % on planes) / `fair` (>= 12 points) / `poor` / `none` (absent) |

Plane frame: local ENU meters with origin at the footprint centroid (`easting`, `northing`) and height
above ground; a point (e, n, h) is on the plane when `normal . (e, n, h) = offset`, so the roof height at
(e, n) is `(offset - ne*e - nn*n) / nup`. `area_m2` is horizontal (projected) area. Absent footprints keep
NaN heights, `roof_type` `unknown` and empty `planes`.

Roof typing: sequential RANSAC on roof points (inlier = within `lidar.ransac_threshold_m` and point normal
within 15 deg of the plane normal, 1-point hypotheses from 3D PCA normals, largest connected patch only),
near-identical planes merged, leftovers absorbed by neighbouring planes. `flat`: planes flatter than
`lidar.flat_roof_max_slope_deg` hold >= 60 % of plane points; `shed`: one facing direction; `hip`: >= 3
facing directions and eaves on >= `lidar.hip_min_eave_frac` of the outline; `gable`: two opposite
facing directions (>= 85 % of sloped points); `complex`: everything else (multi-wing tract homes with
hip + gable wings, cross gables, split levels).

### `missing_buildings.geojson` (WGS84 polygons)

Lidar buildings with no footprint (every `osm_buildings.geojson` footprint, also outside the bbox,
masks them): nDSM > `lidar.missing_building_min_height_m`, planar (1.5 m plane residual < 0.2 m) and
opaque (< 30 % multi-return), opened by a 1 m disk, at least 2.5 m wide, area >= `lidar.missing_building_min_area_m2`,
not within 6 m of a drive-graph road or 10 m of a road bridge (drive-graph `bridge` + Overture
`is_bridge`), not on ground regraded after 2014, not thin or elongated (minimum rotated rectangle width
>= `lidar.missing_building_min_width_m`, length / width <= `lidar.missing_building_max_elongation`: walls,
sign gantries, carport rows), centroid inside the bbox. Polygonized from the 0.5 m mask and simplified
(0.5 m). Properties: `lidar_id` (`lidar_NNNNN`), `area_m2`, `height_m` (= ridge), the same roof columns as
`buildings_roofs.parquet` (from the polygon shrunk 0.25 m; `planes` is a JSON string here),
`penetration`, `centroid_*`, `adjacent_to_footprint` (within 3 m of a mapped footprint: usually an
unmapped wing / carport / patio cover of that building), `lidar_status`, and `quality`:
`good` = stand-alone, >= 40 roof points, not trailer-like; `fair` = adjacent to a footprint or 12-39
points; `poor` = flat, < 4.5 m high and < 150 m2 (parked RVs / trailers / sheds look like this) or
< 12 points. These are **candidates**: consumers that add them as buildings should keep `good` (and
maybe `fair`).

### `trees.parquet` (one row per tree)

Tree tops = variable-window local maxima of the 0.6 m-Gaussian-smoothed CHM (window radius = half the
Popescu & Wynne 2004 crown width for that height, clipped to 1-5 m) taller than `lidar.tree_min_height_m`;
crowns = marker watershed of the inverted CHM on vegetation cells, trimmed below 40 % of the top
height; `crown_radius_m` = sqrt(crown segment area / pi). Trees on building footprints (+1.5 m) or
regraded ground are dropped; inside the bbox only. Tree QA (counts in `lidar_validation.json`
`tree_candidates_dropped`) also drops: candidates with fewer than `lidar.tree_min_upper_returns`
returns in the upper crown (poles, lamp heads, wires), candidates taller than `lidar.tree_max_height_m`
(towers), and tall, very narrow candidates (radius / height < `lidar.wire_max_radius_height_ratio`)
lying on straight >= 400 m runs (the SDG&E transmission-line corridors; sequential RANSAC lines).

| column | meaning |
|---|---|
| `tree_id` | 0..n-1 (stable for a given build) |
| `x`, `z` | scene meters (`pipeline.geo`, region.yaml origin; x east, z south) |
| `lat`, `lon`, `easting`, `northing` | tree top position |
| `ground_y` | ground elevation at the trunk, 3DEP 2024 DEM (scene terrain), NAVD88 m |
| `ground_lidar_m` | 2014 lidar ground there (`abs(ground_y - ground_lidar_m) > lidar.ground_change_m` = stale) |
| `height_m` | max CHM inside the crown (m above ground) |
| `crown_radius_m` | radius of a circle with the crown's area |
| `crown_mean_ratio` | mean crown CHM / top height (cone ~0.33-0.5, dome ~0.7+) |
| `canopy_cover_20m` | share of canopy cells in the 20 m square around the top |
| `upper_returns` | lidar returns within the crown radius above half the height |
| `mid_return_frac` | share of the crown-radius returns above 2 m that lie below half the height (bare trunk ~0) |
| `species_guess` | `palm` / `broadleaf` / `conifer`: **a crude geometric guess, not a classification**. palm = tall (>= `lidar.palm_min_height_m`), small crown (radius <= `lidar.palm_max_crown_radius_m`, <= `lidar.palm_max_radius_height_ratio` x height), fairly isolated (`canopy_cover_20m` < `lidar.palm_max_canopy_cover`), bare-trunk profile (`mid_return_frac` < `lidar.palm_max_mid_return_frac`); conifer = conical crown (`crown_mean_ratio` < `lidar.conifer_max_crown_fill`, narrow, >= `lidar.conifer_min_height_m`); else broadleaf |

### `lidar_validation.json`

Counts (roof types, status, quality, missing buildings, trees by guess), Overture height tag - lidar
ridge statistics (all / per footprint source / per roof type, n, median, MAE, RMSE, share within 1 / 2 m,
correlation; also against p50 and eave heights), height / pitch / crown percentiles, the registration
samples and per-stage timings.
