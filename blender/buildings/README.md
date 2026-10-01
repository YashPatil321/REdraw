# HD buildings (headless Blender)

Every building in the region bbox rebuilt from its real footprint and its real lidar roof, exported as
8 x 8 glTF tiles with two LODs. Owner files: `blender/buildings/`, `pipeline/build_buildings_blender.py`,
`blender/tests/test_buildings_hd.py`, outputs `client/public/assets/buildings_hd/`,
previews `blender/previews/buildings_*.png`.

```bash
.venv/bin/python pipeline/build_buildings_blender.py                 # specs (main venv: footprints, DEM, lidar, streets) ~2 min
.venv/bin/python pipeline/build_buildings_blender.py --preview-only  # terrain / NAIP / roads around the two preview views
.venv-blender/bin/python blender/buildings/build.py                  # tiles (4 bpy workers, resumable) + Draco + manifests
.venv-blender/bin/python blender/buildings/build.py --tiles r3_c3 --force   # rebuild some tiles
.venv-blender/bin/python blender/buildings/preview.py                # Cycles previews -> blender/previews/buildings_*.png
.venv/bin/pytest blender/tests/test_buildings_hd.py
```

`build.py` runs one `bpy` process per tile and skips a tile when its glbs exist and its stamp (spec
file + `geom.py` / `model.py` / `build.py` + params) is unchanged, so a container restart only loses
the tiles in flight. Draco runs through `npx @gltf-transform/cli draco` when available
(`manifest_buildings.json` `draco` says which).

## Pipeline

1. **Specs** (`pipeline/build_buildings_blender.py`, main venv). The building set and ids are exactly the
   pipeline's (`build_buildings.load_osm_buildings` + `prepare_buildings` + `apply_heroes`), so
   `_BUILDING_ID` = `buildings.geojson` id. Footprints inside a hero radius
   (`pipeline/hero_overrides/hero_overrides.json`) are skipped. Lidar-only buildings
   (`data/raw/lidar/missing_buildings.geojson`, quality good / fair, not overlapping a footprint) are
   appended with ids above the pipeline maximum. Per building: ring in scene coords, ground from
   `dem_3dep_10m.tif`, the nearest streets (Overture segments) and driveway, and the roof model:
   - `lidar` - `buildings_roofs.parquet` (status present, quality good / fair; joined by centroid):
     eave / ridge height, roof type, pitch, ridge azimuth, planes, plus a 1 m nDSM height grid of the
     footprint (`ndsm_0p5m.tif`) used for the two-level massing;
   - `tag` - Overture / OSM height, read as a mid-roof height (ridge = tag +
     `buildings_hd.tag_ridge_offset_m`, the measured tag-vs-lidar bias);
   - `heuristic` - building class and footprint size.
   Real-world numbers come from `data/config/assumptions.yaml` (`buildings_hd.*`, `building_style.*`,
   `streets.garage_door_*`) and travel to the model in `specs/index.json` `params`.
2. **Model** (`model.py`, pure numpy + shapely; `geom.py` geometry kernel). Footprint -> dominant axis ->
   rectilinear ring (offsets >= 0.9 m kept, IoU >= 0.86 with the real outline, else the real outline).
   Roofs are upper envelopes of "pieces" over the maximal rectangles of the plan: hip pieces give the
   exact straight-skeleton hip roof of L / T / U / stepped plans, gable pieces ridge along the lidar
   ridge azimuth (cross gables on the wings), `complex` roofs pick hip or gable per wing by testing
   whether a lidar plane drains across that wing's end wall at eave height, `shed` and `flat` as named;
   ridges are capped at the lidar ridge height. Two roof levels: the nDSM in a 1.3 m band inside each
   wing's outline walls gives that wing's eave, so one-storey garage wings / front projections get a
   low roof dying into the two-storey block (step walls with upper-floor windows); a second storey
   set back from all outline walls is found as the tall part of the height grid. Without lidar a
   two-storey house gets a one-storey wing outside its largest rectangle (`two_level_share`).
   Details: 0.55 m eaves with fascia + soffit (roof-parallel), windows per floor (trim surround, glass,
   sloped sill on street walls; fewer on side yards; patio sliders at the back), a recessed garage door
   (2-car, or 2+1 car) on the wall facing the street / driveway with no ground-floor windows on it, the
   entry door next to it (or on the opposite street for alley-loaded lots) with a stoop and a small tile
   porch roof, ledgestone wainscots, PV arrays on south-facing planes, storefront glazing with mullions,
   sign band and canopy on commercial street walls, ribbon windows on schools, parapets with caps and
   rooftop HVAC on flat roofs. LOD1: coarser footprint (2 m offsets), walls to the roof line, roof planes,
   one level at the two-storey eave, no overhangs / openings / parapets.
3. **Tiles** (`build.py`, bpy). Triangle soup -> welded vertices -> one Blender mesh per tile and LOD with
   point attributes -> glTF exporter (`export_attributes`) -> gltf-transform Draco (positions 20 bit).

## Client format (`client/public/assets/buildings_hd/`)

| file | content |
| --- | --- |
| `r{r}_c{c}_lod0.glb`, `r{r}_c{c}_lod1.glb` | one mesh, one primitive, one material (`buildings_hd`, single-sided) per tile and LOD; identity nodes; positions in scene meters (x east, y up, z south) |
| `manifest_buildings.json` | `tiles[]` (`id`, `row`, `col`, `cell` extent, `bounds` incl. y, `buildings`, `lod0` / `lod1` = `{file, triangles, vertices, bytes}`), totals, counts, `draco`, attribute docs, `suggested_lod1_distance_m` |
| `buildings_hd.json` | `buildings[]`: `id`, `tile`, `type`, `eave_h`, `ridge_h` (m above floor), `roof_type` (flat / hip / gable / shed / complex), `source` (lidar / tag / heuristic), `levels`, `two_level`, `origin` (osm / lidar_missing), `garages` / `entries` = `[[x, z, nx, nz, width, floor_y]]` (door center, outward normal) for driveways, parked cars, walks |

Tiles use the shared 8 x 8 grid (`docs/data_contract.md` "HD world build"); a building belongs to the
tile of its footprint centroid, so geometry may stick out of the tile cell by a few meters (`bounds`).

Vertex attributes (all FLOAT; three.js names in brackets):

| attribute | meaning |
| --- | --- |
| `_BUILDING_ID` (`_building_id`) | building id (integral) |
| `_MAT` (`_mat`) | 0 stucco wall, 1 tile roof, 2 flat roof, 3 glass, 4 trim / doors / soffit / fascia / sills, 5 garage door (`materials_manifest.json materials`) |
| `_VARIANT` (`_variant`) | index into `materials[_MAT].variants` (wall finish, tile blend, flat roof membrane, glass, trim stucco / stone, 2-car / single garage door) |
| `COLOR_0` | linear RGB: wall tint on walls (tintable cells: `base = COLOR_0 * texel / cell.mean_albedo_linear`), trim / door color on `_MAT 4`, real fallback colors on roofs / glass / garage doors / stone (non-tintable cells ignore it) |
| `TEXCOORD_0` | `materials_manifest.json uv_conventions`: walls / trim u = m along the wall / 3, v = m above the floor / 3; pitched roofs u = m along the eave / 4, v = m up the slope / 4; flat roofs u = x / 4, v = -z / 4. Exceptions, both inside one atlas cell: window glass maps each pane onto a vision lite of `glass_curtain` (u 0.02-0.48 single pane, 0.02-0.98 two panes with the meeting rail, v 0.27-0.98), garage doors map onto the cell's `opening_rects_m.door` (`garage_2car` spans cells 0-1, the single door uses `garage_3car` cell 2) |
| `NORMAL` | flat per face |

Shader sketch: `cell = cells[_MAT][_VARIANT]`, `atlas_uv = mix(cell.uv_inner.xy, cell.uv_inner.zw, fract(uv))`
(span cells: add `floor(uv.x) * cell_step`), tint per the table, glass roughness ~0.05 with the env map.
Without atlases, `COLOR_0` alone renders a plausible flat-colored city.
