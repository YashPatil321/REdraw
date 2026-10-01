# blender/ - headless Blender asset pipeline

Procedural, seed-fixed props and hero campus models for Redraw, built with the
`bpy` wheel (no Blender UI). Everything here regenerates from code + real open
data; only the scripts, tests and `previews/*.png` are committed.

## Setup and commands

```bash
python3.11 -m venv .venv-blender
.venv-blender/bin/pip install bpy==5.0.1 shapely          # bpy 4.5.x also works

.venv-blender/bin/python blender/build_all_assets.py          # everything (about 5 min on 4 CPUs)
.venv-blender/bin/python blender/build_all_assets.py --only vehicles trees street   # props only, seconds
.venv-blender/bin/python blender/build_all_assets.py --no-previews --samples 32

.venv/bin/python blender/extract_hero_sites.py   # hero footprints/layout (main venv: geopandas); run automatically if missing
.venv/bin/python pipeline/build_props.py         # instance placements (trees, shrubs, lamps, parked cars)
.venv/bin/pytest blender/tests                   # glbs parse, manifest, hero overrides, placements format
```

Stages of `build_all_assets.py`: `vehicles trees street heroes manifest previews`.
Previews render with Cycles on the CPU (EEVEE needs a GPU context headless bpy
does not have) and are re-encoded to stay under 400 KB.

## Outputs

| Path | What |
| --- | --- |
| `client/public/assets/props/vehicles/*.glb` | 7 vehicles |
| `client/public/assets/props/vegetation/*.glb` | 8 plants (alpha-card foliage) |
| `client/public/assets/props/street/street_lamp.glb` | SD cobra-head street light |
| `client/public/assets/props/props_manifest.json` | prop table (below) |
| `client/public/assets/props/placements.json` + `.bin` | instances from `pipeline/build_props.py` (below) |
| `pipeline/hero_overrides/{del_norte_hs,design39,4s_commons}.glb` | hero campuses |
| `pipeline/hero_overrides/hero_overrides.json` | hero placement table the pipeline reads |
| `pipeline/hero_overrides/<id>_trees.json` | campus trees, lot lights, parked-car stalls (merged into placements) |
| `blender/build/` | generated textures, `hero_sites.json` (gitignored) |
| `blender/previews/*.png` | committed beauty renders |

## Conventions (all assets)

- glTF 2.0 binary, meters, **+Y up, forward = -Z**, right = +X. Blender works in
  +X east / +Y north (forward) / +Z up and the exporter (`export_yup`) maps Blender
  +Y to glTF -Z. A three.js object with `rotation.y = 0` faces scene north (-z);
  `rotation.y = atan2(-dx, -dz)` faces scene direction `(dx, dz)`.
- Props: origin on the ground at the footprint center (trunk base, pole base,
  midpoint between the wheels). Heroes: origin at the campus center, local meters
  (+x east, +y up, +z south), so `docs/coordinates.md` holds after a translation.
- No Draco (the pipeline's hero loader rejects it; props are small anyway).
- Fixed seeds everywhere; rebuilding gives identical geometry.

## Vehicles

Lofted bodies (cross-section rings mirrored left/right, wheel arches cut by raising
the ring bottoms around each axle), lathed tires with dished rims, and lights /
grilles / plates projected onto the curved body with a BVH ray cast.

| id | real class | L x W x H (m, body) | tris |
| --- | --- | --- | --- |
| `car_sedan` | Camry / Accord / Model 3 | 4.88 x 1.84 x 1.44 | 1440 |
| `car_suv` | RAV4 / CR-V | 4.75 x 1.88 x 1.70 | 1480 |
| `car_crossover_ev` | Model Y (glass roof, dark rims) | 4.75 x 1.92 x 1.62 | 1322 |
| `car_minivan` | Odyssey / Sienna | 5.16 x 1.99 x 1.74 | 1480 |
| `car_pickup` | F-150 SuperCrew 5.5 ft bed (open bed) | 5.89 x 2.03 x 1.96 | 1422 |
| `school_bus` | Type C Blue Bird Vision, 22.5 in duals | 11.6 x 2.44 x 3.2 | 2418 |
| `shuttle_van` | Transit / Sprinter high roof | 5.98 x 2.06 x 2.78 | 1404 |

(Manifest `width_m` / `length_m` include mirrors and bumpers.)

Materials (glTF PBR; names are a contract): `paint` (near-white metallic + clearcoat,
**tintable**), `glass` / `glass_clear` (dark tint, roughness 0.04), `tire`, `rim`,
`rim_dark`, `chrome`, `trim_black`, `underbody`, `plate`, `bed_liner`,
`headlight` / `drl` / `taillight` / `amber` / `bus_red_lamp` (emissive with
`KHR_materials_emissive_strength` 2-6, so bloom picks them up), bus `paint_bus`
(National School Bus Glossy Yellow) and `paint_bus_roof` (white).

Every vehicle vertex also carries custom attributes (three.js names in brackets),
so a client can merge all primitives into ONE geometry per vehicle type:

| attribute | meaning |
| --- | --- |
| `_LIGHT` (`_light`) | 0 body, 1 headlight/DRL, 2 taillight, 3 glass, 4 dark trim / tire (same codes as `traffic/layer.ts` `aLight`) |
| `_TINT` (`_tint`) | 1.0 on `paint`: multiply the instance paint color here |
| `_ALBEDO` (`_albedo`) | linear RGB base color of the vertex's material |

Paint colors: `props_manifest.json` vehicle entries carry `paint_colors`
(`[{name, hex (sRGB), share}]`, SoCal mix: white 25 %, black 21 %, gray 18 %,
silver 12 %, blue 9 %, red 8 %, ...; also `assumptions.yaml props.vehicle_paint_shares`)
and a `fleet_share` (`props.vehicle_type_shares`).

## Vegetation and street furniture

One mesh + one material per plant: `foliage` (alphaMode MASK, doubleSided) with a
generated 512 x 512 palette PNG atlas (leaf cells in x < 448, a bark strip in the
last 64 px column mapped onto the trunk tubes). Leaf cards are scattered in
ellipsoid clumps; their normals are bent outward from the canopy center so the
card cloud shades like a rounded crown.

| id | species / use | height | tris |
| --- | --- | --- | --- |
| `tree_oak` | coast live oak, canyons / slopes / big yards | 9.6 m, 15 m wide | 1778 |
| `tree_palm_fan` | Mexican fan palm (Washingtonia robusta), arterials / commercial | 19 m | 390 |
| `tree_palm_queen` | queen palm (Syagrus), entries / yards / centers | 12 m | 372 |
| `tree_eucalyptus` | eucalyptus windbreak, edges / slopes | 22.7 m | 1392 |
| `tree_jacaranda` | jacaranda (autumn: green, sparse purple), streets / yards | 9.4 m | 1368 |
| `tree_street` | Brisbane box / elm / pistache parkway tree | 8.8 m | 1208 |
| `shrub` | irrigated shrub mound with bougainvillea accents | 1.6 m | 220 |
| `grass_ornamental` | bunch grass clump | 0.9 m | 14 |
| `street_lamp` | SD cobra head, 30 ft galvanized pole, 8 ft arm toward -z, LED lens emissive | 9.4 m | 190 |

## `props_manifest.json`

A JSON list, one object per prop:

```json
{"id": "car_sedan", "kind": "vehicle", "vehicle_class": "car", "file": "vehicles/car_sedan.glb",
 "triangles": 1440, "footprint_radius_m": 2.68, "length_m": 4.9, "width_m": 2.2, "height_m": 1.44,
 "forward_axis": "-z", "up_axis": "+y", "origin": "ground, footprint center",
 "tintable_region": "material:paint", "suggested_scale_range": [0.97, 1.03],
 "materials": ["underbody", "paint", "..."], "vertex_attributes": {"_LIGHT": "...", "_TINT": "..."},
 "fleet_share": 0.3, "paint_colors": [{"name": "white", "hex": "#E9EAEA", "share": 0.25}], "notes": "..."}
```

`kind` is `vehicle | tree | shrub | lamp`; `footprint_radius_m` is the largest
horizontal distance from the origin; trees also carry `alpha_mode: "MASK"`.

## Hero campuses

1. `extract_hero_sites.py` (main venv) reads `data/raw/osm_buildings.geojson`,
   `osm_schools.geojson` (campus polygons "Del Norte High School", "Design39Campus")
   and Overture `land_use` / `segment` / `place` parquet, and picks for each hero a
   center and `footprint_radius_m` such that every footprint whose centroid falls
   inside the circle belongs to the site (radius in the middle of a centroid gap).
   That matches the pipeline rule: `build_buildings.apply_heroes` drops every
   building with its centroid within the radius and bakes the hero instead.
2. `hero_layout.py` turns the site into a planar ground partition (no overlaps):
   parking lots from real parking aisles (stall stripes every 2.75 m, 5.5 m stalls,
   7.3 m aisles - `assumptions.yaml props.parking_*`), service drives, plazas and walks,
   real pitches classified by size (tennis 23.8 x 11 m, basketball, soccer/football
   with field lines, baseball/softball with dirt infields and foul lines), the
   track with 8 lane lines, playgrounds, grass, planting beds, landscape; public
   streets are cut out so the pipeline's road ribbons show through. It also lays
   out campus trees, lot lights, parked-car stalls (occupancy from
   `props.parked_occupancy`), solar carport canopies over the longest school stall
   rows, goals, hoops, tennis nets, grandstands, play structures, and which wall
   edges face parking (shop fronts / entries).
3. `rdlib/heroes.py` extrudes the real footprints with SoCal Mediterranean-modern
   detail: stucco walls over a stone base band, punched windows with sills
   (schools) or curtain-wall bands (Design39), flat roofs behind parapets with
   rooftop HVAC, metal entry canopies / covered walkways and open lunch shelters
   (OSM `shelter` / `roof`), and for 4S Commons storefront glazing, arcades on
   columns, fabric awnings, tile mansards, tile hip roofs on pads and entry towers.

Heroes use per-face vertex colors (`COLOR_0`) on 3-4 class materials
(`hero_matte`, `hero_glass`, `hero_metal`, `hero_glow`) because the pipeline's hero
loader keeps base color, roughness, `COLOR_0` and textures (not metallic,
emissive or alpha). Trees are NOT in the hero glbs (alpha cards would turn into
opaque quads in the baked tiles): they go to `<id>_trees.json` and from there into
`placements.bin`.

`hero_overrides.json`:

```json
[{"id": "del_norte_hs", "school_id": "del_norte_hs", "name": "Del Norte High School", "type": "school",
  "lat": 33.013574, "lon": -117.122099, "rotation_deg": 0.0, "footprint_radius_m": 244.5,
  "glb": "del_norte_hs.glb", "replaces": "buildings within footprint_radius_m of lat/lon",
  "triangles": 32135, "buildings_modeled": 50, "trees": "del_norte_hs_trees.json",
  "source": "...", "verified": false}]
```

Notes: the brief's 4S Commons guess (33.0195, -117.1260) is the Target / Del Sur
Town Center; Overture places "4S Commons Town Center" (10511 4S Commons Dr) at
33.0195, -117.1129, which is what is modeled. Building heights come from OSM
`height` / `building:levels` where tagged (many Del Norte wings are tagged 3.8-4.6 m),
else 7.0 m (school) / 7.5 m (commercial). Solar carports at the schools, wall colors
and roof types are plausible guesses (`verified: false`).

## Placements (`pipeline/build_props.py`)

`placements.bin`: little-endian float32 records, 24 bytes each:

| field | meaning |
| --- | --- |
| `x`, `z` | scene meters (x east, z south) |
| `y` | terrain elevation at the prop origin (bilinear from the heightmap; heroes: the flattened campus level), sunk 5 cm |
| `rot_y` | radians, three.js `rotation.y` (prop forward -z; lamp arms and car noses point -z at 0) |
| `scale` | uniform, within the manifest `suggested_scale_range` |
| `prop_index` | index into `placements.json props[]` (integral float) |

Records are sorted by `(prop_index, cell)`. `placements.json`:

```json
{"format": "redraw-placements", "version": 1, "synthetic": false, "seed": 20261001,
 "bin": "placements.bin", "count": 304869, "fields": ["x","y","z","rot_y","scale","prop_index"], "stride": 24,
 "record": {"fields": [...], "dtype": "float32", "endianness": "little", "stride_bytes": 24, "notes": {...}},
 "props": [{"index": 0, "id": "car_crossover_ev", "kind": "vehicle", "file": "vehicles/car_crossover_ev.glb",
            "offset": 0, "first_record": 0, "count": 116, "cells": [[cell, first_record, count], ...]}],
 "cells": {"size_m": 500, "min_x": -5900, "min_z": -6200, "cols": 24, "rows": 25,
           "order": "row-major from (min_x, min_z) = north-west corner"},
 "stats": {"street_trees": 75231, "...": 0}}
```

Each prop is a contiguous section (`offset` bytes, `count` records) - one
InstancedMesh per prop - and inside a section records are grouped by 500 m cell
so a client can cull / stream by distance (there are ~300k instances; draw only
cells near the camera, e.g. within 1.5 km). Parked cars carry no color: pick from
`paint_colors` by share, seeded by the record index.

Rules (densities in `data/config/assumptions.yaml -> props.*`, all `verified: false`):

- street trees on both curbs of residential and arterial streets: offset from the
  centerline = travel lanes x `roads.lane_width_m` (+ parking lane / bike lane
  `props.curb_extra_m`) + `props.parkway_offset_m`; spacing
  `props.street_tree_spacing_m` with jitter, `props.street_tree_presence` fill;
  species from `props.species_mix.{residential,arterial}_street`; none on freeways
- yard trees, shrubs and ornamental grass around houses (`props.*_per_house`),
  palms / trees around commercial and apartment buildings
  (`props.commercial_palm_spacing_m`)
- oaks / eucalyptus and scrub on undeveloped slopes steeper than
  `props.slope_min_grade`, at least `props.open_space_building_clearance_m` from
  buildings, clumped with a noise mask
- street lamps at every intersection of 3+ arms (two opposite corners at signals)
  plus mid-block every `props.lamp_midblock_spacing_m`, arm pointing at the road
- trunks keep `props.tree_building_clearance_m` from footprints and
  `props.tree_road_clearance_m` from every travel-way edge; nothing generic is
  placed inside hero radii; hero trees / lot lights / parked cars come from
  `<id>_trees.json`
- props outside the terrain extent are dropped; `synthetic` mirrors
  `region_meta.json`

## Tests

`blender/tests/test_assets.py` (main venv, trimesh): palette and material contracts,
every manifest entry exists and parses, triangle counts match the manifest and the
budgets (cars < 1500, bus < 3000, plants < 2000), vehicles are -z forward with
headlights in front, plants use MASK atlases, hero glbs load through
`pipeline.glb.load_glb_meshes` with `COLOR_0`, placements header/binary agree
(size, sections, cells, scale ranges, extent) and the placement helpers
(`rot_y`, heightmap orientation, cell sorting). Tests that need built outputs skip
with a hint when they are missing.

## Layout of the code

| file | role |
| --- | --- |
| `build_all_assets.py` | entry point / stages / manifest / previews |
| `extract_hero_sites.py`, `hero_layout.py` | main-venv GIS step (geopandas, shapely) -> `build/hero_sites.json` |
| `rdlib/mesh.py` | `Part` (vertices, faces, per-face material key, optional UVs) and primitives |
| `rdlib/materials.py` | PBR prop materials, paint palette (pure python) |
| `rdlib/palette.py` | hero colors (pure python; append-only table) |
| `rdlib/vehicles.py`, `foliage.py`, `props.py`, `heroes.py` | builders |
| `rdlib/bl.py` | Part -> Blender mesh, custom normals, glTF export, Cycles preview helpers |
