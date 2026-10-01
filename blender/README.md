# blender/ - headless Blender asset pipeline

Procedural, seed-fixed props and hero campus models for Redraw, built with the
`bpy` wheel (no Blender UI). Everything here regenerates from code + real open
data; only the scripts, tests and `previews/*` are committed.

## Setup and commands

```bash
python3.11 -m venv .venv-blender
.venv-blender/bin/pip install bpy==5.0.1 shapely          # bpy 4.5.x also works

.venv-blender/bin/python blender/build_all_assets.py          # everything (about 5 min on 4 CPUs)
.venv-blender/bin/python blender/build_all_assets.py --only vehicles trees street   # props only (trees bake LOD1 impostors: ~1-2 min each)
.venv-blender/bin/python blender/build_all_assets.py --only materials   # material atlases, ~30 s
.venv-blender/bin/python blender/build_all_assets.py --only previews --previews street_scene vegetation
.venv-blender/bin/python blender/build_all_assets.py --no-previews --samples 32

.venv/bin/python blender/extract_hero_sites.py   # hero footprints/layout (main venv: geopandas); run automatically if missing
.venv/bin/python pipeline/build_props.py         # instance placements (trees, shrubs, lamps, parked cars)
.venv/bin/pytest blender/tests                   # glbs parse, manifest, hero overrides, placements format
```

Stages of `build_all_assets.py`: `materials vehicles trees street heroes manifest previews`;
`--previews` picks among `sheets street_scene vehicles vegetation heroes`.
Previews render with Cycles on the CPU (EEVEE needs a GPU context headless bpy
does not have) and are re-encoded to stay under 400 KB.

## Outputs

| Path | What |
| --- | --- |
| `client/public/assets/props/vehicles/*.glb` | 7 vehicles |
| `client/public/assets/props/vegetation/*.glb` | 14 plants (alpha-card foliage) + `tree_*_lod1.glb` impostors |
| `client/public/assets/props/street/street_lamp.glb` | SD cobra-head street light |
| `client/public/assets/props/props_manifest.json` | prop table (below) |
| `client/public/assets/props/placements.json` + `.bin` | instances from `pipeline/build_props.py` (below) |
| `pipeline/hero_overrides/{del_norte_hs,design39,4s_commons}.glb` | hero campuses |
| `pipeline/hero_overrides/hero_overrides.json` | hero placement table the pipeline reads |
| `pipeline/hero_overrides/<id>_trees.json` | campus trees, lot lights, parked-car stalls (merged into placements) |
| `blender/build/` | generated textures, `hero_sites.json` (gitignored) |
| `blender/previews/*.{png,jpg}` | committed beauty renders (< 400 KB each) |

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

## Material atlases (`client/public/assets/materials/`)

Tileable PBR atlases for buildings, roofs and ground, generated procedurally by
`rdlib/matgen.py` + `rdlib/texgen.py` (periodic FFT noise, periodic Worley cells, analytic
antialiased shapes, heights in meters -> normal maps and horizon AO with real strength) and
checked in Cycles renders (`previews/materials_*.jpg`, `previews/street_scene.jpg`).
`--only materials` rebuilds them in about 30 s.

| file | content |
| --- | --- |
| `<atlas>_albedo.jpg` | sRGB base color |
| `<atlas>_normal.jpg` | tangent-space normal, OpenGL / glTF convention (+Y up), linear |
| `<atlas>_orm.jpg` | R ambient occlusion, G roughness, B metalness (glTF packing), linear |
| `<atlas>_*_1k.jpg` | half-resolution copies (low preset / far LOD) |
| `facade_openings_mask.png` | R: 1 wall (tint with the wall color), 0.5 paintable near-white (garage / service doors), 0 fixed; G: glass that may glow at night |
| `ground_markings.png` | RGBA lane-marking decals, alpha = worn paint coverage |
| `materials_manifest.json` | everything below, machine readable |

Atlases (`atlases.<name>`): `facade_walls` 2048x1024 (6 neutral stucco finishes, ledgestone,
curtain glazing), `facade_openings` 2048x2048 (vinyl slider / single-hung pair / picture /
small obscure / arched windows, stained entry door with sidelite, patio slider, 2-car and 3-car
raised-panel garage spans, storefront bay, storefront sign band, school ribbon window, school
double door), `roofs` 2048x2048 (4 Spanish S-tile blends, mission barrel, 4 flat concrete tiles,
TPO, TPO with HVAC grime, gravel ballast, modified bitumen, residential PV, standing seam,
concrete deck), `ground` 2048x2048 (fresh / worn / parking asphalt, sidewalk, driveway, curb &
gutter profile, pavers, plaza concrete, lawn, stressed lawn, chaparral, coastal sage, DG, bare dirt,
mulch, pool water).

**Cells.** Every atlas is a grid of 512 px cells; each cell = a 480 px tileable inner square plus a
16 px wrapped gutter. Each `cells[i]` entry has `name`, `index`, `world_size_m` (meters covered by
one inner square), `texels_per_m`, `span_cells`, `tintable`, `mean_albedo_linear`, and per span
cell `px` / `inner_px` (top-left origin) and `uv_inner` = `[u0, v0, u1, v1]` with **v measured from
the bottom** (three.js `TextureLoader` flipY = true, OpenGL, Blender). Shader lookup:
`atlas_uv = mix(uv_inner.xy, uv_inner.zw, fract(mesh_uv))` (use `textureGrad` with the
derivatives of `mesh_uv` to avoid a 1-pixel seam at the fract wrap). `facade_openings` cells
also carry `opening_rects_m` (`glass` / `door` rects in meters from the cell's bottom-left).

**Vertex attributes / material ids** (shared with the pipeline and the buildings agent):

| `_MAT` | meaning | atlas | `_VARIANT` = index into `materials[_MAT].variants` |
| --- | --- | --- | --- |
| 0 | stucco wall | facade_walls | stucco_smooth, stucco_sand, stucco_lace, stucco_catface, stucco_weathered, stucco_scored, stone_veneer |
| 1 | tile roof | roofs | s_tile_terracotta, s_tile_blend, s_tile_brown, s_tile_aged, barrel_mission, flat_tile_brown, flat_tile_grey, flat_tile_charcoal, flat_tile_sandstone, solar_panel |
| 2 | flat roof | roofs | flat_tpo, flat_tpo_grime, flat_gravel, flat_modbit, concrete_deck, standing_seam |
| 3 | glass | facade_walls | glass_curtain |
| 4 | trim | facade_walls | stucco_smooth, stone_veneer |
| 5 | garage door | facade_openings | garage_2car, garage_3car |
| 6 | ground (hero campuses; extension) | ground | `_VARIANT` = ground cell index |
| 7 | vertex color only (extension) | - | - |

`COLOR_0` is the tint: for tintable cells `base = COLOR_0_linear * texel_linear / mean_albedo_linear`
(on `facade_openings`, only where mask R > 0.75). Roof / ground cells carry real colors.

**UV convention** (`TEXCOORD_0`, `rdlib/atlas.py:face_uv` implements it for Blender meshes):
walls / glass / trim / garage: `u = meters along the wall / 3`, `v = meters above the wall base / 3`
(one UV unit = one 3 m x 3 m floor-bay = one cell; `bay = floor(u)`, `floor = floor(v)`); pitched
roofs: `u = meters along the eave / 4`, `v = meters up the slope / 4`; flat roofs: `u = x / 4`,
`v = -z / 4`; ground: planar by the cell's `world_size_m`. Garage spans cover 2 / 3 consecutive
bays: span cell k holds local `x / 3` in `[k, k + 1)`. `bay_grammar` in the manifest suggests
which opening cells go on which (bay, floor) per building type (`rdlib/houses.py` is a working
reference that builds tract houses this way for the street preview). Blender helpers for
previews: `bl.atlas_material(name, manifest, atlas, cell, mat_dir, span_k, tint_hex | tint_attr)`.

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
generated 1024 x 1024 palette PNG atlas (four 448 x 512 leaf cells drawn at 4x
supersampling with per-leaf color jitter, folded blades, midribs, twigs and sun bias;
transparent texels bled with leaf color so mipmaps have no dark fringe; a 128 px bark
strip on the right mapped onto the trunk tubes). Leaf cards are scattered in ellipsoid
clumps (biased to the clump shells); their normals are bent outward from the canopy
center so the card cloud shades like a rounded crown, and `COLOR_0` carries a baked
crown ambient occlusion (inner / underside cards darker) to multiply into the base color.

**LODs (trees).** `lods[0]` = the card tree (`max_distance_m` 140), `lods[1]` =
`<id>_lod1.glb`: a Cycles-baked impostor (orthographic side + top views under a white
sky, i.e. albedo x AO, 512 px per view) on 3 crossed vertical quads + 1 horizontal crown
quad; the vertical quads are doubled back to back and split into a few cells whose normals
follow an ellipsoidal crown, the top quad faces up only and sits high in the crown at 80 %
width (80 triangles), so it lights like LOD0 from any sun / view direction. Cull backfaces,
let impostors cast shadows but not receive them (crossed silhouettes would self-shadow).

| id | species / use | height | tris (LOD0 / LOD1) |
| --- | --- | --- | --- |
| `tree_oak` | coast live oak (Quercus agrifolia), canyons / slopes / big yards | 9.5 m | 2558 / 80 |
| `tree_palm_fan` | Mexican fan palm (Washingtonia robusta), arterials / commercial | 19.5 m | 450 / 80 |
| `tree_palm_queen` | queen palm (Syagrus romanzoffiana), entries / yards / centers | 11.8 m | 924 / 80 |
| `tree_pine_canary` | Canary Island pine (Pinus canariensis), parks / slopes / school edges | 20.5 m | 3384 / 80 |
| `tree_eucalyptus` | eucalyptus windbreak, edges / slopes / canyon rims | 22.6 m | 2152 / 80 |
| `tree_jacaranda` | jacaranda (autumn: green, sparse late bloom), streets / yards | 9.2 m | 2028 / 80 |
| `tree_ficus` | Indian laurel fig (Ficus microcarpa), commercial / parking lots | 8.7 m | 2858 / 80 |
| `tree_street` | Brisbane box (Lophostemon) parkway tree, stands in for elm / pistache | 8.8 m | ~2100 / 80 |
| `shrub` | irrigated shrub mound | 1.5 m | 340 |
| `shrub_bougainvillea` | bougainvillea mound, magenta bracts (walls, entries, slopes) | 2.6 m | 824 |
| `succulent_agave` | Agave americana rosettes + echeveria (xeriscape) | 0.9 m | 1900 |
| `hedge` | clipped hedge SEGMENT, 2.0 m along local x: tile end to end | 1.6 m | 350 |
| `grass_ornamental` | bunch grass clump | 0.9 m | 14 |
| `grass_tuft` | lawn tuft, kind `groundcover` (client scatters it procedurally near the camera; not placed) | 0.2 m | 6 |
| `street_lamp` | SD cobra head, 30 ft galvanized pole, 8 ft arm toward -z, LED lens emissive | 9.4 m | 190 |

Palms stay well under 2k triangles on purpose: their crowns are a few dozen big
fronds, extra cards add nothing visible. Tree entries also carry `crown_radius_m`
(95th percentile of the card radius) so placements can scale a model to a measured crown.

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

`kind` is `vehicle | tree | shrub | lamp | groundcover`; `footprint_radius_m` is the largest
horizontal distance from the origin; plants carry `alpha_mode: "MASK"`, trees also
`crown_radius_m`, `lods` and `lod1_file`.

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
emissive or alpha). On top of that every vertex carries the material-atlas contract
(the loader keeps `_`-prefixed attributes): `_MAT` / `_VARIANT` from each palette key
(`rdlib/atlas.py KEY_MAT`: stucco / stone walls, trim, glass, TPO / gravel / standing-seam
flat roofs, tile roofs, solar panels, and `_MAT` 6 ground cells for asphalt, concrete,
pavers, lawn, sage scrub, mulch, DG; sports surfaces and paint lines stay `_MAT` 7
vertex color) and atlas `TEXCOORD_0` (facade u = meters along the wall / 3, roofs / 4,
ground planar by the cell's `world_size_m`). Vertices are split between faces of
different (`_MAT`, `_VARIANT`), so the point attributes are exact; `COLOR_0` stays the
tint (walls / trim) and the color-only fallback. The hero previews are rendered from
these attributes with the real atlases. Trees are NOT in the hero glbs (alpha cards would turn into
opaque quads in the baked tiles): they go to `<id>_trees.json` and from there into
`placements.bin`. `<id>_trees.json` also lists `keepout` polygons (local x east / y north:
modeled buildings, fields, courts, track, rubber play areas, parking) where
`build_props.py` must not put real lidar trees.

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

**Real trees.** When `data/raw/lidar/trees.parquet` exists (pipeline/lidar_features.py:
individual trees from the 2014 USGS 3DEP QL2 point cloud: easting / northing, height,
crown radius, palm / broadleaf / conifer guess, ground before / after), every tree inside
the region bbox becomes an instance at its real position (x, z recomputed from UTM with the
current scene origin):

- trees on ground regraded since the survey (`|ground_y - ground_lidar_m| > lidar.ground_change_m`)
  are stale and dropped; tops inside / within `props.lidar_trees.building_clearance_m` of a
  footprint are dropped (roof artifacts); tops over a travel lane are moved back to the curb
  (+ `road_clearance_m`); inside hero radii the hero `keepout` polygons apply instead
- species: `palm` -> fan / queen palm by height (`palm_fan_min_height_m`), `conifer` -> Canary
  pine, `broadleaf` -> a context mix (`props.lidar_trees.mix.broadleaf_{street,yard,commercial,open}`
  from the distance to roads / buildings and the nearest building type); broadleaf taller than
  `eucalyptus_min_height_m` -> mostly eucalyptus; palms out in open space are treated as
  broadleaf; within a class each species is weighted by how well its model crown/height ratio
  matches the tree (`aspect_sigma`), seeded
- scale = `(h / h_model)^0.7 * (r / r_model)^0.3` (`height_weight`, clamped to `scale_clamp`),
  so these records may exceed `suggested_scale_range` (header `scale_note`). Watershed crowns
  come out narrower than drip lines, so measured radii are first rescaled per class so the class
  median crown/height ratio matches the models (`stats.lidar_crown_scale_*`)
- the rule-based tree rules below then only fill GAPS: houses built after
  `props.lidar_trees.survey_year` (`parcel_year_built`), developed `gap_cell_m` cells without
  any lidar tree, and anything outside the lidar AOI; never within a few meters of a real tree;
  gap trees are young (`young_scale`). Hero layout trees survive only where no real tree stands
  within 20 m. Slope trees come only from lidar inside the AOI. Shrubs, hedges, agave,
  bougainvillea, grass, lamps and parked cars are always rule-based.

The header records `trees_source` (`lidar+gaps` or `procedural`), `region_bbox` and the
lidar counts in `stats`. The run refuses to start while `region_meta.json` describes another
bbox than `region.yaml` (`--allow-region-mismatch` for dev only), and every record is clipped
to the region bbox polygon.

Rules (densities in `data/config/assumptions.yaml -> props.*`, all `verified: false`):

- street trees on both curbs of residential and arterial streets: offset from the
  centerline = travel lanes x `roads.lane_width_m` (+ parking lane / bike lane
  `props.curb_extra_m`) + `props.parkway_offset_m`; spacing
  `props.street_tree_spacing_m` with jitter, `props.street_tree_presence` fill;
  species from `props.species_mix.{residential,arterial}_street`; none on freeways
- yard trees, shrubs and ornamental grass around houses (`props.*_per_house`; shrub
  species from `props.species_mix.yard_shrub`: shrub / agave / bougainvillea / clipped
  hedge runs of `props.hedge_run_segments` 2 m segments along a house wall), palms /
  trees around commercial and apartment buildings (`props.commercial_palm_spacing_m`)
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
budgets (cars < 1500, bus < 3000, trees < 4000 with broadleaf / conifer LOD0 2-4k,
LOD1 <= 128, shrubs < 2000), every species present, tree LOD tables, vehicles are -z
forward with headlights in front, plants use MASK atlases, hero glbs load through
`pipeline.glb.load_glb_meshes` with `COLOR_0`, `_MAT` / `_VARIANT` (in range) and UVs,
hero keep-out polygons, materials manifest names / indices stable (append-only), atlas
textures sized as declared, placements header/binary agree (size, sections, cells, scale
ranges, extent, inside the region bbox), the lidar placement rules on a synthetic mini
world (building / road / region filters, species, scale, gaps), the lidar loader (frame,
stale trees) and the placement helpers (`rot_y`, heightmap orientation, cell sorting). Tests that need built outputs skip
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
| `rdlib/bl.py` | Part -> Blender mesh (incl. hero `_MAT` / `_VARIANT` / atlas UVs), custom normals, glTF export, Cycles preview helpers |
| `rdlib/impostor.py` | LOD1 tree impostors (Cycles bake + crossed quads) |
| `rdlib/matgen.py`, `texgen.py`, `atlas.py`, `houses.py`, `previews.py` | material atlases, atlas conventions, tract houses and preview scenes |
