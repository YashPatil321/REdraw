# Open questions and unverified facts

Every guessed real-world fact is marked `verified: false` in `data/config/` and
listed here (spec rule 14.7). The report card footer also lists every
unverified input used in a run. Check items off as a person confirms them.

## Team questions (spec 15)
- [ ] Real bell times and drop off layouts for Del Norte, Oak Valley, Design39, and nearby elementary schools
- [ ] Attendance boundary data from Poway Unified, if published
- [ ] Which open weight model runs best on the GPU server for 500 residents
- [ ] Real typical travel times on 6 to 10 routes for calibration (`data/config/calibration_targets.yaml`, all `observed_minutes` are null today)
- [ ] Whether SanGIS building outlines and parcels can be downloaded in bulk for the bbox

## Region (`data/config/region.yaml`)
- [ ] Bbox still the spec starting box; verify against OSM place boundaries and expand to include I 15, SR 56, Black Mountain Open Space, San Dieguito River
- [ ] Arterial label names vs OSM `name` tags (Camino Del Norte, Camino Del Sur, Dove Canyon Road, 4S Ranch Parkway, Bernardo Center Drive, Carmel Valley Road, Black Mountain Road, I 15 / Escondido Freeway, SR 56 / Ted Williams Parkway)
- [ ] Exit node locations: I 15 north, I 15 south, SR 56 west, Camino Del Norte east

## Schools (`data/config/schools.yaml`)
All of these were written from memory and are placeholders:
- [ ] Coordinates of every school and drop-off entrance (Del Norte HS, Oak Valley MS, Design39, Stone Ranch ES, Monterey Ridge ES, Del Sur ES, Willow Grove ES)
- [ ] Bell start times (08:30 Del Norte, 08:00 Oak Valley, 08:15 Design39, 08:40 elementaries)
- [ ] Curb spots and average unload seconds per entrance
- [ ] Which other schools in the bbox are missing (the real pipeline adds OSM `amenity=school` with defaults)

## Modeling assumptions (`data/config/assumptions.yaml`)
Every leaf with `verified: false`. Highest impact first:
- [ ] Worker departure time distribution (mean 07:30, sd 35 min)
- [ ] Mode choice constants and coefficients (student and worker ASCs, time and cost betas, habit)
- [ ] HS self-drive share, work from home share, external through-trip volume
- [ ] Road class defaults (lanes, speeds, capacities) where OSM tags are missing
- [ ] Signal delay parameters, drop-off spillback simplifications
- [ ] All tool costs and tool effect sizes (shuttle, carpool, curb spots, turn lanes, bike and walk routes)
- [ ] Resident approval weights

## Data access
- [ ] The cloud dev sandbox used to build v1 could not reach OSM, USGS, Census or Planetary Computer, so the real-data pipeline was written but not run end to end there. Run `python pipeline/build_all.py` on a machine with internet and record results here.

## Found while building on real data (2026-10-01)
- [ ] Census ACS/LODES were unreachable from the build sandbox; households come from a labeled footprint estimate (37,418 households in the expanded bbox). Rerun with `--population-source acs` where census.gov is reachable.
- [ ] 532 traffic signals are inferred (Overture has none); many tertiary junctions are probably all-way stops
- [ ] Lanes and most speed limits are class defaults (Overture lacks them)
- [ ] Every school drop-off entrance is inferred (side of campus facing the highest-class road). Del Norte lands on Camino San Bernardo; the real main drop-off is probably Nighthawk Lane
- [ ] 18 schools added from OSM use default bell 08:15, 6 curb spots, 40 s unload
- [ ] Design39 is a lottery school; nearest-school assignment overstates its local enrollment
- [ ] Gated communities (Santaluz, The Crosby) have no public roads in Overture; ~870 households snap > 500 m to the network
- [ ] Del Norte's max curb line (~100 cars) depends on placeholder curb data (10 spots, 45 s)
- [ ] Imagery is Sentinel-2 (2.5 m resampled); NAIP 0.6 m was not reachable
- [ ] Hero campus models (Del Norte, Design39, 4S Commons) are approximations from OSM footprints

## Props and hero campuses (blender/, pipeline/build_props.py)
- [ ] Hero campus details are guesses on top of real OSM/Overture footprints: wall colors, window patterns, solar carports over the longest school stall rows, entry canopies, tile roofs / arcades at 4S Commons. Check against photos.
- [ ] Del Norte building heights: many OSM wings are tagged 3.8-4.6 m (single story); untagged school buildings default to 7.0 m. Several Del Norte buildings may be two-story.
- [ ] 4S Commons was modeled at Overture's "4S Commons Town Center" (33.0195, -117.1129), not the brief's 33.0195, -117.1260 (that is the Target / Del Sur Town Center).
- [ ] Street tree spacing, yard tree / shrub counts, slope vegetation density, lamp spacing and species mixes (`assumptions.yaml props.*`) are eyeballed from aerial imagery, not surveyed; City of San Diego street light and street tree inventories would replace them.
- [ ] Vehicle color and type shares (`props.vehicle_paint_shares`, `props.vehicle_type_shares`) are national color reports plus a guess for the local fleet mix (DMV registrations by ZIP 92127 would pin it down).
- [ ] Lidar tree species are mapped from a crude geometric palm / broadleaf / conifer guess plus context (street / yard / commercial / open space) and height rules (`props.lidar_trees.*`: broadleaf over 15 m -> mostly eucalyptus, palms over 12 m -> mostly Mexican fan palm). A street tree inventory or NAIP color-infrared classification would replace the guess. Instances use one uniform scale (height-weighted), so crown width only approximately matches the measured crown.
- [ ] Trees planted after the 2014 lidar survey are filled in by rules (gap cells: developed 80 m cells without any lidar tree, houses with `parcel_year_built` after 2014) as young trees (`props.lidar_trees.young_scale`); their positions are not real.
- [ ] Front-yard planting mix (`props.species_mix.yard_shrub`: shrubs, agave / succulents, bougainvillea, clipped hedges) is a guess from street imagery.

## Lidar features (pipeline/fetch_lidar.py, pipeline/lidar_features.py)
- [ ] Point cloud is USGS CA_SanDiegoQL2_2014 (flown 2014, ~4.6 pts/m2), not the denser CA_SanDiego_2015_C17_1 named in the brief: the 2015 EPT has no nodes over the bbox, and the 2024 CA_SanDiegoCo_D24 LAZ tiles are only on rockyweb.usgs.gov (blocked from the sandbox). Buildings / trees newer than 2014 are absent (`lidar_status` = absent*); a 2024 point cloud would fix it.
- [ ] Lidar -> footprint registration shift (~1 m W, ~1 m N, `data/raw/lidar/work/registration.json`) is a pixel-level (0.5 m) IoU fit on 5 residential samples; whether the offset is in the footprints (imagery tracing) or the lidar is not known.
- [ ] Roof types (flat / gable / hip / complex / shed) come from RANSAC planes and an eave-outline test (`lidar.hip_min_eave_frac`), calibrated by eye on a few 4S Ranch blocks; no ground-truth roof-type sample was checked.
- [ ] Tree `species_guess` thresholds (`lidar.palm_*`, `lidar.conifer_*`) are geometric guesses (crown size / shape / isolation) without spectral data; no field or street-tree-inventory check.

## HD world build (pipeline/build_streets.py, build_landcover.py, build_terrain.py)
- [ ] `deldios_north` exit: Del Dios Highway only clips the NW bbox corner and has no junction with the network inside the bbox (nor in the raw data extent), so the exit is routed `via` `sandieguito_west` (exits.json). Check that trips toward Del Dios / Rancho Santa Fe Lakes really leave via San Dieguito Road west.
- [ ] Private (gated) streets are now rendered from Overture but stay out of the routable network; residents of Santaluz / The Crosby still snap to the nearest public node. Decide whether `access=private` residential streets should join the drive graph for their own residents.
- [ ] Paseo / park path width (`hd_world.paseo_width_m` 2.4 m) and trail width (`hd_world.trail_width_m` 1.8 m) are guesses from imagery; trail tread width varies from single track to fire roads.
- [ ] Land-cover splat thresholds: imagery color classes (`build_landcover.classify_imagery`), Overture land_cover weights (`LANDCOVER_PRIOR`), lidar nDSM shrub / canopy cutoffs (`hd_world.ndsm_*`) and the "bright grey water = covered reservoir" rule were tuned by eye on a few tiles of 10 m Sentinel-2 imagery. NAIP 0.6 m would make the color classes much sharper.
- [ ] House driveways are inferred (garage on the wall nearest the street, straight to the curb), not mapped; alley-loaded and side-entry garages are only right when the nearest street is the alley.
- [ ] Medians exist only where OSM/Overture maps the arterial as two one-way carriageways; raised medians on single-line two-way roads are not modeled.
