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
