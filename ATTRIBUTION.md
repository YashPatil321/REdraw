# Data attribution

Redraw's real-mode world is built from open data. The exact files, URLs, releases and
per-record licenses used by a build are recorded in `data/processed/region_meta.json`
(`sources`, `attribution`), read from `data/raw/aws_sources.json` and the
`data/raw/*.source.json` sidecars that the fetchers write. Show the short attribution lines
(`region_meta.json -> attribution`) wherever the map is displayed.

The `--synthetic` mode uses no third-party data (procedural, CC BY 4.0 Redraw project).

## Map data: Overture Maps Foundation (release 2026-09-23.1)

Fetched from the Overture open-data mirror on AWS (`s3://overturemaps-us-west-2/release/...`)
by `pipeline/fetch_aws.py`. See <https://docs.overturemaps.org/attribution/>.

| Theme / type | Used for | License |
| --- | --- | --- |
| transportation / segment, connector | drive, walk and bike networks (`roads_drive.graphml`, `network_*.parquet`, `roads.geojson`, road meshes) | ODbL 1.0, derived from OpenStreetMap (a few TomTom-sourced segments, also ODbL) |
| buildings / building | building footprints, heights, `roof:shape`, colours (`buildings.geojson`, building meshes), and the footprint population estimate | ODbL 1.0 (theme). Sources: OpenStreetMap (ODbL), **Microsoft ML Buildings via Overture** (ODbL), Esri Community Maps (CC BY 4.0 with OpenStreetMap waivers); the per-record mix is tallied in `aws_sources.json` |
| base / land_use, water | school campus polygons, land-use typing of generic buildings | ODbL 1.0, derived from OpenStreetMap |
| places / place | extra school names and locations | Mixed: CDLA-Permissive-2.0 (Overture, Meta, Microsoft, BrightQuery and others), Apache 2.0 (Foursquare records), CC0 1.0 (AllThePlaces records); tallied per record in `aws_sources.json` |

Required attribution: **(c) OpenStreetMap contributors, Overture Maps Foundation**.
OpenStreetMap data is available under the Open Database License,
<https://www.openstreetmap.org/copyright>. Derived databases (our processed road and building
files) are also ODbL. Produced works such as the rendered map are not restricted, but they must
carry the attribution above.

When the raw data comes from the primary fetchers instead (`pipeline/fetch_osm.py`, Overpass
API), the map data is OpenStreetMap under ODbL 1.0 with the same attribution.

## Elevation: USGS 3D Elevation Program (3DEP)

USGS 3DEP 1 m lidar DEM, project `CA_SanDiegoCo_D24`
(`https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/CA_SanDiegoCo_D24/`),
read at 2 m. Gaps can be filled from the 3DEP 1/3 arc-second tiles. **Public domain**
(US Government work). Credit: "USGS 3D Elevation Program".

## Imagery: Copernicus Sentinel-2

The terrain albedo texture comes from a Sentinel-2 L2A true-colour scene (bands B04/B03/B02, 10 m,
scene id and date in `naip_mosaic.source.json`), read from the `sentinel-cogs` AWS
open-data bucket. The data is free, full and open under the Copernicus Sentinel Data Terms.
Required attribution: **"Contains modified Copernicus Sentinel data 2026"**.

The raw file is named `naip_mosaic.tif` for pipeline compatibility. USDA NAIP (public domain)
is the spec's primary imagery source, but it is requester-pays on AWS and was not used for this
build.

## Population

Census ACS 5-year, TIGER/Line and LEHD LODES (US Census Bureau, public domain) are the spec's
sources for the synthetic population, but they were unreachable for this build. With
`build_all.py --population-source footprints`, households are **estimated** from the
Overture/OSM residential building footprints above plus `data/config/assumptions.yaml`.
They are not census data, and `region_meta.json` records
`population_source: "footprint_estimate"`.

## Optional client layer: Google Photorealistic 3D Tiles

The web client can optionally display Google Photorealistic 3D Tiles (Google Maps Platform
Map Tiles API) as a visual layer. It is used under the Google Maps Platform Terms of Service with
the user's own API key, and it must show Google's attribution and logo while visible. Redraw does
**not** download, cache, redistribute or derive data from these tiles. No pipeline output and
no simulation input uses them. The public build relies on the open data above only.
