# Redraw static viewer (deploy branch)

Built output only. Generated from the main development branch with `cd client && npm run build:viewer`
(snapshot of a running API) and served by Vercel. Do not edit by hand; rebuild and replace `site/`.

- Google Photorealistic 3D Tiles: set `GOOGLE_MAPS_API_KEY` in the Vercel project's environment
  variables (Map Tiles API enabled, key restricted by HTTP referrer to the site's domains), then redeploy.
  The build step writes `site/runtime-config.json` from it.
- Live plan runs need the Python API; the viewer shows the baseline and the example plans.
- Licenses: code Apache 2.0, assets and results CC BY 4.0, OSM-derived data ODbL (see ATTRIBUTION.md on
  the main branch).
