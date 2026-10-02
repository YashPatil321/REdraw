// Vercel build step: write site/runtime-config.json from the project's environment variables,
// so the Google Maps key can be set in Vercel without rebuilding the app.
import { writeFileSync } from 'node:fs';
const googleMapsKey = (process.env.GOOGLE_MAPS_API_KEY || process.env.VITE_GOOGLE_MAPS_API_KEY || '').trim();
writeFileSync('site/runtime-config.json', JSON.stringify({ googleMapsKey }));
console.log(`runtime-config.json written (google key ${googleMapsKey ? 'set' : 'not set'})`);
