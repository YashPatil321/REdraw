// Copies the three.js Draco decoder into public/draco/ so DRACOLoader can
// fetch it at runtime (dev and build). Generated output, gitignored.
import { cpSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'node_modules/three/examples/jsm/libs/draco/gltf');
const dst = join(root, 'public/draco');
if (!existsSync(src)) {
  console.error(`copy-draco: ${src} not found; run npm install first`);
  process.exit(1);
}
mkdirSync(dst, { recursive: true });
for (const f of ['draco_decoder.js', 'draco_decoder.wasm', 'draco_wasm_wrapper.js']) {
  cpSync(join(src, f), join(dst, f));
}
console.log('copy-draco: decoder copied to public/draco/');
