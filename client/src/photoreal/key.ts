/**
 * Google Maps Platform API key for Photorealistic 3D Tiles. Never committed:
 * it comes from `VITE_GOOGLE_MAPS_API_KEY` at build time, `runtime-config.json` at deploy time, or `?gkey=<key>`
 * at runtime (remembered in localStorage so the link can be shared without it).
 * `?gkey=` with an empty value (or `?gkey=clear`) forgets a stored key.
 */

export const GKEY_STORAGE = 'redraw-google-maps-key';

/** Key from an optional `runtime-config.json` next to index.html (written at deploy time, e.g.
 * from a Vercel environment variable), so a static build needs no rebuild to get a key. */
let runtimeKey: string | undefined;

export async function loadRuntimeConfig(): Promise<void> {
  if (typeof fetch === 'undefined' || typeof document === 'undefined') return;
  try {
    const r = await fetch(new URL('runtime-config.json', document.baseURI), { cache: 'no-store' });
    if (!r.ok) return;
    const j = (await r.json()) as { googleMapsKey?: unknown };
    if (typeof j.googleMapsKey === 'string' && j.googleMapsKey.trim()) runtimeKey = j.googleMapsKey.trim();
  } catch {
    /* no runtime config (dev server answers with index.html): ignore */
  }
}

export interface KeySources {
  search: string;
  envKey: string | undefined;
  storage: Pick<Storage, 'getItem' | 'setItem' | 'removeItem'> | null;
}

export interface ResolvedKey {
  key: string | null;
  source: 'url' | 'storage' | 'env' | null;
}

export function resolveGoogleKey(src: KeySources): ResolvedKey {
  const params = new URLSearchParams(src.search);
  if (params.has('gkey')) {
    const v = (params.get('gkey') ?? '').trim();
    if (!v || v === 'clear') {
      try {
        src.storage?.removeItem(GKEY_STORAGE);
      } catch {
        /* storage unavailable */
      }
    } else {
      try {
        src.storage?.setItem(GKEY_STORAGE, v);
      } catch {
        /* storage unavailable: key still works for this page load */
      }
      return { key: v, source: 'url' };
    }
  }
  try {
    const stored = src.storage?.getItem(GKEY_STORAGE);
    if (stored) return { key: stored, source: 'storage' };
  } catch {
    /* ignore */
  }
  const env = (src.envKey ?? '').trim();
  if (env) return { key: env, source: 'env' };
  return { key: null, source: null };
}

/** Remove `gkey` from the address bar so the key is not shared by copy-paste. */
export function stripKeyFromUrl(): void {
  if (typeof location === 'undefined' || typeof history === 'undefined') return;
  const u = new URL(location.href);
  if (!u.searchParams.has('gkey')) return;
  u.searchParams.delete('gkey');
  history.replaceState(history.state, '', u.toString());
}

export function browserGoogleKey(): ResolvedKey {
  let storage: Storage | null = null;
  try {
    storage = typeof localStorage !== 'undefined' ? localStorage : null;
  } catch {
    storage = null;
  }
  const r = resolveGoogleKey({
    search: typeof location !== 'undefined' ? location.search : '',
    envKey: (import.meta.env.VITE_GOOGLE_MAPS_API_KEY as string | undefined) || runtimeKey,
    storage,
  });
  stripKeyFromUrl();
  return r;
}

export const NO_KEY_HELP =
  'Photoreal mode needs a Google Maps Platform API key with the Map Tiles API enabled. ' +
  'Open the app once with ?gkey=YOUR_KEY (stored in this browser only), or set VITE_GOOGLE_MAPS_API_KEY in client/.env.local and restart Vite.';
