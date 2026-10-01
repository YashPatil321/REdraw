/** Hash routing: `#/` explore, `#/traffic`, `#/build`, `#/report`, `#/browse`, `#/plan/{id}`. */

import type { View } from './state';

export type Route = { view: View; planId?: string };

const VIEW_PATHS: Record<string, View> = {
  '': 'explore',
  explore: 'explore',
  traffic: 'traffic',
  build: 'plan',
  report: 'report',
  browse: 'browse',
};

export function parseHash(hash: string): Route {
  const path = hash.replace(/^#/, '').replace(/^\/+/, '').replace(/\/+$/, '');
  const parts = path.split('/').filter(Boolean);
  if (parts[0] === 'plan' && parts[1]) return { view: 'plan', planId: decodeURIComponent(parts[1]) };
  return { view: VIEW_PATHS[parts[0] ?? ''] ?? 'explore' };
}

export function hashFor(route: Route): string {
  if (route.planId) return `#/plan/${encodeURIComponent(route.planId)}`;
  switch (route.view) {
    case 'explore':
      return '#/';
    case 'traffic':
      return '#/traffic';
    case 'plan':
      return '#/build';
    case 'report':
      return '#/report';
    case 'browse':
      return '#/browse';
  }
}
