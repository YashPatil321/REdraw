import { describe, expect, it } from 'vitest';
import { hashFor, parseHash } from './router';

describe('hash routing', () => {
  it('parses views and plan links', () => {
    expect(parseHash('')).toEqual({ view: 'explore' });
    expect(parseHash('#/')).toEqual({ view: 'explore' });
    expect(parseHash('#/traffic')).toEqual({ view: 'traffic' });
    expect(parseHash('#/build')).toEqual({ view: 'plan' });
    expect(parseHash('#/browse/')).toEqual({ view: 'browse' });
    expect(parseHash('#/plan/abc-123')).toEqual({ view: 'plan', planId: 'abc-123' });
    expect(parseHash('#/nonsense')).toEqual({ view: 'explore' });
  });
  it('round-trips', () => {
    expect(hashFor({ view: 'plan', planId: 'a b' })).toBe('#/plan/a%20b');
    expect(parseHash(hashFor({ view: 'plan', planId: 'a b' }))).toEqual({ view: 'plan', planId: 'a b' });
    expect(hashFor({ view: 'explore' })).toBe('#/');
  });
});
