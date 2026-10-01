/**
 * GPU frame time via EXT_disjoint_timer_query_webgl2 (where available).
 * One query per frame around the whole render; results are read a few frames
 * later without stalling. `ms` is a smoothed GPU time, or null if unsupported.
 */

interface TimerExt {
  TIME_ELAPSED_EXT: number;
  GPU_DISJOINT_EXT: number;
}

export class GpuTimer {
  private gl: WebGL2RenderingContext | null;
  private ext: TimerExt | null = null;
  private pending: WebGLQuery[] = [];
  private active: WebGLQuery | null = null;
  ms: number | null = null;

  constructor(gl: WebGLRenderingContext | WebGL2RenderingContext) {
    this.gl = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext ? gl : null;
    try {
      this.ext = (this.gl?.getExtension('EXT_disjoint_timer_query_webgl2') as TimerExt | null) ?? null;
    } catch {
      this.ext = null;
    }
  }

  get supported(): boolean {
    return !!this.ext;
  }

  begin(): void {
    const gl = this.gl;
    const ext = this.ext;
    if (!gl || !ext || this.active || this.pending.length > 4) return;
    const q = gl.createQuery();
    if (!q) return;
    gl.beginQuery(ext.TIME_ELAPSED_EXT, q);
    this.active = q;
  }

  end(): void {
    const gl = this.gl;
    const ext = this.ext;
    if (!gl || !ext || !this.active) return;
    gl.endQuery(ext.TIME_ELAPSED_EXT);
    this.pending.push(this.active);
    this.active = null;
    this.poll();
  }

  private poll(): void {
    const gl = this.gl!;
    const ext = this.ext!;
    while (this.pending.length) {
      const q = this.pending[0]!;
      if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) break;
      this.pending.shift();
      const disjoint = gl.getParameter(ext.GPU_DISJOINT_EXT) as boolean;
      if (!disjoint) {
        const ns = gl.getQueryParameter(q, gl.QUERY_RESULT) as number;
        const ms = ns / 1e6;
        this.ms = this.ms === null ? ms : this.ms * 0.9 + ms * 0.1;
      }
      gl.deleteQuery(q);
    }
  }
}
