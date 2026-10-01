// Minimal typings for the parts of n8ao (ISC) used by src/scene/postfx.ts.
declare module 'n8ao' {
  import type { Camera, Scene, Color } from 'three';
  import { Pass } from 'three/examples/jsm/postprocessing/Pass.js';
  export class N8AOPass extends Pass {
    constructor(scene: Scene, camera: Camera, width?: number, height?: number);
    configuration: {
      aoRadius: number;
      distanceFalloff: number;
      intensity: number;
      halfRes: boolean;
      gammaCorrection: boolean;
      screenSpaceRadius: boolean;
      aoSamples: number;
      denoiseSamples: number;
      denoiseRadius: number;
      color: Color;
      depthAwareUpsampling: boolean;
      [k: string]: unknown;
    };
    setQualityMode(mode: 'Performance' | 'Low' | 'Medium' | 'High' | 'Ultra'): void;
    setSize(width: number, height: number): void;
    dispose(): void;
  }
}
