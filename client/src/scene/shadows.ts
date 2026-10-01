/**
 * Sun light + cascaded shadow maps (three/examples CSM) with soft PCF.
 *
 * Materials are patched lazily (`patchScene`, called every frame): any lit
 * material in the scene gets the CSM defines and shared uniforms, chained in
 * front of its own onBeforeCompile. With shadows off there is a single plain
 * directional light and the same patched shaders (CSM_CASCADES = 1) still work.
 */

import * as THREE from 'three';
import { CSM } from 'three/examples/jsm/csm/CSM.js';

type LitMaterial = THREE.MeshStandardMaterial | THREE.MeshLambertMaterial | THREE.MeshPhongMaterial;

interface PatchState {
  orig: ((shader: THREE.WebGLProgramParametersWithUniforms, renderer: THREE.WebGLRenderer) => void) | null;
  ver: number;
}

/** CSM fields not in the public typings. */
interface CsmInternals {
  _getExtendedBreaks(target: THREE.Vector2[]): void;
  breaks: number[];
}

export class SunShadows {
  readonly group = new THREE.Group();
  /** direction toward the sun (unit) */
  readonly sunDir = new THREE.Vector3(0, 1, 0);
  readonly color = new THREE.Color(1, 1, 1);
  intensity = 3;
  private csm: CSM | null = null;
  private plain = new THREE.DirectionalLight(0xffffff, 3);
  private cascades = 1;
  private ver = 1;
  private maxFar = 2000;
  private mapSize = 2048;
  private enabled = false;
  private lastNear = 0;
  private lastFar = 0;
  private lastMaxFar = 0;
  readonly uniforms = {
    CSM_cascades: { value: [new THREE.Vector2(0, 1)] as THREE.Vector2[] },
    cameraNear: { value: 1 },
    shadowFar: { value: 2000 },
  };

  constructor(
    private scene: THREE.Scene,
    private camera: THREE.PerspectiveCamera,
  ) {
    this.group.name = 'sun';
    this.plain.castShadow = false;
    this.group.add(this.plain, this.plain.target);
    scene.add(this.group);
  }

  get lights(): THREE.DirectionalLight[] {
    return this.csm ? this.csm.lights : [this.plain];
  }

  get shadowsOn(): boolean {
    return this.enabled;
  }

  /** (Re)build cascades for a quality preset. */
  configure(enabled: boolean, mapSize: number, cascades: number, maxFar: number): void {
    const n = enabled ? Math.max(1, cascades) : 1;
    const same = enabled === this.enabled && n === this.cascades && mapSize === this.mapSize;
    this.maxFar = maxFar;
    if (same && (this.csm || !enabled)) {
      if (this.csm) this.csm.maxFar = maxFar;
      return;
    }
    this.disposeCsm();
    this.enabled = enabled;
    this.mapSize = mapSize;
    this.cascades = n;
    if (enabled) {
      this.csm = new CSM({
        camera: this.camera,
        parent: this.group,
        cascades: n,
        maxFar,
        mode: 'practical',
        shadowMapSize: mapSize,
        lightDirection: this.sunDir.clone().negate(),
        lightIntensity: this.intensity,
        lightNear: 1,
        lightFar: 20000,
        lightMargin: 600,
        shadowBias: -0.00025,
      });
      this.csm.fade = true;
      for (const l of this.csm.lights) {
        l.shadow.normalBias = 0.35;
        l.shadow.radius = 2.5;
        l.shadow.blurSamples = 10;
      }
      this.plain.visible = false;
    } else {
      this.plain.visible = true;
    }
    this.uniforms.CSM_cascades.value = Array.from({ length: n }, () => new THREE.Vector2(0, 1));
    this.lastNear = this.lastFar = this.lastMaxFar = 0;
    this.ver++;
  }

  private disposeCsm(): void {
    if (!this.csm) return;
    for (const l of this.csm.lights) {
      l.shadow.map?.dispose();
      this.group.remove(l, l.target);
    }
    this.csm = null;
  }

  /** Shadows reach this far from the camera (m); orbit views pass a distance-dependent value. */
  setMaxFar(m: number): void {
    this.maxFar = m;
  }

  /** Per frame: light direction / color, cascade fit, material patching. */
  update(): void {
    const cam = this.camera;
    for (const l of this.lights) l.color.copy(this.color);
    if (this.csm) {
      const csm = this.csm;
      csm.lightDirection.copy(this.sunDir).negate();
      for (const l of csm.lights) l.intensity = this.intensity;
      const maxFar = Math.min(this.maxFar, cam.far);
      if (Math.abs(cam.near - this.lastNear) > 1e-3 || cam.far !== this.lastFar || Math.abs(maxFar - this.lastMaxFar) / Math.max(1, maxFar) > 0.06) {
        csm.maxFar = maxFar;
        csm.updateFrustums();
        this.lastNear = cam.near;
        this.lastFar = cam.far;
        this.lastMaxFar = maxFar;
      }
      csm.update();
      (csm as unknown as CsmInternals)._getExtendedBreaks(this.uniforms.CSM_cascades.value);
      this.uniforms.cameraNear.value = cam.near;
      this.uniforms.shadowFar.value = Math.min(cam.far, csm.maxFar);
    } else {
      this.plain.intensity = this.intensity;
      this.plain.position.copy(this.sunDir).multiplyScalar(1000);
      this.plain.target.position.set(0, 0, 0);
      this.uniforms.cameraNear.value = cam.near;
      this.uniforms.shadowFar.value = cam.far;
    }
    this.patchScene();
  }

  /** Make every lit material CSM-aware (idempotent; cheap check per material). */
  patchScene(root: THREE.Object3D = this.scene): void {
    root.traverse((o) => {
      const m = (o as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
      if (!m) return;
      if (Array.isArray(m)) m.forEach((x) => this.patch(x));
      else this.patch(m);
    });
  }

  patch(mat: THREE.Material): void {
    const lit = mat as LitMaterial;
    if (!(lit.isMeshStandardMaterial || lit.isMeshLambertMaterial || lit.isMeshPhongMaterial)) return;
    const st = mat.userData.rdCsm as PatchState | undefined;
    if (st && st.ver === this.ver) return;
    const orig = st ? st.orig : (mat.onBeforeCompile as PatchState['orig']);
    mat.userData.rdCsm = { orig, ver: this.ver } satisfies PatchState;
    mat.defines = mat.defines ?? {};
    mat.defines['USE_CSM'] = 1;
    mat.defines['CSM_CASCADES'] = this.cascades;
    if (this.csm) mat.defines['CSM_FADE'] = '';
    else delete mat.defines['CSM_FADE'];
    const uniforms = this.uniforms;
    mat.onBeforeCompile = function (this: THREE.Material, shader, renderer) {
      orig?.call(this, shader, renderer);
      shader.uniforms['CSM_cascades'] = uniforms.CSM_cascades;
      shader.uniforms['cameraNear'] = uniforms.cameraNear;
      shader.uniforms['shadowFar'] = uniforms.shadowFar;
    };
    mat.needsUpdate = true;
  }

  dispose(): void {
    this.disposeCsm();
    this.scene.remove(this.group);
  }
}
