/**
 * Street-level first-person camera: WASD / arrow keys to move, mouse to look
 * (pointer lock after a click, or drag-to-look when pointer lock is not
 * available), Shift to run, Q/E or PageUp/PageDown to change eye height a bit.
 * The eye stays 1.7 m above the ground, clamped every frame by a height
 * callback (photoreal tiles raycast, or our terrain).
 */

import * as THREE from 'three';

export const EYE_HEIGHT_M = 1.7;
export const WALK_SPEED = 1.6; // m/s, brisk walk
export const RUN_SPEED = 9; // m/s, "run" (more of a jog-bike; the region is 9 km wide)

/** Ground height at x, z; `fromY` (when given) casts down from there, so canopies / bridges overhead are ignored. */
export type GroundFn = (x: number, z: number, fromY?: number) => number | null;

export interface WalkPose {
  x: number;
  z: number;
  /** compass heading the walker faces, radians, 0 = north (-z), pi/2 = east (+x) */
  heading: number;
  /** look pitch, radians (+ up) */
  pitch?: number;
}

/** Pure step function (unit tested): new x/z after moving with keys for dt seconds. */
export function walkStep(
  pose: { x: number; z: number; heading: number },
  keys: { fwd: number; right: number; run: boolean },
  dt: number,
): { x: number; z: number } {
  const len = Math.hypot(keys.fwd, keys.right);
  if (len === 0) return { x: pose.x, z: pose.z };
  const speed = keys.run ? RUN_SPEED : WALK_SPEED;
  const f = keys.fwd / len;
  const r = keys.right / len;
  // forward vector for heading h: (sin h, -cos h) in x/z; right: (cos h, sin h)
  const fx = Math.sin(pose.heading);
  const fz = -Math.cos(pose.heading);
  const rx = Math.cos(pose.heading);
  const rz = Math.sin(pose.heading);
  return {
    x: pose.x + (fx * f + rx * r) * speed * dt,
    z: pose.z + (fz * f + rz * r) * speed * dt,
  };
}

export class WalkControls {
  enabled = false;
  heading = 0;
  pitch = 0;
  x = 0;
  z = 0;
  eye = EYE_HEIGHT_M;
  private groundY = 0;
  private hasGround = false;
  private smoothY: number | null = null;
  private keys = new Set<string>();
  private dragging: { x: number; y: number } | null = null;
  private locked = false;
  private onKeyDown = (e: KeyboardEvent): void => this.keyDown(e);
  private onKeyUp = (e: KeyboardEvent): void => {
    this.keys.delete(e.code);
  };
  private onMouseMove = (e: MouseEvent): void => this.mouseMove(e);
  private onPointerDown = (e: PointerEvent): void => this.pointerDown(e);
  private onPointerUp = (): void => {
    this.dragging = null;
  };
  private onLockChange = (): void => {
    this.locked = document.pointerLockElement === this.dom;
  };
  private onBlur = (): void => this.keys.clear();
  /** called when the user presses Escape while not pointer-locked (leave walk mode) */
  onExit: (() => void) | null = null;

  constructor(
    private camera: THREE.PerspectiveCamera,
    private dom: HTMLElement,
  ) {}

  enable(pose: WalkPose, ground: GroundFn): void {
    this.x = pose.x;
    this.z = pose.z;
    this.heading = pose.heading;
    this.pitch = pose.pitch ?? 0;
    this.smoothY = null;
    const g = ground(this.x, this.z);
    this.groundY = g ?? this.camera.position.y - this.eye;
    this.hasGround = g !== null;
    if (this.enabled) return;
    this.enabled = true;
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('blur', this.onBlur);
    document.addEventListener('mousemove', this.onMouseMove);
    document.addEventListener('pointerlockchange', this.onLockChange);
    this.dom.addEventListener('pointerdown', this.onPointerDown);
    window.addEventListener('pointerup', this.onPointerUp);
  }

  disable(): void {
    if (!this.enabled) return;
    this.enabled = false;
    this.keys.clear();
    this.dragging = null;
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    window.removeEventListener('blur', this.onBlur);
    document.removeEventListener('mousemove', this.onMouseMove);
    document.removeEventListener('pointerlockchange', this.onLockChange);
    this.dom.removeEventListener('pointerdown', this.onPointerDown);
    window.removeEventListener('pointerup', this.onPointerUp);
    if (document.pointerLockElement === this.dom) document.exitPointerLock?.();
  }

  get isPointerLocked(): boolean {
    return this.locked;
  }

  private isTyping(e: KeyboardEvent): boolean {
    const path = e.composedPath();
    return path.some((t) => t instanceof HTMLElement && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable));
  }

  private keyDown(e: KeyboardEvent): void {
    if (this.isTyping(e)) return;
    if (e.code === 'Escape' && !this.locked) {
      this.onExit?.();
      return;
    }
    const handled = ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ShiftLeft', 'ShiftRight', 'KeyQ', 'KeyE', 'PageUp', 'PageDown'];
    if (handled.includes(e.code)) {
      this.keys.add(e.code);
      e.preventDefault();
    }
  }

  private pointerDown(e: PointerEvent): void {
    if (e.button !== 0 && e.button !== 2) return;
    // try pointer lock (real browsers); fall back to drag-to-look
    if (!this.locked && this.dom.requestPointerLock && e.button === 0 && !e.shiftKey) {
      try {
        const r = this.dom.requestPointerLock() as unknown as Promise<void> | undefined;
        r?.catch?.(() => undefined);
      } catch {
        /* not allowed (iframe, headless) */
      }
    }
    this.dragging = { x: e.clientX, y: e.clientY };
  }

  private mouseMove(e: MouseEvent): void {
    let dx = 0;
    let dy = 0;
    if (this.locked) {
      dx = e.movementX;
      dy = e.movementY;
    } else if (this.dragging) {
      dx = e.clientX - this.dragging.x;
      dy = e.clientY - this.dragging.y;
      this.dragging = { x: e.clientX, y: e.clientY };
    } else return;
    this.look(dx, dy);
  }

  /** Turn by mouse deltas (pixels). */
  look(dx: number, dy: number): void {
    const k = 0.0025;
    this.heading += dx * k;
    this.pitch = THREE.MathUtils.clamp(this.pitch - dy * k, -1.35, 1.35);
  }

  private axis(pos: string[], neg: string[]): number {
    return (pos.some((k) => this.keys.has(k)) ? 1 : 0) - (neg.some((k) => this.keys.has(k)) ? 1 : 0);
  }

  update(dt: number, ground: GroundFn): void {
    if (!this.enabled) return;
    const fwd = this.axis(['KeyW', 'ArrowUp'], ['KeyS', 'ArrowDown']);
    const right = this.axis(['KeyD'], ['KeyA']);
    const turn = this.axis(['ArrowRight'], ['ArrowLeft']);
    if (turn) this.heading += turn * 1.4 * dt;
    const run = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight');
    const next = walkStep({ x: this.x, z: this.z, heading: this.heading }, { fwd, right, run }, dt);
    // cast from a little above the current ground: walking under a tree must not climb it
    const g = ground(next.x, next.z, this.hasGround ? this.groundY + 2.5 : undefined);
    if (g !== null) {
      this.hasGround = true;
      // refuse steps up walls (a jump of more than 1.2 m in one step), allow stairs/slopes
      if (g - this.groundY < 1.2 + (run ? 0.6 : 0) || fwd === 0) {
        this.x = next.x;
        this.z = next.z;
        this.groundY = g;
      }
    } else {
      this.x = next.x;
      this.z = next.z;
    }
    const up = this.axis(['KeyE', 'PageUp'], ['KeyQ', 'PageDown']);
    if (up) this.eye = THREE.MathUtils.clamp(this.eye + up * 2 * dt, 1.2, 12);
    const target = this.groundY + this.eye;
    // smooth vertical motion (tile surfaces are bumpy)
    this.smoothY = this.smoothY === null ? target : this.smoothY + (target - this.smoothY) * Math.min(1, dt * 10);
    this.apply();
  }

  /** Write the pose into the camera. */
  apply(): void {
    const cam = this.camera;
    cam.position.set(this.x, this.smoothY ?? this.groundY + this.eye, this.z);
    // three.js camera looks down -z; heading 0 = north (-z)
    cam.rotation.set(0, 0, 0);
    cam.rotation.order = 'YXZ';
    cam.rotation.y = -this.heading;
    cam.rotation.x = this.pitch;
    cam.updateMatrixWorld();
  }

  dispose(): void {
    this.disable();
  }
}
