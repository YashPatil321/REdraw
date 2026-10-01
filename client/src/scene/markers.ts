/**
 * Map annotations: clickable school markers (3D pin + HTML label), arterial
 * labels, the plan's map inputs (stops, routes, edges, intersections), an
 * edge highlight for "show on map", and a generic location pin.
 */

import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { latLonToScene, type Origin } from '../geo';
import type { RoadNetwork } from '../traffic/network';
import type { School, ToolDef, ToolInstance } from '../types';

export const PLAN_COLOR = 0xff5fa2; // plan accent (pink)
export const BASELINE_COLOR = 0x4fb3ff; // baseline accent (blue)

function makeLabel(text: string, cls: string): HTMLElement {
  const el = document.createElement('div');
  el.className = cls;
  el.textContent = text;
  return el;
}

export class SchoolMarkers {
  readonly group = new THREE.Group();
  private pins: THREE.InstancedMesh | null = null;
  private labels: CSS2DObject[] = [];

  constructor(schools: School[], onClick: (s: School) => void, heroId?: string) {
    this.group.name = 'school-markers';
    if (!schools.length) return;
    const geo = new THREE.CylinderGeometry(3, 3, 60, 8).translate(0, 30, 0);
    this.pins = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ color: 0xffe08a, fog: false }), schools.length);
    const m = new THREE.Matrix4();
    schools.forEach((s, i) => {
      m.makeTranslation(s.x, s.y ?? 0, s.z);
      this.pins!.setMatrixAt(i, m);
      const el = makeLabel(s.name, `school-label${s.id === heroId ? ' hero' : ''}`);
      el.setAttribute('role', 'button');
      el.tabIndex = 0;
      el.title = `${s.name}: bell ${s.bell_start}, ${s.students} students`;
      el.addEventListener('pointerdown', (ev) => ev.stopPropagation());
      el.addEventListener('click', (ev) => {
        ev.stopPropagation();
        onClick(s);
      });
      el.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter') onClick(s);
      });
      const o = new CSS2DObject(el);
      o.center.set(0.5, 1.15);
      o.position.set(s.x, (s.y ?? 0) + 62, s.z);
      this.group.add(o);
      this.labels.push(o);
    });
    this.pins.computeBoundingSphere();
    this.group.add(this.pins);
  }

  dispose(): void {
    this.pins?.geometry.dispose();
    (this.pins?.material as THREE.Material | undefined)?.dispose();
    for (const l of this.labels) l.element.remove();
    this.group.clear();
  }
}

/** One label per arterial (`label` field), at the midpoint of its longest edge. */
export class ArterialLabels {
  readonly group = new THREE.Group();
  constructor(net: RoadNetwork) {
    const best = new Map<string, { e: number; len: number }>();
    for (let e = 0; e < net.nEdges; e++) {
      const label = net.edges[e]!.label;
      if (!label) continue;
      const len = net.length[e]!;
      const cur = best.get(label);
      if (!cur || len > cur.len) best.set(label, { e, len });
    }
    for (const [label, { e }] of best) {
      const p = net.midpoint(e);
      const o = new CSS2DObject(makeLabel(label, 'road-label'));
      o.position.set(p.x, p.y + 8, p.z);
      this.group.add(o);
    }
  }
  dispose(): void {
    this.group.children.forEach((c) => (c as CSS2DObject).element?.remove());
    this.group.clear();
  }
}

/** Visualizes the draft plan's map inputs. Rebuilt on every draft change (small). */
export class PlanOverlay {
  readonly group = new THREE.Group();
  private mat = new THREE.MeshBasicMaterial({ color: PLAN_COLOR, fog: false, depthTest: false, transparent: true, opacity: 0.95 });
  private lineMat = new THREE.MeshBasicMaterial({ color: PLAN_COLOR, fog: false, transparent: true, opacity: 0.85, depthTest: false });
  private sphere = new THREE.SphereGeometry(7, 12, 8);
  private ring = new THREE.TorusGeometry(16, 3, 6, 24).rotateX(Math.PI / 2);

  constructor(
    private net: RoadNetwork | null,
    private origin: Origin,
    private heightAt: (x: number, z: number) => number,
  ) {
    this.group.name = 'plan-overlay';
    this.group.renderOrder = 10;
  }

  private clear(): void {
    for (const c of [...this.group.children]) {
      const m = c as THREE.Mesh;
      if (m.geometry && m.geometry !== this.sphere && m.geometry !== this.ring) m.geometry.dispose();
      (c as CSS2DObject).element?.remove();
    }
    this.group.clear();
  }

  private ll(v: number[]): THREE.Vector3 {
    const p = latLonToScene(v[0]!, v[1]!, this.origin);
    return new THREE.Vector3(p.x, this.heightAt(p.x, p.z) + 3, p.z);
  }

  private tube(points: THREE.Vector3[], radius: number): void {
    if (points.length < 2) return;
    const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.1);
    const g = new THREE.TubeGeometry(curve, Math.max(8, points.length * 8), radius, 6, false);
    const m = new THREE.Mesh(g, this.lineMat);
    m.renderOrder = 10;
    this.group.add(m);
  }

  private edgePoints(e: number): THREE.Vector3[] {
    const out: THREE.Vector3[] = [];
    if (!this.net || e < 0 || e >= this.net.nEdges) return out;
    const s = this.net.ptStart[e]!;
    const n = this.net.ptCount[e]!;
    for (let k = 0; k < n; k++) {
      out.push(new THREE.Vector3(this.net.pts[(s + k) * 3]!, this.net.pts[(s + k) * 3 + 1]! + 3, this.net.pts[(s + k) * 3 + 2]!));
    }
    return out;
  }

  update(tools: ToolInstance[], defs: ToolDef[], selected: number | null): void {
    this.clear();
    tools.forEach((inst, ti) => {
      const def = defs.find((d) => d.id === inst.tool);
      if (!def) return;
      const dim = selected !== null && selected !== ti;
      for (const p of def.params) {
        const v = inst.params[p.id];
        if (v === undefined || v === null) continue;
        const add = (obj: THREE.Mesh): void => {
          obj.renderOrder = 10;
          if (dim) obj.scale.multiplyScalar(0.7);
          this.group.add(obj);
        };
        if (p.type === 'point' && Array.isArray(v) && v.length === 2) {
          const m = new THREE.Mesh(this.sphere, this.mat);
          m.position.copy(this.ll(v as number[]));
          add(m);
        } else if ((p.type === 'points' || p.type === 'polyline') && Array.isArray(v)) {
          const pts = (v as number[][]).map((x) => this.ll(x));
          pts.forEach((pt, i) => {
            const m = new THREE.Mesh(this.sphere, this.mat);
            m.position.copy(pt);
            add(m);
            if (p.type === 'points') {
              const lbl = new CSS2DObject(makeLabel(String(i + 1), 'pin-label'));
              lbl.position.copy(pt).add(new THREE.Vector3(0, 14, 0));
              this.group.add(lbl);
            }
          });
          if (p.type === 'polyline') this.tube(pts, 3.5);
        } else if (p.type === 'edge' && typeof v === 'number') {
          this.tube(this.edgePoints(v), 4.5);
        } else if (p.type === 'node' && typeof v === 'number' && this.net) {
          const nd = this.net.nodeById.get(v);
          if (nd) {
            const m = new THREE.Mesh(this.ring, this.mat);
            m.position.set(nd.x, nd.y + 3, nd.z);
            add(m);
          }
        }
      }
    });
  }

  dispose(): void {
    this.clear();
    this.mat.dispose();
    this.lineMat.dispose();
    this.sphere.dispose();
    this.ring.dispose();
  }
}

/** Glowing tube along one edge (side effects "show on map"). */
export class EdgeHighlight {
  readonly group = new THREE.Group();
  private mat = new THREE.MeshBasicMaterial({ color: 0xffffff, fog: false, transparent: true, opacity: 0.9, depthTest: false });
  private mesh: THREE.Mesh | null = null;

  constructor(private net: RoadNetwork | null) {}

  set(edge: number | null): void {
    if (this.mesh) {
      this.mesh.geometry.dispose();
      this.group.remove(this.mesh);
      this.mesh = null;
    }
    if (edge === null || !this.net || edge < 0 || edge >= this.net.nEdges) return;
    const s = this.net.ptStart[edge]!;
    const n = this.net.ptCount[edge]!;
    const pts: THREE.Vector3[] = [];
    for (let k = 0; k < n; k++) pts.push(new THREE.Vector3(this.net.pts[(s + k) * 3]!, this.net.pts[(s + k) * 3 + 1]! + 4, this.net.pts[(s + k) * 3 + 2]!));
    if (pts.length < 2) return;
    const g = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts, false, 'catmullrom', 0.05), Math.max(8, n * 6), 6, 6, false);
    this.mesh = new THREE.Mesh(g, this.mat);
    this.mesh.renderOrder = 11;
    this.group.add(this.mesh);
  }

  tick(now: number): void {
    this.mat.opacity = 0.55 + 0.4 * Math.sin(now / 180);
  }

  dispose(): void {
    this.set(null);
    this.mat.dispose();
  }
}

/** A single pin (selected building / resident block). */
export class LocationPin {
  readonly group = new THREE.Group();
  private mesh: THREE.Mesh;
  constructor(color: THREE.ColorRepresentation) {
    const g = new THREE.ConeGeometry(6, 18, 12).rotateX(Math.PI).translate(0, 9, 0);
    this.mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, fog: false, depthTest: false, transparent: true }));
    this.mesh.renderOrder = 12;
    this.group.add(this.mesh);
    this.group.visible = false;
  }
  show(x: number, y: number, z: number): void {
    this.group.position.set(x, y, z);
    this.group.visible = true;
  }
  hide(): void {
    this.group.visible = false;
  }
  tick(now: number, scale: number): void {
    this.mesh.position.y = 4 + Math.abs(Math.sin(now / 300)) * 6 * scale;
    this.mesh.scale.setScalar(scale);
  }
  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}
