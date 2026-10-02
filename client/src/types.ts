/** Types for the HTTP API contract (docs/api.md). Fields the client does not
 * rely on are typed loosely so server-side additions never break parsing. */

import type { BBox } from './geo';

export interface ExtentScene {
  min_x: number;
  max_x: number;
  min_z: number;
  max_z: number;
}

export interface CalibrationStatus {
  status: 'uncalibrated' | 'calibrated' | 'above_target' | string;
  median_error_pct: number | null;
  targets_with_data?: number;
  targets_total?: number;
  table?: Array<Record<string, unknown>>;
}

export interface Mission {
  id: string;
  title: string;
  brief: string;
  budget_usd_upfront: number;
  budget_usd_per_year: number;
  constraints: string[];
  goals_suggested: string[];
}

export interface MetricDef {
  id: string;
  label: string;
  unit: string;
  better: 'lower' | 'higher' | string;
}

export interface TimeConfig {
  bin_start_s: number;
  bin_s: number;
  n_bins: number;
  report_start_s: number;
  report_end_s: number;
}

export interface WorldMeta {
  region: {
    name: string;
    display_name: string;
    bbox: BBox;
    origin: { lat: number; lon: number; easting?: number; northing?: number };
    timezone: string;
    extent_scene: ExtentScene;
  };
  synthetic: boolean;
  /** Client-only: true when served by the in-browser mock (`?mock=1`). */
  mock?: boolean;
  assets: { base_url: string; manifest: string };
  counts: Record<string, unknown>;
  calibration: CalibrationStatus;
  mission: Mission;
  metrics: MetricDef[];
  time: TimeConfig;
  hero: { school_id: string; x: number; z: number };
  unverified_inputs: string[];
}

export interface NetworkEdge {
  i: number;
  u: number;
  v: number;
  name: string;
  label: string;
  highway: string;
  lanes: number;
  len: number;
  /** flattened [x0,y0,z0, x1,y1,z1, ...] scene meters, from u to v */
  pts: number[];
}

export interface NetworkNode {
  id: number;
  x: number;
  y: number;
  z: number;
  signal: boolean;
}

export interface NetworkJson {
  n_edges: number;
  edges: NetworkEdge[];
  nodes: NetworkNode[];
}

export interface BuildingInfo {
  id: number;
  type?: string;
  height_m?: number;
  levels?: number | null;
  address?: string | null;
  name?: string | null;
  area_m2?: number;
  centroid_x?: number;
  centroid_z?: number;
  base_elev_m?: number;
  school_id?: string | null;
  households?: number;
  block?: string;
  [k: string]: unknown;
}

export interface EntranceBaseline {
  max_queue_cars: number;
  max_spillback_m: number;
  avg_wait_min: number;
}

export interface Entrance {
  id: string;
  key: string;
  x: number;
  y: number;
  z: number;
  curb_spots: number;
  unload_seconds: number;
  verified: boolean;
  baseline?: EntranceBaseline | null;
}

export interface School {
  id: string;
  name: string;
  grades: number[];
  bell_start: string;
  x: number;
  y: number;
  z: number;
  verified: boolean;
  students: number;
  entrances: Entrance[];
}

export interface Range3 {
  median: number;
  p10: number;
  p90: number;
}

/** Range whose values may be null (e.g. baseline of resident_approval_pct). */
export interface MaybeRange3 {
  median: number | null;
  p10: number | null;
  p90: number | null;
}

export interface MetricBlock {
  metrics: Array<MetricDef & { value: Range3 }>;
  per_school: unknown[];
  mode_share: unknown[];
}

export interface BaselineResponse {
  summary: MetricBlock;
  playback_url: string;
  calibration: CalibrationStatus;
}

export type ParamType =
  | 'school'
  | 'entrance'
  | 'time'
  | 'int'
  | 'float'
  | 'select'
  | 'bool'
  | 'text'
  | 'point'
  | 'points'
  | 'polyline'
  | 'edge'
  | 'node';

export interface SelectOption {
  value: string;
  label: string;
}

export interface ToolParamDef {
  id: string;
  label: string;
  type: ParamType | string;
  required?: boolean;
  default?: unknown;
  min?: number | string;
  max?: number | string;
  step?: number;
  options?: SelectOption[];
  map_input?: boolean;
  max_count?: number;
  max_length?: number;
  allow_all?: boolean;
}

export interface ToolDef {
  id: string;
  name: string;
  category: string;
  description: string;
  cost_note?: string;
  enabled_in_mvp?: boolean;
  params: ToolParamDef[];
}

export interface ToolsResponse {
  mission: string;
  tools: ToolDef[];
}

export type ParamValue = string | number | boolean | null | number[] | number[][] | CustomEstimate;

/** One model lever proposed by the custom tool preview (spec 7.3). */
export type CustomLever =
  | { type: 'mode_utility_shift'; mode: string; applies_to: 'students' | 'workers' | 'all'; school: string | null; utils: number }
  | { type: 'capacity_change'; target: 'edge' | 'entrance'; edge_idx?: number | null; entrance?: string | null; factor: number };

/** LLM estimate for a custom idea; the player confirms it, then it is stored in params.estimate. */
export interface CustomEstimate {
  summary: string;
  levers: CustomLever[];
  adoption_range: [number, number];
  cost_upfront_usd: number;
  cost_per_year_usd: number;
  assumptions: string[];
}

export interface CustomPreviewResponse {
  ok: boolean;
  label: string;
  description: string;
  estimate: CustomEstimate;
  tool: { tool: 'custom'; params: { description: string; estimate: CustomEstimate } };
}

export interface ToolInstance {
  tool: string;
  params: Record<string, ParamValue>;
}

export interface PlanInput {
  mission: string;
  title: string;
  pitch: string;
  tools: ToolInstance[];
}

export interface PlanCheck {
  ok: boolean;
  errors: string[];
  warnings: string[];
  cost_upfront_usd: number;
  cost_per_year_usd: number;
  over_budget: boolean;
  budget_upfront_usd: number;
  budget_per_year_usd: number;
  constraint_violations: string[];
  resolved_tools: Array<Record<string, unknown>>;
}

export type PlanStatus = 'draft' | 'queued' | 'running' | 'done' | 'failed';

export interface ReportMetric extends MetricDef {
  baseline: MaybeRange3;
  plan: MaybeRange3;
  delta: MaybeRange3;
  note?: string;
}

export interface BaselinePlan {
  baseline: Range3;
  plan: Range3;
}

export interface PerSchoolRow {
  school_id: string;
  name: string;
  [metric: string]: BaselinePlan | string;
}

export interface ModeShareRow {
  mode: string;
  label?: string;
  baseline_pct: number;
  plan_pct: number;
  delta_pp: number;
}

export interface SideEffect {
  edge_idx: number;
  name: string;
  baseline_vc: number;
  plan_vc: number;
  bin_s: number;
  x: number;
  z: number;
}

export interface CostLine {
  tool: string;
  upfront_usd: number;
  per_year_usd: number;
  note?: string;
}

export interface Report {
  plan_id: string;
  seeds: number;
  generated_at: string;
  synthetic: boolean;
  cost: {
    upfront_usd: number;
    per_year_usd: number;
    over_budget: boolean;
    budget_upfront_usd: number;
    budget_per_year_usd: number;
    lines: CostLine[];
  };
  constraint_violations: string[];
  metrics: ReportMetric[];
  per_school: PerSchoolRow[];
  mode_share: ModeShareRow[];
  winners: Range3;
  losers: Range3;
  side_effects: SideEffect[];
  peak_overlap?: { baseline: number; plan: number; note?: string };
  unverified_inputs: string[];
  llm_estimated_tools: string[];
  calibration: CalibrationStatus;
}

export interface Plan {
  id: string;
  mission: string;
  title: string;
  pitch: string;
  author_id: string | null;
  tools: ToolInstance[];
  created_at: string;
  report: Report | null;
  status: PlanStatus;
  check: PlanCheck | null;
  votes: number;
  job_id?: string | null;
  is_mine?: boolean;
  my_vote?: number;
}

export interface Job {
  id: string;
  plan_id: string;
  status: 'queued' | 'running' | 'done' | 'failed';
  progress: number;
  message: string;
  error: string | null;
}

export interface Reaction {
  persona_id: number;
  first_name: string;
  age: number;
  block: string;
  x: number;
  z: number;
  school_ids: string[];
  values: string[];
  approval: number;
  approves: boolean;
  deltas: { commute_min?: number | null; dropoff_min?: number | null; cost_usd_year?: number | null; street_change?: boolean };
  text: string | null;
}

export interface ResidentsResponse {
  approval_pct: number;
  reactions: Reaction[];
  text_status?: 'complete' | 'pending' | 'partial' | 'unavailable' | string;
  llm_available?: boolean;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  plan_id?: string | null;
  at?: string;
}

export interface ChatResponse {
  persona_id: number;
  reply?: string | null;
  messages: ChatMessage[];
  llm_available?: boolean;
  error?: string | null;
}

export interface TownhallSpeaker extends Reaction {
  side: 'for' | 'against';
  comment: string | null;
}

export interface TownhallFollowup {
  persona_id: number;
  message: string;
  text: string | null;
}

export interface TownhallResponse {
  plan_id: string;
  speakers: TownhallSpeaker[];
  followup: TownhallFollowup | null;
  llm_available?: boolean;
}

export interface PlanListItem {
  id: string;
  title: string;
  pitch: string;
  created_at: string;
  votes: number;
  status: PlanStatus;
  headline: Record<string, number>;
}

export interface ManifestTerrainLod {
  lod: number;
  path: string;
  triangles?: number;
  spacing_m?: number;
  texture_px?: number;
}

export interface ManifestTile {
  id: string;
  row: number;
  col: number;
  bounds: { min_x: number; max_x: number; min_z: number; max_z: number; min_y: number; max_y: number };
  terrain?: string;
  buildings?: string;
  /** HD build: terrain LODs (0 = finest, RTIN; 1, 2 = coarser grids) */
  terrain_lods?: ManifestTerrainLod[];
  albedo?: string;
  /** HD build: landcover splat masks [a (lawn, chaparral, dirt), b (paved, water, canopy)] */
  splat?: string[];
  /** HD build: road surfaces + markings, and sidewalks / driveways / medians / pools */
  roads?: string | null;
  ground?: string | null;
}

export interface Manifest {
  contract_version: number;
  hd_version?: number;
  synthetic: boolean;
  draco: boolean;
  tiles: ManifestTile[];
  /** HD build: the shared tile grid (scene meters) */
  grid?: { rows: number; cols: number; min_x: number; max_x: number; min_z: number; max_z: number };
  roads: string[];
  terrain_meta?: string;
  triangles?: Record<string, unknown>;
  terrain_lod?: { suggested_switch_distance_m?: Record<string, number> };
  splat?: { channels?: Record<string, string[]>; suggested_ground_cells?: Record<string, string> };
  materials?: string | null;
  /** Blender hero campuses baked into pipeline building tiles (meshes named `hero_<id>_<n>`) */
  heroes?: Array<{ id: string; name?: string; building_id?: number; tile: string; school_id?: string | null }>;
  /** path (under assets/) of the Blender HD buildings manifest, when built */
  buildings_hd?: string;
}
