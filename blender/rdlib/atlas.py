"""Material-atlas conventions shared by houses (previews), heroes and tests (pure python + numpy).

`_MAT` ids (materials_manifest.json -> materials):
    0 stucco wall, 1 tile roof, 2 flat roof, 3 glass, 4 trim, 5 garage door   (pipeline contract)
    6 ground (hero campuses: planar UV, _VARIANT = ground atlas cell)          (extension)
    7 vertex color only (paint lines, sports surfaces, small site props)       (extension)

UVs (TEXCOORD_0), Blender axes (+X east, +Y north, +Z up):
    walls / glass / trim / garage: u = meters along the wall / 3, v = meters above ground / 3
    pitched roofs: u = meters along the eave / 4, v = meters up the slope / 4
    flat roofs: u = x / 4, v = y / 4 (north)
    ground: u = x / size, v = y / size with the cell's world size
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MAT_DIR = REPO / "client" / "public" / "assets" / "materials"

MAT_WALL, MAT_TILE_ROOF, MAT_FLAT_ROOF, MAT_GLASS, MAT_TRIM, MAT_GARAGE, MAT_GROUND, MAT_VCOL = range(8)
FACADE_M = 3.0
ROOF_M = 4.0

# hero palette key -> (_MAT, variant name). Unlisted keys -> MAT_VCOL.
KEY_MAT: dict[str, tuple[int, str]] = {
    # walls (variant names from materials_manifest materials[0].variants)
    "stucco_white": (MAT_WALL, "stucco_sand"), "stucco_cream": (MAT_WALL, "stucco_sand"), "stucco_sand": (MAT_WALL, "stucco_sand"),
    "stucco_tan": (MAT_WALL, "stucco_lace"), "stucco_clay": (MAT_WALL, "stucco_smooth"), "stucco_gray": (MAT_WALL, "stucco_scored"),
    "stucco_warm": (MAT_WALL, "stucco_scored"), "stucco_olive": (MAT_WALL, "stucco_smooth"), "panel_white": (MAT_WALL, "stucco_smooth"),
    "accent_red": (MAT_WALL, "stucco_smooth"), "accent_orange": (MAT_WALL, "stucco_smooth"), "accent_teal": (MAT_WALL, "stucco_smooth"),
    "accent_navy": (MAT_WALL, "stucco_smooth"), "accent_yellow": (MAT_WALL, "stucco_smooth"), "accent_green": (MAT_WALL, "stucco_smooth"),
    "stone_base": (MAT_WALL, "stone_veneer"), "retaining_wall": (MAT_WALL, "stone_veneer"),
    "concrete_dark": (MAT_WALL, "stucco_scored"),
    # trim
    "trim_white": (MAT_TRIM, "stucco_smooth"), "trim_dark": (MAT_TRIM, "stucco_smooth"), "metal_dark": (MAT_TRIM, "stucco_smooth"),
    "steel_white": (MAT_TRIM, "stucco_smooth"),
    # glass
    "glass_dark": (MAT_GLASS, "glass_curtain"), "glass": (MAT_GLASS, "glass_curtain"), "glass_light": (MAT_GLASS, "glass_curtain"),
    # roofs
    "roof_tile": (MAT_TILE_ROOF, "s_tile_blend"), "roof_tile_dark": (MAT_TILE_ROOF, "s_tile_brown"),
    "roof_tpo": (MAT_FLAT_ROOF, "flat_tpo_grime"), "roof_flat": (MAT_FLAT_ROOF, "flat_tpo"), "roof_gravel": (MAT_FLAT_ROOF, "flat_gravel"),
    "roof_metal": (MAT_FLAT_ROOF, "standing_seam"), "hvac": (MAT_FLAT_ROOF, "standing_seam"),
    "solar_panel": (MAT_TILE_ROOF, "solar_panel"),
    # ground
    "asphalt": (MAT_GROUND, "asphalt_parking"), "asphalt_light": (MAT_GROUND, "asphalt_worn"),
    "concrete": (MAT_GROUND, "concrete_plaza"), "curb": (MAT_GROUND, "concrete_sidewalk"), "pavers": (MAT_GROUND, "pavers"),
    "grass": (MAT_GROUND, "grass_lawn"), "grass_dark": (MAT_GROUND, "grass_lawn"), "landscape": (MAT_GROUND, "coastal_sage"),
    "mulch": (MAT_GROUND, "mulch"), "dg_path": (MAT_GROUND, "decomposed_granite"), "dry_grass": (MAT_GROUND, "grass_patchy"),
    "soil_plinth": (MAT_GROUND, "bare_dirt"), "water": (MAT_GROUND, "pool_water"),
}


def load_manifest(mat_dir: Path = MAT_DIR) -> dict[str, Any]:
    p = mat_dir / "materials_manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; build it with: .venv-blender/bin/python blender/build_all_assets.py --only materials")
    return json.loads(p.read_text())


def cell(man: dict[str, Any], atlas: str, name: str) -> dict[str, Any]:
    for c in man["atlases"][atlas]["cells"]:
        if c["name"] == name:
            return c
    raise KeyError(f"{atlas}/{name}")


def variant_index(man: dict[str, Any], mat: int, name: str) -> int:
    if mat == MAT_GROUND:
        return cell(man, "ground", name)["index"]
    return man["materials"][str(mat)]["variants"].index(name)


def mat_of_key(key: str) -> tuple[int, str | None]:
    if key in KEY_MAT:
        return KEY_MAT[key]
    return MAT_VCOL, None


def face_uv(P: np.ndarray, mat: int, ground_size: float = 4.0) -> np.ndarray:
    """UVs for one planar face (P: (k, 3) Blender coords) following the conventions above."""
    P = np.asarray(P, float)
    n = np.zeros(3)
    for a in range(len(P)):  # Newell normal
        u, v = P[a], P[(a + 1) % len(P)]
        n += np.array([(u[1] - v[1]) * (u[2] + v[2]), (u[2] - v[2]) * (u[0] + v[0]), (u[0] - v[0]) * (u[1] + v[1])])
    ln = np.linalg.norm(n)
    n = n / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
    if mat == MAT_GROUND:
        return P[:, :2] / ground_size
    if mat == MAT_FLAT_ROOF or (mat in (MAT_TILE_ROOF, MAT_VCOL) and abs(n[2]) > 0.97):
        return P[:, :2] / ROOF_M
    if mat == MAT_TILE_ROOF or (abs(n[2]) > 0.35 and mat not in (MAT_WALL, MAT_GLASS, MAT_TRIM, MAT_GARAGE)):
        e = np.array([-n[1], n[0], 0.0])  # horizontal, along the eave
        le = np.linalg.norm(e)
        e = e / le if le > 1e-9 else np.array([1.0, 0.0, 0.0])
        s = np.cross(n, e)
        if s[2] < 0:
            s, e = -s, -e
        return np.stack([P @ e, P @ s], axis=1) / ROOF_M
    if abs(n[2]) > 0.97:  # horizontal faces of walls/trim (sills, parapet caps): planar
        return P[:, :2] / FACADE_M
    r = np.cross(-n, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r) + 1e-12
    return np.stack([P @ r, P[:, 2]], axis=1) / FACADE_M
