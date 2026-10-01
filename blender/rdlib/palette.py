"""Shared color palette for every Redraw Blender asset.

Pure Python (no bpy) so tests and other tools can import it.

Every asset uses ONE material whose base color and metallic/roughness come from
two tiny palette textures (`palette_basecolor.png`, `palette_mr.png`). Each
palette entry is a CELL x CELL square; a face picks its color by putting all of
its UVs on the center of that square. One material means one draw call per mesh
(important for instancing thousands of trees and cars).

Colors are sRGB hex. Roughness/metallic are glTF linear values.
Edit the table below to restyle everything, then rerun
`.venv-blender/bin/python blender/build_all_assets.py`.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

GRID = 16  # cells per side
CELL = 8  # pixels per cell
SIZE = GRID * CELL  # texture size in px (128)


@dataclass(frozen=True)
class Swatch:
    hex: str
    roughness: float = 0.85
    metallic: float = 0.0


# Order matters only for texture layout; append new entries at the end so
# existing UVs in built assets stay valid.
PALETTE: dict[str, Swatch] = {
    # --- architecture -----------------------------------------------------
    "stucco_white": Swatch("#EEE8DD", 0.9),
    "stucco_cream": Swatch("#E6D6B8", 0.9),
    "stucco_sand": Swatch("#D9C29C", 0.9),
    "stucco_tan": Swatch("#C9A97E", 0.9),
    "stucco_clay": Swatch("#C47F5E", 0.9),
    "stucco_gray": Swatch("#B9B5AE", 0.9),
    "stone_base": Swatch("#A8957B", 0.95),
    "trim_dark": Swatch("#4E463F", 0.6, 0.3),
    "trim_white": Swatch("#E4E0D6", 0.7),
    "roof_tile": Swatch("#9E4A30", 0.8),
    "roof_tile_dark": Swatch("#8E4128", 0.8),
    "roof_flat": Swatch("#D2CCC1", 0.95),
    "roof_metal": Swatch("#8E979B", 0.45, 0.6),
    "roof_gravel": Swatch("#A9A399", 1.0),
    "glass": Swatch("#36556A", 0.08, 0.2),
    "glass_light": Swatch("#7FA4B8", 0.1, 0.2),
    # --- accents ------------------------------------------------------------
    "accent_navy": Swatch("#2C3E66", 0.6),
    "accent_teal": Swatch("#2E8A87", 0.6),
    "accent_orange": Swatch("#E27A35", 0.6),
    "accent_yellow": Swatch("#EBBB45", 0.6),
    "accent_green": Swatch("#72A04D", 0.6),
    "accent_red": Swatch("#B8432F", 0.6),
    "awning_red": Swatch("#A8432F", 0.85),
    "awning_green": Swatch("#3D6A4F", 0.85),
    "awning_blue": Swatch("#33607F", 0.85),
    "awning_cream": Swatch("#EADFC6", 0.85),
    "wood": Swatch("#8B6644", 0.8),
    "wood_light": Swatch("#B88D5F", 0.8),
    # --- hardscape ----------------------------------------------------------
    "concrete": Swatch("#CBC6BC", 0.9),
    "concrete_dark": Swatch("#A19B91", 0.9),
    "pavers": Swatch("#C2A889", 0.9),
    "curb": Swatch("#D9D5CB", 0.9),
    "asphalt": Swatch("#45474B", 0.95),
    "asphalt_light": Swatch("#5A5C61", 0.95),
    "stripe_white": Swatch("#F2F2EE", 0.8),
    "stripe_yellow": Swatch("#E9C443", 0.8),
    "stripe_blue": Swatch("#3D6FB6", 0.8),
    "court_green": Swatch("#4F7F5B", 0.85),
    "court_blue": Swatch("#3F6E9A", 0.85),
    "track_red": Swatch("#B5523D", 0.95),
    "dirt_infield": Swatch("#C38D5D", 1.0),
    "dg_path": Swatch("#CDB28A", 1.0),
    "mulch": Swatch("#7B5B40", 1.0),
    "rubber_play": Swatch("#3E7C8E", 0.95),
    "water": Swatch("#4C93B5", 0.05, 0.1),
    # --- ground / landscape ---------------------------------------------------
    "grass": Swatch("#719E46", 0.95),
    "grass_dark": Swatch("#5E8B3B", 0.95),
    "turf_field": Swatch("#4F9B45", 0.9),
    "turf_field_dark": Swatch("#458C3D", 0.9),
    "dry_grass": Swatch("#C2A76D", 1.0),
    "soil_plinth": Swatch("#A78A69", 1.0),
    "retaining_wall": Swatch("#B8A88F", 0.95),
    # --- foliage --------------------------------------------------------------
    "leaf_oak": Swatch("#4C6A2D", 0.9),
    "leaf_oak_light": Swatch("#647F37", 0.9),
    "leaf_euc": Swatch("#8BA07C", 0.9),
    "leaf_euc_dark": Swatch("#6D8566", 0.9),
    "leaf_palm": Swatch("#6B8A38", 0.85),
    "leaf_palm_dark": Swatch("#557331", 0.85),
    "palm_dead": Swatch("#B09466", 0.95),
    "leaf_jacaranda": Swatch("#9474C9", 0.85),
    "leaf_jacaranda_dark": Swatch("#7458A9", 0.85),
    "leaf_pine": Swatch("#3E5D36", 0.9),
    "leaf_street": Swatch("#5E8A3D", 0.9),
    "leaf_street_light": Swatch("#7AA24B", 0.9),
    "shrub": Swatch("#5B7C39", 0.9),
    "shrub_flower": Swatch("#D86A8E", 0.85),
    "chaparral": Swatch("#7E8557", 0.95),
    "chaparral_dark": Swatch("#5F6745", 0.95),
    "bark_brown": Swatch("#5D4939", 0.95),
    "bark_gray": Swatch("#7E766B", 0.95),
    "bark_euc": Swatch("#DCCDB9", 0.9),
    "palm_trunk": Swatch("#8C7660", 0.95),
    # --- vehicles -------------------------------------------------------------
    "paint_tint": Swatch("#E4E4E2", 0.35, 0.3),  # near white: client instanceColor tints it
    "paint_bus": Swatch("#F3B21C", 0.4, 0.1),
    "paint_white": Swatch("#F1F1EF", 0.4, 0.1),
    "paint_black": Swatch("#26272A", 0.35, 0.3),
    "car_glass": Swatch("#28333D", 0.08, 0.3),
    "tire": Swatch("#1E1E20", 0.9),
    "rim": Swatch("#B9BDC1", 0.35, 0.9),
    "plastic_black": Swatch("#2D2E31", 0.7),
    "headlight": Swatch("#FFF5DA", 0.2),
    "taillight": Swatch("#B5222A", 0.3),
    "amber": Swatch("#F09A2A", 0.3),
    # --- street furniture / site ----------------------------------------------
    "pole_gray": Swatch("#8C9195", 0.45, 0.7),
    "pole_dark": Swatch("#3C4044", 0.5, 0.6),
    "lamp_lens": Swatch("#FFEFC2", 0.2),
    "solar_panel": Swatch("#1E2B45", 0.15, 0.4),
    "solar_frame": Swatch("#B8BDC3", 0.4, 0.8),
    "steel_white": Swatch("#E7E7E3", 0.5, 0.4),
    "aluminum": Swatch("#BBC0C5", 0.35, 0.8),
    "shade_sail": Swatch("#F2EEE5", 0.9),
    "fence_dark": Swatch("#2F3A35", 0.6, 0.5),
    "play_red": Swatch("#D2473A", 0.6),
    "play_yellow": Swatch("#F0C23B", 0.6),
    "play_blue": Swatch("#3B78C2", 0.6),
    "scoreboard": Swatch("#22324F", 0.6),
    "signage": Swatch("#F7F4EC", 0.6),
    # --- hero campuses (appended; keep order) -------------------------------------
    "landscape": Swatch("#7F8350", 1.0),  # irrigated groundcover / slope planting
    "glass_dark": Swatch("#24323D", 0.08, 0.0),
    "panel_white": Swatch("#EEEEEA", 0.6),
    "metal_dark": Swatch("#3B4045", 0.45, 0.6),
    "stucco_warm": Swatch("#DCC7A6", 0.9),
    "stucco_olive": Swatch("#B9AE8C", 0.9),
    "hvac": Swatch("#C9CBC8", 0.5, 0.5),
    "roof_tpo": Swatch("#C4C1B9", 0.85),
    "net_dark": Swatch("#2B2E2F", 0.8),
}


def keys() -> list[str]:
    return list(PALETTE)


def index_of(key: str) -> int:
    try:
        return keys().index(key)
    except ValueError as e:
        raise KeyError(f"unknown palette key '{key}'") from e


def cell_uv(key: str) -> tuple[float, float]:
    """UV (glTF/Blender convention: v=0 at the bottom of the image in Blender) of a cell center."""
    i = index_of(key)
    col, row = i % GRID, i // GRID
    u = (col + 0.5) / GRID
    v = 1.0 - (row + 0.5) / GRID  # Blender UV origin is bottom-left; row 0 is the top image row
    return u, v


def hex_to_rgb8(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _png_bytes(width: int, height: int, rows: list[bytes]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + r for r in rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def _image_rows(pixel_of: callable) -> list[bytes]:  # type: ignore[valid-type]
    ks = keys()
    rows: list[bytes] = []
    for y in range(SIZE):
        row = bytearray()
        for x in range(SIZE):
            i = (y // CELL) * GRID + (x // CELL)
            row += bytes(pixel_of(PALETTE[ks[i]]) if i < len(ks) else (255, 0, 255))
        rows.append(bytes(row))
    return rows


def write_textures(out_dir: Path) -> tuple[Path, Path]:
    """Write palette_basecolor.png (sRGB) and palette_mr.png (glTF: G=roughness, B=metallic)."""
    if len(PALETTE) > GRID * GRID:
        raise ValueError(f"palette has {len(PALETTE)} entries, max {GRID * GRID}")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "palette_basecolor.png"
    mr = out_dir / "palette_mr.png"
    base.write_bytes(_png_bytes(SIZE, SIZE, _image_rows(lambda s: hex_to_rgb8(s.hex))))
    mr.write_bytes(
        _png_bytes(SIZE, SIZE, _image_rows(lambda s: (255, round(s.roughness * 255), round(s.metallic * 255))))
    )
    return base, mr
