"""PBR material library for props (vehicles, lamps, trees).

Pure Python (no bpy) so tests can import it. `bl.py` turns these specs into
Blender Principled BSDF materials; the glTF exporter writes them as glTF PBR
(metallic-roughness) plus KHR_materials_emissive_strength (lights) and
KHR_materials_clearcoat (car paint).

Material NAMES are part of the client contract (props_manifest.json
`tintable_region` = "material:paint"): the client multiplies its per-instance
color into the `paint` material only. Keep names stable.

Colors are sRGB hex; roughness / metallic are linear glTF factors.
Hero campuses do not use this table: they use vertex colors from palette.py
(the pipeline's hero loader keeps base color, roughness, COLOR_0 and textures only).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mat:
    hex: str
    roughness: float = 0.6
    metallic: float = 0.0
    emissive_hex: str | None = None
    emissive_strength: float = 0.0
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.05
    double_sided: bool = False
    alpha_mask: bool = False  # alphaMode MASK (leaf cards), base color texture alpha
    alpha_cutoff: float = 0.5
    texture: str | None = None  # key of a generated texture (foliage atlas)
    light: int = 0  # `_LIGHT` vertex attribute value (client aLight: 0 body, 1 head, 2 tail, 3 glass, 4 dark trim)
    tint: bool = False  # `_TINT` vertex attribute 1.0 (client multiplies instance color here)


MATERIALS: dict[str, Mat] = {
    # --- vehicle --------------------------------------------------------------------
    # Near-white metallic paint with clearcoat: the client multiplies a per-instance
    # SoCal color (props_manifest.json paint_colors) into this material only.
    "paint": Mat("#E8E8E6", 0.32, 0.55, clearcoat=1.0, tint=True),
    "paint_bus": Mat("#F2A900", 0.38, 0.0, clearcoat=0.8),  # National School Bus Glossy Yellow (FS 13432)
    "paint_bus_roof": Mat("#F1F0EA", 0.5, 0.0),  # white reflective roof common on CA buses
    "paint_white": Mat("#F2F2F0", 0.35, 0.05, clearcoat=0.8, tint=True),  # shuttle body (tintable fleet color)
    "glass": Mat("#0B0F13", 0.04, 0.0, light=3),  # dark privacy tint, mirror-like
    "glass_clear": Mat("#1A242C", 0.04, 0.0, light=3),  # windshield: slightly lighter
    "tire": Mat("#1B1B1C", 0.92, 0.0, light=4),
    "rim": Mat("#C4C7CB", 0.28, 1.0),
    "rim_dark": Mat("#3A3C40", 0.35, 0.9, light=4),
    "chrome": Mat("#D9DCDF", 0.12, 1.0),
    "trim_black": Mat("#151618", 0.55, 0.0, light=4),  # bumpers, cladding, grille, mirrors, rub rails
    "underbody": Mat("#0E0E0F", 0.95, 0.0, light=4),
    "plate": Mat("#E9E7DF", 0.5, 0.0),  # California plate: white
    "headlight": Mat("#FFF6E5", 0.08, 0.0, emissive_hex="#FFF3DC", emissive_strength=4.0, light=1),
    "drl": Mat("#FFFFFF", 0.1, 0.0, emissive_hex="#F4F7FF", emissive_strength=6.0, light=1),
    "taillight": Mat("#8E0D12", 0.12, 0.0, emissive_hex="#FF1A10", emissive_strength=3.0, light=2),
    "amber": Mat("#E07A12", 0.15, 0.0, emissive_hex="#FF8A14", emissive_strength=2.0),
    "bus_red_lamp": Mat("#A3110F", 0.15, 0.0, emissive_hex="#FF2010", emissive_strength=2.0),
    "stop_arm": Mat("#B3141B", 0.4, 0.0),
    "sign_black": Mat("#111111", 0.5, 0.0, light=4),
    "bed_liner": Mat("#202122", 0.95, 0.0, light=4),
    # --- street furniture -------------------------------------------------------------
    "pole_galv": Mat("#9EA3A6", 0.45, 0.85),  # galvanized steel davit pole (SD standard)
    "luminaire": Mat("#8D9296", 0.4, 0.6),  # cobra head housing
    "lamp_lens": Mat("#FFF1D6", 0.15, 0.0, emissive_hex="#FFE2B0", emissive_strength=3.0, light=1),
    "concrete": Mat("#B9B4AA", 0.9, 0.0),
    # --- vegetation (one atlas material per tree: bark + leaf cards share it) ----------
    "foliage": Mat("#FFFFFF", 0.82, 0.0, double_sided=True, alpha_mask=True, alpha_cutoff=0.45, texture="foliage_atlas"),
}


def get(key: str) -> Mat:
    try:
        return MATERIALS[key]
    except KeyError as e:
        raise KeyError(f"unknown material '{key}'") from e


def hex_to_linear(h: str) -> tuple[float, float, float]:
    """sRGB hex -> linear RGB (what Blender's Principled BSDF and glTF factors expect)."""
    h = h.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i : i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return out[0], out[1], out[2]


# Typical new/used vehicle color shares for Southern California suburbs. Used by the
# client to tint `paint` per instance. Mirrored (append-only) in
# data/config/assumptions.yaml -> props.vehicle_paint_shares (verified: false).
PAINT_COLORS: list[dict[str, object]] = [
    {"name": "white", "hex": "#E9EAEA", "share": 0.25},
    {"name": "black", "hex": "#16171A", "share": 0.21},
    {"name": "gray", "hex": "#5E6267", "share": 0.18},
    {"name": "silver", "hex": "#A9ADB1", "share": 0.12},
    {"name": "blue", "hex": "#1F3B66", "share": 0.09},
    {"name": "red", "hex": "#8E1B1E", "share": 0.08},
    {"name": "pearl_white", "hex": "#F1EFE8", "share": 0.03},
    {"name": "green", "hex": "#3F4F3E", "share": 0.02},
    {"name": "beige", "hex": "#B7A88E", "share": 0.02},
]
