"""Prop registry (vehicles, vegetation, street furniture) and the street lamp.

Every prop's origin is on the ground at the footprint center. Blender +Y is the
prop's forward / reach direction (vehicles' noses, the lamp's mast arm), which the
glTF exporter turns into -Z (three.js forward). See blender/README.md.
"""

from __future__ import annotations

import numpy as np

from .mesh import Part, box, cylinder, tube


def street_lamp() -> Part:
    """San Diego cobra-head street light: galvanized round tapered pole, ~9.1 m (30 ft) mounting
    height, 2.4 m (8 ft) davit mast arm reaching +Y, flat LED cobra-head luminaire."""
    p = cylinder(0.3, 0.45, "concrete", n=10).moved(0, 0, -0.15)  # foundation pedestal
    p += cylinder(0.14, 0.12, "pole_galv", n=10).moved(0, 0, 0.28)  # base flange
    p += cylinder(0.105, 8.6, "pole_galv", n=10, r_top=0.065).moved(0, 0, 0.38)
    arm = tube([(0, 0, 8.5), (0, 0.25, 8.95), (0, 0.9, 9.22), (0, 1.8, 9.3), (0, 2.35, 9.28)],
               [0.055, 0.05, 0.045, 0.042, 0.04], "pole_galv", n=6, cap=True)
    p += arm
    head = box(0.34, 0.72, 0.13, "luminaire", bevel=0.05, segments=1)
    head = head.moved(0, 2.62, 9.18)
    p += head
    p += box(0.26, 0.48, 0.02, "lamp_lens", bottom=True).moved(0, 2.66, 9.165)  # LED array underneath
    return p


def make_shared_normals(p: Part) -> np.ndarray:
    """Zero normals (= automatic) for every vertex of a Part (used where custom normals are optional)."""
    return np.zeros((len(p.V), 3))


STREET: dict[str, tuple] = {
    "street_lamp": (street_lamp, {"kind": "lamp", "height_m": 9.4, "radius_m": 0.4, "reach_m": 2.9,
                                  "notes": "SD cobra-head LED street light, 30 ft pole, 8 ft mast arm toward +forward (-Z)"}),
}
