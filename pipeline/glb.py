"""Minimal glTF 2.0 binary (GLB) writer and reader.

We write GLB ourselves (instead of via trimesh) so we fully control custom
vertex attributes (`_BUILDING_ID`), vertex colors and embedded JPEG textures.
Positions are scene meters (x east, y up, z south); nodes have identity
transforms (docs/data_contract.md, mesh conventions).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GL_FLOAT = 5126
GL_UNSIGNED_INT = 5125
GL_UNSIGNED_BYTE = 5121
GL_UNSIGNED_SHORT = 5123
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963

_TYPE_BY_WIDTH = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}
_WIDTH_BY_TYPE = {v: k for k, v in _TYPE_BY_WIDTH.items()}
_DTYPE_BY_COMPONENT = {GL_FLOAT: np.float32, GL_UNSIGNED_INT: np.uint32, GL_UNSIGNED_BYTE: np.uint8, GL_UNSIGNED_SHORT: np.uint16}


@dataclass
class MeshData:
    """One mesh primitive. Arrays are per vertex unless noted."""

    name: str
    positions: np.ndarray  # (N, 3) float32
    indices: np.ndarray  # (M,) uint32, triangles
    normals: np.ndarray | None = None  # (N, 3) float32
    uvs: np.ndarray | None = None  # (N, 2) float32
    colors: np.ndarray | None = None  # (N, 4) uint8 (normalized) -> COLOR_0
    custom: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (N,) float32
    texture_jpeg: bytes | None = None
    texture_mime: str = "image/jpeg"
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    roughness: float = 1.0
    double_sided: bool = False
    texture_repeat: bool = False  # REPEAT wrapping (road markings) instead of CLAMP (terrain)

    @property
    def triangle_count(self) -> int:
        return int(len(self.indices) // 3)


class _Builder:
    def __init__(self) -> None:
        self.bin = bytearray()
        self.buffer_views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def _align(self) -> None:
        while len(self.bin) % 4:
            self.bin.append(0)

    def view(self, data: bytes, target: int | None) -> int:
        self._align()
        bv: dict[str, Any] = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target is not None:
            bv["target"] = target
        self.bin.extend(data)
        self.buffer_views.append(bv)
        return len(self.buffer_views) - 1

    def accessor(self, arr: np.ndarray, component: int, target: int | None, normalized: bool = False, minmax: bool = False) -> int:
        arr = np.ascontiguousarray(arr)
        width = 1 if arr.ndim == 1 else arr.shape[1]
        bv = self.view(arr.tobytes(), target)
        acc: dict[str, Any] = {
            "bufferView": bv,
            "componentType": component,
            "count": int(arr.shape[0]),
            "type": _TYPE_BY_WIDTH[width],
        }
        if normalized:
            acc["normalized"] = True
        if minmax and arr.shape[0] > 0:
            a2 = arr.reshape(arr.shape[0], -1)
            acc["min"] = [float(v) for v in a2.min(axis=0)]
            acc["max"] = [float(v) for v in a2.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def write_glb(path: Path, meshes: list[MeshData], extras: dict[str, Any] | None = None) -> int:
    """Write meshes (one node each, identity transform) to a GLB. Returns triangle count."""
    b = _Builder()
    gl_meshes: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    textures: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    tris = 0
    for m in meshes:
        if len(m.positions) == 0 or len(m.indices) == 0:
            continue
        attrs: dict[str, int] = {}
        attrs["POSITION"] = b.accessor(m.positions.astype(np.float32), GL_FLOAT, ARRAY_BUFFER, minmax=True)
        if m.normals is not None:
            attrs["NORMAL"] = b.accessor(m.normals.astype(np.float32), GL_FLOAT, ARRAY_BUFFER)
        if m.uvs is not None:
            attrs["TEXCOORD_0"] = b.accessor(m.uvs.astype(np.float32), GL_FLOAT, ARRAY_BUFFER)
        if m.colors is not None:
            attrs["COLOR_0"] = b.accessor(m.colors.astype(np.uint8), GL_UNSIGNED_BYTE, ARRAY_BUFFER, normalized=True)
        for name, arr in m.custom.items():
            attrs[name] = b.accessor(arr.astype(np.float32).reshape(-1), GL_FLOAT, ARRAY_BUFFER, minmax=True)
        idx = b.accessor(m.indices.astype(np.uint32).reshape(-1), GL_UNSIGNED_INT, ELEMENT_ARRAY_BUFFER)
        mat: dict[str, Any] = {
            "name": f"{m.name}_mat",
            "pbrMetallicRoughness": {"baseColorFactor": list(m.base_color), "metallicFactor": 0.0, "roughnessFactor": m.roughness},
        }
        if m.double_sided:
            mat["doubleSided"] = True
        if m.texture_jpeg is not None:
            img_bv = b.view(m.texture_jpeg, None)
            images.append({"bufferView": img_bv, "mimeType": m.texture_mime})
            textures.append({"source": len(images) - 1, "sampler": 1 if m.texture_repeat else 0})
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(textures) - 1}
        materials.append(mat)
        gl_meshes.append({"name": m.name, "primitives": [{"attributes": attrs, "indices": idx, "material": len(materials) - 1, "mode": 4}]})
        nodes.append({"name": m.name, "mesh": len(gl_meshes) - 1})
        tris += m.triangle_count

    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "redraw-pipeline"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gl_meshes,
        "materials": materials,
        "accessors": b.accessors,
        "bufferViews": b.buffer_views,
        "buffers": [{"byteLength": len(b.bin)}],
    }
    if images:
        gltf["images"] = images
        gltf["textures"] = textures
        gltf["samplers"] = [
            {"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071},
            {"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 10497},
        ]
    if extras:
        gltf["asset"]["extras"] = extras
    if not nodes:
        # glTF requires at least valid structure; keep an empty scene.
        gltf.pop("meshes")
        gltf.pop("materials")
        gltf.pop("accessors")
        gltf.pop("bufferViews")
        gltf.pop("buffers")

    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    binb = bytes(b.bin) + b"\x00" * ((4 - len(b.bin) % 4) % 4)
    chunks = struct.pack("<II", len(js), 0x4E4F534A) + js
    if nodes:
        chunks += struct.pack("<II", len(binb), 0x004E4942) + binb
    header = struct.pack("<III", 0x46546C67, 2, 12 + len(chunks))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + chunks)
    return tris


def read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"{path} is not a glTF 2.0 binary")
    off = 12
    gltf: dict[str, Any] = {}
    binb = b""
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8 : off + 8 + clen]
        if ctype == 0x4E4F534A:
            gltf = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            binb = chunk
        off += 8 + clen
    return gltf, binb


def read_accessor(gltf: dict[str, Any], binb: bytes, idx: int) -> np.ndarray:
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    dtype = _DTYPE_BY_COMPONENT[acc["componentType"]]
    width = _WIDTH_BY_TYPE[acc["type"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    arr = np.frombuffer(binb, dtype=dtype, count=acc["count"] * width, offset=start)
    return arr.reshape(acc["count"], width) if width > 1 else arr
