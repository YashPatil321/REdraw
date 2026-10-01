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


def id_attribute(ids: np.ndarray) -> np.ndarray:
    """Per-vertex building ids as uint16 when they fit (exact through Draco), else float32."""
    a = np.asarray(ids)
    if len(a) == 0 or float(a.max()) <= 65535:
        return a.astype(np.uint16)
    return a.astype(np.float32)


@dataclass
class MeshData:
    """One mesh primitive. Arrays are per vertex unless noted."""

    name: str
    positions: np.ndarray  # (N, 3) float32
    indices: np.ndarray  # (M,) uint32, triangles
    normals: np.ndarray | None = None  # (N, 3) float32
    uvs: np.ndarray | None = None  # (N, 2) float32
    colors: np.ndarray | None = None  # (N, 4) uint8 (normalized) -> COLOR_0
    custom: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (N,) float32, uint16 or uint8 (kept as is)
    texture_jpeg: bytes | None = None
    texture_mime: str = "image/jpeg"
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    roughness: float = 1.0
    double_sided: bool = False
    texture_repeat: bool = False  # REPEAT wrapping along v (road markings) instead of CLAMP (terrain)
    texture_repeat_both: bool = False  # REPEAT in u and v (imported hero model textures)

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
            a = np.asarray(arr).reshape(-1)
            if a.dtype == np.uint8:
                attrs[name] = b.accessor(a, GL_UNSIGNED_BYTE, ARRAY_BUFFER, minmax=True)
            elif a.dtype == np.uint16:
                attrs[name] = b.accessor(a, GL_UNSIGNED_SHORT, ARRAY_BUFFER, minmax=True)
            else:
                attrs[name] = b.accessor(a.astype(np.float32), GL_FLOAT, ARRAY_BUFFER, minmax=True)
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
            textures.append({"source": len(images) - 1, "sampler": 2 if m.texture_repeat_both else (1 if m.texture_repeat else 0)})
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(textures) - 1}
        materials.append(mat)
        gl_meshes.append({"name": m.name, "primitives": [{"attributes": attrs, "indices": idx, "material": len(materials) - 1, "mode": 4}]})
        nodes.append({"name": m.name, "mesh": len(gl_meshes) - 1})
        tris += m.triangle_count

    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "redraw-pipeline"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))} if nodes else {}],
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
            {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497},
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
        gltf.pop("nodes")

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


_NP_BY_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NORM_DIV = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}


def read_accessor_any(gltf: dict[str, Any], binb: bytes, idx: int) -> np.ndarray:
    """Accessor -> float64/uint array, honoring byteStride and `normalized`."""
    acc = gltf["accessors"][idx]
    if "bufferView" not in acc:
        raise ValueError("sparse / bufferView-less accessors are not supported")
    bv = gltf["bufferViews"][acc["bufferView"]]
    if bv.get("buffer", 0) != 0:
        raise ValueError("only single-buffer GLB files are supported")
    comp = acc["componentType"]
    dtype = np.dtype(_NP_BY_COMPONENT[comp])
    width = _WIDTH_BY_TYPE.get(acc["type"])
    if width is None:
        raise ValueError(f"unsupported accessor type {acc['type']}")
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    stride = bv.get("byteStride") or dtype.itemsize * width
    if stride == dtype.itemsize * width:
        arr = np.frombuffer(binb, dtype=dtype, count=count * width, offset=start).reshape(count, width)
    else:
        raw = np.frombuffer(binb, dtype=np.uint8, count=stride * (count - 1) + dtype.itemsize * width, offset=start)
        rows = np.lib.stride_tricks.as_strided(raw, shape=(count, dtype.itemsize * width), strides=(stride, 1))
        arr = np.ascontiguousarray(rows).view(dtype).reshape(count, width)
    if acc.get("normalized") and comp in _NORM_DIV:
        return arr.astype(np.float64) / _NORM_DIV[comp]
    return arr


def _node_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T  # column-major
    t = np.asarray(node.get("translation", [0, 0, 0]), dtype=np.float64)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    s = np.asarray(node.get("scale", [1, 1, 1]), dtype=np.float64)
    m = np.eye(4)
    m[:3, :3] = r * s[None, :]
    m[:3, 3] = t
    return m


def load_glb_meshes(path: Path) -> list[MeshData]:
    """Read every triangle primitive of a GLB, baked to world space (node transforms applied).

    Keeps per-primitive material base color, roughness, embedded base color texture,
    vertex colors and UVs. Draco/meshopt compressed files are rejected with a clear error.
    """
    gltf, binb = read_glb(path)
    used = set(gltf.get("extensionsRequired", [])) | set(gltf.get("extensionsUsed", []))
    for ext in ("KHR_draco_mesh_compression", "EXT_meshopt_compression", "KHR_mesh_quantization"):
        if ext in used:
            raise ValueError(f"{path}: {ext} is not supported for hero models; export an uncompressed GLB")
    out: list[MeshData] = []
    nodes = gltf.get("nodes", [])
    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)] if gltf.get("scenes") else {"nodes": list(range(len(nodes)))}

    def walk(ni: int, parent: np.ndarray) -> None:
        node = nodes[ni]
        m = parent @ _node_matrix(node)
        if "mesh" in node:
            nm = np.linalg.inv(m[:3, :3]).T
            for k, prim in enumerate(gltf["meshes"][node["mesh"]]["primitives"]):
                if prim.get("mode", 4) != 4 or "POSITION" not in prim["attributes"]:
                    continue
                a = prim["attributes"]
                pos = read_accessor_any(gltf, binb, a["POSITION"]).astype(np.float64)
                pos = pos @ m[:3, :3].T + m[:3, 3]
                nrm = None
                if "NORMAL" in a:
                    nrm = read_accessor_any(gltf, binb, a["NORMAL"]).astype(np.float64) @ nm.T
                    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                uv = read_accessor_any(gltf, binb, a["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in a else None
                col = None
                if "COLOR_0" in a:
                    c = read_accessor_any(gltf, binb, a["COLOR_0"])
                    c = c.astype(np.float64) if c.dtype.kind == "f" else c.astype(np.float64)
                    if c.shape[1] == 3:
                        c = np.column_stack([c, np.ones(len(c))])
                    col = np.clip(np.round(c * 255.0), 0, 255).astype(np.uint8)
                idx = read_accessor_any(gltf, binb, prim["indices"]).reshape(-1).astype(np.uint32) if "indices" in prim else np.arange(len(pos), dtype=np.uint32)
                if np.linalg.det(m[:3, :3]) < 0:
                    idx = idx.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1)
                md = MeshData(name=f"{Path(path).stem}_{ni}_{k}", positions=pos, indices=idx, normals=nrm, uvs=uv, colors=col)
                # keep application-specific attributes (_MAT, _VARIANT, ...) of hero models
                for an, ai in a.items():
                    if not an.startswith("_") or an == "_BUILDING_ID":
                        continue
                    arr = read_accessor_any(gltf, binb, ai)
                    arr = arr[:, 0] if arr.ndim == 2 else arr
                    if arr.dtype.kind == "f" and len(arr) and np.all(np.mod(arr, 1.0) == 0) and arr.min() >= 0 and arr.max() <= 255:
                        arr = arr.astype(np.uint8)
                    md.custom[an] = np.asarray(arr)
                if "material" in prim:
                    mat = gltf["materials"][prim["material"]]
                    pbr = mat.get("pbrMetallicRoughness", {})
                    md.base_color = tuple(pbr.get("baseColorFactor", [1, 1, 1, 1]))  # type: ignore[assignment]
                    md.roughness = float(pbr.get("roughnessFactor", 1.0))
                    md.double_sided = bool(mat.get("doubleSided", False))
                    tex = pbr.get("baseColorTexture")
                    if tex is not None and uv is not None:
                        img = gltf["images"][gltf["textures"][tex["index"]]["source"]]
                        if "bufferView" in img:
                            bv = gltf["bufferViews"][img["bufferView"]]
                            o = bv.get("byteOffset", 0)
                            md.texture_jpeg = bytes(binb[o : o + bv["byteLength"]])
                            md.texture_mime = img.get("mimeType", "image/png")
                            md.texture_repeat_both = True
                out.append(md)
        for ch in node.get("children", []):
            walk(ch, m)

    for ni in scene.get("nodes", []):
        walk(ni, np.eye(4))
    return out
