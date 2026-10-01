"""RDPB v1 traffic playback encoder/decoder (docs/playback_format.md).

Layout: "RDPB" | uint32 version | uint32 header_len | JSON header | zero pad to 4 |
data region of sections (little-endian, each at a 4-byte aligned offset from
the data region start). Readers look sections up by name and ignore unknown ones.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

MAGIC = b"RDPB"
VERSION = 1
DEFAULT_KINDS = {"0": "car", "1": "dropoff_car", "2": "shuttle", "3": "school_bus", "4": "carpool"}
_DTYPES = {"float32": "<f4", "uint32": "<u4", "int32": "<i4"}


class PlaybackFormatError(ValueError):
    pass


def encode_playback(
    *,
    plan_id: str,
    seed: int,
    synthetic: bool,
    time: dict[str, int],
    edge_vc: np.ndarray,
    edge_speed_kph: np.ndarray,
    queue_len: np.ndarray,
    entrance_ids: list[str],
    traj_offsets: np.ndarray,
    traj_kind: np.ndarray,
    traj_edge: np.ndarray,
    traj_enter_s: np.ndarray,
    traj_exit_s: np.ndarray,
    kinds: dict[str, str] | None = None,
    extra_sections: dict[str, np.ndarray] | None = None,
    extra_header: dict[str, Any] | None = None,
) -> bytes:
    """Encode one playback. ``edge_vc``/``edge_speed_kph`` are (n_bins, n_edges); ``queue_len`` (n_bins, n_entrances)."""
    n_bins = int(time["n_bins"])
    edge_vc = np.asarray(edge_vc, dtype="<f4")
    edge_speed_kph = np.asarray(edge_speed_kph, dtype="<f4")
    if edge_vc.ndim != 2 or edge_vc.shape[0] != n_bins or edge_speed_kph.shape != edge_vc.shape:
        raise PlaybackFormatError("edge arrays must be (n_bins, n_edges)")
    n_edges = int(edge_vc.shape[1])
    queue_len = np.asarray(queue_len, dtype="<f4").reshape(n_bins, len(entrance_ids))
    offs = np.asarray(traj_offsets, dtype="<u4")
    n_traj = len(offs) - 1
    n_points = int(offs[-1]) if len(offs) else 0
    if len(offs) == 0:
        offs = np.zeros(1, dtype="<u4")
        n_traj = 0
    sections_data: list[tuple[str, str, np.ndarray]] = [
        ("edge_vc", "float32", edge_vc.ravel()),
        ("edge_speed_kph", "float32", edge_speed_kph.ravel()),
        ("queue_len", "float32", queue_len.ravel()),
        ("traj_offsets", "uint32", offs),
        ("traj_kind", "uint32", np.asarray(traj_kind, dtype="<u4")),
        ("traj_edge", "int32", np.asarray(traj_edge, dtype="<i4")),
        ("traj_enter_s", "float32", np.asarray(traj_enter_s, dtype="<f4")),
        ("traj_exit_s", "float32", np.asarray(traj_exit_s, dtype="<f4")),
    ]
    for name, arr in (extra_sections or {}).items():
        dt = {np.dtype("float32"): "float32", np.dtype("uint32"): "uint32", np.dtype("int32"): "int32"}.get(np.asarray(arr).dtype)
        if dt is None:
            raise PlaybackFormatError(f"extra section {name}: dtype must be float32/uint32/int32")
        sections_data.append((name, dt, np.asarray(arr, dtype=_DTYPES[dt]).ravel()))
    if len(sections_data[4][2]) != n_traj or len(sections_data[6][2]) != n_points or len(sections_data[7][2]) != n_points or len(sections_data[5][2]) != n_points:
        raise PlaybackFormatError("trajectory arrays are inconsistent with traj_offsets")
    sections = []
    off = 0
    for name, dt, arr in sections_data:
        sections.append({"name": name, "dtype": dt, "offset": off, "count": int(arr.size)})
        off += arr.size * 4
    header = {
        "version": VERSION, "plan_id": str(plan_id), "seed": int(seed), "synthetic": bool(synthetic),
        "bin_start_s": int(time["bin_start_s"]), "bin_s": int(time["bin_s"]), "n_bins": n_bins,
        "report_start_s": int(time["report_start_s"]), "report_end_s": int(time["report_end_s"]),
        "n_edges": n_edges, "entrance_ids": list(entrance_ids), "n_trajectories": int(n_traj), "n_points": n_points,
        "kinds": kinds or DEFAULT_KINDS, "sections": sections,
    }
    if extra_header:
        header.update({k: v for k, v in extra_header.items() if k not in header})
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-(12 + len(hb))) % 4
    parts = [MAGIC, struct.pack("<II", VERSION, len(hb)), hb, b"\x00" * pad]
    parts.extend(arr.astype(_DTYPES[dt], copy=False).tobytes() for _, dt, arr in sections_data)
    return b"".join(parts)


def decode_playback(buf: bytes) -> dict[str, Any]:
    """Decode RDPB bytes -> {"header": {...}, "<section name>": np.ndarray, ...} (unknown sections included)."""
    if len(buf) < 12 or buf[:4] != MAGIC:
        raise PlaybackFormatError("not an RDPB file")
    version, hlen = struct.unpack_from("<II", buf, 4)
    if version != VERSION:
        raise PlaybackFormatError(f"unsupported RDPB version {version}")
    header = json.loads(buf[12 : 12 + hlen].decode("utf-8"))
    data_start = 12 + hlen + ((-(12 + hlen)) % 4)
    out: dict[str, Any] = {"header": header}
    for s in header["sections"]:
        dt = _DTYPES.get(s["dtype"])
        if dt is None:
            continue
        start = data_start + int(s["offset"])
        if start % 4:
            raise PlaybackFormatError(f"section {s['name']} is not 4-byte aligned")
        out[s["name"]] = np.frombuffer(buf, dtype=dt, count=int(s["count"]), offset=start)
    return out


def playback_from_seed(seed_playback: dict[str, Any], *, plan_id: str, seed: int, synthetic: bool, time: dict[str, int],
                       kinds: dict[str, str] | None = None) -> bytes:
    tr = seed_playback["traj"]
    return encode_playback(
        plan_id=plan_id, seed=seed, synthetic=synthetic, time=time,
        edge_vc=seed_playback["edge_vc"], edge_speed_kph=seed_playback["edge_speed_kph"],
        queue_len=seed_playback["queue_len"], entrance_ids=seed_playback["entrance_ids"],
        traj_offsets=tr["offsets"], traj_kind=tr["kind"], traj_edge=tr["edge"], traj_enter_s=tr["enter"],
        traj_exit_s=tr["exit"], kinds=kinds,
    )
