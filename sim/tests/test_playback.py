"""RDPB v1 round trip and the shared small fixture file."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from sim.playback import PlaybackFormatError, decode_playback, encode_playback

FIXTURE = Path(__file__).parent / "fixtures" / "playback_small.rdpb"
FIXTURE_JSON = Path(__file__).parent / "fixtures" / "playback_small.json"


def make_small_playback() -> tuple[bytes, dict]:
    """Deterministic tiny playback: 4 bins, 5 edges, 2 entrances, 3 trajectories."""
    n_bins, n_edges = 4, 5
    vc = (np.arange(n_bins * n_edges, dtype=np.float32) / 10.0).reshape(n_bins, n_edges)
    speed = (50.0 - np.arange(n_bins * n_edges, dtype=np.float32)).reshape(n_bins, n_edges)
    queue = np.array([[0, 1], [2.5, 3], [4, 0], [0, 0]], dtype=np.float32)
    offsets = np.array([0, 2, 5, 6], dtype=np.uint32)
    kind = np.array([0, 1, 2], dtype=np.uint32)
    edge = np.array([0, 1, 1, 2, 3, 4], dtype=np.int32)
    enter = np.array([27000, 27030, 27100, 27160, 27200, 28000], dtype=np.float32)
    exit_ = np.array([27030, 27060, 27160, 27200, 27500, 28100], dtype=np.float32)
    time = {"bin_start_s": 27000, "bin_s": 300, "n_bins": n_bins, "report_start_s": 27000, "report_end_s": 28200}
    buf = encode_playback(plan_id="fixture", seed=0, synthetic=True, time=time, edge_vc=vc, edge_speed_kph=speed,
                          queue_len=queue, entrance_ids=["del_norte_hs/main_dropoff", "oak_valley_ms/main_dropoff"],
                          traj_offsets=offsets, traj_kind=kind, traj_edge=edge, traj_enter_s=enter, traj_exit_s=exit_)
    arrays = {"edge_vc": vc.ravel().tolist(), "edge_speed_kph": speed.ravel().tolist(), "queue_len": queue.ravel().tolist(),
              "traj_offsets": offsets.tolist(), "traj_kind": kind.tolist(), "traj_edge": edge.tolist(),
              "traj_enter_s": enter.tolist(), "traj_exit_s": exit_.tolist()}
    return buf, arrays


def test_round_trip_random():
    rng = np.random.default_rng(0)
    B, E, K = 48, 37, 3
    offs = np.r_[0, np.cumsum(rng.integers(1, 9, 20))].astype(np.uint32)
    P = int(offs[-1])
    time = {"bin_start_s": 21600, "bin_s": 300, "n_bins": B, "report_start_s": 23400, "report_end_s": 34200}
    args = dict(edge_vc=rng.random((B, E)).astype(np.float32), edge_speed_kph=rng.random((B, E)).astype(np.float32) * 80,
                queue_len=rng.random((B, K)).astype(np.float32), traj_offsets=offs,
                traj_kind=rng.integers(0, 5, 20).astype(np.uint32), traj_edge=rng.integers(0, E, P).astype(np.int32),
                traj_enter_s=rng.random(P).astype(np.float32) * 1e4, traj_exit_s=rng.random(P).astype(np.float32) * 1e4)
    buf = encode_playback(plan_id="p1", seed=7, synthetic=False, time=time, entrance_ids=["a/x", "b/y", "c/z"], **args)
    assert buf[:4] == b"RDPB"
    version, hlen = struct.unpack_from("<II", buf, 4)
    assert version == 1
    data_start = 12 + hlen + (-(12 + hlen)) % 4
    assert data_start % 4 == 0
    d = decode_playback(buf)
    h = d["header"]
    assert h["n_edges"] == E and h["n_bins"] == B and h["n_trajectories"] == 20 and h["n_points"] == P
    assert all(s["offset"] % 4 == 0 for s in h["sections"])
    assert np.array_equal(d["edge_vc"], args["edge_vc"].ravel())
    assert np.array_equal(d["edge_speed_kph"], args["edge_speed_kph"].ravel())
    assert np.array_equal(d["queue_len"], args["queue_len"].ravel())
    for k in ("traj_offsets", "traj_kind", "traj_edge", "traj_enter_s", "traj_exit_s"):
        assert np.array_equal(d[k], args[k])
    # bin-major indexing: b * n_edges + e
    assert d["edge_vc"][5 * E + 3] == args["edge_vc"][5, 3]
    assert len(buf) == data_start + sum(s["count"] * 4 for s in h["sections"])


def test_unknown_sections_are_ignored_and_bad_magic_fails():
    buf, _ = make_small_playback()
    d = decode_playback(buf)
    assert "edge_vc" in d
    with pytest.raises(PlaybackFormatError):
        decode_playback(b"NOPE" + buf[4:])
    time = {"bin_start_s": 0, "bin_s": 300, "n_bins": 1, "report_start_s": 0, "report_end_s": 300}
    extra = encode_playback(plan_id="x", seed=0, synthetic=True, time=time, edge_vc=np.zeros((1, 2), np.float32),
                            edge_speed_kph=np.zeros((1, 2), np.float32), queue_len=np.zeros((1, 0), np.float32), entrance_ids=[],
                            traj_offsets=np.zeros(1, np.uint32), traj_kind=np.zeros(0, np.uint32), traj_edge=np.zeros(0, np.int32),
                            traj_enter_s=np.zeros(0, np.float32), traj_exit_s=np.zeros(0, np.float32),
                            extra_sections={"future_thing": np.arange(3, dtype=np.int32)})
    assert np.array_equal(decode_playback(extra)["future_thing"], [0, 1, 2])


def test_shared_fixture_file_matches_generator():
    buf, arrays = make_small_playback()
    if not FIXTURE.exists():  # regenerate (tests own the fixture)
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_bytes(buf)
        FIXTURE_JSON.write_text(json.dumps({"header": decode_playback(buf)["header"], "arrays": arrays}, indent=1))
    assert FIXTURE.read_bytes() == buf
    d = decode_playback(FIXTURE.read_bytes())
    for k, v in arrays.items():
        assert np.allclose(d[k], v)
    assert json.loads(FIXTURE_JSON.read_text())["arrays"] == arrays


def test_engine_playback_decodes(world):
    from sim.engine import run_seed
    from sim.playback import playback_from_seed

    r = run_seed(world, 0, keep_playback=True)
    buf = playback_from_seed(r.playback, plan_id="baseline", seed=0, synthetic=True, time=world.time.as_dict())
    d = decode_playback(buf)
    h = d["header"]
    assert h["n_edges"] == world.net.n_edges and 0 < h["n_trajectories"] <= 3000
    assert len(d["queue_len"]) == h["n_bins"] * len(h["entrance_ids"])
    off = d["traj_offsets"]
    assert off[0] == 0 and off[-1] == h["n_points"] and np.all(np.diff(off) >= 1)
    assert np.all(d["traj_exit_s"] >= d["traj_enter_s"])
    assert d["traj_edge"].min() >= 0 and d["traj_edge"].max() < world.net.n_edges
    # consecutive edges of a trajectory are connected (v of one == u of next) and times chain
    net = world.net
    for i in range(min(200, h["n_trajectories"])):
        a, b = off[i], off[i + 1]
        es = d["traj_edge"][a:b]
        if len(es) > 1:
            assert np.all(net.ev[es[:-1]] == net.eu[es[1:]])
            assert np.all(d["traj_enter_s"][a + 1 : b] >= d["traj_exit_s"][a : b - 1] - 1e-3)
