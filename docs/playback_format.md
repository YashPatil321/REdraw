# Traffic playback binary format

Format id: `RDPB`, version **1**. Both the web client and the future Unreal
client parse exactly this layout (spec 13A.4). Served by
`GET /plans/{id}/playback` and `GET /baseline/playback` with
`Content-Type: application/octet-stream`.

All multi-byte values are **little-endian**. All sections start on a 4-byte
boundary.

## Layout

```
offset  size            field
0       4               magic: ASCII "RDPB"
4       4               uint32 version (= 1)
8       4               uint32 header_len (bytes of JSON, before padding)
12      header_len      UTF-8 JSON header
...     0-3             zero padding so the data region starts at a multiple of 4
D       ...             data region (sections, see header.sections)
```

`D = 12 + header_len` rounded up to the next multiple of 4.

## Header JSON

```json
{
  "version": 1,
  "plan_id": "baseline",
  "seed": 0,
  "synthetic": false,
  "bin_start_s": 21600,
  "bin_s": 300,
  "n_bins": 48,
  "report_start_s": 23400,
  "report_end_s": 34200,
  "n_edges": 8123,
  "entrance_ids": ["del_norte_hs/main_dropoff", "oak_valley_ms/main_dropoff"],
  "n_trajectories": 3000,
  "n_points": 120000,
  "kinds": {"0": "car", "1": "dropoff_car", "2": "shuttle", "3": "school_bus", "4": "carpool"},
  "sections": [
    {"name": "edge_vc",        "dtype": "float32", "offset": 0,       "count": 389904},
    {"name": "edge_speed_kph", "dtype": "float32", "offset": 1559616, "count": 389904},
    {"name": "queue_len",      "dtype": "float32", "offset": 3119232, "count": 336},
    {"name": "traj_offsets",   "dtype": "uint32",  "offset": 3120576, "count": 3001},
    {"name": "traj_kind",      "dtype": "uint32",  "offset": 3132580, "count": 3000},
    {"name": "traj_edge",      "dtype": "int32",   "offset": 3144580, "count": 120000},
    {"name": "traj_enter_s",   "dtype": "float32", "offset": 3624580, "count": 120000},
    {"name": "traj_exit_s",    "dtype": "float32", "offset": 4104580, "count": 120000}
  ]
}
```

- `offset` is in bytes from the start of the data region `D`; always a multiple of 4.
- Readers must look sections up **by name** and ignore unknown sections
  (forward compatible). Writers may add sections without bumping the version;
  removing or changing a section's meaning bumps the version.
- Times are **seconds since local midnight** (06:00 = 21600).
- Bin `b` covers `[bin_start_s + b*bin_s, bin_start_s + (b+1)*bin_s)`.

## Sections

| name | dtype | count | meaning |
| --- | --- | --- | --- |
| edge_vc | float32 | n_bins * n_edges | volume/capacity ratio. Index `b * n_edges + e` (bin-major). |
| edge_speed_kph | float32 | n_bins * n_edges | congested speed. Same indexing. |
| queue_len | float32 | n_bins * n_entrances | drop-off queue length in **cars** at the end of each bin. Index `b * n_entrances + k`, `k` indexes `entrance_ids`. |
| traj_offsets | uint32 | n_trajectories + 1 | trajectory `i` uses points `[traj_offsets[i], traj_offsets[i+1])`. |
| traj_kind | uint32 | n_trajectories | key into `kinds`. |
| traj_edge | int32 | n_points | `edge_idx` traversed (see data_contract: network_edges). |
| traj_enter_s | float32 | n_points | time the vehicle enters that edge. |
| traj_exit_s | float32 | n_points | time the vehicle leaves that edge. |

A vehicle at time `t` with `enter_s <= t < exit_s` on edge `e` is at fraction
`(t - enter_s) / (exit_s - enter_s)` along the edge polyline (from `u` to `v`).
Edge polylines come from `GET /world/network`. Vehicles waiting in a drop-off
queue appear as a stretched final edge (slow movement), which is intended.

## Reference implementations

- Writer: `sim/playback.py` (`encode_playback`, `decode_playback`).
- Reader: `client/src/traffic/playback.ts` (`parsePlayback`).
- Both have round-trip tests against the same fixture.
