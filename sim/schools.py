"""School drop-off queue model (spec 6.5).

Each entrance is a deterministic fluid queue fed by the drop-off arrivals the
assignment produces. Service rate ``mu = curb_spots / unload_seconds`` cars/s.
For vehicles sorted by arrival time ``A_k`` with weight ``w_k`` (carpools are
fractional vehicles) the service completion of the fluid server is

    C_k = max(A_k, C_{k-1}) + w_k / mu      (vectorized as a running max)

service starts at ``B_k = C_k - w_k / mu``; the car waits ``B_k - A_k`` in the
queue and then occupies a curb spot for ``unload_seconds``.

Queue length (cars waiting, not yet at the curb) times ``car_length_m`` is the
spillback. Balking: a parent who reaches the line when the expected wait is
longer than ``sim_engine.dropoff_balk_wait_min`` drops the kid off on a nearby
street instead (``informal_dropoff_stop_s`` stop, kid walks
``informal_dropoff_walk_min`` extra), so the formal queue is bounded by
behaviour, as observed at real schools. When spillback exceeds the approach edge length the approach
edge's capacity is multiplied by ``queue_capacity_reduction_per_spill`` and
the edges feeding the approach get ``upstream_spill_delay_s_per_car`` per
spilled car, in that 5-minute bin. These effects are fed back inside the
assignment's MSA loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.assumptions import Af
from sim.world import Network, TimeGrid


@dataclass
class QueueOutput:
    wait_s: np.ndarray  # per arrival, in input order
    start_s: np.ndarray  # service start (car reaches the curb)
    done_s: np.ndarray  # car leaves the curb (start + unload)
    queue_end: np.ndarray  # (n_bins,) cars waiting at the end of each bin
    queue_max: np.ndarray  # (n_bins,) max cars waiting during the bin
    spill_m: np.ndarray  # (n_bins,) max spillback (m) during the bin
    avg_wait_s: np.ndarray  # (n_bins,) mean wait of vehicles arriving in the bin (NaN if none)
    arrivals: np.ndarray  # (n_bins,) vehicles (weighted) joining the line in the bin
    balked: np.ndarray  # per arrival: dropped off informally on a nearby street instead of queueing


def service_rate(curb_spots: float, unload_s: float) -> float:
    """Cars per second an entrance can unload."""
    return max(float(curb_spots), 1e-6) / max(float(unload_s), 1e-6)


def _serve_with_balking(A_: np.ndarray, s: np.ndarray, balk_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sequential fluid server where arrivals facing a wait > their balk_s leave (informal drop-off)."""
    n = len(A_)
    A_l, s_l, bk_l = A_.tolist(), s.tolist(), np.broadcast_to(balk_s, (n,)).tolist()
    B = np.empty(n)
    balked = np.zeros(n, dtype=bool)
    c = -np.inf
    last_b = -np.inf
    for k in range(n):
        a = A_l[k]
        wait = c - a if c > a else 0.0
        if wait > bk_l[k]:
            balked[k] = True
            B[k] = max(last_b, a) if last_b > -np.inf else a  # keeps B sorted; zero weight
            continue
        B[k] = a + wait
        last_b = B[k]
        c = B[k] + s_l[k]
    return B, balked


def run_queue(arr_t: np.ndarray, w: np.ndarray, curb_spots: float, unload_s: float, tg: TimeGrid,
              balk_wait_s: float | np.ndarray | None = None) -> QueueOutput:
    """Fluid curb queue. With ``balk_wait_s`` (scalar or per arrival) set, arrivals facing a longer wait balk."""
    n_b = tg.n_bins
    car_len = Af("schools.car_length_m")
    if len(arr_t) == 0:
        z = np.zeros(n_b)
        return QueueOutput(np.zeros(0), np.zeros(0), np.zeros(0), z, z.copy(), z.copy(), np.full(n_b, np.nan), z.copy(),
                           np.zeros(0, bool))
    mu = service_rate(curb_spots, unload_s)
    order = np.argsort(arr_t, kind="stable")
    A_ = arr_t[order].astype(np.float64)
    W = w[order].astype(np.float64)
    s = W / mu
    if balk_wait_s is None:
        S = np.cumsum(s)
        C = S + np.maximum.accumulate(A_ - (S - s))
        B = C - s
        balked_sorted = np.zeros(len(A_), dtype=bool)
    else:
        bw = np.broadcast_to(np.asarray(balk_wait_s, dtype=np.float64), arr_t.shape)[order]
        B, balked_sorted = _serve_with_balking(A_, s, bw)
    wait_sorted = np.where(balked_sorted, 0.0, np.maximum(B - A_, 0.0))
    W = np.where(balked_sorted, 0.0, W)  # balked cars never join the line
    cumA = np.cumsum(W)
    cumB = cumA  # B is nondecreasing in the same order (balked cars have zero weight)
    started_at_arrival = np.where(
        (k := np.searchsorted(B, A_, side="right")) > 0, cumB[np.maximum(k - 1, 0)], 0.0)
    q_after = cumA - started_at_arrival
    edges = tg.bin_start_s + np.arange(n_b + 1) * tg.bin_s
    ia = np.searchsorted(A_, edges[1:], side="right")
    ib = np.searchsorted(B, edges[1:], side="right")
    arrived = np.where(ia > 0, cumA[np.maximum(ia - 1, 0)], 0.0)
    started = np.where(ib > 0, cumB[np.maximum(ib - 1, 0)], 0.0)
    q_end = np.maximum(arrived - started, 0.0)
    b = tg.bin_of(A_)
    q_max = np.zeros(n_b)
    np.maximum.at(q_max, b, q_after)
    q_max = np.maximum(q_max, np.r_[0.0, q_end[:-1]])
    q_max = np.maximum(q_max, q_end)
    arrivals = np.bincount(b, weights=W, minlength=n_b)
    wsum = np.bincount(b, weights=W * wait_sorted, minlength=n_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_wait = np.where(arrivals > 0, wsum / np.maximum(arrivals, 1e-12), np.nan)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return QueueOutput(
        wait_s=wait_sorted[inv], start_s=B[inv], done_s=B[inv] + unload_s,
        queue_end=q_end, queue_max=q_max, spill_m=q_max * car_len, avg_wait_s=avg_wait, arrivals=arrivals,
        balked=balked_sorted[inv],
    )


def upstream_edges(net: Network, approach_edge: int) -> np.ndarray:
    """Edges feeding the approach edge's upstream node (where a spilled queue backs up), no U-turns."""
    u = net.eu[approach_edge]
    v = net.ev[approach_edge]
    return np.nonzero((net.ev == u) & (net.eu != v))[0]


def spill_effects(net: Network, approach_edge: int, spill_m: np.ndarray, capf: np.ndarray, extra: np.ndarray) -> np.ndarray:
    """Apply spillback effects of one entrance in place on (B, E) capacity factor and extra delay arrays.

    Returns the per-bin extra delay (s) added to each upstream edge, so the assignment can
    exempt this entrance's own drop-off cars: their time in the line is the point-queue wait,
    and charging them the spillback delay as well would count the same queue twice.
    """
    L = float(net.length_m[approach_edge])
    over = spill_m > L
    per_bin = np.zeros(len(spill_m))
    if not over.any():
        return per_bin
    car_len = Af("schools.car_length_m")
    red = Af("schools.queue_capacity_reduction_per_spill")
    per_car = Af("schools.upstream_spill_delay_s_per_car")
    capf[over, approach_edge] *= red
    up = upstream_edges(net, approach_edge)
    if len(up):
        per_bin[over] = per_car * (spill_m[over] - L) / car_len
        extra[np.ix_(np.nonzero(over)[0], up)] += per_bin[over][:, None]
    return per_bin
