"""End-to-end streaming pipeline with per-stage latency instrumentation.

The challenge scores real-time performance against a 40 ms budget, so latency is
measured the way a reviewer would want it measured: wall-clock per processed
window, broken out by stage, reported as a distribution rather than a mean.

## Multi-scale detection

Detection runs at one or more *scales*. A scale is an accumulation depth: how
many consecutive filtered windows the clusterer sees at once. Depth 1 is the
original behaviour — cluster this window, decide, move on.

Depth matters because integration time is a signal-to-noise choice, and the
right choice depends on how fast the target crosses the sensor. Signal grows
linearly with integration time while the target stays inside one aperture;
background grows linearly too, but once the target smears the aperture has to
grow with it, so background starts growing faster. There is an optimum, and on
real telescope data it is two orders of magnitude apart for a tracked RSO
(6 px/s, optimum ~100-250 ms) and a transiting object (640 px/s, optimum
10 ms). Measured, not assumed: `docs/CALIBRATION.md`.

Running both scales concurrently costs one extra clustering pass over a
slightly longer buffer, which is cheap, and it keeps the update rate at the
window cadence — a depth-25 scale still emits a decision every 10 ms, it just
decides using the last 250 ms of evidence.

Cross-scale duplicates are merged by preferring the more compact cluster: the
same target seen at two scales gives a tight blob at the matched scale and a
smear at the mismatched one, and the tight blob is the better position fix.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
from time import perf_counter
import numpy as np

from . import filters as F
from .detect import GridClusterDetector
from .track import PDATracker
from .feast import FEAST
from .profiles import ScaleSpec


@dataclass
class Config:
    window_us: int = 10_000          # 10 ms slice -> 100 Hz update rate
    baf_dt_us: int = 5_000
    baf_radius: int = 1
    baf_include_self: bool = False
    refractory_us: int = 800
    hot_sigma: float = 8.0
    hot_min_count: int = 50
    hot_neighbour_ratio: float = 0.0   # >0 spares count outliers with neighbour support
    cell: int = 8
    min_events: int = 5
    min_cluster_events: int = 10
    max_cluster_cells: int = 64
    scales: tuple | None = None      # None -> one depth-1 scale from the fields above
    merge_px: float = 0.0            # cross-scale duplicate suppression radius
    use_feast: bool = False
    feast_neurons: int = 16
    feast_radius: int = 4
    gate_chi2: float = 16.0
    confirm_hits: int = 3
    max_misses: int = 4
    q_accel: float = 250.0
    r_pos: float = 2.0
    max_assoc_px: float = 25.0
    max_speed_px_s: float = 4000.0

    @classmethod
    def from_profile(cls, name: str, **overrides):
        from . import profiles
        d = profiles.get(name)
        d.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**d)

    def scale_specs(self) -> tuple:
        if self.scales:
            return tuple(self.scales)
        return (ScaleSpec(windows=1, cell=self.cell, min_events=self.min_events,
                          min_cluster_events=self.min_cluster_events,
                          max_cluster_cells=self.max_cluster_cells,
                          label="base"),)


class Pipeline:
    def __init__(self, shape, cfg: Config = None):
        self.shape = shape
        self.cfg = cfg or Config()
        c = self.cfg
        self.filt = F.Pipeline(shape, dt_us=c.baf_dt_us, refractory_us=c.refractory_us,
                               baf_radius=c.baf_radius,
                               baf_include_self=c.baf_include_self,
                               hot_sigma=c.hot_sigma, hot_min_count=c.hot_min_count,
                               hot_neighbour_ratio=c.hot_neighbour_ratio)
        self.feast = FEAST(shape, n_neurons=c.feast_neurons,
                           radius=c.feast_radius) if c.use_feast else None

        self.specs = c.scale_specs()
        self.detectors = [
            GridClusterDetector(shape, cell=s.cell, min_events=s.min_events,
                                min_cluster_events=s.min_cluster_events,
                                max_cluster_cells=s.max_cluster_cells,
                                label=s.label, time_bins=s.time_bins,
                                min_time_bins=s.min_time_bins,
                                snr_k=s.snr_k, bg_cells=s.bg_cells,
                                max_detections=s.max_detections)
            for s in self.specs
        ]
        # One tracker per scale. A 6 px/s tracked RSO and a 640 px/s transit
        # need different process noise and different association gates, and a
        # tracker fed at 4 Hz by the slow scale must not accrue a miss for
        # every window the fast scale filled in.
        self.trackers = [
            PDATracker(gate_chi2=s.gate_chi2, confirm_hits=s.confirm_hits,
                       confirm_window=s.confirm_window, max_misses=s.max_misses,
                       confirm_score=s.confirm_score,
                       min_confirm_hits=s.min_confirm_hits, q_accel=s.q_accel, r_pos=s.r_pos,
                       max_assoc_px=s.max_assoc_px,
                       min_speed_px_s=s.min_speed_px_s,
                       max_speed_px_s=s.max_speed_px_s)
            for s in self.specs
        ]
        # keep track ids globally unique across scales
        for i, tk in enumerate(self.trackers):
            tk._next_id = 1 + i * 1_000_000
        # How far each scale's state estimate trails the object. A scale
        # centroids its whole integration span but stamps the result at the
        # span's end, so its fix sits half an integration behind — and by a
        # different amount at every scale. _merge_tracks needs this to compare
        # two scales at a common instant.
        self.scale_lag_s = [s.windows * self.cfg.window_us / 2e6
                            for s in self.specs]
        self.max_depth = max(s.windows for s in self.specs)
        self.history: deque = deque(maxlen=self.max_depth)
        self._n = 0
        self.timings = {k: [] for k in ("filter", "feast", "detect", "track", "total")}
        self.window_events = []

    def calibrate(self, ev: np.ndarray):
        """Fit the hot-pixel mask on a lead-in slice. Cheap, done once."""
        self.filt.fit_hot_pixels(ev)
        return self

    # -- cross-scale merge ---------------------------------------------------
    def _merge(self, dets):
        """Suppress duplicates across scales, keeping the more compact cluster.

        Ranking is (footprint ascending, event count descending). A target seen
        at its matched scale is a tight blob; the same target seen at a
        mismatched scale is a smear whose centroid lags. Preferring compactness
        picks the better position fix rather than merely the louder one.
        """
        r = self.cfg.merge_px
        if r <= 0 or len(dets) < 2:
            return dets
        order = sorted(dets, key=lambda d: (d.n_cells, -d.n_events))
        kept = []
        for d in order:
            if all((d.x - k.x) ** 2 + (d.y - k.y) ** 2 > r * r for k in kept):
                kept.append(d)
        kept.sort(key=lambda d: -d.n_events)
        return kept

    def _merge_tracks(self, tracks, lags=None):
        """Suppress the same physical object confirmed at two scales at once.

        The comparison is on **lag-corrected** positions. Every scale reports
        the centroid of its whole integration span but stamps it at the span's
        end, so its state estimate trails the object by `v * span/2` — by a
        different amount at each scale. Comparing raw positions therefore fails
        to merge precisely the targets where two scales disagree most: a 1 s
        scale and a 500 ms scale sit 1.75 px apart on a 7 px/s object but 15 px
        apart at 60 px/s, and `merge_px` is 4. Projecting each track forward by
        its own scale's lag puts both at a common instant before the distance
        test, which is what the test always meant to ask.

        The projection is local to the comparison. `tr.x` is never mutated —
        the reported track position and the viewer both read it, and a track
        that jumps forward on merge would be a worse lie than a duplicate.

        The merge is deliberately **stateless**. Making it sticky — preferring
        an id that survived the previous merge, so an object cannot change
        identity mid-flight — was implemented, tested and measured: it moved one
        false track in 39 across the subset and left true-tracks-per-object at
        1.56. It does not help because the identity change it guards against is
        not an ordering flip; it is a track *death* at one scale surfacing the
        suppressed track at another, and by then the sticky id no longer exists.
        Removed rather than shipped unmeasured. See CALIBRATION.md 3.12.
        """
        r = self.cfg.merge_px
        if r <= 0 or len(tracks) < 2:
            return tracks
        if lags is None:
            lags = [0.0] * len(tracks)
        kept, kept_pos = [], []
        for tr, lag in sorted(zip(tracks, lags), key=lambda p: -p[0].hits):
            x, y = tr.pos
            vx, vy = tr.vel
            px, py = x + vx * lag, y + vy * lag
            if all((px - kx) ** 2 + (py - ky) ** 2 > r * r
                   for kx, ky in kept_pos):
                kept.append(tr)
                kept_pos.append((px, py))
        self._merge_kept = {tr.track_id for tr in kept}
        return kept

    def process_window(self, ev: np.ndarray, t_end: int):
        t0 = perf_counter()
        f = self.filt(ev)
        t1 = perf_counter()
        if self.feast is not None and len(f):
            f, _ = self.feast(f)
        t2 = perf_counter()

        self.history.append(f)
        self._n += 1
        per_scale = []
        for spec, det in zip(self.specs, self.detectors):
            if self._n % spec.step:
                per_scale.append(None)          # this scale does not fire now
                continue
            if spec.windows <= 1:
                buf = f
            else:
                if len(self.history) < spec.windows:
                    per_scale.append(None)      # not enough history to integrate honestly
                    continue
                parts = list(self.history)[-spec.windows:]
                buf = parts[0] if len(parts) == 1 else np.concatenate(parts)
            t_start = t_end - spec.windows * self.cfg.window_us
            per_scale.append(det(buf, t_end=t_end, t_start=t_start) if len(buf) else [])
        t3 = perf_counter()

        tracks, lags = [], []
        for tk, d, lag in zip(self.trackers, per_scale, self.scale_lag_s):
            got = tk.coast(t_end) if d is None else tk.update(d, t_end)
            tracks.extend(got)
            lags.extend([lag] * len(got))
        dets = self._merge([d for lst in per_scale if lst for d in lst])
        tracks = self._merge_tracks(tracks, lags)
        t4 = perf_counter()

        self.timings["filter"].append((t1 - t0) * 1e3)
        self.timings["feast"].append((t2 - t1) * 1e3)
        self.timings["detect"].append((t3 - t2) * 1e3)
        self.timings["track"].append((t4 - t3) * 1e3)
        self.timings["total"].append((t4 - t0) * 1e3)
        self.window_events.append(len(ev))
        return dets, tracks, len(f)

    def run(self, ev: np.ndarray, progress=False):
        """Iterate the whole recording in fixed time windows."""
        if len(ev) == 0:
            return []
        W = self.cfg.window_us
        t0, t1 = int(ev["t"][0]), int(ev["t"][-1])
        edges = np.arange(t0, t1 + W, W)
        idx = np.searchsorted(ev["t"], edges)
        out = []
        for i in range(len(edges) - 1):
            chunk = ev[idx[i]:idx[i + 1]]
            dets, tracks, n_kept = self.process_window(chunk, int(edges[i + 1]))
            out.append(dict(t=int(edges[i + 1]), n_in=len(chunk), n_kept=n_kept,
                            detections=[d.as_dict() for d in dets],
                            tracks=[tr.as_dict() for tr in tracks]))
            if progress and i % 500 == 0:
                print(f"  window {i}/{len(edges)-1}  in={len(chunk)} kept={n_kept} "
                      f"det={len(dets)} trk={len(tracks)}", flush=True)
        return out

    # -- reporting ---------------------------------------------------------
    def latency_report(self) -> dict:
        rep = {}
        for k, v in self.timings.items():
            if not v:
                continue
            a = np.asarray(v)
            rep[k] = dict(mean_ms=float(a.mean()), p50_ms=float(np.percentile(a, 50)),
                          p95_ms=float(np.percentile(a, 95)),
                          p99_ms=float(np.percentile(a, 99)), max_ms=float(a.max()))
        tot = np.asarray(self.timings["total"]) if self.timings["total"] else np.zeros(1)
        n_ev = int(np.sum(self.window_events))
        wall_s = float(tot.sum()) / 1e3
        rep["summary"] = dict(
            windows=len(tot), window_us=self.cfg.window_us, events=n_ev,
            scales=[dict(label=s.label, windows=s.windows, stride=s.step,
                         integration_us=s.windows * self.cfg.window_us,
                         rate_hz=1e6 / (s.step * self.cfg.window_us))
                    for s in self.specs],
            throughput_ev_per_s=float(n_ev / wall_s) if wall_s > 0 else 0.0,
            realtime_factor=float((len(tot) * self.cfg.window_us * 1e-6) / wall_s)
            if wall_s > 0 else 0.0,
            budget_ms=40.0,
            frac_windows_over_budget=float((tot > 40.0).mean()),
        )
        return rep
