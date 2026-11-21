"""Detection: sparse spatio-temporal clustering over a short event window.

Grid clustering, chosen over DBSCAN/K-Means for the same reason the FPGA
literature chooses it — O(N) with a single pass, no k, no neighbour graph, and
an initiation interval that does not blow up when the noise rate spikes.

A window of filtered events is accumulated into coarse cells; cells above an
occupancy floor are connected with 8-connectivity; each connected component
yields one detection with an event-count-weighted sub-pixel centroid.

## Temporal spread

When the detector integrates over a long window, a cluster floor alone stops
discriminating: four events in one cell is four events whether they arrived
evenly across 250 ms or all inside one microsecond burst. On real sensor data
those two cases are physically different. A faint point source emits at a low
but steady rate, so its events are spread across the integration period. What
survives the noise gate, by construction, is *temporally correlated* — bursts.

So a cluster can be required to occupy at least `min_time_bins` of the
`time_bins` sub-intervals of the window. It costs one bincount over the events
already gathered, and it is the cheapest discriminator available at this stage
because it uses the one axis a frame-based detector throws away.

Set `min_time_bins <= 1` to disable, which is the default and what the
single-window synthetic profile uses.

## Adaptive thresholding

A fixed occupancy floor assumes the background is uniform and known. On a real
sky it is neither: the event rate varies across the sensor with sky glow, optics
vignetting and the local star density, and it varies over the recording as
conditions change. A floor low enough to catch the target in a quiet corner
fires on every cell in a bright one.

With `snr_k > 0` each cell is instead compared against its own local
neighbourhood — a `bg_cells`-wide box mean and variance, with a Poisson floor
under the variance — and kept when it stands `snr_k` sigma above it. This is
the standard construction for point-source detection on a structured
background, it costs two box filters over the coarse grid (hundreds of cells,
not millions of pixels), and it removes the single most fragile hand-tuned
number in the pipeline. `min_events` still applies underneath as an absolute
floor so that a quiet region cannot promote a two-event cell.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage


@dataclass
class Detection:
    t: int             # window end timestamp, microseconds
    x: float           # sub-pixel centroid
    y: float
    n_events: int
    bbox: tuple        # (x0, y0, x1, y1)
    peak_density: float
    score: float = 0.0 # peak local-background SNR; 0 when adaptive mode is off
    n_cells: int = 1   # cluster footprint; a compact cluster is a better fix
    scale: str = "base"    # which detection scale produced it
    time_bins_occupied: int = 0   # temporal spread; 0 when the test is off

    def as_dict(self):
        return dict(t=int(self.t), x=float(self.x), y=float(self.y),
                    n_events=int(self.n_events), bbox=[int(v) for v in self.bbox],
                    peak_density=float(self.peak_density),
                    n_cells=int(self.n_cells), scale=self.scale,
                    score=round(float(self.score), 3),
                    time_bins_occupied=int(self.time_bins_occupied))


class GridClusterDetector:
    def __init__(self, shape, cell=8, min_events=6, min_cluster_events=12,
                 max_cluster_cells=64, label="base",
                 time_bins=1, min_time_bins=0, snr_k=0.0, bg_cells=15,
                 max_detections=0):
        self.H, self.W = shape
        self.cell = int(cell)
        self.min_events = int(min_events)              # per-cell occupancy floor
        self.min_cluster_events = int(min_cluster_events)
        self.max_cluster_cells = int(max_cluster_cells)
        self.label = str(label)
        self.snr_k = float(snr_k)
        self.bg_cells = int(bg_cells)
        self.time_bins = max(1, int(time_bins))
        self.min_time_bins = int(min_time_bins)
        self.max_detections = int(max_detections)
        self.gh = (self.H + self.cell - 1) // self.cell
        self.gw = (self.W + self.cell - 1) // self.cell

    def __call__(self, ev: np.ndarray, t_end: int | None = None,
                 t_start: int | None = None) -> list[Detection]:
        if len(ev) == 0:
            return []
        t_end = int(ev["t"][-1]) if t_end is None else int(t_end)
        check_spread = self.min_time_bins > 1 and self.time_bins > 1
        if check_spread:
            # Bin against the NOMINAL integration window, not the observed
            # event span. Using the span rescales a short burst so that it
            # fills every bin, which is exactly the case the test exists to
            # reject — the discriminator is only meaningful against the window
            # the detector was asked to integrate over.
            t0b = int(ev["t"][0]) if t_start is None else int(t_start)
            span = max(t_end - t0b, 1)
            tbin = ((ev["t"].astype(np.int64) - t0b) * self.time_bins
                    // span).clip(0, self.time_bins - 1)
        gy = ev["y"].astype(np.int64) // self.cell
        gx = ev["x"].astype(np.int64) // self.cell
        flat = gy * self.gw + gx
        counts = np.bincount(flat, minlength=self.gh * self.gw).reshape(self.gh, self.gw)
        if self.snr_k > 0:
            c = counts.astype(np.float32)
            bg = ndimage.uniform_filter(c, size=self.bg_cells, mode="nearest")
            var = ndimage.uniform_filter(c * c, size=self.bg_cells,
                                         mode="nearest") - bg * bg
            # Poisson floor: the variance of a counting process is at least its
            # mean, so an empty patch cannot produce a divide-by-nothing sigma.
            sigma = np.sqrt(np.maximum(var, np.maximum(bg, 1.0)))
            snr_map = (c - bg) / sigma
            active = (snr_map >= self.snr_k) & (counts >= self.min_events)
        else:
            snr_map = None
            active = counts >= self.min_events
        if not active.any():
            return []
        lab, n = ndimage.label(active, structure=np.ones((3, 3), dtype=int))
        if n == 0:
            return []
        ev_lab = lab[gy, gx]
        dets = []
        for cid in range(1, n + 1):
            m = ev_lab == cid
            k = int(m.sum())
            if k < self.min_cluster_events:
                continue
            n_cells = int((lab == cid).sum())
            if n_cells > self.max_cluster_cells:
                continue                                  # a cloud, not a target
            if check_spread:
                occupied = int(np.unique(tbin[m]).size)
                if occupied < self.min_time_bins:
                    continue                              # a burst, not a source
            xs = ev["x"][m].astype(np.float64)
            ys = ev["y"][m].astype(np.float64)
            dets.append(Detection(
                t=t_end, x=xs.mean(), y=ys.mean(), n_events=k,
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                peak_density=k / max(n_cells, 1),
                score=float(snr_map[lab == cid].max()) if snr_map is not None else 0.0,
                n_cells=n_cells, scale=self.label,
                time_bins_occupied=int(np.unique(tbin[m]).size) if check_spread else 0,
            ))
        dets.sort(key=lambda d: -d.n_events)
        if self.max_detections and len(dets) > self.max_detections:
            # A per-firing budget, kept by SNR rather than by size.
            #
            # A firing that yields fifty clusters is describing noise, not fifty
            # targets: on a dense recording a shallow scale produces exactly
            # that, and the tracker then has fifty chances per firing to confirm
            # a coincidence. Capping the number of candidates bounds how much
            # clutter can reach association at all, and ranking by score keeps
            # the strongest rather than merely the largest — a smeared noise
            # blob can out-count a real point source.
            dets = sorted(dets, key=lambda d: -d.score)[:self.max_detections]
            dets.sort(key=lambda d: -d.n_events)
        return dets
