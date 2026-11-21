"""Observing profiles — one named parameter set per sensing regime.

The pipeline has one architecture and two very different jobs, and the
difference is not a matter of taste. It is set by the physics of what is in
front of the sensor.

| | synthetic scene | real telescope NVS (EBSSA) |
|---|---|---|
| target signal | ~10^2 events per 10 ms window | ~10^0 events per 10 ms window |
| signal fraction | ~5% of all events | **0.08%** of all events |
| target rate | 30-300 px/s | 6 px/s (tracked) to 640 px/s (transit) |
| noise | uniform shot noise | structured: hot pixels, stars, sky glow |
| useful integration | 10 ms | 10 ms *and* 250 ms, simultaneously |

A single parameter set cannot serve both. Rather than retuning the defaults —
which would silently invalidate every number in `RESULTS.md`, the proposal and
the reviewer's `docker run` line — each regime is a named profile.

`synthetic` is FROZEN. It reproduces the canonical scene bit-for-bit and its
values must not move without moving every quoted number in the same commit.

`sparse-sky` is calibrated on EBSSA. Every value in it comes from a sweep
recorded in `docs/CALIBRATION.md`, not from taste.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleSpec:
    """One detection scale: an integration time, a detector, and a tracker.

    ## windows and stride

    `windows` is the accumulation depth in pipeline windows — how much history
    the clusterer sees at once. `stride` is how often that scale runs.

    Stride matters more than it looks. With `stride=1` a depth-25 scale
    re-clusters an almost identical buffer every window: consecutive buffers
    share 96% of their events, so a chance cluster of noise reappears 25 times
    in a row and sails through any m-of-n confirmation. Overlapping
    integration does not just cost 25x the compute, it destroys the statistical
    independence that track confirmation depends on.

    Defaulting `stride` to `windows` makes the integration non-overlapping:
    each long-scale decision rests on fresh evidence, so surviving m-of-n means
    something again, and the scale costs one clustering pass per integration
    period instead of per window.

    ## per-scale tracker

    Each scale carries its own tracker, because a 6 px/s tracked RSO and a
    640 px/s transit need different process noise, different association gates
    and different speed limits — and because a tracker fed at 4 Hz by the slow
    scale would otherwise register a miss on every window the fast scale filled
    in. Confirmed tracks from all scales are unioned; ids stay unique.
    """
    windows: int = 1
    stride: int | None = None          # None -> non-overlapping (== windows)
    # detector
    cell: int = 8
    min_events: int = 5
    min_cluster_events: int = 10
    max_cluster_cells: int = 64
    time_bins: int = 1                 # sub-intervals of the integration window
    min_time_bins: int = 0             # <=1 disables the temporal-spread test
    snr_k: float = 0.0                 # >0 -> local-background threshold, k sigma
    max_detections: int = 0            # >0 -> keep only the N strongest per firing
    bg_cells: int = 15                 # box width for the local background
    # tracker
    gate_chi2: float = 16.0
    confirm_hits: int = 3
    confirm_window: int = 5
    confirm_score: float = 0.0         # accumulated SNR that confirms early; 0 = off
    min_confirm_hits: int = 2
    max_misses: int = 4
    q_accel: float = 250.0
    r_pos: float = 2.0
    max_assoc_px: float = 25.0
    min_speed_px_s: float = 0.0
    max_speed_px_s: float = 4000.0
    label: str = "base"

    @property
    def step(self) -> int:
        return int(self.stride) if self.stride else max(1, int(self.windows))


# ── synthetic — FROZEN ───────────────────────────────────────────────────────
# Reproduces 640x480 - 2.0 s - mags 6.0/7.5/9.0 - seed 0 - 245,161 events
#   -> P 1.000 - R 0.638 - F1 0.779 - 3/3 objects - 0.96 px
SYNTHETIC = dict(
    window_us=10_000,
    baf_dt_us=5_000,
    baf_radius=1,
    baf_include_self=False,
    refractory_us=800,
    hot_sigma=8.0,
    hot_min_count=50,
    hot_neighbour_ratio=0.0,     # OFF: the classic count-outlier test, frozen
    cell=8,
    min_events=5,
    min_cluster_events=10,
    max_cluster_cells=64,
    scales=None,                 # None -> a single depth-1 scale from the fields above
    merge_px=0.0,                # no cross-scale merge with one scale
    use_feast=False,
    gate_chi2=16.0,
    confirm_hits=3,
    max_misses=4,
    q_accel=250.0,
    r_pos=2.0,
    max_assoc_px=25.0,
    max_speed_px_s=4000.0,
)

# ── sparse-sky — calibrated on EBSSA ─────────────────────────────────────────
# Noise gate: swept against signal-retention / noise-retention, never against
# total event count. Winner keeps 42% of RSO events while keeping 3.0% of
# everything else — selectivity 14.2x. Radius 2 with the self-term OFF beat
# radius 1 with it ON: an unresolved point source jitters across two pixels, so
# radius 2 gives it real spatial support, whereas the self-term mostly admits
# single-pixel shot noise. Full sweep in docs/CALIBRATION.md.
SPARSE_SKY = dict(
    SYNTHETIC,
    baf_dt_us=20_000,
    baf_radius=2,
    baf_include_self=False,
    refractory_us=0,
    hot_sigma=6.0,
    hot_min_count=30,
    # A stuck pixel and a near-stationary RSO are the same object to a per-pixel
    # count test, and the mask was deleting the target before the detector ran —
    # on refslew_2106_atis all 2,243 ground-truth samples sat within 1 px of a
    # masked pixel. They separate on spatial support: measured across every
    # labelled recording, best-neighbour/own count is 1.307 on a labelled object
    # against 0.117 elsewhere, so 0.3 sits inside the gap.
    # Swept on the 33 real RSO recordings: mean F1 0.166 -> 0.206, objects
    # 16 -> 20, recordings above zero 12 -> 15. It costs precision — false
    # tracks 15 -> 32, one per 244 s instead of one per 521 s — and it is a
    # deliberate trade, not a free win. It also hurts the 7 star-reference
    # recordings, where the labelled "object" is a saturated bright star that
    # this test then declines to mask. docs/CALIBRATION.md 3.8.
    hot_neighbour_ratio=0.3,
    scales=(
        # Tracked RSOs — 1 s of non-overlapping integration, 1 Hz.
        # A tracked RSO emits ~15 events/s onto two pixels against ~24,000
        # events/s of background. Detectability goes as signal/sqrt(background),
        # which keeps improving with integration time until the target smears
        # out of its own aperture; at 6 px/s that is about a second.
        # confirm_hits=7 is deliberately strict. Detector recall per firing is
        # only ~0.36, so demanding 7 hits inside a 17-firing window costs real
        # recall — and takes the run from 18 reported tracks (15 of them false)
        # down to 5 (3 false). The standing rule in docs/ROADMAP.md is that an
        # operator drowning in false tracks switches the system off.
        # time_bins/min_time_bins are left OFF: the temporal-spread test is
        # implemented and swept, and it removes the target along with the
        # clutter (F1 0.063 -> 0.000 at every setting tried). The reason is that
        # the noise gate ahead of it already selects for temporal correlation,
        # so what reaches the detector is bursty whether it is signal or not.
        # A third measured negative result, kept as evidence. See CALIBRATION.md.
        ScaleSpec(windows=100, cell=8, min_events=4, min_cluster_events=6,
                  max_cluster_cells=16, snr_k=6.5, bg_cells=15,
                  q_accel=4.0, r_pos=2.0, gate_chi2=16.0,
                  confirm_hits=7, confirm_window=10, max_misses=3,
                  max_assoc_px=10.0,
                  min_speed_px_s=0.0, max_speed_px_s=60.0,
                  label="tracked"),
    ),
    merge_px=4.0,
)

# ── the transit scale, kept as a reproducible negative result ────────────────
# Ten of the twelve EBSSA objects cross at 244-641 px/s. 288 configurations were
# swept against them; the best scored F1 0.030 / precision 0.025. The reason is
# arithmetic, not tuning: a transit deposits 0.6-3.0 events in a 10 ms window
# before filtering and 0.5-1.3 after, so there is nothing to cluster, and
# integrating longer smears it across hundreds of pixels faster than it
# accumulates. Detecting these needs a streak-aware detector (Hough/Radon over
# the event cloud), which is a different instrument — deferred in docs/STACK.md.
#
# Shipped OFF, alongside FEAST, as a measured negative result rather than a
# silent omission. `--profile sparse-sky-transit` reproduces it.
TRANSIT_SCALE = ScaleSpec(
    windows=1, stride=1, cell=6, min_events=2, min_cluster_events=3,
    max_cluster_cells=48, snr_k=5.5, bg_cells=15,
    q_accel=400.0, r_pos=2.0, gate_chi2=16.0, confirm_hits=4,
    max_misses=3, max_assoc_px=20.0,
    min_speed_px_s=60.0, max_speed_px_s=1500.0, label="transit")

SPARSE_SKY_TRANSIT = dict(SPARSE_SKY,
                          scales=(TRANSIT_SCALE,) + tuple(SPARSE_SKY["scales"]))

# ── the short scale, kept so its result is reproducible ──────────────────────
# CALIBRATION.md 3.10 showed that confirm_hits=7 on a 1 Hz scale imposes a
# seven-second minimum object lifetime, which makes recordings like slew2107
# (objects live 3.8-6.5 s) unwinnable by construction. A 500 ms scale fires at
# 2 Hz, so the same seven hits inside a ten-firing window need only 3.5 s.
#
# It reaches those objects. 3.11 still shipped it OFF, because every
# configuration reported 1.5-1.9 true tracks per object against a baseline of
# 1.04 -- the shallow scale re-detecting what the tracked scale already held,
# which _merge_tracks failed to suppress because it compared positions two
# scales stamp half an integration apart. That merge defect is now fixed, so
# this profile exists to re-measure the trade rather than to assert it.
#
# Only depth, threshold and confirmation differ from the tracked scale; every
# other field is copied so the comparison isolates one change.
SHORT_SCALE = ScaleSpec(
    windows=50, cell=8, min_events=4, min_cluster_events=6,
    max_cluster_cells=16, snr_k=6.5, bg_cells=15,
    q_accel=4.0, r_pos=2.0, gate_chi2=16.0,
    confirm_hits=7, confirm_window=10, max_misses=3,
    max_assoc_px=10.0,
    min_speed_px_s=0.0, max_speed_px_s=60.0,
    label="short")

SPARSE_SKY_SHORT = dict(SPARSE_SKY,
                        scales=tuple(SPARSE_SKY["scales"]) + (SHORT_SCALE,))

PROFILES = {"synthetic": SYNTHETIC,
            "sparse-sky": SPARSE_SKY,
            "sparse-sky-transit": SPARSE_SKY_TRANSIT,
            "sparse-sky-short": SPARSE_SKY_SHORT}


def get(name: str) -> dict:
    if name not in PROFILES:
        raise SystemExit(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    return dict(PROFILES[name])
