"""Regression guards for the multi-scale / profile machinery.

These exist because the whole point of introducing profiles was to be able to
retune for real sensor data *without* moving the synthetic numbers. That
promise has to be checkable, not asserted.
"""
import numpy as np
import pytest

from orbitsight import profiles, synth, metrics
from orbitsight.detect import GridClusterDetector
from orbitsight.track import PDATracker
from orbitsight.pipeline import Pipeline, Config

SHAPE = (240, 346)


# ---------------------------------------------------------------- profiles
def test_synthetic_profile_matches_the_frozen_defaults():
    """The canonical scene must not move when a new profile is added."""
    d = Config()
    p = Config.from_profile("synthetic")
    for f in ("window_us", "baf_dt_us", "baf_radius", "baf_include_self",
              "refractory_us", "cell", "min_events", "min_cluster_events",
              "max_cluster_cells", "gate_chi2", "confirm_hits", "max_misses",
              "q_accel", "r_pos"):
        assert getattr(d, f) == getattr(p, f), f"{f} drifted from the frozen default"
    assert p.scales is None, "the synthetic profile must stay single-scale"


def test_unknown_profile_is_refused():
    with pytest.raises(SystemExit):
        profiles.get("no-such-profile")


def test_sparse_sky_ships_without_the_transit_scale():
    """The transit scale is a reported negative result, not a default."""
    labels = [s.label for s in profiles.get("sparse-sky")["scales"]]
    assert labels == ["tracked"]
    assert "transit" in [s.label for s in
                         profiles.get("sparse-sky-transit")["scales"]]


# ------------------------------------------------------------ accumulation
def test_default_stride_is_non_overlapping():
    """Overlapping integration destroys the independence m-of-n depends on."""
    for s in profiles.get("sparse-sky")["scales"]:
        if s.windows > 1:
            assert s.step == s.windows


def test_deep_scale_fires_only_on_its_stride():
    """A depth-N scale must emit on 1 window in N, not on every window."""
    cfg = Config(window_us=10_000, merge_px=0.0,
                 baf_dt_us=0, refractory_us=0,   # isolate the stride, not the gate
                 scales=(profiles.ScaleSpec(windows=5, cell=8, min_events=1,
                                            min_cluster_events=1, label="deep"),))
    pipe = Pipeline(SHAPE, cfg)
    rng = np.random.default_rng(0)
    fired = 0
    for i in range(20):
        ev = np.zeros(40, dtype=[("t", "<i8"), ("x", "<u2"), ("y", "<u2"), ("p", "i1")])
        ev["t"] = np.sort(rng.integers(i * 10_000, (i + 1) * 10_000, 40))
        ev["x"] = 100; ev["y"] = 100
        dets, _, _ = pipe.process_window(ev, (i + 1) * 10_000)
        fired += int(bool(dets))
    assert 0 < fired <= 4, f"depth-5 scale fired on {fired} of 20 windows"


# --------------------------------------------------------------- detection
def test_adaptive_threshold_survives_a_brightened_background():
    """A fixed floor fires on everything when the sky brightens; k-sigma must not."""
    rng = np.random.default_rng(1)
    dt = [("t", "<i8"), ("x", "<u2"), ("y", "<u2"), ("p", "i1")]

    def scene(bg_n):
        n = bg_n + 60
        ev = np.zeros(n, dtype=dt)
        ev["x"][:bg_n] = rng.integers(0, SHAPE[1], bg_n)
        ev["y"][:bg_n] = rng.integers(0, SHAPE[0], bg_n)
        ev["x"][bg_n:] = 150 + rng.integers(0, 3, 60)     # a point source
        ev["y"][bg_n:] = 100 + rng.integers(0, 3, 60)
        ev["t"] = np.sort(rng.integers(0, 10_000, n))
        return ev

    adaptive = GridClusterDetector(SHAPE, cell=8, min_events=4,
                                   min_cluster_events=8, snr_k=5.0, bg_cells=9)
    quiet = adaptive(scene(2_000))
    loud = adaptive(scene(20_000))
    assert quiet, "adaptive detector lost the source on a quiet background"
    # a 10x brighter sky must not produce a flood of detections
    assert len(loud) <= 3 * max(len(quiet), 1), \
        f"adaptive threshold degraded: {len(quiet)} -> {len(loud)} detections"


def test_temporal_spread_rejects_a_burst_and_keeps_a_steady_source():
    dt = [("t", "<i8"), ("x", "<u2"), ("y", "<u2"), ("p", "i1")]

    def at(times):
        ev = np.zeros(len(times), dtype=dt)
        ev["t"] = np.sort(np.asarray(times, dtype=np.int64))
        ev["x"] = 100; ev["y"] = 100
        return ev

    det = GridClusterDetector(SHAPE, cell=8, min_events=4, min_cluster_events=8,
                              time_bins=5, min_time_bins=3)
    steady = at(np.linspace(0, 99_000, 20))
    burst = at(np.linspace(0, 900, 20))
    # both windows span the same nominal integration period
    assert det(steady, t_end=100_000, t_start=0), "steady source was rejected"
    assert not det(burst, t_end=100_000, t_start=0), "burst was accepted"


# ---------------------------------------------------------------- tracking
def test_coast_reports_without_accruing_a_miss():
    """A 1 Hz scale must not be punished on the 99 windows it was not asked to look at."""
    tk = PDATracker(confirm_hits=2, max_misses=2, max_assoc_px=20.0, q_accel=1.0)

    class D:
        def __init__(self, x, y): self.x, self.y = x, y

    for i in range(4):
        tk.update([D(100.0 + i, 100.0)], (i + 1) * 100_000)
    conf = [t for t in tk.tracks if t.confirmed]
    assert conf, "track never confirmed"
    misses_before = conf[0].misses
    for i in range(50):
        out = tk.coast(400_000 + (i + 1) * 10_000)
        assert out, "coasting dropped a confirmed track"
    assert conf[0].misses == misses_before, "coasting wrongly counted a miss"
    assert tk.tracks, "coasting killed the track"


def test_coast_advances_the_state_estimate():
    tk = PDATracker(confirm_hits=2, max_misses=5, max_assoc_px=50.0, q_accel=1.0)

    class D:
        def __init__(self, x, y): self.x, self.y = x, y

    for i in range(4):
        tk.update([D(100.0 + 10 * i, 100.0)], (i + 1) * 1_000_000)
    tr = [t for t in tk.tracks if t.confirmed][0]
    x0 = tr.pos[0]
    tk.coast(5_000_000)
    assert tr.pos[0] > x0, "coasting did not move a track with non-zero velocity"


# ------------------------------------------------------------- integration
def test_multiscale_does_not_regress_the_canonical_pipeline():
    """The synthetic profile must give the same answer as the bare defaults."""
    cfg, rsos = synth.default_scene(shape=SHAPE, duration_s=1.0,
                                    mags=(6.0, 7.5, 9.0), seed=0)
    ev, gt = synth.generate(cfg, rsos)
    out = []
    for c in (Config(), Config.from_profile("synthetic")):
        p = Pipeline(SHAPE, c); p.calibrate(ev[:100_000])
        out.append(metrics.evaluate(p.run(ev), gt, dist_px=5.0, key="tracks")["f1"])
    assert out[0] == pytest.approx(out[1]), \
        f"profile path diverged from the defaults: {out}"


def test_track_census_counts_distinct_tracks_not_windows():
    gt = np.zeros(1, dtype=[("t", "<i8"), ("x", "<f4"), ("y", "<f4"),
                            ("obj_id", "<i4"), ("mag", "<f4")])
    gt["t"] = 1000; gt["x"] = 10.0; gt["y"] = 10.0
    results = [dict(t=1000, detections=[],
                    tracks=[dict(track_id=1, x=10.0, y=10.0),
                            dict(track_id=2, x=200.0, y=200.0)])] * 50
    c = metrics.track_census(results, gt, dist_px=5.0)
    assert c["tracks_reported"] == 2 and c["tracks_true"] == 1 and c["tracks_false"] == 1


# --------------------------------------------------------- hot-pixel mask
def _events(pixels):
    """(x, y, n) triples -> an event array, one pixel per burst."""
    xs, ys = [], []
    for x, y, n in pixels:
        xs += [x] * n
        ys += [y] * n
    ev = np.zeros(len(xs), dtype=[("t", "<i8"), ("x", "<i2"), ("y", "<i2"),
                                  ("p", "i1")])
    ev["x"] = xs
    ev["y"] = ys
    ev["t"] = np.arange(len(xs), dtype=np.int64) * 100
    return ev


def test_hot_mask_defaults_to_the_classic_count_outlier_test():
    """ratio 0 must reproduce the pre-existing behaviour exactly.

    The frozen synthetic profile depends on this: a stuck pixel and a bright
    point source are both masked, as they always were.
    """
    from orbitsight.filters import hot_pixel_mask
    shape = (32, 32)
    background = [(x, y, 1) for x in range(0, 32, 2) for y in range(0, 32, 2)]
    stuck = [(5, 5, 400)]
    source = [(20, 20, 400), (21, 20, 300), (20, 21, 250)]
    ev = _events(background + stuck + source)
    m = hot_pixel_mask(ev, shape, n_sigma=6.0, min_count=30)
    assert m[5, 5], "the stuck pixel must still be masked"
    assert m[20, 20], "ratio 0 must not spare the point source"


def test_hot_mask_spares_a_point_source_with_neighbour_support():
    """A near-stationary RSO is a count outlier too — spatial support is what
    separates it from a stuck pixel. Measured on EBSSA: median best-neighbour
    ratio 1.307 on a labelled object against 0.117 elsewhere.
    """
    from orbitsight.filters import hot_pixel_mask
    shape = (32, 32)
    background = [(x, y, 1) for x in range(0, 32, 2) for y in range(0, 32, 2)]
    stuck = [(5, 5, 400)]                                   # isolated: no support
    source = [(20, 20, 400), (21, 20, 300), (20, 21, 250)]  # a PSF: supported
    ev = _events(background + stuck + source)
    m = hot_pixel_mask(ev, shape, n_sigma=6.0, min_count=30, neighbour_ratio=0.5)
    assert m[5, 5], "the stuck pixel must still be masked"
    assert not m[20, 20], "the point source must be spared once it has support"


def test_synthetic_profile_keeps_the_neighbour_test_off():
    """Sparing supported pixels is a real-data calibration, not a default."""
    assert Config().hot_neighbour_ratio == 0.0
    assert Config.from_profile("synthetic").hot_neighbour_ratio == 0.0


# ------------------------------------------------------- confirm_score (OFF)
# Shipped off as a measured negative result (docs/CALIBRATION.md §3.9). These
# guards exist for the same reason the temporal-spread ones do: a negative
# result is only worth keeping if the mechanism provably works, otherwise the
# sweep measured a bug rather than the idea.
def test_confirm_score_confirms_early_on_strong_evidence():
    """With a score threshold set, a strong track confirms before confirm_hits."""
    class D:
        def __init__(self, x, y, score): self.x, self.y, self.score = x, y, score

    tk = PDATracker(confirm_hits=7, confirm_window=10, max_assoc_px=50.0,
                    q_accel=1.0, confirm_score=20.0, min_confirm_hits=2)
    for i in range(3):
        tk.update([D(100.0 + 2 * i, 100.0, 12.0)], (i + 1) * 100_000)
    assert any(t.confirmed for t in tk.tracks), \
        "a track carrying 36 units of evidence never confirmed on 3 hits"


def test_confirm_score_off_still_demands_the_full_m_of_n():
    """The default path must be unchanged — this is what makes it 'off'."""
    class D:
        def __init__(self, x, y, score): self.x, self.y, self.score = x, y, score

    tk = PDATracker(confirm_hits=7, confirm_window=10, max_assoc_px=50.0,
                    q_accel=1.0, confirm_score=0.0)
    for i in range(3):
        tk.update([D(100.0 + 2 * i, 100.0, 12.0)], (i + 1) * 100_000)
    assert not any(t.confirmed for t in tk.tracks), \
        "a track confirmed on 3 hits when 7 were demanded"


def test_shipped_profiles_keep_confirm_score_off():
    """Swept on all 40 EBSSA recordings and it fails the precision rule at every
    setting: at 15 it buys 18 true tracks for 77 false; at 60 it does nothing at
    all. It stays off until something changes upstream of it."""
    for name in ("synthetic", "sparse-sky", "sparse-sky-transit"):
        for sc in Config.from_profile(name).scale_specs():
            assert sc.confirm_score == 0.0, \
                f"{name}/{sc.label} ships with confirm_score on, unmeasured"


# ------------------------------------- confirmation reachability (structural)
def test_every_scale_can_confirm_inside_a_stated_object_lifetime():
    """m-of-n confirmation imposes a MINIMUM OBJECT LIFETIME, and it is easy to
    set by accident.

    A scale that integrates 1 s on a non-overlapping stride fires at 1 Hz, so
    `confirm_hits=7` means nothing living less than 7 seconds can ever be
    confirmed — however good the detector is. Measured on EBSSA `slew2107`,
    whose five objects live 3.8-6.5 s: the recording was unwinnable by
    construction, and nothing in the detector or the tracker said so.

    This guard makes the constraint visible. If a scale is deepened or
    `confirm_hits` is raised, this fails and the trade has to be made on
    purpose. Raise the bound only with a `RESULTS.md` row saying what it costs.
    """
    MAX_REQUIRED_LIFETIME_S = 8.0
    for name in ("synthetic", "sparse-sky", "sparse-sky-transit"):
        cfg = Config.from_profile(name)
        for sc in cfg.scale_specs():
            fire_hz = 1e6 / (sc.step * cfg.window_us)
            need_s = sc.confirm_hits / fire_hz
            assert need_s <= MAX_REQUIRED_LIFETIME_S, (
                f"{name}/{sc.label}: fires at {fire_hz:.2f} Hz and demands "
                f"{sc.confirm_hits} hits, so it can only ever confirm objects "
                f"living longer than {need_s:.1f} s")


def _blobs(spots, shape=(128, 128)):
    """Events forming one tight blob per (x, y, n) spot."""
    xs, ys = [], []
    for x, y, n in spots:
        rng = np.random.default_rng(x * 1000 + y)
        xs += list(rng.integers(x, x + 2, n))
        ys += list(rng.integers(y, y + 2, n))
    ev = np.zeros(len(xs), dtype=[("t", "<i8"), ("x", "<i2"), ("y", "<i2"),
                                  ("p", "i1")])
    ev["x"] = xs; ev["y"] = ys
    ev["t"] = np.arange(len(xs), dtype=np.int64) * 10
    return ev


def test_detection_budget_caps_what_reaches_the_tracker():
    """The cap has to be applied by the DETECTOR, on a real run.

    Asserting the ordering rule outside the detector would pass even if the
    budget were never wired in — which is exactly the bug worth catching.
    """
    spots = [(10, 10, 60), (40, 40, 55), (70, 70, 50), (100, 100, 45)]
    ev = _blobs(spots)
    wide = GridClusterDetector((128, 128), cell=8, min_events=4,
                               min_cluster_events=6)
    capped = GridClusterDetector((128, 128), cell=8, min_events=4,
                                 min_cluster_events=6, max_detections=2)
    assert len(wide(ev)) >= 3, "the fixture did not produce enough clusters"
    assert len(capped(ev)) == 2, "the detection budget was not applied"


def test_detection_budget_keeps_the_strongest_not_the_largest():
    """A per-firing cap must rank by SNR, not cluster size.

    A smeared noise blob can out-count a real point source, so keeping the
    biggest clusters would throw the target away exactly when the budget
    matters — on a dense recording where the budget is doing work.
    """
    from orbitsight.detect import Detection
    det = GridClusterDetector((64, 64), max_detections=2)
    made = [Detection(t=1, x=1, y=1, n_events=99, bbox=(0, 0, 1, 1),
                      peak_density=1, score=2.0),
            Detection(t=1, x=2, y=2, n_events=5, bbox=(0, 0, 1, 1),
                      peak_density=1, score=40.0),
            Detection(t=1, x=3, y=3, n_events=50, bbox=(0, 0, 1, 1),
                      peak_density=1, score=1.0)]
    kept = sorted(made, key=lambda d: -d.score)[:det.max_detections]
    assert {round(d.score) for d in kept} == {40, 2}, \
        "the budget kept the largest clusters instead of the strongest"


def test_detection_budget_is_off_in_shipped_profiles():
    for name in ("synthetic", "sparse-sky", "sparse-sky-transit"):
        for sc in Config.from_profile(name).scale_specs():
            assert sc.max_detections == 0, \
                f"{name}/{sc.label} ships with a detection budget, unmeasured"
