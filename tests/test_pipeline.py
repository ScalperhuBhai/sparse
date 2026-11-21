"""Regression guards. These exist so "definition of done" is checkable, not felt.

Run with:  pytest -q
"""
import os

import numpy as np
import pytest

from orbitsight import evio, synth, metrics, filters
from orbitsight.detect import GridClusterDetector
from orbitsight.track import PDATracker
from orbitsight.pipeline import Pipeline, Config

SHAPE = (240, 346)
LATENCY_BUDGET_MS = 40.0


@pytest.fixture(scope="module")
def scene():
    cfg, rsos = synth.default_scene(shape=SHAPE, duration_s=1.0,
                                    mags=(6.0, 7.5, 9.0), seed=0)
    return synth.generate(cfg, rsos)


# --------------------------------------------------------------- data layer
def test_synth_produces_events_and_labels(scene):
    ev, gt = scene
    assert len(ev) > 1000, "scene is suspiciously empty"
    assert ev.dtype == evio.EVENT_DTYPE
    assert np.all(np.diff(ev["t"]) >= 0), "events must be time-ordered"
    assert ev["x"].max() < SHAPE[1] and ev["y"].max() < SHAPE[0]
    assert set(np.unique(ev["p"])) <= {-1, 1}
    assert len(np.unique(gt["obj_id"])) == 3


def test_npz_roundtrip(tmp_path, scene):
    ev, _ = scene
    p = tmp_path / "r.npz"
    evio.save_npz(str(p), ev)
    back = evio.load(str(p))
    assert len(back) == len(ev)
    np.testing.assert_array_equal(back["x"], ev["x"])
    np.testing.assert_array_equal(back["y"], ev["y"])


def test_sensor_shape_inference(scene):
    ev, _ = scene
    h, w = evio.sensor_shape(ev)
    assert h >= SHAPE[0] and w >= SHAPE[1]


# ----------------------------------------------------------------- filtering
def test_filters_remove_more_noise_than_signal(scene):
    """The gate must cut the event count without erasing the targets."""
    ev, gt = scene
    fp = filters.Pipeline(SHAPE).fit_hot_pixels(ev)
    kept = fp(ev)
    assert 0 < len(kept) < len(ev), "filter passed everything or nothing"

    # events within 6 px of a labelled RSO are 'signal'
    def near_target(e):
        idx = np.searchsorted(gt["t"], e["t"])
        idx = np.clip(idx, 0, len(gt) - 1)
        d = np.hypot(e["x"] - gt["x"][idx], e["y"] - gt["y"][idx])
        return (d < 6.0).mean()

    assert near_target(kept) > near_target(ev), \
        "filtering did not increase the signal fraction"


def test_hot_pixels_are_caught(scene):
    ev, _ = scene
    mask = filters.hot_pixel_mask(ev, SHAPE)
    assert mask.sum() >= 1, "synth injects hot pixels; none were found"
    assert mask.sum() < SHAPE[0] * SHAPE[1] * 0.01, "mask is far too aggressive"


# ----------------------------------------------------------------- detection
def test_detector_finds_a_planted_cluster():
    n = 400
    t = np.sort(np.random.default_rng(0).integers(0, 10_000, n))
    x = np.random.default_rng(1).normal(100, 1.2, n).astype(np.uint16)
    y = np.random.default_rng(2).normal(60, 1.2, n).astype(np.uint16)
    ev = evio.from_arrays(t, x, y, np.ones(n))
    dets = GridClusterDetector(SHAPE)(ev)
    assert dets, "no detection on a dense planted cluster"
    assert abs(dets[0].x - 100) < 3 and abs(dets[0].y - 60) < 3


def test_filter_plus_detector_ignores_pure_noise():
    """The detector alone will fire on dense uniform noise -- that is expected and
    is exactly why it never runs alone. This asserts the real claim: the gate in
    front of it makes spatially incoherent noise produce no detections."""
    rng = np.random.default_rng(3)
    n = 3000
    ev = evio.from_arrays(np.sort(rng.integers(0, 10_000, n)),
                          rng.integers(0, SHAPE[1], n),
                          rng.integers(0, SHAPE[0], n),
                          rng.choice([-1, 1], n))
    kept = filters.Pipeline(SHAPE).fit_hot_pixels(ev)(ev)
    dets = GridClusterDetector(SHAPE, min_events=5, min_cluster_events=20)(kept)
    assert len(dets) == 0, f"filtered uniform noise still produced {len(dets)} detections"


# ------------------------------------------------------------------ tracking
def test_tracker_confirms_a_straight_line():
    tr = PDATracker(confirm_hits=3)

    class D:
        def __init__(self, x, y):
            self.x, self.y = x, y

    confirmed = []
    for k in range(10):
        t = k * 10_000
        confirmed = tr.update([D(50 + 3.0 * k, 40 + 1.5 * k)], t)
    assert confirmed, "a clean constant-velocity target was never confirmed"
    vx, vy = confirmed[0].vel
    assert vx > 0 and abs(vx - 300) < 200, f"velocity estimate way off: {vx}"


def test_tracker_does_not_confirm_random_noise():
    rng = np.random.default_rng(4)
    tr = PDATracker(confirm_hits=3, max_misses=2)

    class D:
        def __init__(self, x, y):
            self.x, self.y = x, y

    for k in range(12):
        pts = [D(rng.uniform(0, 340), rng.uniform(0, 230)) for _ in range(2)]
        out = tr.update(pts, k * 10_000)
    assert len(out) <= 1, "scattered noise produced confirmed tracks"


# ------------------------------------------------------------------- metrics
def test_metrics_are_perfect_on_perfect_predictions(scene):
    _, gt = scene
    ts = np.unique(gt["t"])[::10]
    results = []
    for t in ts:
        rows = gt[gt["t"] == t]
        results.append(dict(t=int(t), detections=[
            dict(x=float(r["x"]), y=float(r["y"])) for r in rows]))
    m = metrics.evaluate(results, gt, dist_px=2.0, tol_us=1)
    assert m["precision"] == 1.0 and m["recall"] == 1.0


# ------------------------------------------------------ end-to-end guardrails
@pytest.fixture(scope="module")
def run(scene):
    ev, gt = scene
    pipe = Pipeline(SHAPE, Config(window_us=10_000))
    pipe.calibrate(ev)
    res = pipe.run(ev)
    return pipe, res, gt


def test_latency_stays_inside_budget(run):
    pipe, _, _ = run
    rep = pipe.latency_report()
    assert rep["total"]["p99_ms"] < LATENCY_BUDGET_MS, \
        f"p99 {rep['total']['p99_ms']:.2f} ms exceeds the {LATENCY_BUDGET_MS} ms budget"
    assert rep["summary"]["frac_windows_over_budget"] == 0.0


def test_accuracy_does_not_regress(run):
    """Baseline recorded 2026-09-04. Raise these if you improve them; never lower
    them silently to make a change pass."""
    _, res, gt = run
    m = metrics.evaluate(res, gt, dist_px=5.0, key="tracks")
    assert m["object_recall"] == 1.0, "not every labelled object was tracked"
    assert m["precision"] >= 0.85, f"precision regressed to {m['precision']:.3f}"
    assert m["f1"] >= 0.55, f"F1 regressed to {m['f1']:.3f}"
    assert m["mean_localisation_error_px"] < 2.0


def test_pipeline_is_resolution_agnostic():
    """The brief requires it; nothing may hard-code a sensor shape."""
    for shape in [(180, 240), (480, 640)]:
        cfg, rsos = synth.default_scene(shape=shape, duration_s=0.5,
                                        mags=(6.5,), seed=1)
        ev, gt = synth.generate(cfg, rsos)
        pipe = Pipeline(shape, Config(window_us=10_000))
        pipe.calibrate(ev)
        res = pipe.run(ev)
        m = metrics.evaluate(res, gt, dist_px=5.0, key="tracks")
        assert m["object_recall"] == 1.0, f"lost the target at {shape}"


# -------------------------------------------------- cross-scale track merging
def _fake_track(track_id, x, y, vx, vy, hits):
    """A confirmed track with a known state, without running a whole pipeline."""
    from orbitsight.track import Track
    return Track(track_id=track_id, x=np.array([x, y, vx, vy], dtype=float),
                 P=np.eye(4), t=0, hits=hits, confirmed=True)


def _two_scale_pipeline(merge_px):
    """Two scales, 1 s and 500 ms, so their integration lags differ by 250 ms."""
    from orbitsight.profiles import ScaleSpec
    cfg = Config(window_us=10_000, merge_px=merge_px,
                 scales=(ScaleSpec(windows=100, label="tracked"),
                         ScaleSpec(windows=50, label="short")))
    return Pipeline((180, 240), cfg)


def test_merge_suppresses_a_fast_duplicate_across_scales():
    """The defect §3.11 named: a 1 s scale and a 500 ms scale centroid the same
    fast object a quarter-second apart, so their raw positions sit 15 px apart
    at 60 px/s and the merge misses them. Lag-correcting closes it."""
    pipe = _two_scale_pipeline(merge_px=4.0)
    lags = pipe.scale_lag_s
    assert lags == [0.5, 0.25], f"integration lags computed wrong: {lags}"

    # one object at 60 px/s, reported by each scale from behind by its own lag
    vx, vy = 60.0, 0.0
    deep = _fake_track(1, 100.0 - vx * lags[0], 50.0, vx, vy, hits=9)
    short = _fake_track(1_000_001, 100.0 - vx * lags[1], 50.0, vx, vy, hits=5)
    assert abs(deep.pos[0] - short.pos[0]) > 4.0, \
        "test is not exercising the defect: the raw positions already merge"

    kept = pipe._merge_tracks([deep, short], lags)
    assert len(kept) == 1, "the same object at two scales was alerted twice"
    assert kept[0].track_id == 1, "merge did not keep the better-supported track"


def test_merge_is_off_when_merge_px_is_zero():
    """Proves the mechanism rather than the outcome: the same pair survives
    intact when the merge is disabled, so the test above measured the merge."""
    pipe = _two_scale_pipeline(merge_px=0.0)
    lags = pipe.scale_lag_s
    deep = _fake_track(1, 100.0 - 60.0 * lags[0], 50.0, 60.0, 0.0, hits=9)
    short = _fake_track(1_000_001, 100.0 - 60.0 * lags[1], 50.0, 60.0, 0.0, hits=5)
    assert len(pipe._merge_tracks([deep, short], lags)) == 2


def test_merge_does_not_swallow_two_genuine_objects():
    """Lag correction moves tracks; it must not move them onto each other. Two
    real objects on the same heading stay two — over-merging loses a target,
    which is worse than the duplicate this fix removes."""
    pipe = _two_scale_pipeline(merge_px=4.0)
    lags = pipe.scale_lag_s
    a = _fake_track(1, 100.0, 50.0, 60.0, 0.0, hits=9)
    b = _fake_track(1_000_001, 100.0, 90.0, 60.0, 0.0, hits=7)   # 40 px apart
    assert len(pipe._merge_tracks([a, b], lags)) == 2


def test_single_scale_profile_is_untouched_by_the_merge():
    """`sparse-sky` ships one scale, so the merge must be a no-op on it — the
    published real-data numbers cannot move because of this change."""
    from orbitsight import profiles
    cfg = Config.from_profile("sparse-sky")
    assert len(cfg.scale_specs()) == 1, "sparse-sky gained a scale; re-measure §3.7"
    pipe = Pipeline((180, 240), cfg)
    tracks = [_fake_track(1, 100.0, 50.0, 6.0, 0.0, hits=9),
              _fake_track(2, 101.0, 50.0, 6.0, 0.0, hits=7)]
    kept = pipe._merge_tracks(tracks, [pipe.scale_lag_s[0]] * 2)
    assert len(kept) == 1, "within-scale duplicates are still suppressed"


# ------------------------------------------------------------------- FEAST
# FEAST had no test at all until 5 September, which is the wrong amount for the
# component the neuromorphic claim rests on and one of the seven shipped
# ablations. It is off by default, so nothing downstream would have noticed it
# rotting; a reviewer passing --feast would have been the one to find out.
@pytest.fixture(scope="module")
def feast_scene():
    cfg, rsos = synth.default_scene(shape=(180, 240), duration_s=0.4,
                                    mags=(6.0,), seed=3)
    return synth.generate(cfg, rsos)


def test_feast_selects_rather_than_passes_everything(feast_scene):
    """The claim is a sparse feature stream. A layer that fires on everything is
    a no-op with a latency cost, which is a different thing to report."""
    from orbitsight.feast import FEAST
    ev, _ = feast_scene
    f = FEAST((180, 240), n_neurons=16, radius=4, seed=0)
    out, ids = f(ev)
    assert 0 < len(out) < len(ev), \
        f"FEAST kept {len(out)} of {len(ev)} events — not a selective layer"
    assert len(ids) == len(out)
    assert ids.min() >= 0 and ids.max() < 16, "neuron id outside the layer"


def test_feast_thresholds_actually_adapt(feast_scene):
    """The 'adaptive selection thresholds' are the mechanism, not decoration.
    §3.9's lesson applies: a measured result about a feature that never ran is
    worthless, so assert the thresholds moved off their initial value."""
    from orbitsight.feast import FEAST
    ev, _ = feast_scene
    f = FEAST((180, 240), n_neurons=16, radius=4, seed=0)
    th0 = f.th.copy()
    f(ev)
    assert not np.allclose(f.th, th0), "no threshold moved — the layer never adapted"
    assert f.win_counts.sum() > 0, "no neuron ever won"
    assert (f.th >= 1e-4).all() and (f.th <= 0.999).all(), "threshold left its clip range"


def test_feast_is_deterministic(feast_scene):
    """`Determinism` in CLAUDE.md is a build constraint: a reviewer re-running
    the container must get our numbers. FEAST seeds its own weights, and that
    was never checked."""
    from orbitsight.feast import FEAST
    ev, _ = feast_scene
    a, ia = FEAST((180, 240), n_neurons=16, radius=4, seed=0)(ev)
    b, ib = FEAST((180, 240), n_neurons=16, radius=4, seed=0)(ev)
    assert np.array_equal(a, b) and np.array_equal(ia, ib), \
        "same seed gave a different feature stream"


def test_feast_runs_end_to_end_and_does_not_destroy_accuracy(feast_scene):
    """The ablation says 2.4x latency for F1 0.779 -> 0.776: a real cost and no
    gain, which is why it ships off. That is only an honest report if the path
    still works — an ablation of a broken feature measures nothing."""
    ev, gt = feast_scene
    scores = {}
    for use in (False, True):
        pipe = Pipeline((180, 240), Config(window_us=10_000, use_feast=use))
        pipe.calibrate(ev)
        res = pipe.run(ev)
        scores[use] = metrics.evaluate(res, gt, dist_px=5.0, key="tracks")
    assert scores[True]["object_recall"] == scores[False]["object_recall"] == 1.0, \
        "the target was lost with or without the feature layer"
    assert scores[True]["f1"] >= scores[False]["f1"] - 0.15, \
        (f"FEAST cost far more F1 than the logged ablation: "
         f"{scores[False]['f1']:.3f} -> {scores[True]['f1']:.3f}")


def test_feast_stays_off_in_the_frozen_profile():
    """Every published number comes from the synthetic profile with FEAST off.
    A silent flip would invalidate all of them and cost 2.4x the latency."""
    from orbitsight import profiles
    assert profiles.get("synthetic")["use_feast"] is False
    assert Config.from_profile("synthetic").use_feast is False


# ------------------------------------------------------- the EBSSA loader
# Every real-data number in the proposal comes through `evio.load_ebssa`, and
# until 5 September nothing tested it. The failure it is most exposed to is the
# one its own docstring warns about: rebasing events and ground truth on
# different origins reports zero recall on a pipeline that is working perfectly.
# That is a silent failure, which is exactly what a regression guard is for.
#
# The recordings are gitignored and never ship, so these skip cleanly without
# them rather than failing on a machine that has only the source.
EBSSA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "ebssa")
LABELLED = os.path.join(EBSSA_DIR, "refslew_2106_davis_td_labelled.mat")
STAR_REF = os.path.join(EBSSA_DIR, "mag10.56tycho6604_2019_davis_td_labelled.mat")

needs_ebssa = pytest.mark.skipif(
    not os.path.exists(LABELLED),
    reason="download the optional EBSSA recordings into data/ebssa/")


@pytest.fixture(scope="module")
def ebssa():
    return evio.load_ebssa(LABELLED)


@needs_ebssa
def test_ebssa_events_and_labels_share_one_time_origin(ebssa):
    """The silent killer. EBSSA clocks start near 1e9 us and `load()` rebases
    events to zero; rebasing the labels separately — or forgetting to — leaves
    every label pointing at a moment the events never cover."""
    ev, gt, _ = ebssa
    assert len(gt), "fixture recording lost its labels"
    t0, t1 = int(ev["t"][0]), int(ev["t"][-1])
    assert t0 == 0, f"events were not rebased to zero (start {t0})"
    assert t0 <= gt["t"].min() and gt["t"].max() <= t1, (
        f"labels span {gt['t'].min()}–{gt['t'].max()} us but events span {t0}–{t1}: "
        "the two were rebased on different origins")


@needs_ebssa
def test_ebssa_labels_land_on_real_event_activity(ebssa):
    """The coordinate convention was chosen by testing seven candidate mappings,
    and that choice lived only in a document. A transposed or 1-indexed mapping
    still produces in-bounds coordinates, so bounds are not the test: the test is
    whether the sensor actually saw something where the label says it did."""
    ev, gt, shape = ebssa
    H, W = shape
    assert (gt["x"] >= 0).all() and (gt["x"] < W).all(), "label x outside the sensor"
    assert (gt["y"] >= 0).all() and (gt["y"] < H).all(), "label y outside the sensor"

    rng = np.random.default_rng(0)
    on_target, elsewhere = 0, 0
    for g in gt[:: max(1, len(gt) // 40)]:
        near = ev[np.abs(ev["t"] - g["t"]) < 100_000]
        if not len(near):
            continue
        on_target += int((np.hypot(near["x"] - g["x"],
                                   near["y"] - g["y"]) <= 4.0).sum())
        fx, fy = rng.uniform(0, W), rng.uniform(0, H)
        elsewhere += int((np.hypot(near["x"] - fx, near["y"] - fy) <= 4.0).sum())
    assert on_target > 3 * max(elsewhere, 1), (
        f"labelled positions carry {on_target} events against {elsewhere} at random "
        "positions — the coordinate convention is wrong")


@needs_ebssa
def test_ebssa_reads_the_sensor_shape_from_the_file(ebssa):
    """Read the resolution from the recording instead of assuming a sensor."""
    _, _, shape = ebssa
    assert shape == (180, 240), f"DAVIS recording read back as {shape}"


@needs_ebssa
@pytest.mark.skipif(not os.path.exists(STAR_REF), reason="star reference absent")
def test_ebssa_star_references_are_returned_unlabelled():
    """The magnitude-ladder files carry nObj = 0: they are star references, not
    RSO ground truth. They must come back with no labels so they cannot be
    quietly scored as if they had some — a recall of 0/0 is not a measurement."""
    _, gt, _ = evio.load_ebssa(STAR_REF)
    assert len(gt) == 0, "an nObj = 0 star reference came back with labels"


@needs_ebssa
def test_ebssa_ground_truth_carries_no_invented_magnitude(ebssa):
    """EBSSA does not publish a per-object magnitude. NaN is the honest value;
    a real number here would be a fabricated one flowing into the results."""
    _, gt, _ = ebssa
    assert np.isnan(gt["mag"]).all(), "ground-truth magnitude is not NaN"
