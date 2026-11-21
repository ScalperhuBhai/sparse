#!/usr/bin/env python3
"""Score the `sparse-sky` profile across every labelled EBSSA recording.

One recording is an anecdote. `ISS_2` is a hard one — a faint tracked object
against dense shot noise — and quoting only it understates the pipeline as much
as quoting only an easy GEO stare would overstate it. This runs the whole set
with **one frozen profile**, no per-recording tuning, and writes a table.

Per-recording tuning would be the obvious way to make this table look better and
would make it worthless: the number a reviewer cares about is what one
configuration does on data it was not fitted to. Only `ISS_2` was used for
calibration; everything else here is held out.

  python3 scripts/eval_ebssa.py --out results/ebssa_sweep.json
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import evio, metrics
from orbitsight.pipeline import Pipeline, Config


def run_one(path: str, profile: str, dist_px: float,
            overrides: dict | None = None) -> dict | None:
    try:
        ev, gt, shape = evio.load_ebssa(path)
    except Exception as e:
        return dict(file=os.path.basename(path), error=str(e))
    if len(gt) == 0:
        return None                       # nObj = 0: a star reference, not scoreable

    cfg = Config.from_profile(profile, **(overrides or {}))
    pipe = Pipeline(shape, cfg)
    pipe.calibrate(ev[:min(len(ev), 200_000)])
    t0 = time.time()
    res = pipe.run(ev)
    wall = time.time() - t0
    rep = pipe.latency_report()
    m = metrics.evaluate(res, gt, dist_px=dist_px, key="tracks")
    c = metrics.track_census(res, gt, dist_px=dist_px)

    # object speeds, so the table can separate what is reachable from what is not
    speeds = []
    for oid in np.unique(gt["obj_id"]):
        g = gt[gt["obj_id"] == oid]
        dt = np.diff(g["t"]).astype(np.float64)
        dp = np.hypot(np.diff(g["x"]), np.diff(g["y"]))
        ok = dt > 0
        if ok.any():
            speeds.append(float(np.median(dp[ok] / (dt[ok] / 1e6))))
    slow = sum(1 for s in speeds if s < 60.0)

    dur = float(ev["t"][-1] - ev["t"][0]) / 1e6
    return dict(
        file=os.path.basename(path), sensor=f"{shape[1]}x{shape[0]}",
        events=len(ev), duration_s=round(dur, 1),
        objects=int(len(np.unique(gt["obj_id"]))), gt_points=len(gt),
        slow_objects=slow, median_speed_px_s=round(float(np.median(speeds)), 1) if speeds else None,
        precision=round(m["precision"], 4), recall=round(m["recall"], 4),
        f1=round(m["f1"], 4), objects_detected=m["objects_detected"],
        object_recall=round(m["object_recall"], 3),
        loc_err_px=round(m["mean_localisation_error_px"], 2)
        if m["mean_localisation_error_px"] else None,
        tracks_reported=c["tracks_reported"], tracks_true=c["tracks_true"],
        tracks_false=c["tracks_false"],
        s_between_false=round(c["seconds_between_false_tracks"], 1)
        if c["seconds_between_false_tracks"] else None,
        mean_ms=round(rep["total"]["mean_ms"], 3),
        p99_ms=round(rep["total"]["p99_ms"], 3),
        over_budget_pct=round(rep["summary"]["frac_windows_over_budget"] * 100, 3),
        realtime_x=round(rep["summary"]["realtime_factor"], 1),
        wall_s=round(wall, 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/ebssa/*.mat")
    ap.add_argument("--files", nargs="*", default=[], metavar="SUBSTRING",
                    help="score only recordings whose filename contains one of "
                         "these, so a named subset is reproducible from the "
                         "command line. A full-set sweep loads multi-million-"
                         "event recordings back to back and has been killed by "
                         "this host's memory guard before now.")
    ap.add_argument("--profile", default="sparse-sky")
    ap.add_argument("--dist-px", type=float, default=5.0)
    ap.add_argument("--out", default="results/ebssa_sweep.json")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override a Config field, e.g. --set hot_neighbour_ratio=0.3")
    a = ap.parse_args()

    overrides = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        try:
            overrides[k] = int(v) if v.isdigit() else float(v)
        except ValueError:
            overrides[k] = v
    if overrides:
        print(f"overrides: {overrides}")

    files = sorted(glob.glob(a.glob))
    if a.files:
        want = [w.lower() for w in a.files]
        files = [f for f in files
                 if any(w in os.path.basename(f).lower() for w in want)]
        missing = [w for w in want
                   if not any(w in os.path.basename(f).lower() for f in files)]
        if missing:
            sys.exit(f"no recording matches: {', '.join(missing)}")
    if not files:
        sys.exit(f"no recordings match {a.glob!r}; download EBSSA recordings first")
    print(f"{len(files)} recordings, profile '{a.profile}', no per-file tuning\n")
    rows, skipped = [], 0
    for p in files:
        r = run_one(p, a.profile, a.dist_px, overrides)
        if r is None:
            skipped += 1
            continue
        if "error" in r:
            print(f"  ERROR {r['file']}: {r['error']}")
            continue
        rows.append(r)
        print(f"  {r['file']:46s} {r['sensor']:>8s} {r['objects']:2d}obj "
              f"({r['slow_objects']} slow)  F1 {r['f1']:.3f}  "
              f"objR {r['object_recall']:.2f}  "
              f"trk {r['tracks_true']}/{r['tracks_reported']}  "
              f"{r['mean_ms']:.2f}ms  {r['realtime_x']:.0f}x", flush=True)

    print(f"\n{len(rows)} scoreable, {skipped} skipped as nObj=0 star references")
    if not rows:
        sys.exit("no labelled recordings could be scored")
    if rows:
        f1 = np.array([r["f1"] for r in rows])
        objr = np.array([r["object_recall"] for r in rows])
        ms = np.array([r["mean_ms"] for r in rows])
        ob = np.array([r["over_budget_pct"] for r in rows])
        # weight object recall by how many objects are actually within the
        # detection regime, since the transits are a known, stated limitation
        tot_obj = sum(r["objects"] for r in rows)
        tot_slow = sum(r["slow_objects"] for r in rows)
        tot_det = sum(r["objects_detected"] for r in rows)
        print(f"\n  F1            median {np.median(f1):.3f}  mean {f1.mean():.3f}  "
              f"best {f1.max():.3f}")
        print(f"  object recall median {np.median(objr):.3f}  mean {objr.mean():.3f}")
        print(f"  objects       {tot_det} detected of {tot_obj} labelled "
              f"({tot_slow} of which move < 60 px/s and are in regime)")
        print(f"  latency       mean {ms.mean():.3f} ms   worst over-budget {ob.max():.3f} %")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
