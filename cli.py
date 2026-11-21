#!/usr/bin/env python3
"""OrbitSight command line.

  python cli.py synth   --out data/scene.npz --shape 480x640 --duration 2.0
  python cli.py run     --input data/scene.npz --out out/ [--gt data/scene_gt.npy]
  python cli.py ablate  --out out/ablation.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

from orbitsight import evio, synth, metrics
from orbitsight.pipeline import Pipeline, Config


def _shape(s):
    h, w = s.lower().split("x")
    return (int(h), int(w))


def cmd_synth(a):
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cfg, rsos = synth.default_scene(shape=_shape(a.shape), duration_s=a.duration,
                                    mags=tuple(a.mags), seed=a.seed)
    cfg.noise_rate_hz = a.noise
    t0 = time.time()
    ev, gt = synth.generate(cfg, rsos)
    evio.save_npz(a.out, ev)
    np.save(a.out.replace(".npz", "_gt.npy"), gt)
    print(f"generated {len(ev):,} events, {len(gt):,} gt samples in {time.time()-t0:.1f}s")
    print(f"  -> {a.out}\n  -> {a.out.replace('.npz','_gt.npy')}")
    print(f"  event rate: {len(ev)/a.duration:,.0f} ev/s   shape {cfg.shape}")


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    gt_inline = None
    if a.input.lower().endswith(".mat"):
        # EBSSA recordings carry their own ground truth, and it must be rebased
        # on the same clock as the events — so load them together rather than
        # letting --gt point at a separately-rebased file.
        ev, gt_inline, mat_shape = evio.load_ebssa(a.input)
        shape = _shape(a.shape) if a.shape else mat_shape
        if len(gt_inline) == 0:
            print(f"note: {os.path.basename(a.input)} has no labelled objects "
                  f"(nObj=0) — running detection only, no accuracy metrics")
            gt_inline = None
    else:
        ev = evio.load(a.input)
        shape = _shape(a.shape) if a.shape else evio.sensor_shape(ev)
    cfg = Config.from_profile(a.profile, window_us=a.window, use_feast=a.feast or None,
                              cell=a.cell, min_events=a.min_events,
                              min_cluster_events=a.min_cluster_events)
    pipe = Pipeline(shape, cfg)
    lead = ev[:min(len(ev), 200_000)]
    pipe.calibrate(lead)
    scales = ", ".join(f"{s.label}:{s.windows}w"
                       f"({s.windows*cfg.window_us/1000:.0f}ms)" for s in cfg.scale_specs())
    print(f"loaded {len(ev):,} events, sensor {shape}, window {cfg.window_us} us")
    print(f"profile '{a.profile}'  scales [{scales}]  "
          f"baf {cfg.baf_dt_us}us r{cfg.baf_radius}"
          f"{'+self' if cfg.baf_include_self else ''}")
    t0 = time.time()
    results = pipe.run(ev, progress=a.verbose)
    wall = time.time() - t0
    rep = pipe.latency_report()

    with open(os.path.join(a.out, "results.jsonl"), "w") as f:
        for w in results:
            f.write(json.dumps(w) + "\n")
    with open(os.path.join(a.out, "latency.json"), "w") as f:
        json.dump(rep, f, indent=2)

    s = rep["summary"]
    print(f"\nprocessed {s['windows']} windows in {wall:.2f}s "
          f"({s['realtime_factor']:.1f}x real time)")
    print(f"throughput {s['throughput_ev_per_s']:,.0f} ev/s")
    for k in ("filter", "feast", "detect", "track", "total"):
        if k in rep and rep[k]["mean_ms"] > 0:
            r = rep[k]
            print(f"  {k:<7} mean {r['mean_ms']:6.2f} ms  p95 {r['p95_ms']:6.2f}  "
                  f"p99 {r['p99_ms']:6.2f}  max {r['max_ms']:6.2f}")
    print(f"windows over the 40 ms budget: {s['frac_windows_over_budget']*100:.2f}%")

    if a.gt or gt_inline is not None:
        gt = gt_inline if gt_inline is not None else np.load(a.gt)
        for key in ("detections", "tracks"):
            m = metrics.evaluate(results, gt, dist_px=a.dist_px, key=key)
            print(f"\n[{key}] P {m['precision']:.3f}  R {m['recall']:.3f}  "
                  f"F1 {m['f1']:.3f}  obj-recall {m['object_recall']:.2f} "
                  f"({m['objects_detected']}/{m['objects_in_gt']})")
            if m["mean_localisation_error_px"] is not None:
                print(f"          localisation err mean {m['mean_localisation_error_px']:.2f} px"
                      f"  p95 {m['p95_localisation_error_px']:.2f} px")
            if key == "tracks":
                c = metrics.track_census(results, gt, dist_px=a.dist_px)
                m["census"] = c
                rate = c["seconds_between_false_tracks"]
                print(f"          distinct tracks {c['tracks_reported']} "
                      f"({c['tracks_true']} true, {c['tracks_false']} false)"
                      + (f"  -> one false track every {rate:.0f} s" if rate else ""))
            with open(os.path.join(a.out, f"metrics_{key}.json"), "w") as f:
                json.dump(m, f, indent=2)


def cmd_ablate(a):
    """Sweep brightness and sensor resolution — the two axes the brief names."""
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = []
    for shape in [(180, 240), (240, 346), (480, 640)]:
        for mag in [6.0, 8.0, 10.0, 11.5, 12.5]:
            cfg, rsos = synth.default_scene(shape=shape, duration_s=a.duration,
                                            mags=(mag,), seed=a.seed)
            ev, gt = synth.generate(cfg, rsos)
            pipe = Pipeline(shape, Config(window_us=a.window))
            pipe.calibrate(ev[:100_000])
            res = pipe.run(ev)
            m = metrics.evaluate(res, gt, dist_px=5.0, key="tracks")
            rep = pipe.latency_report()
            row = dict(shape=f"{shape[0]}x{shape[1]}", mag=mag, events=len(ev),
                       f1=m["f1"], precision=m["precision"], recall=m["recall"],
                       obj_recall=m["object_recall"],
                       loc_err=m["mean_localisation_error_px"],
                       p95_latency_ms=rep["total"]["p95_ms"],
                       mean_latency_ms=rep["total"]["mean_ms"])
            rows.append(row)
            print(f"{row['shape']:>9}  mag {mag:5.1f}  F1 {row['f1']:.3f}  "
                  f"objR {row['obj_recall']:.2f}  p95 {row['p95_latency_ms']:6.2f} ms",
                  flush=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {a.out}")


def main():
    p = argparse.ArgumentParser(prog="orbitsight")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate a labelled synthetic recording")
    s.add_argument("--out", default="data/scene.npz")
    s.add_argument("--shape", default="480x640")
    s.add_argument("--duration", type=float, default=2.0)
    s.add_argument("--mags", type=float, nargs="+", default=[6.0, 9.0, 11.5])
    s.add_argument("--noise", type=float, default=0.12)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    r = sub.add_parser("run", help="run the detection + tracking pipeline")
    r.add_argument("--input", required=True)
    r.add_argument("--out", default="out")
    r.add_argument("--gt", default=None)
    r.add_argument("--shape", default=None)
    r.add_argument("--profile", default="synthetic",
                   help="observing profile: synthetic (default, the canonical scene) "
                        "or sparse-sky (real telescope NVS, calibrated on EBSSA)")
    r.add_argument("--window", type=int, default=None,
                   help="window length in us; default comes from the profile")
    r.add_argument("--cell", type=int, default=None)
    r.add_argument("--min-events", dest="min_events", type=int, default=None)
    r.add_argument("--min-cluster-events", dest="min_cluster_events", type=int, default=None)
    r.add_argument("--dist-px", dest="dist_px", type=float, default=5.0)
    r.add_argument("--feast", action="store_true")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    ab = sub.add_parser("ablate", help="brightness x resolution sweep")
    ab.add_argument("--out", default="out/ablation.json")
    ab.add_argument("--duration", type=float, default=1.5)
    ab.add_argument("--window", type=int, default=10_000)
    ab.add_argument("--seed", type=int, default=0)
    ab.set_defaults(func=cmd_ablate)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
