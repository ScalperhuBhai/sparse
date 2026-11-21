"""Detection and tracking metrics against sub-pixel ground truth."""
from __future__ import annotations

import numpy as np


def _gt_at(gt: np.ndarray, t: int, tol_us: int):
    m = np.abs(gt["t"] - t) <= tol_us
    if not m.any():
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int32)
    sub = gt[m]
    # one entry per object: the sample nearest in time
    out_xy, out_id = [], []
    for oid in np.unique(sub["obj_id"]):
        s = sub[sub["obj_id"] == oid]
        j = int(np.argmin(np.abs(s["t"] - t)))
        out_xy.append((s["x"][j], s["y"][j]))
        out_id.append(oid)
    return np.array(out_xy, dtype=np.float64), np.array(out_id, dtype=np.int32)


def evaluate(results, gt: np.ndarray, dist_px: float = 5.0,
             tol_us: int = 5000, key: str = "detections") -> dict:
    """Greedy nearest-neighbour matching per window, at a pixel-distance gate."""
    TP = FP = FN = 0
    errors = []
    matched_ids = set()
    seen_ids = set()
    for w in results:
        t = w["t"]
        gxy, gid = _gt_at(gt, t, tol_us)
        seen_ids.update(int(i) for i in gid)
        pxy = np.array([[d["x"], d["y"]] for d in w[key]], dtype=np.float64) \
            if w[key] else np.zeros((0, 2))
        if len(gxy) == 0:
            FP += len(pxy)
            continue
        if len(pxy) == 0:
            FN += len(gxy)
            continue
        D = np.linalg.norm(pxy[:, None, :] - gxy[None, :, :], axis=2)
        used_p, used_g = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
        for pi, gi in order:
            if D[pi, gi] > dist_px:
                break
            if pi in used_p or gi in used_g:
                continue
            used_p.add(int(pi)); used_g.add(int(gi))
            errors.append(float(D[pi, gi]))
            matched_ids.add(int(gid[gi]))
        TP += len(used_g)
        FP += len(pxy) - len(used_p)
        FN += len(gxy) - len(used_g)

    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(
        TP=TP, FP=FP, FN=FN, precision=prec, recall=rec, f1=f1,
        mean_localisation_error_px=float(np.mean(errors)) if errors else None,
        p95_localisation_error_px=float(np.percentile(errors, 95)) if errors else None,
        objects_in_gt=len(seen_ids), objects_detected=len(matched_ids),
        object_recall=len(matched_ids) / len(seen_ids) if seen_ids else 0.0,
    )


def track_census(results, gt: np.ndarray, dist_px: float = 5.0,
                 tol_us: int = 5000) -> dict:
    """Count *distinct* tracks, not per-window hits.

    Per-window precision is the metric the scoring asks for, but on its own it
    misreads what an operator sees. A track that confirms once and then coasts
    for three seconds contributes three hundred window-level false positives
    while being, to the person watching the screen, exactly one spurious track.
    Both numbers are true and they answer different questions; this one answers
    "how often does the system cry wolf".

    A track is counted true if it ever came within `dist_px` of any ground-truth
    object at a time it was being reported.
    """
    seen, good = set(), set()
    span_us = 0
    if results:
        span_us = max(w["t"] for w in results) - min(w["t"] for w in results)
    for w in results:
        if not w["tracks"]:
            continue
        m = np.abs(gt["t"] - w["t"]) <= tol_us
        g = gt[m]
        for tr in w["tracks"]:
            seen.add(tr["track_id"])
            if len(g) and np.min(np.hypot(g["x"] - tr["x"],
                                          g["y"] - tr["y"])) <= dist_px:
                good.add(tr["track_id"])
    false = len(seen) - len(good)
    minutes = span_us / 60e6 if span_us else 0.0
    return dict(
        tracks_reported=len(seen), tracks_true=len(good), tracks_false=false,
        false_track_rate_per_min=float(false / minutes) if minutes else None,
        seconds_between_false_tracks=float(span_us / 1e6 / false) if false else None,
    )
