"""FEAST — Feature Extraction with Adaptive Selection Thresholds.

An unsupervised, online, spiking feature layer (Afshar et al.). Each neuron owns
a weight vector over a local time-surface patch and its own selection threshold.
On each event the winning neuron (max cosine similarity) either fires — pulling
its weights toward the input and *raising* its threshold — or, if nobody clears
threshold, every threshold drops a little until the layer starts responding.

Why it belongs in a three-day CPU build:
  * no labels, no backprop, no GPU — it adapts on the recording it is shown;
  * it is a genuine spiking mechanism, so the neuromorphic story is real rather
    than decorative;
  * output is a sparse event stream, so the downstream cost keeps falling.

Chunk-vectorised: similarity is one matmul per chunk instead of per event.
"""
from __future__ import annotations

import numpy as np


class FEAST:
    def __init__(self, shape, n_neurons=16, radius=4, tau_us=50_000,
                 eta=0.05, th_open=0.002, th_close=0.005, seed=0):
        self.H, self.W = shape
        self.R = int(radius)
        self.D = (2 * self.R + 1) ** 2
        self.tau = float(tau_us)
        self.eta = float(eta)
        self.th_open = float(th_open)     # threshold drop when nobody wins
        self.th_close = float(th_close)   # threshold rise when a neuron wins
        rng = np.random.default_rng(seed)
        W0 = rng.random((n_neurons, self.D)).astype(np.float32)
        self.Wt = W0 / np.linalg.norm(W0, axis=1, keepdims=True)
        self.th = np.full(n_neurons, 0.001, dtype=np.float32)
        self.T = np.full((self.H + 2 * self.R, self.W + 2 * self.R),
                         -(1 << 62), dtype=np.int64)
        self.win_counts = np.zeros(n_neurons, dtype=np.int64)

    def reset_surface(self):
        self.T.fill(-(1 << 62))

    def _contexts(self, ev_chunk):
        """Exponentially-decayed time-surface patches for a chunk of events."""
        R, k = self.R, 2 * self.R + 1
        yy = ev_chunk["y"].astype(np.int64) + R
        xx = ev_chunk["x"].astype(np.int64) + R
        dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
        rows = yy[:, None] + dy.ravel()[None, :]
        cols = xx[:, None] + dx.ravel()[None, :]
        Tp = self.T[rows, cols]
        dt = ev_chunk["t"][:, None] - Tp
        with np.errstate(over="ignore"):
            C = np.exp(-np.clip(dt, 0, None) / self.tau).astype(np.float32)
        C[dt > 20 * self.tau] = 0.0
        n = np.linalg.norm(C, axis=1, keepdims=True)
        np.maximum(n, 1e-6, out=n)
        return C / n, yy, xx

    def __call__(self, ev: np.ndarray, chunk=1024, learn=True):
        """Return (feature_events, neuron_id) for events that made a neuron fire."""
        if len(ev) == 0:
            return ev, np.zeros(0, dtype=np.int32)
        keep = np.zeros(len(ev), dtype=bool)
        ids = np.zeros(len(ev), dtype=np.int32)
        for s in range(0, len(ev), chunk):
            sl = slice(s, s + chunk)
            C, yy, xx = self._contexts(ev[sl])
            sim = C @ self.Wt.T                      # (n_ev, n_neurons)
            win = np.argmax(sim, axis=1)
            best = sim[np.arange(len(win)), win]
            fired = best > self.th[win]
            keep[sl] = fired
            ids[sl] = win
            if learn:
                if fired.any():
                    wf, Cf = win[fired], C[fired]
                    # one averaged update per winning neuron per chunk
                    for n_id in np.unique(wf):
                        m = wf == n_id
                        upd = Cf[m].mean(axis=0)
                        self.Wt[n_id] += self.eta * (upd - self.Wt[n_id])
                        self.Wt[n_id] /= np.linalg.norm(self.Wt[n_id]) + 1e-9
                        self.th[n_id] += self.th_close * m.sum() / len(win)
                        self.win_counts[n_id] += int(m.sum())
                miss = (~fired).sum()
                if miss:
                    self.th -= self.th_open * miss / len(win)
                    np.clip(self.th, 1e-4, 0.999, out=self.th)
            self.T[yy, xx] = ev["t"][sl]
        return ev[keep], ids[keep]
