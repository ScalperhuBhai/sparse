"""Tracking: constant-velocity Kalman filter with probabilistic data association.

Motion model is [x, y, vx, vy] with a variable time step, because event windows
do not arrive on a fixed cadence. Association is Mahalanobis-gated; inside a
gate, PDA blends the candidate measurements by likelihood rather than committing
to the nearest one, which is what keeps a faint target alive when a star
scintillates next to it. Tracks confirm on m-of-n and die on consecutive misses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Track:
    track_id: int
    x: np.ndarray                       # state [x, y, vx, vy]
    P: np.ndarray                       # covariance
    t: int
    hits: int = 1
    misses: int = 0
    age: int = 1
    score: float = 0.0                  # accumulated detection evidence
    history: list = field(default_factory=list)
    confirmed: bool = False

    @property
    def pos(self):
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self):
        return float(self.x[2]), float(self.x[3])

    def as_dict(self):
        return dict(track_id=self.track_id, t=int(self.t),
                    x=float(self.x[0]), y=float(self.x[1]),
                    vx=float(self.x[2]), vy=float(self.x[3]),
                    hits=self.hits, misses=self.misses, confirmed=self.confirmed,
                    score=round(float(self.score), 2))


class PDATracker:
    def __init__(self, q_accel=250.0, r_pos=2.0, gate_chi2=16.0,
                 confirm_hits=3, confirm_window=5, max_misses=4,
                 p_detect=0.85, clutter_density=1e-5,
                 min_speed_px_s=0.0, max_speed_px_s=4000.0,
                 max_assoc_px=25.0, init_speed_sigma_px_s=None,
                 confirm_score=0.0, min_confirm_hits=2):
        self.q = float(q_accel)              # px/s^2, process noise
        self.R = np.eye(2) * float(r_pos) ** 2
        self.gate = float(gate_chi2)         # chi-square gate, 2 dof
        self.confirm_hits = confirm_hits
        self.confirm_window = confirm_window
        self.max_misses = max_misses
        self.pd = p_detect
        self.lam = clutter_density
        self.min_speed = min_speed_px_s
        self.max_speed = max_speed_px_s
        # A hard spatial gate alongside the chi-square one. Without it a fresh
        # track's velocity covariance inflates S so far that every measurement
        # in the frame falls inside the ellipse, and clutter confirms tracks.
        self.max_assoc_px = float(max_assoc_px)
        # Evidence-based confirmation. A fixed hit count says a target must be
        # seen N times, which at a 1 Hz scale means it must exist for N seconds.
        # Real recordings are full of objects that cross in three. Accumulating
        # detection strength lets an unambiguous target confirm in two or three
        # firings while a marginal one still has to earn the full count — the
        # threshold is on evidence, which is what m-of-n was always a proxy for.
        # 0 disables, leaving pure m-of-n (what the synthetic profile uses).
        self.confirm_score = float(confirm_score)
        self.min_confirm_hits = int(min_confirm_hits)
        self.init_speed_sigma = float(init_speed_sigma_px_s) \
            if init_speed_sigma_px_s is not None else max_speed_px_s / 6.0
        self.tracks: list[Track] = []
        self._next_id = 1

    # -- Kalman primitives -------------------------------------------------
    def _predict(self, tr: Track, t: int):
        dt = max((t - tr.t) * 1e-6, 1e-6)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        Q = G @ (np.eye(2) * self.q ** 2) @ G.T
        tr.x = F @ tr.x
        tr.P = F @ tr.P @ F.T + Q
        tr.t = t

    def _innovation(self, tr: Track):
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        S = H @ tr.P @ H.T + self.R
        K = tr.P @ H.T @ np.linalg.inv(S)
        return H, S, K

    def coast(self, t: int) -> list[Track]:
        """Predict confirmed tracks forward without consuming a detection.

        A scale that integrates over 250 ms only produces measurements at 4 Hz,
        but the state estimate it carries is still valid in between — that is
        what a motion model is for. Coasting reports the current position at
        the window cadence without accruing a miss, so a slow scale is not
        punished for reporting slowly. Misses are only counted on windows where
        the scale actually looked and saw nothing.
        """
        out = []
        for tr in self.tracks:
            if tr.confirmed and t > tr.t:
                self._predict(tr, t)
                tr.history.append((int(t), float(tr.x[0]), float(tr.x[1])))
            if tr.confirmed:
                out.append(tr)
        return out

    # -- main step ---------------------------------------------------------
    def update(self, detections, t: int) -> list[Track]:
        meas = np.array([[d.x, d.y] for d in detections], dtype=np.float64) \
            if detections else np.zeros((0, 2))
        for tr in self.tracks:
            self._predict(tr, t)

        claimed = np.zeros(len(meas), dtype=bool)
        for tr in self.tracks:
            H, S, K = self._innovation(tr)
            if len(meas):
                nu = meas - (H @ tr.x)[None, :]
                Sinv = np.linalg.inv(S)
                d2 = np.einsum("ij,jk,ik->i", nu, Sinv, nu)
                dist = np.linalg.norm(nu, axis=1)
                inside = (d2 <= self.gate) & (dist <= self.max_assoc_px)
            else:
                inside = np.zeros(0, dtype=bool)

            if inside.any():
                nu_in = nu[inside]
                d2_in = d2[inside]
                # PDA association weights
                norm = 1.0 / np.sqrt(np.linalg.det(2 * np.pi * S))
                like = self.pd * norm * np.exp(-0.5 * d2_in)
                b0 = self.lam * (1.0 - self.pd)
                beta = like / (like.sum() + b0)
                nu_bar = (beta[:, None] * nu_in).sum(axis=0)
                tr.x = tr.x + K @ nu_bar
                # covariance with the standard PDA spread-of-innovations term
                b_none = b0 / (like.sum() + b0)
                Pc = tr.P - K @ S @ K.T
                spread = sum(beta[i] * np.outer(nu_in[i], nu_in[i])
                             for i in range(len(beta))) - np.outer(nu_bar, nu_bar)
                tr.P = b_none * tr.P + (1 - b_none) * Pc + K @ spread @ K.T
                tr.hits += 1
                tr.misses = 0
                # strongest measurement inside the gate; PDA already blended the
                # positions, but evidence should reflect the best look we got
                # getattr, not attribute access: the tracker only needs x and y
                # from a measurement, and demanding a score would couple it to
                # one detector's output shape.
                tr.score += max((getattr(detections[i], "score", 0.0)
                                 for i in np.nonzero(inside)[0]), default=0.0)
                claimed[np.nonzero(inside)[0]] = True
            else:
                tr.misses += 1
            tr.age += 1
            enough = tr.hits >= self.confirm_hits
            if self.confirm_score > 0 and tr.hits >= self.min_confirm_hits \
                    and tr.score >= self.confirm_score:
                enough = True
            if not tr.confirmed and enough \
                    and tr.age <= self.confirm_window + self.confirm_hits:
                speed = np.hypot(tr.x[2], tr.x[3])
                if self.min_speed <= speed <= self.max_speed:
                    tr.confirmed = True
            if tr.confirmed:
                tr.history.append((int(t), float(tr.x[0]), float(tr.x[1])))

        # spawn tentative tracks from unclaimed detections
        for i, used in enumerate(claimed):
            if used:
                continue
            P0 = np.diag([4.0, 4.0, self.init_speed_sigma ** 2, self.init_speed_sigma ** 2])
            self.tracks.append(Track(self._next_id, np.array([meas[i, 0], meas[i, 1], 0.0, 0.0]),
                                     P0, t))
            self._next_id += 1

        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]
        return [tr for tr in self.tracks if tr.confirmed]
