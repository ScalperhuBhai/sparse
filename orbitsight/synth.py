"""Synthetic NVS scene generator for a sidereally-tracked telescope.

Why this exists: the challenge explicitly asks for performance "across varying
object brightness levels" and "compatibility across camera resolutions". Real
recordings rarely sweep those axes cleanly. This generator produces labelled
event streams on demand, so ablations become a loop instead of a data-collection
campaign, and every detection has exact sub-pixel ground truth.

Sensor model is the standard DVS one: an event fires whenever the log-intensity
at a pixel departs from its stored reference by the contrast threshold C, and
the reference is then advanced by C in the direction of the change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .evio import from_arrays


@dataclass
class RSO:
    """One resident space object crossing the field of view."""
    x0: float
    y0: float
    vx: float               # pixels / second
    vy: float
    mag: float = 8.0        # apparent magnitude; higher = fainter
    sigma: float = 1.1      # PSF sigma in pixels
    obj_id: int = 0


@dataclass
class SceneConfig:
    shape: tuple[int, int] = (480, 640)     # (H, W)
    duration_s: float = 2.0
    dt_us: int = 500                        # simulation step
    contrast: float = 0.18                  # log-intensity threshold C
    background: float = 40.0                # DN
    n_stars: int = 220
    star_mag_range: tuple[float, float] = (4.0, 11.0)
    star_jitter: float = 0.035              # scintillation, fractional
    noise_rate_hz: float = 0.12             # per pixel per second (background activity)
    n_hot_pixels: int = 12
    hot_rate_hz: float = 900.0
    mag_zero_point: float = 3.0             # mag at which peak amplitude == 1e4 DN
    seed: int = 0


def _amplitude(mag: float, zp: float) -> float:
    return 1e4 * 10.0 ** (-0.4 * (mag - zp))


def _gauss_patch(cx: float, cy: float, sigma: float, shape, radius=None):
    H, W = shape
    r = radius if radius is not None else max(2, int(np.ceil(4 * sigma)))
    x0, x1 = int(np.floor(cx)) - r, int(np.floor(cx)) + r + 1
    y0, y1 = int(np.floor(cy)) - r, int(np.floor(cy)) + r + 1
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    return (slice(y0, y1), slice(x0, x1)), g.astype(np.float32)


def generate(cfg: SceneConfig, rsos: list[RSO]) -> tuple[np.ndarray, np.ndarray]:
    """Return (events, ground_truth).

    ground_truth dtype: t(int64 us), x(f4), y(f4), obj_id(i4), mag(f4)
    """
    rng = np.random.default_rng(cfg.seed)
    H, W = cfg.shape
    eps = 1e-3

    # ---- static scene: background + star field -------------------------
    scene = np.full((H, W), cfg.background, dtype=np.float32)
    star_slices = []
    for _ in range(cfg.n_stars):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        mag = rng.uniform(*cfg.star_mag_range)
        sl, g = _gauss_patch(cx, cy, rng.uniform(0.9, 1.6), (H, W)) or (None, None)
        if sl is None:
            continue
        amp = _amplitude(mag, cfg.mag_zero_point)
        scene[sl] += amp * g
        if amp > 3 * cfg.background:      # only bright stars twinkle visibly
            star_slices.append((sl, (amp * g).astype(np.float32)))

    Lref = np.log(scene + eps).astype(np.float32)

    n_steps = int(cfg.duration_s * 1e6 / cfg.dt_us)
    ev_t, ev_x, ev_y, ev_p = [], [], [], []
    gt = []

    def emit(sl, L_new, t_us):
        """Compare a patch against the reference surface and fire events."""
        ref = Lref[sl]
        diff = L_new - ref
        n = np.floor(np.abs(diff) / cfg.contrast).astype(np.int32)
        if not n.any():
            return
        yy, xx = np.nonzero(n)
        counts = n[yy, xx]
        sgn = np.sign(diff[yy, xx]).astype(np.int8)
        rep = np.repeat(np.arange(len(counts)), counts)
        ys = (yy[rep] + sl[0].start).astype(np.uint16)
        xs = (xx[rep] + sl[1].start).astype(np.uint16)
        ps = sgn[rep]
        # spread event times inside the step so ordering stays realistic
        ts = t_us + rng.integers(0, cfg.dt_us, size=len(rep))
        ev_t.append(ts); ev_x.append(xs); ev_y.append(ys); ev_p.append(ps)
        Lref[sl][yy, xx] = ref[yy, xx] + sgn * counts * cfg.contrast

    for step in range(n_steps):
        t_us = int(step * cfg.dt_us)
        t_s = t_us * 1e-6

        # --- moving targets ---
        for r in rsos:
            cx, cy = r.x0 + r.vx * t_s, r.y0 + r.vy * t_s
            if not (-5 <= cx < W + 5 and -5 <= cy < H + 5):
                continue
            out = _gauss_patch(cx, cy, r.sigma, (H, W))
            if out is None:
                continue
            sl, g = out
            amp = _amplitude(r.mag, cfg.mag_zero_point)
            L_new = np.log(scene[sl] + amp * g + eps).astype(np.float32)
            emit(sl, L_new, t_us)
            if 0 <= cx < W and 0 <= cy < H:
                gt.append((t_us, cx, cy, r.obj_id, r.mag))

        # --- star scintillation (the dominant real-world clutter source) ---
        if cfg.star_jitter > 0 and star_slices:
            for sl, starlight in star_slices:
                f = 1.0 + cfg.star_jitter * rng.standard_normal()
                L_new = np.log(scene[sl] + (f - 1.0) * starlight + eps).astype(np.float32)
                emit(sl, L_new, t_us)

    # --- sensor background activity noise -------------------------------
    n_noise = rng.poisson(cfg.noise_rate_hz * H * W * cfg.duration_s)
    if n_noise:
        ev_t.append(rng.integers(0, int(cfg.duration_s * 1e6), size=n_noise))
        ev_x.append(rng.integers(0, W, size=n_noise).astype(np.uint16))
        ev_y.append(rng.integers(0, H, size=n_noise).astype(np.uint16))
        ev_p.append(rng.choice(np.array([-1, 1], dtype=np.int8), size=n_noise))

    # --- hot pixels ------------------------------------------------------
    for _ in range(cfg.n_hot_pixels):
        hx, hy = rng.integers(0, W), rng.integers(0, H)
        n_h = rng.poisson(cfg.hot_rate_hz * cfg.duration_s)
        if not n_h:
            continue
        ev_t.append(rng.integers(0, int(cfg.duration_s * 1e6), size=n_h))
        ev_x.append(np.full(n_h, hx, dtype=np.uint16))
        ev_y.append(np.full(n_h, hy, dtype=np.uint16))
        ev_p.append(rng.choice(np.array([-1, 1], dtype=np.int8), size=n_h))

    if not ev_t:
        return from_arrays([], [], [], []), np.zeros(0, dtype=[("t", "<i8")])

    events = from_arrays(np.concatenate(ev_t), np.concatenate(ev_x),
                         np.concatenate(ev_y), np.concatenate(ev_p))
    events = events[np.argsort(events["t"], kind="stable")]

    gt_arr = np.array(gt, dtype=[("t", "<i8"), ("x", "<f4"), ("y", "<f4"),
                                 ("obj_id", "<i4"), ("mag", "<f4")])
    return events, gt_arr


def default_scene(shape=(480, 640), duration_s=2.0, mags=(6.0, 9.0, 11.5), seed=0):
    """A three-target leap-frog scene spanning bright / medium / faint."""
    H, W = shape
    rng = np.random.default_rng(seed)
    rsos = []
    for i, m in enumerate(mags):
        y0 = H * (0.25 + 0.25 * i)
        speed = rng.uniform(0.25, 0.55) * W / duration_s
        rsos.append(RSO(x0=rng.uniform(0.02, 0.15) * W, y0=y0,
                        vx=speed, vy=rng.uniform(-0.06, 0.06) * H / duration_s,
                        mag=m, sigma=1.1, obj_id=i))
    cfg = SceneConfig(shape=shape, duration_s=duration_s, seed=seed)
    return cfg, rsos
