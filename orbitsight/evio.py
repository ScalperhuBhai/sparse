"""Event I/O: read heterogeneous NVS formats into one structured NumPy array.

Canonical event array dtype:
    t : int64   microseconds, monotonically non-decreasing
    x : uint16  column
    y : uint16  row
    p : int8    polarity, +1 / -1

Deliberately dependency-light. Prophesee RAW/DAT is handled by `expelliarmus`
(pip-installable, no OpenEB build required); iniVation AEDAT4 by `dv-processing`
if present. Both are optional imports so the container still builds without them.
"""
from __future__ import annotations

import os
import numpy as np

EVENT_DTYPE = np.dtype([("t", "<i8"), ("x", "<u2"), ("y", "<u2"), ("p", "i1")])


def empty(n: int = 0) -> np.ndarray:
    return np.zeros(n, dtype=EVENT_DTYPE)


def from_arrays(t, x, y, p) -> np.ndarray:
    ev = empty(len(t))
    ev["t"] = np.asarray(t, dtype=np.int64)
    ev["x"] = np.asarray(x, dtype=np.uint16)
    ev["y"] = np.asarray(y, dtype=np.uint16)
    pol = np.asarray(p)
    if pol.dtype == bool:
        pol = pol.astype(np.int8) * 2 - 1
    pol = pol.astype(np.int8)
    pol[pol == 0] = -1
    ev["p"] = pol
    return ev


def _normalise(ev: np.ndarray) -> np.ndarray:
    """Sort by time and rebase timestamps to zero."""
    if len(ev) == 0:
        return ev
    if not np.all(np.diff(ev["t"]) >= 0):
        ev = ev[np.argsort(ev["t"], kind="stable")]
    ev = ev.copy()
    ev["t"] -= ev["t"][0]
    return ev


# --------------------------------------------------------------------------
# format readers
# --------------------------------------------------------------------------

def _read_npz(path: str) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    if isinstance(z, np.ndarray):
        if z.dtype.names and set(z.dtype.names) >= {"t", "x", "y", "p"}:
            return from_arrays(z["t"], z["x"], z["y"], z["p"])
        if z.ndim == 2 and z.shape[1] >= 4:  # assume t,x,y,p columns
            return from_arrays(z[:, 0], z[:, 1], z[:, 2], z[:, 3])
        raise ValueError(f"unrecognised array layout in {path}: {z.shape} {z.dtype}")
    keys = {k.lower(): k for k in z.files}
    def pick(*names):
        for n in names:
            if n in keys:
                return z[keys[n]]
        raise KeyError(f"none of {names} in {z.files}")
    return from_arrays(pick("t", "ts", "timestamp"), pick("x", "xs"),
                       pick("y", "ys"), pick("p", "pol", "polarity"))


def _read_hdf5(path: str) -> np.ndarray:
    import h5py  # optional
    with h5py.File(path, "r") as f:
        # walk for the first group that has t/x/y/p-ish members
        cand = []
        f.visit(lambda name: cand.append(name))
        names = set(cand)
        for base in ("events/", "CD/events/", ""):
            keys = {k.split("/")[-1] for k in names if k.startswith(base)}
            if {"x", "y"} <= keys and ({"t"} <= keys or {"ts"} <= keys):
                g = f[base] if base else f
                tkey = "t" if "t" in g else "ts"
                pkey = "p" if "p" in g else ("pol" if "pol" in g else "polarity")
                return from_arrays(g[tkey][:], g["x"][:], g["y"][:], g[pkey][:])
        # Prophesee HDF5 packs a compound dataset
        for k in cand:
            d = f[k]
            if isinstance(d, h5py.Dataset) and d.dtype.names and "x" in d.dtype.names:
                a = d[:]
                tkey = "t" if "t" in a.dtype.names else "ts"
                pkey = "p" if "p" in a.dtype.names else "polarity"
                return from_arrays(a[tkey], a["x"], a["y"], a[pkey])
    raise ValueError(f"no event dataset found in {path}")


def _read_prophesee(path: str) -> np.ndarray:
    """Prophesee .raw (EVT2/EVT2.1/EVT3) or .dat via expelliarmus."""
    from expelliarmus import Wizard  # optional
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    for encoding in (("evt3", "evt2", "evt21") if ext == "raw" else ("dat",)):
        try:
            wiz = Wizard(encoding=encoding, fpath=path)
            arr = wiz.read()
            if arr is not None and len(arr):
                return from_arrays(arr["t"], arr["x"], arr["y"], arr["p"])
        except Exception:
            continue
    raise ValueError(f"expelliarmus could not decode {path}")


def _read_aedat4(path: str) -> np.ndarray:
    import dv_processing as dv  # optional
    reader = dv.io.MonoCameraRecording(path)
    chunks = []
    while reader.isRunning():
        batch = reader.getNextEventBatch()
        if batch is None:
            continue
        a = batch.numpy()
        chunks.append(from_arrays(a["timestamp"], a["x"], a["y"], a["polarity"]))
    return np.concatenate(chunks) if chunks else empty()


def _read_mat(path: str) -> np.ndarray:
    """EBSSA-style MATLAB v7 recording (Afshar et al., Western Sydney ICNS).

    Uses `scipy.io.loadmat` — SciPy is already a hard dependency, so reading
    this format costs no new package and no new licence.

    Layout: a `TD` struct of x / y / ts / p, plus `Obj` ground truth, `nObj`,
    and `xMax` / `yMax` giving the sensor resolution. **Coordinates are
    1-indexed**, because MATLAB is; they are converted to 0-indexed here so the
    rest of the pipeline sees the same convention as every other reader.
    """
    from scipy.io import loadmat  # scipy is already required
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "TD" not in m:
        raise ValueError(f"no TD struct in {path}; not an EBSSA-style recording")
    TD = m["TD"]
    x = np.atleast_1d(TD.x).astype(np.int64) - 1
    y = np.atleast_1d(TD.y).astype(np.int64) - 1
    return from_arrays(np.atleast_1d(TD.ts), np.clip(x, 0, None),
                       np.clip(y, 0, None), np.atleast_1d(TD.p))


def _read_csv(path: str) -> np.ndarray:
    a = np.loadtxt(path, delimiter=",", skiprows=0, ndmin=2)
    return from_arrays(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


READERS = {
    ".npz": _read_npz, ".npy": _read_npz,
    ".h5": _read_hdf5, ".hdf5": _read_hdf5,
    ".raw": _read_prophesee, ".dat": _read_prophesee,
    ".aedat4": _read_aedat4,
    ".mat": _read_mat,
    ".csv": _read_csv, ".txt": _read_csv,
}


def load(path: str) -> np.ndarray:
    """Load any supported recording into the canonical event array."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in READERS:
        raise ValueError(f"unsupported extension {ext!r}; supported: {sorted(READERS)}")
    return _normalise(READERS[ext](path))


def save_npz(path: str, ev: np.ndarray, **meta) -> None:
    np.savez_compressed(path, t=ev["t"], x=ev["x"], y=ev["y"], p=ev["p"], **meta)


def sensor_shape(ev: np.ndarray, fallback=(720, 1280)) -> tuple[int, int]:
    """Infer (H, W) from the data, rounded up to a sane sensor size."""
    if len(ev) == 0:
        return fallback
    h, w = int(ev["y"].max()) + 1, int(ev["x"].max()) + 1
    for cand in [(180, 240), (240, 346), (480, 640), (720, 1280), (1080, 1920)]:
        if h <= cand[0] and w <= cand[1]:
            return cand
    return (h, w)


# --------------------------------------------------------------------------
# EBSSA labelled recordings: events and ground truth, on one clock
# --------------------------------------------------------------------------

GT_DTYPE = np.dtype([("t", "<i8"), ("x", "<f4"), ("y", "<f4"),
                     ("obj_id", "<i4"), ("mag", "<f4")])


def load_ebssa(path: str):
    """Load an EBSSA `.mat` recording as (events, ground_truth, sensor_shape).

    Both arrays are rebased by the SAME time origin. That matters more than it
    looks: `load()` rebases event timestamps to zero, and EBSSA clocks start
    around 1e9 us, so rebasing the ground truth independently — or not at all —
    silently shifts every label off its events and reports zero recall on a
    pipeline that is working perfectly.

    Ground truth carries no per-object magnitude (EBSSA does not label one), so
    `mag` is filled with NaN rather than a fabricated number.
    """
    from scipy.io import loadmat
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    ev = _normalise(_read_mat(path))
    raw_t = np.atleast_1d(m["TD"].ts).astype(np.int64)
    t0 = int(raw_t.min())                      # the origin _normalise used

    Obj = m["Obj"]
    ox = np.atleast_1d(Obj.x).astype(np.float32)
    gt = np.zeros(len(ox), dtype=GT_DTYPE)
    if len(ox):
        gt["t"] = np.atleast_1d(Obj.ts).astype(np.int64) - t0
        gt["x"] = ox - 1.0                     # 1-indexed -> 0-indexed
        gt["y"] = np.atleast_1d(Obj.y).astype(np.float32) - 1.0
        gt["obj_id"] = np.atleast_1d(Obj.id).astype(np.int32)
        gt["mag"] = np.nan
        gt = gt[np.argsort(gt["t"], kind="stable")]

    shape = (int(m["yMax"]), int(m["xMax"]))   # (H, W)
    return ev, gt, shape
