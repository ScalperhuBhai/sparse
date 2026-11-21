# SPARSE

SPARSE is a CPU-based research tool for detecting and tracking satellites and orbital debris in event-camera recordings. It filters sensor noise, clusters events into detections, and links those detections into tracks. The pipeline works on timestamped events and runs without a GPU or trained model.

The release includes a seeded synthetic example, an offline viewer, and tools for evaluating a CPU baseline on the Event-Based Space Situational Awareness (EBSSA) dataset. [Open the browser demo](https://vixitai.com/solutions/neuromorphic/explore/).

## Quick start

From the repository root, use Python 3.10 or 3.11 to create a virtual environment and install the dependencies.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run `make demo` to generate a labelled recording and run the tracker, or run these commands directly.

```bash
python cli.py synth --out out/demo/scene.npz --shape 480x640 --duration 2.0 \
  --mags 6.0 7.5 9.0 --seed 0
python cli.py run --input out/demo/scene.npz --gt out/demo/scene_gt.npy --out out/demo/results
python scripts/make_viz.py --input out/demo/scene.npz \
  --results out/demo/results/results.jsonl --latency out/demo/results/latency.json \
  --metrics out/demo/results/metrics_tracks.json --out out/demo/results/viewer.html
```

Open `out/demo/results/viewer.html` in a browser. The reference scene has a track F1 of **0.779**, with all three objects detected at least once. SPARSE writes per-window detections and tracks to `results.jsonl`, timing data to `latency.json`, and accuracy metrics to JSON when ground truth is available.

## Docker

Build the research image from this checkout. The container exposes the same CLI as the local Python installation.

```bash
docker build -t sparse:0.1.0 .
docker run --rm sparse:0.1.0 --help
mkdir -p out/docker
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/out/docker:/work" \
  sparse:0.1.0 synth --out /work/scene.npz --shape 480x640 --duration 2.0 --mags 6 7.5 9 --seed 0
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/out/docker:/work" \
  sparse:0.1.0 run --input /work/scene.npz --gt /work/scene_gt.npy --out /work/results
```

## Run on telescope data

Download `ISS_2_td_labelled.mat` from the [Western Sydney University EBSSA dataset](https://researchers.westernsydney.edu.au/en/datasets/event-based-space-situational-awareness-ebssa-dataset/). Replace the example path with the location of your download.

```bash
python cli.py run --input /path/to/ISS_2_td_labelled.mat --profile sparse-sky --out out/ebssa
```

The reference run found **1 of 12 labelled objects** on this recording. The [benchmark guide](BENCHMARKS.md) includes a smaller two-recording comparison and explains how to report your own results.

The core reads NPZ, CSV and EBSSA MAT files. Install `h5py` for HDF5, `expelliarmus` for supported Prophesee RAW and DAT files, or `dv-processing` for AEDAT4 if your recording needs it.

## Test

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

To enable the optional EBSSA tests, place `refslew_2106_davis_td_labelled.mat` and `mag10.56tycho6604_2019_davis_td_labelled.mat` in `data/ebssa/`.

Repository maintainers can enable GitHub Actions by copying [the CI template](.github/ci-template.yml) into `.github/workflows/ci.yml`.

## Contribute

If you run SPARSE on a recording you can share publicly, open an issue with the release version, dataset and recording name, command, CPU model, and resulting metrics. Reproducible examples of missed detections and false tracks are useful. Link to recordings through their original source.

## License

The SPARSE source is released under the [Apache License 2.0](LICENSE). External recordings are available from their respective providers and have their own terms.
