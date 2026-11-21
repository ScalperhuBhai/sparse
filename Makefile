PYTHON ?= python
DEMO_DIR ?= out/demo

.PHONY: test demo

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) cli.py synth --out "$(DEMO_DIR)/scene.npz" --shape 480x640 --duration 2.0 --mags 6 7.5 9 --seed 0
	$(PYTHON) cli.py run --input "$(DEMO_DIR)/scene.npz" --gt "$(DEMO_DIR)/scene_gt.npy" --out "$(DEMO_DIR)/results"
	$(PYTHON) scripts/make_viz.py --input "$(DEMO_DIR)/scene.npz" --results "$(DEMO_DIR)/results/results.jsonl" --latency "$(DEMO_DIR)/results/latency.json" --metrics "$(DEMO_DIR)/results/metrics_tracks.json" --out "$(DEMO_DIR)/results/viewer.html"
