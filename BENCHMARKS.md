# Benchmarks

VixitAI Labs measured these results with the v0.1.0 source on an AMD EPYC host, with one BLAS thread. Timing depends on the machine and recording. Track F1 combines window-level precision and recall. The object count shows how many labelled objects were matched at least once.

| Input | Track F1 | Labelled objects detected | Mean processing per window | p99 processing per window |
| --- | ---: | ---: | ---: | ---: |
| Seeded synthetic scene | 0.779 | 3 of 3 | 1.32 ms | 2.43 ms |
| EBSSA `ISS_2_td_labelled.mat` | 0.063 | 1 of 12 | 0.30 ms | 3.00 ms |
| EBSSA `geo_abs6_25924_2017_davis_td_labelled.mat` | 0.357 | 2 of 3 | 0.25 ms | 1.77 ms |
| EBSSA `refslew_2106_davis_td_labelled.mat` | 0.000 | 0 of 1 | 0.48 ms | 4.34 ms |

The synthetic scene uses 480 rows by 640 columns, two seconds of events, seed 0, and 10 ms processing windows. It contains 245,161 events with three simulated objects. Use the exact command in the [quick start](README.md#quick-start) to reproduce it.

The EBSSA runs used the same `sparse-sky` profile. `ISS_2` was the calibration recording. The two held-out examples below show a partial detection and a complete miss. These examples demonstrate the evaluation workflow. A broader sweep is needed to characterize recall, especially on faint targets and fast transits.

## Run a two-recording comparison

Download `geo_abs6_25924_2017_davis_td_labelled.mat` and `refslew_2106_davis_td_labelled.mat` from the [Western Sydney University EBSSA dataset](https://researchers.westernsydney.edu.au/en/datasets/event-based-space-situational-awareness-ebssa-dataset/). Place them in the same directory and run the evaluator.

```bash
python scripts/eval_ebssa.py --glob '/path/to/labelled/*.mat' \
  --files geo_abs6_25924_2017_davis refslew_2106_davis \
  --profile sparse-sky --out out/ebssa-pair.json
```

## Reproduce the EBSSA sweep

Obtain the labelled MATLAB recordings from [Western Sydney University](https://researchers.westernsydney.edu.au/en/datasets/event-based-space-situational-awareness-ebssa-dataset/). Run the evaluator against the directory containing those files.

```bash
python scripts/eval_ebssa.py --glob '/path/to/labelled/*.mat' \
  --profile sparse-sky --out out/ebssa-sweep.json
```

The evaluator skips star-reference files with zero object labels. To compare two runs, use the same recording list and report the release or commit, CPU, profile, and any parameter overrides. Include the generated JSON or a public link to it so others can reproduce the result.
