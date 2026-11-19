# Glacier references and classical comparator

Run from the repository root. Configurations are in `experiment/`.

| Step / command | Inputs | Outputs | Computational cost |
|---|---|---|---|
| `python glaciers/runs/workflow/01_reference.py --config glaciers/runs/experiment/aletsch_search_reference.yaml --precision fp64 --gpu 0` | Initial Aletsch input | Canonical trajectory, frames, convergence and independent-start diagnostics | Full 300-year trajectory; original recorded cost ≈4.3 GPU h |
| Repeat for `valais_search_reference.yaml`, `jakobshavn_search_reference.yaml` | Their initial inputs | Their canonical FP64 trajectories | Original costs ≈41 h / 2 h; hardware dependent |
| Same driver with `aletsch_warm_fp32_comparator.yaml --precision fp32`; repeat for Valais/Sermeq | Initial inputs | Warm-started comparator and external FP64 checks | Original wall costs ≈0.49 / 4.93 / 0.64 h, including audits |
| `python glaciers/runs/workflow/04_compare.py --fp32 PATH32 --fp64 PATH64 --output results/comparison/aletsch` | Matching complete trajectories | Frame metrics and signed `comparison_summary.csv` | CPU; all saved frames |
| `python glaciers/runs/workflow/02_export.py` | Bundled compact measurements | Classical table data in `results/exports/glaciers/` | Seconds, CPU |
| `python glaciers/runs/workflow/03_export_overview.py --reference PATH64 --output results/maps/map_aletsch.npz` | Completed canonical trajectory | Portable overview arrays and provenance; repeat for three cases | CPU; reads saved frames |

Full-period defaults: `results/rerun/CASE/paper/reference/fp64` and `fp32`. Use `--output PATH --namespace NAME` for separate runs; matching interrupted work can use `--resume` (failed attempts additionally need `--retry-failed`).

FP64 reference solves start from zero; FP32 comparators warm-start velocity and solve every time step. Solver tolerances and convergence checks are specified in the YAML configurations.

The Sermeq FP32 comparator's independent-start discrepancy is about 6.9 m/yr against a 1 m/yr criterion.

To export a classical-comparison table, concatenate the three `comparison_summary.csv` files with one header:

```bash
python glaciers/runs/workflow/02_export.py --source results/rerun \
  --comparisons results/comparisons.csv \
  --neural-timings results/exports/searches/neural_steady_timings.csv \
  --output results/exports/fresh-glaciers
```

Use comparisons and neural timings from the associated runs, with matching hardware and power settings.
