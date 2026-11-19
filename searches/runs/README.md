# Uniform and NSGA-II searches

Eight architectures on three glaciers. `experiment/{aletsch,valais,jakobshavn}.yaml` contains budgets and physics; `experiment/distributions.yaml` defines the search spaces. Timings include startup and initial training.

| Command (repository root) | Inputs | Outputs | Work |
|---|---|---|---|
| `python searches/runs/workflow/01_search.py --config searches/runs/experiment/aletsch.yaml --sampler uniform --gpu 0` | Validated canonical FP64 reference | Fresh SQLite studies, parameters, failures and trajectories | 50 attempts × 8 architectures |
| Same with `--sampler pareto` | Same reference | NSGA-II studies, population 20 | 500 attempts × 8 architectures |
| Repeat with `valais.yaml` and `jakobshavn.yaml` | Their canonical references | Complete three-glacier populations | 13,200 attempts total; up to 4 h/trial |
| `python searches/runs/workflow/02_export.py --classical glaciers/figure/data/table5_3_classical.csv` | Bundled trials and classical benchmark | Tables 5.1/SM3.1, populations, correlations and selected steady timings | Seconds, CPU |
| `python searches/runs/workflow/03_replay.py --config searches/runs/experiment/aletsch.yaml --sampler pareto --architectures CNN --trials 0 --gpu 0` | Optional recorded parameters and original seed | Replay trajectory | One full trajectory |

For a short test, use `--architectures CNN --trials 2`. Use one worker per SQLite study and `--resume` to continue an interrupted search.

Use `--reference-source PATH64` for another reference and `--output PATH --namespace NAME` for another output destination. To export completed runs, use `02_export.py --source PATH --output results/fresh-search-export --classical PATH_TO_ASSOCIATED_BENCHMARK.csv`. Speedups require the associated classical benchmark.

Selected trajectories can also be run directly:

```bash
cd searches/runs
CUDA_VISIBLE_DEVICES=0 igm_run +experiment=aletsch_DahuNet_cnn_5f study.args.gpu=0
# valais_DahuNet_cnn_5f or jakobshavn_DahuNet_fno_5f
```

Set `study.args.reference_source` for another reference and `study.args.resume=true` to resume.

Tables require all expected attempts, including failures. `T10` is the tenth-cheapest qualifying trial.
