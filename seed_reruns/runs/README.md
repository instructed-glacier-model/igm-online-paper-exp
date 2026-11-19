# Seed reruns

`experiment/cohort.yaml` contains the eight fixed selected architecture/glacier groups: CNN, DahuNet-CNN, FNO and DahuNet-FNO on Aletsch and Sermeq, each with seeds 0, 1, 2 (**24 trajectories**). The Aletsch DahuNet-FNO selection additionally used vector error, as stated in Table SM4.1.

| Command (repository root) | Inputs | Outputs | Work |
|---|---|---|---|
| `python seed_reruns/runs/workflow/01_rerun.py --config seed_reruns/runs/experiment/cohort.yaml --case aletsch --gpu 0` | Canonical reference and fixed selected recipes | Twelve native trajectories with independent model seeds | 12 full trajectories |
| Repeat with `--case jakobshavn` | Sermeq reference | Remaining twelve trajectories | 12 full trajectories |
| `python seed_reruns/runs/workflow/02_export.py` | Bundled measurements | Table SM4.1, including all three seeds per group | Seconds, CPU |

Use `--reference-source PATH64`, `--output PATH --namespace NAME` and `--resume` as in the other studies. `--architectures CNN` selects one architecture. To export completed runs, supply `--source PATH --output results/fresh-seeds`. Tables require all three seeds per group.
