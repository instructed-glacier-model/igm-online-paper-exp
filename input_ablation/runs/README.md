# Controlled input ablation

A common CNN backbone is used on Aletsch/Valais and a common FNO backbone on Sermeq. `experiment/cohort.yaml` gives all model specifications and seeds. Each glacier/input has nine paired cases: three anchors × seeds 0, 1, 2. Raw, three-feature, five-feature and scrambled inputs give **108 cases**.

| Command (repository root) | Inputs | Outputs | Work |
|---|---|---|---|
| `python input_ablation/runs/workflow/01_run.py --config input_ablation/runs/experiment/cohort.yaml --case aletsch --gpu 0` | Canonical FP64 trajectory; regenerated converged anchor/target states | Anchor weights, frozen normalization, independent correction checkpoints and curves | 36 anchors × 20,000 Adam steps; 50 fresh-Adam corrections each |
| Repeat with `--case valais`, `--case jakobshavn` | Their references | Remaining 72 paired cases | Same declared step budgets |
| Same Aletsch command with `--rate-controls --resume` | Saved anchor checkpoints | **36 additional Aletsch** branches at learning rate 10⁻⁵ | 36 × 50 corrections; no repeated anchor training |
| `python input_ablation/runs/workflow/02_export.py` | Bundled cases and curves | Table 5.2 and figure data in `results/exports/input_ablation/` | Seconds, CPU |
| `python input_ablation/runs/workflow/03_rate_summary.py` | Bundled rate curves | Supplementary Aletsch rate summary | Seconds, CPU |

Anchor learning rates are 2×10⁻⁴ / 2×10⁻³ / 10⁻³ for Aletsch/Valais/Sermeq. Main corrections use 5×10⁻⁵. Anchors are 200, 350, 490 yr for Alpine cases and 1, 5, 9 yr for Sermeq; the target is one prescribed time step later (0.1 / 0.05 / 0.05 yr). Each correction branch restores the same anchor weights and normalization and starts a new Adam optimizer.

Use `--reference-source PATH64`, `--output PATH --namespace NAME`, `--resume` and `--retry-failed` as in the glacier study. `--architectures Backbone_5f --transitions 0` selects one case. To export completed runs, add `--source PATH --output results/fresh-ablation` to `02_export.py` or `03_rate_summary.py`. Tables require the complete cohort. Paired reductions match anchor and seed; ratios with zero denominators are undefined.
