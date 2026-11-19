# Ten-step transfer

The cohort has eight tuned configurations for each glacier: 30 Aletsch + 30 Valais + 9 Sermeq transitions per configuration, **552 cases**. `experiment/cohort.yaml` specifies the selected models, seeds and glacier-specific anchor learning rates.

| Command (repository root) | Inputs | Outputs | Work |
|---|---|---|---|
| `python transfer/runs/workflow/01_run.py --config transfer/runs/experiment/cohort.yaml --case aletsch --gpu 0` | Canonical FP64 trajectory; converged prescribed states | Trained anchors, normalization/weights, transfer and correction fields, FP64 energy gaps | 240 anchors × 20,000 Adam steps; ten corrections each |
| Repeat with `--case valais` | Valais reference | 240 cases | Same step budgets |
| Repeat with `--case jakobshavn` | Sermeq reference | 72 cases | Same step budgets |
| `python transfer/runs/workflow/02_export.py` | Bundled compact vector and energy measurements | Table SM5.1 and defined/undefined counts | Seconds, CPU |

At each anchor, train from the recorded random initialization for 20,000 Adam steps, carry weights unchanged to the target, then apply **ten** fresh-Adam corrections at 5×10⁻⁵. Anchor rates are 2×10⁻⁴ / 2×10⁻³ / 10⁻³. Alpine anchors are years 200:10:490; Sermeq anchors are years 1:1:9. The next state is 0.1 / 0.05 / 0.05 year later. Normalization measured at the anchor is retained at the target.

Held and transferred errors are both measured against the **target** reference. Anticipation statistics are undefined when the reference change is unresolved; energy reduction ratios are undefined for nonpositive or roundoff-sized transferred gaps. Tables require all expected cases. Valais has 19 unresolved reference changes per architecture, hence 11 defined cosines per architecture.

Use `--reference-source PATH64` and `--output PATH --namespace NAME` for separate runs. `--architectures NAME --transitions 0` selects one case. `--resume` requires identical metadata; failed attempts need `--retry-failed` too. Export completed runs: `02_export.py --source PATH --output results/fresh-transfer`.
