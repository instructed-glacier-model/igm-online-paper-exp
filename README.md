# Physics-enriched online neural solvers: paper experiments

Material for *Physics-Enriched Inputs for Online Neural Solvers with Application to Ice Flow*. Each study has `runs/` to repeat its experiments and `figure/` to redraw the published outputs from compact data.

| Study | Manuscript outputs | Instructions |
|---|---|---|
| Schematics | Figures 2.1, 3.1 | [schematic](schematic/README.md) |
| Glaciers and classical solver | Figure 4.1; Tables 5.3, SM1.1–SM1.3, SM2.1 | [runs](glaciers/runs/README.md), [figure](glaciers/figure/README.md) |
| Uniform and NSGA-II searches | Figure 5.1; Tables 5.1, SM3.1 | [runs](searches/runs/README.md), [figure](searches/figure/README.md) |
| Controlled input ablation | Figure 5.2; Table 5.2; Aletsch correction-rate check | [runs](input_ablation/runs/README.md), [figure](input_ablation/figure/README.md) |
| Seed reruns | Table SM4.1 | [runs](seed_reruns/runs/README.md), [table](seed_reruns/figure/README.md) |
| Ten-step transfer | Table SM5.1 | [runs](transfer/runs/README.md), [table](transfer/figure/README.md) |

## Redraw without simulations

```bash
python -m pip install -r requirements.txt
python glaciers/figure/make_figure.py
python searches/figure/make_figure.py
python input_ablation/figure/make_figure.py
python seed_reruns/figure/make_tables.py
python transfer/figure/make_tables.py
```

Build the two TikZ schematics as described in [schematic/README.md](schematic/README.md). LaTeX tables use `booktabs` and `amsmath`; the glacier-settings table also uses `array`.

## Repeat the simulations

Use **Python 3.10, IGM 3.3.0, TensorFlow 2.17.0 and Keras 3.12.2** in a dedicated environment:

```bash
python -m pip install -r glaciers/runs/requirements.txt
```

Generate the FP64 references and FP32 comparators first, then run the other studies using the instructions above. For a selected neural trajectory:

```bash
cd searches/runs
CUDA_VISIBLE_DEVICES=0 igm_run +experiment=aletsch_DahuNet_cnn_5f study.args.gpu=0
```

Runs write to `results/`; use `--output PATH --namespace NAME` for another destination, `--resume` to continue a matching run, and `--retry-failed` to retry failures. Configurations are in each study's `runs/experiment/` directory.

Figure scripts read adjacent `data/` directories and write outputs beside themselves. To use new exports, copy the study's `figure/` directory and replace the copy's data.

## Measurements and definitions

- Accuracy: mean depth-averaged speed RMSE. Qualification: median speed error below 5% at every saved frame, with a 10 m/yr denominator offset.
- Timing includes startup and initial training; steady computation is reported separately. FP32 comparator timings exclude external FP64 checks.

## License and data

Code: GPL-3.0-or-later; see [LICENSE](LICENSE). Input data, source citations and dataset terms are in [glaciers/runs/data](glaciers/runs/data/README.md).
