"""Paper table fragments generated solely from the adjacent numeric data."""
from pathlib import Path
import csv, math

CASES = ["aletsch", "valais", "jakobshavn"]
ALL = [
    "MLP",
    "CNN",
    "DahuNet_cnn_3f",
    "DahuNet_cnn_5f",
    "DahuNet_scramble",
    "FNO2",
    "DahuNet_fno_3f",
    "DahuNet_fno_5f",
]
MAIN = ["MLP", "CNN", "DahuNet_cnn_5f", "FNO2", "DahuNet_fno_5f"]
TITLES = dict(aletsch="Aletsch", valais="Valais", jakobshavn="Sermeq Kujalleq")
LABELS = dict(
    MLP="MLP, raw",
    CNN="CNN, raw",
    FNO2="FNO, raw",
    DahuNet_cnn_3f="CNN, three features",
    DahuNet_cnn_5f="DahuNet-CNN",
    DahuNet_fno_3f="FNO, three features",
    DahuNet_fno_5f="DahuNet-FNO",
    DahuNet_scramble="CNN, scrambled",
)


def read(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f))


def fmt(v, d=0):
    return "---" if v in ["", None] else f"{float(v):.{d}f}"


def integer(v):
    return "---" if v in ["", None] else f"{int(float(v)):,}".replace(",", r"\,")


def sci(v):
    if v in ["", None]:
        return "---"
    v = float(v)
    if v == 0:
        mant, e = "0.00", 0
    elif abs(v) >= 0.1:
        mant, e = f"{v:.2f}", 0
    else:
        e = math.floor(math.log10(abs(v)))
        mant = f"{v/10**e:.1f}"
    mant += r"\phantom{0}" * (2 - len(mant.split(".")[1]))
    power = str(e) if e < 0 else r"\phantom{-}" + str(e)
    return "$" + mant + r"\times10^{" + power + "}$"


def sci_sig(v):
    if v in ["", None]:
        return "---"
    x = float(v)
    if x == 0:
        return "$0.0\\times10^{0}$"
    e = math.floor(math.log10(abs(x)))
    m = round(x / 10**e, 1)
    if abs(m) >= 10:
        m /= 10
        e += 1
    return "$" + f"{m:.1f}" + r"\times10^{" + str(e) + "}$"


def decimal(v, d):
    return fmt(v, d) + r"\phantom{0}" * (2 - d)


def cos(v, n, show):
    return fmt(v, 2) + (f" ({n})" if show else r"\phantom{ (11)}")


def section(n, c, suffix=""):
    return r"\multicolumn{" + str(n) + r"}{@{}l}{\emph{" + TITLES[c] + suffix + r"}}\\"


def write(folder, name, n, header, groups):
    lines = [
        r"% Generated from data/" + name + ".csv. Requires booktabs and amsmath.",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l" + "r" * (n - 1) + r"@{}}",
        r"\toprule",
        *header,
        r"\midrule",
    ]
    for i, group in enumerate(groups):
        if i:
            lines.append(r"\midrule")
        lines.extend(group)
    lines.extend([r"\bottomrule", r"\end{tabular*}"])
    (folder / (name + ".tex")).write_text("\n".join(lines) + "\n")


def render(folder):
    f = Path(folder)
    for path in sorted((f / "data").glob("table*.csv")):
        name = path.stem
        rows = read(path)
        lookup = {(r.get("case", ""), r.get("architecture", r.get("input", ""))): r for r in rows}
        groups = []
        if name in ["table5_1_searches", "tableSM3_1_all_variants"]:
            main = name == "table5_1_searches"
            n = 9 if main else 8
            for case in CASES if main else ["aletsch", "jakobshavn", "valais"]:
                g = [section(n, case)]
                for a in MAIN if main else ALL:
                    r = lookup[case, a]
                    if main:
                        vals = [
                            LABELS[a],
                            fmt(r["uniform_median_error_percent"]),
                            r["uniform_qualifying"],
                            r["pareto_qualifying"],
                            fmt(r["t10_seconds"]),
                            fmt(r["speedup_over_fp32_comparator"]),
                            integer(r["smallest_qualifying_parameters"]),
                            fmt(r["top10_m_over_period"], 1),
                            fmt(r["top10_m0"]),
                        ]
                    else:
                        rmse = float(r["uniform_median_speed_rmse"])
                        best = float(r["best_speed_rmse"])
                        vals = [
                            LABELS[a],
                            fmt(r["uniform_median_error_percent"]),
                            fmt(rmse, 1 if rmse < 100 else 0),
                            r["uniform_qualifying"],
                            r["pareto_qualifying"],
                            fmt(r["t10_seconds"]),
                            integer(r["smallest_qualifying_parameters"]),
                            fmt(best, 2 if best < 10 else 1),
                        ]
                    g.append(" & ".join(vals) + r" \\")
                groups.append(g)
            header = (
                [
                    r"Backbone, inputs & error (\%) & qual. & qual. & $T_{10}$ (s) & speed-up & smallest & $m/\ell$ & $m_0$\\"
                ]
                if main
                else [
                    r"Backbone, inputs & error (\%) & RMSE & qual. & qual. & $T_{10}$ (s) & smallest & best RMSE\\"
                ]
            )
            write(f, name, n, header, groups)
        elif name == "table5_2_ablation":
            for case in CASES:
                g = [
                    section(7, case, ", FNO backbone" if case == "jakobshavn" else ", CNN backbone")
                ]
                for inp, title in [
                    ("raw", "Raw state"),
                    ("three_features", "Three features"),
                    ("five_features", "Five features"),
                    ("scrambled", "Scrambled"),
                ]:
                    r = lookup[case, inp]
                    ks = [
                        "median_held_error_rms",
                        "median_transfer_ratio",
                        "median_correction_factor",
                        "median_final_vector_rms",
                        "median_final_speed_error_percent",
                    ]
                    vals = [
                        fmt(r[k], 1 if case == "jakobshavn" and k in [ks[0], ks[3]] else 2)
                        for k in ks
                    ]
                    g.append(
                        " & ".join(
                            [
                                title,
                                *vals,
                                "---"
                                if inp == "raw"
                                else fmt(r["median_paired_speed_error_reduction_percent"]),
                            ]
                        )
                        + r" \\"
                    )
                groups.append(g)
            write(
                f,
                name,
                7,
                [
                    r"Input & $\|\mathbf e^{h}\|$ & $\|\mathbf e^{t}\|/\|\mathbf e^{h}\|$ & $\gamma_{50}$ & $\|\mathbf e^{(50)}\|$ & Speed error (\%) & Reduction (\%)\\"
                ],
                groups,
            )
        elif name == "tableSM4_1_seed_reruns":
            for case in ["aletsch", "jakobshavn"]:
                g = [section(4, case)]
                for a in ["CNN", "DahuNet_cnn_5f", "FNO2", "DahuNet_fno_5f"]:
                    r = lookup[case, a]
                    g.append(
                        " & ".join(
                            [
                                LABELS[a],
                                fmt(r["median_total_seconds"], 1),
                                fmt(
                                    r["worst_speed_error_percent"],
                                    3
                                    if abs(float(r["worst_speed_error_percent"]) - 5) < 0.01
                                    else 2,
                                ),
                                f"{r['qualifying_seeds']}/{r['seeds']}",
                            ]
                        )
                        + r" \\"
                    )
                groups.append(g)
            write(
                f,
                name,
                4,
                [
                    r"Backbone, inputs & Median total (s) & Worst-frame error (\%) & Qualifying seeds\\"
                ],
                groups,
            )
        elif name == "tableSM5_1_transfer":
            for case in ["aletsch", "jakobshavn", "valais"]:
                g = [section(8, case)]
                for a in ALL:
                    r = lookup[case, a]
                    g.append(
                        " & ".join(
                            [
                                LABELS[a],
                                sci(r["median_G_s"]),
                                sci(r["median_G_t"]),
                                fmt(r["median_gamma_adam"], 2),
                                decimal(
                                    r["median_transferred_error_rms"], 2 if case == "aletsch" else 1
                                ),
                                decimal(r["median_adam_error_rms"], 2 if case == "aletsch" else 1),
                                decimal(
                                    r["median_input_response_rms"], 1 if case == "jakobshavn" else 2
                                ),
                                cos(r["median_cosine"], r["defined_cosine"], case == "valais"),
                            ]
                        )
                        + r" \\"
                    )
                groups.append(g)
            write(
                f,
                name,
                8,
                [
                    r"Parametrization & $G^{s}$ & $G^{t}$ & $\gamma$ & $\|\mathbf e^{t}\|$ & after 10 & $\|\Delta\mathbf U\|$ & cos ($n$)\\"
                ],
                groups,
            )
        elif name == "tableSM1_1_search_settings":
            import json

            g = []
            for r in rows:
                if r["kind"] == "categorical":
                    values = json.loads(r["choices"])
                    choices = ", ".join(
                        "yes" if v is True else "no" if v is False else str(v) for v in values
                    )
                    bounds = r"$\{" + choices + r"\}$"
                    sampling = "categorical"
                else:

                    def value(v):
                        x = float(v)
                        if abs(x) >= 0.01:
                            return fmt(x, 0) if x.is_integer() else str(x)
                        exponent = math.floor(math.log10(x))
                        coefficient = x / 10**exponent
                        return (
                            (str(int(round(coefficient))) + r"\times" if coefficient != 1 else "")
                            + r"10^{"
                            + str(exponent)
                            + "}"
                        )

                    bounds = "$[" + value(r["low"]) + ", " + value(r["high"]) + "]$"
                    sampling = (
                        "log-uniform"
                        if r.get("log") == "True"
                        else "integer" + (", step " + fmt(r["step"]) if r.get("step") else "")
                    )
                g.append(" & ".join([r["parameter"], bounds, sampling]) + r" \\")
            write(f, name, 3, [r"Parameter & Range or choices & Sampling\\"], [g])
        elif name == "tableSM1_2_normalization":
            g = [
                " & ".join(
                    [
                        r["input"],
                        r["unit"],
                        str(float(r["offset"])).removesuffix(".0"),
                        str(float(r["scale"])).removesuffix(".0"),
                    ]
                )
                + r" \\"
                for r in rows
            ]
            write(f, name, 4, [r"Input & Unit & $\mu$ & $\sigma$\\"], [g])
        elif name == "tableSM1_3_glacier_settings":
            bycase = {r["case"]: r for r in rows}
            g = []
            fields = [
                ("Type", "type"),
                ("Grid, spacing", None),
                ("Forcing", "forcing"),
                ("Simulation period (yr)", None),
                ("Saved frames", "saved_frames"),
                ("Softness $A$ (MPa$^{-3}$ yr$^{-1}$)", "arrhenius"),
                ("Transfer anchors (yr)", None),
                ("Increment (one time step, yr)", "increment_years"),
                ("Anchor learning rate", "anchor_lr"),
                ("Ablation backbone", "ablation_backbone"),
                ("Ablation anchors (yr)", "ablation_anchors"),
            ]
            for label, key in fields:
                vals = []
                for case in CASES:
                    r = bycase[case]
                    if label == "Grid, spacing":
                        v = (
                            r["grid"].replace(" x ", r" $\times$ ")
                            + ", "
                            + fmt(r["spacing_m"])
                            + " m"
                        )
                    elif label == "Simulation period (yr)":
                        v = fmt(r["start_year"]) + "--" + fmt(r["end_year"])
                    elif label == "Transfer anchors (yr)":
                        v = (
                            "1, 2, $\ldots$, 9"
                            if case == "jakobshavn"
                            else "200, 210, $\ldots$, 490"
                        )
                    elif key == "anchor_lr":
                        v = sci(r[key])
                    elif key == "arrhenius":
                        v = fmt(r[key])
                    else:
                        v = r[key]
                    vals.append(v)
                g.append(" & ".join([label, *vals]) + r" \\")
            lines = [
                r"% Generated from data/" + name + ".csv. Requires booktabs and array.",
                r"\begin{tabular}{@{}p{.25\linewidth}p{.22\linewidth}p{.22\linewidth}p{.22\linewidth}@{}}",
                r"\toprule",
                r" & Aletsch & Valais & Sermeq Kujalleq\\",
                r"\midrule",
                *g,
                r"\bottomrule",
                r"\end{tabular}",
            ]
            (f / (name + ".tex")).write_text("\n".join(lines) + "\n")
        elif name == "tableSM2_1_stationarity":
            g = []
            for case in CASES:
                r = lookup[case, ""]
                x = float(r["maximum_last_change"])
                # Published maxima are conservative upper roundings at two significant digits.
                place = 10 ** (math.floor(math.log10(x)) - 1)
                upper = math.ceil(x / place) * place
                last = sci(upper) if case == "aletsch" else fmt(upper, 3 if case == "valais" else 2)
                g.append(
                    " & ".join(
                        [
                            TITLES[case],
                            r["solves"],
                            sci_sig(r["median_relative_gradient"]),
                            sci_sig(r["maximum_relative_gradient"]),
                            "---"
                            if case == "aletsch"
                            else fmt(r["median_last_change"], 3 if case == "valais" else 2),
                            last,
                        ]
                    )
                    + r" \\"
                )
            write(
                f,
                name,
                6,
                [
                    r" & & \multicolumn{2}{c}{Relative gradient} & \multicolumn{2}{c}{Last change (m yr$^{-1}$)}\\",
                    r"Glacier & Solves & median & maximum & median & maximum\\",
                ],
                [g],
            )
        elif name == "table5_3_classical":
            neural = {r["case"]: r for r in read(f / "data/neural_steady_timings.csv")}
            g = []
            for case in CASES:
                r = lookup[case, ""]
                err = float(r["worst_speed_median_percent"])
                g.append(
                    " & ".join(
                        [
                            TITLES[case],
                            r["velocity_solves"],
                            fmt(float(r["total_excluding_external_fp64_audits"]) / 60),
                            fmt(r["updates_per_solve"], 1),
                            fmt(err, 4 if err < 0.001 else 3 if err < 0.01 else 2),
                            fmt(r["steady_seconds_per_step"], 1) + " s",
                            fmt(1000 * float(neural[case]["seconds_per_step"])) + " ms",
                        ]
                    )
                    + r" \\"
                )
            write(
                f,
                name,
                7,
                [
                    r"Glacier & Solves & $T_{\rm cl}$ (min) & Updates/solve & Error (\%) & classical/step & DahuNet/step\\"
                ],
                [g],
            )
