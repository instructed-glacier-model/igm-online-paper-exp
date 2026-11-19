"""CPU-only aggregation of compact paper records or newly generated run records."""
from pathlib import Path
import csv, json, gzip, math, collections
import numpy as np

CASES = ["aletsch", "valais", "jakobshavn"]
ARCH = [
    "MLP",
    "CNN",
    "FNO2",
    "DahuNet_cnn_3f",
    "DahuNet_cnn_5f",
    "DahuNet_fno_3f",
    "DahuNet_fno_5f",
    "DahuNet_scramble",
]
MAIN = ["MLP", "CNN", "FNO2", "DahuNet_cnn_5f", "DahuNet_fno_5f"]


def read(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f))


def number(x):
    return float(x) if x not in ["", None, "None"] else None


def med(v):
    return float(np.median(v)) if v and all(x is not None and np.isfinite(x) for x in v) else None


def csvout(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()}
            )


def js(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")


def boolval(v):
    return v is True or v == "True"


def gz(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(s) for s in f]


def fresh_trials(source):
    """Read retained attempts, including failed attempts, from fresh Optuna studies."""
    import sqlite3

    rows = []
    for db in sorted(Path(source).rglob("study.sqlite3")):
        d = db.parent
        sc = json.loads((d / "manifest.json").read_text())["scientific"]
        case = sc["config"]["case"]
        arch = sc["architecture"]
        sampler = sc["sampler"]
        with sqlite3.connect(db) as c:
            states = c.execute("select number,state from trials order by number").fetchall()
        for i, state in states:
            if state not in ["COMPLETE", "FAIL"]:
                raise ValueError(f"Unfinished attempt {d}/{i}: {state}")
            p = d / "trials" / f"{i:04d}" / "result.json"
            r = json.loads(p.read_text()) if p.exists() else {}
            t = r.get("trajectory") or {}
            valid = state == "COMPLETE" and bool(t.get("valid"))
            jobpath = d / "trials" / f"{i:04d}" / "job.json"
            job = json.loads(jobpath.read_text()) if jobpath.exists() else {}
            rows.append(
                dict(
                    case=case,
                    architecture=arch,
                    sampler=sampler,
                    trial=i,
                    optuna_state=state,
                    valid=valid,
                    qualified=valid
                    and t.get("median_only_qualifies", False)
                    and t.get("worst_speed_median_percent", math.inf) < 5,
                    worst_speed_median_percent=t.get("worst_speed_median_percent")
                    if valid
                    else math.inf,
                    mean_speed_rmse=t.get("mean_speed_rmse") if valid else math.inf,
                    total_seconds=max(r.get("process_seconds", 0), r.get("total_seconds", 0))
                    if r
                    else None,
                    parameters=r.get("parameters"),
                    online=r.get("online", job.get("online")),
                    spec=r.get("spec", job.get("spec")),
                    seed=r.get("seed", job.get("seed")),
                    steps=r.get("steps"),
                    steady_seconds=r.get("steady_seconds"),
                )
            )
    return rows


def searches(source, out, baseline=None):
    rows = read(source / "trials.csv") if (source / "trials.csv").exists() else fresh_trials(source)
    if not rows:
        raise ValueError("No search attempts found")
    require_cohort(
        rows,
        {
            (c, a, t, i)
            for c in CASES
            for a in ARCH
            for t, n in [("uniform", 50), ("pareto", 500)]
            for i in range(n)
        },
        ["case", "architecture", "sampler", "trial"],
    )
    for r in rows:
        if r.get("total_seconds") is None:
            r["total_seconds"] = r.get("total_inclusive_seconds")
        for k in ["worst_speed_median_percent", "mean_speed_rmse", "total_seconds", "parameters"]:
            r[k] = number(r.get(k))
        for k in ["qualified", "valid"]:
            r[k] = boolval(r[k])
        r["qualified"] = r["qualified"] and r["valid"] and r["optuna_state"] == "COMPLETE"
        for key in ["worst_speed_median_percent", "mean_speed_rmse"]:
            if r[key] is None or not np.isfinite(r[key]):
                r[key] = math.inf
                r["qualified"] = False
        if not r.get("online"):
            r["online"] = {k: number(r.get(k)) for k in ["nbit", "retrain_freq", "nbit_init"]}
        elif isinstance(r["online"], str):
            r["online"] = json.loads(r["online"])
    summaries = []
    for (case, arch, sampler), g in groups(rows, ["case", "architecture", "sampler"]):
        q = sorted([r for r in g if r["qualified"]], key=lambda r: r["total_seconds"])
        ten = q[:10]
        summaries.append(
            dict(
                case=case,
                architecture=arch,
                sampler=sampler,
                attempts=len(g),
                failed=sum(r["optuna_state"] == "FAIL" for r in g),
                qualifying=len(q),
                median_worst_speed_error_percent=float(
                    np.median([r["worst_speed_median_percent"] for r in g])
                ),
                median_speed_rmse=float(np.median([r["mean_speed_rmse"] for r in g])),
                best_speed_rmse=min(r["mean_speed_rmse"] for r in g),
                best_worst_speed_error_percent=min(r["worst_speed_median_percent"] for r in g),
                tenth_cheapest_qualifying_seconds=q[9]["total_seconds"] if len(q) >= 10 else None,
                smallest_qualifying_parameters=min(r["parameters"] for r in q) if q else None,
                top10_m_over_period_median=med(
                    [r["online"]["nbit"] / r["online"]["retrain_freq"] for r in ten]
                )
                if len(q) >= 10
                else None,
                top10_m0_median=med([r["online"]["nbit_init"] for r in ten])
                if len(q) >= 10
                else None,
            )
        )
    classics = (
        {r["case"]: number(r["total_excluding_external_fp64_audits"]) for r in read(baseline)}
        if baseline
        else {}
    )
    if baseline and set(classics) != set(CASES):
        raise ValueError("Classical benchmark must contain all three cases")
    tables = []
    for case in CASES:
        for a in ARCH:
            u = next(
                (
                    r
                    for r in summaries
                    if (r["case"], r["architecture"], r["sampler"]) == (case, a, "uniform")
                ),
                None,
            )
            p = next(
                (
                    r
                    for r in summaries
                    if (r["case"], r["architecture"], r["sampler"]) == (case, a, "pareto")
                ),
                None,
            )
            if not u or not p:
                continue
            t = p["tenth_cheapest_qualifying_seconds"]
            tables.append(
                dict(
                    case=case,
                    architecture=a,
                    uniform_median_error_percent=u["median_worst_speed_error_percent"],
                    uniform_median_speed_rmse=u["median_speed_rmse"],
                    uniform_qualifying=u["qualifying"],
                    pareto_qualifying=p["qualifying"],
                    pareto_attempts=p["attempts"],
                    pareto_failed=p["failed"],
                    t10_seconds=t,
                    speedup_over_fp32_comparator=classics[case] / t
                    if t and case in classics
                    else None,
                    classical_cost_seconds=classics.get(case),
                    smallest_qualifying_parameters=p["smallest_qualifying_parameters"],
                    top10_m_over_period=p["top10_m_over_period_median"],
                    top10_m0=p["top10_m0_median"],
                    best_speed_rmse=min(u["best_speed_rmse"], p["best_speed_rmse"]),
                    best_pareto_median_error_percent=p["best_worst_speed_error_percent"],
                )
            )
    csvout(out / "population_summary.csv", summaries)
    csvout(out / "tableSM3_1_all_variants.csv", tables)
    csvout(out / "table5_1_searches.csv", [r for r in tables if r["architecture"] in MAIN])
    if source.resolve() != out.resolve():
        canonical = []
        for row in rows:
            entry = {
                k: v
                for k, v in row.items()
                if k not in ["online", "spec", "total_inclusive_seconds"]
            }
            entry.update(row.get("online") or {})
            if isinstance(row.get("spec"), dict):
                entry.update(row["spec"]["params"])
            canonical.append(entry)
        csvout(out / "trials.csv", canonical)
    timing_rows = []
    for case in CASES:
        name = "DahuNet_fno_5f" if case == "jakobshavn" else "DahuNet_cnn_5f"
        candidates = [
            v
            for v in rows
            if v["case"] == case
            and v["architecture"] == name
            and v["sampler"] == "pareto"
            and v["qualified"]
        ]
        if candidates:
            best = min(candidates, key=lambda v: v["total_seconds"])
            steps = number(best.get("steps"))
            seconds = number(best.get("steady_seconds"))
            timing_rows.append(
                dict(
                    case=case,
                    architecture=name,
                    trial=best["trial"],
                    steps=steps,
                    steady_seconds=seconds,
                    seconds_per_step=seconds / steps if seconds is not None and steps else None,
                )
            )
    if timing_rows:
        csvout(out / "neural_steady_timings.csv", timing_rows)
    if baseline:
        import hashlib

        js(
            out / "classical_benchmark.json",
            dict(
                path=str(Path(baseline).resolve()),
                sha256=hashlib.sha256(Path(baseline).read_bytes()).hexdigest(),
                scope="Explicit benchmark supplied to exporter; hardware must be matched separately",
            ),
        )
    from scipy.stats import spearmanr

    csvout(
        out / "metric_correlations.csv",
        [
            dict(
                case=c,
                spearman_rmse_vs_worst_median=float(
                    spearmanr(
                        [v["mean_speed_rmse"] for v in rows if v["case"] == c],
                        [v["worst_speed_median_percent"] for v in rows if v["case"] == c],
                    ).statistic
                ),
            )
            for c in CASES
        ],
    )


def groups(rows, keys):
    d = collections.defaultdict(list)
    for r in rows:
        d[tuple(r[k] for k in keys)].append(r)
    return d.items()


def ablation(source, out):
    if (source / "cases.csv").exists():
        rows = read(source / "cases.csv")
        curves = read(source / "correction_curves.csv")
    else:
        rows = []
        curves = []
        for p in sorted(source.rglob("adam_5e-05/result.json")):
            if "/input_ablation/" not in str(p):
                continue
            r = json.loads(p.read_text())
            h = r["curves"]
            name = r["architecture"]
            label = (
                "raw"
                if name.endswith("_raw")
                else "three_features"
                if name.endswith("_3f")
                else "five_features"
                if name.endswith("_5f")
                else "scrambled"
            )
            common = dict(
                case=r["case"], architecture=name, input=label, seed=r["seed"], index=r["index"]
            )
            rows.append(
                dict(
                    **common,
                    status=r["status"],
                    held_error_rms=r["held_error_rms"],
                    anchor_error_rms=r["anchor_error_rms"],
                    transfer_ratio=r["transfer_ratio"],
                    correction_factor=r["correction_factor"],
                    final_vector_rms=h[-1]["vector_rms"],
                    initial_speed_error_percent=h[0]["speed_median_percent"],
                    final_speed_error_percent=h[-1]["speed_median_percent"],
                )
            )
            curves.extend(dict(**common, **v) for v in h)
        for r in rows:
            raw = next(
                v
                for v in rows
                if v["case"] == r["case"]
                and v["input"] == "raw"
                and v["seed"] == r["seed"]
                and v["index"] == r["index"]
            )
            r["paired_speed_error_reduction_percent"] = (
                100 * (1 - r["final_speed_error_percent"] / raw["final_speed_error_percent"])
                if raw["final_speed_error_percent"]
                else None
            )
    if not rows:
        raise ValueError("No controlled ablation records found")
    if any(row.get("status", "complete") != "complete" for row in rows):
        raise ValueError(
            "Unresolved ablation cases retained in source; final paper table is not ready"
        )
    require_cohort(
        rows,
        {
            (c, i, k, z)
            for c in CASES
            for i in ["raw", "three_features", "five_features", "scrambled"]
            for k in ([0, 4, 8] if c == "jakobshavn" else [0, 15, 29])
            for z in range(3)
        },
        ["case", "input", "index", "seed"],
    )
    require_curves(curves, rows, 50)
    tables = []
    for (case, inp), g in groups(rows, ["case", "input"]):
        complete = sum(r["status"] == "complete" for r in g)
        t = dict(
            case=case,
            input=inp,
            expected_cases=9,
            completed_cases=complete,
            status="complete" if len(g) == complete == 9 else "pending_or_failed",
        )
        for k in [
            "held_error_rms",
            "anchor_error_rms",
            "transfer_ratio",
            "correction_factor",
            "final_vector_rms",
            "initial_speed_error_percent",
            "final_speed_error_percent",
            "paired_speed_error_reduction_percent",
        ]:
            t["median_" + k] = (
                med([number(r.get(k)) for r in g]) if len(g) == complete == 9 else None
            )
        tables.append(t)
    csvout(out / "cases.csv", rows)
    csvout(out / "correction_curves.csv", curves)
    csvout(out / "table5_2_ablation.csv", tables)


def seeds(source, out):
    if (source / "cases.csv").exists():
        rows = read(source / "cases.csv")
    else:
        rows = []
        for p in sorted(source.rglob("result.json")):
            r = json.loads(p.read_text())
            if "trajectory" not in r or "architecture" not in r or "/seed_reruns/" not in str(p):
                continue
            rows.append(
                dict(
                    case=r["case"],
                    architecture=r["architecture"],
                    seed=r["seed"],
                    qualified=r["trajectory"]["median_only_qualifies"],
                    total_seconds=max(r.get("process_seconds", 0), r["total_seconds"]),
                    worst_speed_median_percent=r["trajectory"]["worst_speed_median_percent"],
                )
            )
    if not rows:
        raise ValueError("No seed reruns found")
    require_cohort(
        rows,
        {
            (c, a, z)
            for c in ["aletsch", "jakobshavn"]
            for a in ["CNN", "FNO2", "DahuNet_cnn_5f", "DahuNet_fno_5f"]
            for z in range(3)
        },
        ["case", "architecture", "seed"],
    )
    tables = []
    for (case, a), g in groups(rows, ["case", "architecture"]):
        tables.append(
            dict(
                case=case,
                architecture=a,
                seeds=len(g),
                qualifying_seeds=sum(boolval(r["qualified"]) for r in g),
                median_total_seconds=med([number(r["total_seconds"]) for r in g]),
                worst_speed_error_percent=max(
                    number(r["worst_speed_median_percent"])
                    if number(r["worst_speed_median_percent"]) is not None
                    else math.inf
                    for r in g
                ),
            )
        )
    csvout(out / "cases.csv", rows)
    csvout(out / "tableSM4_1_seed_reruns.csv", tables)


def transfer(source, out):
    if (source / "cases.csv").exists():
        vectors = read(source / "cases.csv")
        energy = vectors
    else:
        vectors = []
        energy = []
        for p in sorted(source.rglob("adam_5e-05/result.json")):
            if "/transfer/" not in str(p):
                continue
            r = json.loads(p.read_text())
            d = r["decomposition"]
            common = {k: r[k] for k in ["case", "architecture", "index"]}
            common["status"] = r.get("status", "complete")
            vectors.append(
                dict(
                    **common,
                    held_error_rms=r["held_error_rms"],
                    transferred_error_rms=r["transferred_error_rms"],
                    adam_error_rms=d["final_error_rms"],
                    input_response_rms=d["input_response_rms"],
                    reference_change_rms=d["reference_change_rms"],
                    cosine=d["anticipation_cosine"],
                    anticipation_ratio=d["anticipation_ratio"],
                    ratio_status=d["ratio_status"],
                    reference_uncertainty=d["reference_uncertainty"],
                    input_change_benefit_percent=r["input_change_benefit_percent"],
                )
            )
            energy.append(
                dict(
                    **common,
                    **{
                        k: r[k]
                        for k in ["G_s", "G_t", "G_o_adam", "gamma_adam", "adam_energy_increased"]
                    },
                )
            )
    if not vectors:
        raise ValueError("No transfer records found")
    if any(row.get("status", "complete") != "complete" for row in vectors):
        raise ValueError(
            "Unresolved transfer cases retained in source; final paper table is not ready"
        )
    require_cohort(
        vectors,
        {(c, a, k) for c in CASES for a in ARCH for k in range(9 if c == "jakobshavn" else 30)},
        ["case", "architecture", "index"],
    )
    tables = []
    for (case, a), g in groups(vectors, ["case", "architecture"]):
        r = dict(
            case=case,
            architecture=a,
            transitions=len(g),
            undefined_cosines=sum(number(v["cosine"]) is None for v in g),
            undefined_anticipation_ratios=sum(number(v["anticipation_ratio"]) is None for v in g),
            worsening_input_change=sum(
                number(v["input_change_benefit_percent"]) is not None
                and number(v["input_change_benefit_percent"]) < 0
                for v in g
            ),
        )
        for k in [
            "held_error_rms",
            "transferred_error_rms",
            "adam_error_rms",
            "input_response_rms",
            "reference_change_rms",
            "cosine",
            "anticipation_ratio",
            "input_change_benefit_percent",
        ]:
            v = [number(v[k]) for v in g if number(v[k]) is not None]
            r["median_" + k] = med(v)
            r["defined_" + k] = len(v)
        e = [v for v in energy if v["case"] == case and v["architecture"] == a]
        for k in ["G_s", "G_t", "G_o_adam", "gamma_adam"]:
            v = [number(v[k]) for v in e if number(v[k]) is not None]
            r["median_" + k] = med(v)
        r.update(
            undefined_ratios=sum(number(v["gamma_adam"]) is None for v in e),
            adam_energy_increases=sum(boolval(v["adam_energy_increased"]) for v in e),
            ratio_scope="Positive gaps above FP64 roundoff only; does not bound reference uncertainty",
        )
        tables.append(r)
    csvout(
        out / "cases.csv",
        [
            {
                **v,
                **{
                    k: e.get(k)
                    for k in ["G_s", "G_t", "G_o_adam", "gamma_adam", "adam_energy_increased"]
                },
            }
            for v, e in zip(vectors, energy)
        ],
    )
    csvout(out / "tableSM5_1_transfer.csv", tables)


def references(source, out, comparisons=None, neural=None):
    if (source / "runs.csv").exists():
        rows = read(source / "runs.csv")
    else:
        rows = []
        for p in source.rglob("reference/*/result.json"):
            r = json.loads(p.read_text())
            d = p.parent
            meta = json.loads((d / "manifest.json").read_text())["scientific"]
            audits = [json.loads(q.read_text()) for q in (d / "fp64_checks").glob("*.json")]
            cost = sum(
                a.get(
                    "audit_total_seconds",
                    a.get("solve_seconds", 0) + a.get("problem_initialization_seconds", 0),
                )
                for a in audits
            )
            n = r["steps"] + 1
            rows.append(
                dict(
                    case=r["case"],
                    role="warm_fp32_comparator" if r["precision"] == "fp32" else "search_reference",
                    precision=r["precision"],
                    validated=r["validated"],
                    velocity_solves=n,
                    newton_iterations=r["newton_iterations"],
                    updates_per_solve=r["newton_iterations"] / n,
                    total_recorded_seconds=r["total_seconds"],
                    fp64_audit_seconds=cost,
                    total_excluding_external_fp64_audits=r["total_seconds"] - cost,
                    steady_seconds=r["steady_seconds"],
                    steady_seconds_per_step=r["steady_seconds"] / r["steps"],
                    initialization_seconds=r["initialization_seconds"],
                    source_sha256=meta["source_sha256"],
                    source_commit=meta["source_commit"],
                    input_sha256=meta["input_sha256"],
                )
            )
    if not rows:
        raise ValueError("No reference records found")
    require_cohort(rows, {(c, p) for c in CASES for p in ["fp32", "fp64"]}, ["case", "precision"])
    fresh = not (source / "runs.csv").exists()
    if fresh:
        if comparisons is None or neural is None:
            raise ValueError(
                "Fresh reference export needs --comparisons and --neural-timings; published timings are never substituted"
            )
        checks = read(comparisons)
        require_cohort(checks, {(c,) for c in CASES}, ["case"])
        for r in rows:
            if r["precision"] == "fp32":
                comparison = next(v for v in checks if v["case"] == r["case"])
                candidates = [
                    p
                    for p in source.rglob("reference/fp32/manifest.json")
                    if json.loads(p.read_text())["scientific"]["config"]["case"] == r["case"]
                ]
                if (
                    len(candidates) != 1
                    or comparison.get("fp32_signature")
                    != json.loads(candidates[0].read_text())["signature"]
                ):
                    raise ValueError("Comparison is not associated with the FP32 trajectory")
                target = [
                    p
                    for p in source.rglob("reference/fp64/manifest.json")
                    if json.loads(p.read_text())["scientific"]["config"]["case"] == r["case"]
                ]
                if (
                    len(target) != 1
                    or comparison.get("fp64_signature")
                    != json.loads(target[0].read_text())["signature"]
                ):
                    raise ValueError(
                        "Comparison is not associated with the canonical FP64 trajectory"
                    )
                r.update({k: v for k, v in comparison.items() if k not in ["case"]})
    if neural is not None:
        nr = read(neural)
        require_cohort(nr, {(c,) for c in CASES}, ["case"])
        csvout(out / "neural_steady_timings.csv", nr)
    elif not fresh:
        csvout(out / "neural_steady_timings.csv", read(source / "neural_steady_timings.csv"))
    for r in rows:
        assert np.isclose(
            number(r["total_recorded_seconds"]) - number(r["fp64_audit_seconds"]),
            number(r["total_excluding_external_fp64_audits"]),
            rtol=1e-12,
        )
        assert np.isclose(
            number(r["newton_iterations"]) / number(r["velocity_solves"]),
            number(r["updates_per_solve"]),
            rtol=1e-12,
        )
    csvout(out / "runs.csv", rows)
    csvout(out / "table5_3_classical.csv", [r for r in rows if r["role"] == "warm_fp32_comparator"])
    if fresh:
        convergence = []
        independent = []
        for path in sorted(source.rglob("reference/*/result.json")):
            result = json.loads(path.read_text())
            case = result["case"]
            precision = result["precision"]
            for diagpath in sorted((path.parent / "diagnostics").glob("step*.json")):
                diag = json.loads(diagpath.read_text())
                if precision == "fp64":
                    convergence.append(
                        dict(
                            case=case,
                            solve=diagpath.name,
                            status=diag["status"],
                            relative_gradient=diag.get("relative_gradient"),
                            velocity_change_rms=diag.get("velocity_change_rms"),
                            refinement_rms=diag.get("velocity_refinement_rms"),
                            newton_iterations=diag.get("newton_iterations"),
                            cg_iterations=diag.get("cg_iterations"),
                        )
                    )
                if diag.get("independent"):
                    check = diag["independent"]
                    independent.append(
                        dict(
                            case=case,
                            precision=precision,
                            solve=diagpath.name,
                            status=diag["status"],
                            independent_status=check.get("status"),
                            velocity_rms=check.get("velocity_rms"),
                        )
                    )
        # Reuse the CSV writer, then compress only the compact diagnostic series.
        csvout(out / "convergence.csv", convergence)
        with gzip.open(out / "convergence.csv.gz", "wb") as stream:
            stream.write((out / "convergence.csv").read_bytes())
        (out / "convergence.csv").unlink()
        csvout(out / "independent_checks.csv", independent)
        stationarity(out, out)
        settings(out)


def require_cohort(rows, expected, keys):
    """Refuse incomplete, duplicated or extraneous publication cohorts."""

    def key(row):
        return tuple(
            int(row[k]) if k in ("index", "seed", "trial", "step") else row[k] for k in keys
        )

    found = [key(row) for row in rows]
    if len(found) != len(set(found)):
        raise ValueError(f"Duplicate records for {keys}")
    missing = expected - set(found)
    extra = set(found) - expected
    if missing or extra:
        raise ValueError(
            f"Incomplete publication cohort: {len(missing)} missing, {len(extra)} unexpected; examples {sorted(missing)[:3]} {sorted(extra)[:3]}"
        )


def require_curves(curves, cases, steps):
    expected = {
        (r["case"], r["input"], int(r["index"]), int(r["seed"]), k)
        for r in cases
        for k in range(steps + 1)
    }
    require_cohort(curves, expected, ["case", "input", "index", "seed", "step"])


def rate_summary(source, out):
    if (source / "rate_curves.csv").exists():
        rows = read(source / "rate_curves.csv")
    else:
        rows = []
        for path in sorted(source.rglob("adam_1e-05/result.json")):
            if "/input_ablation/" not in str(path):
                continue
            result = json.loads(path.read_text())
            if result["case"] != "aletsch":
                continue
            for point in result["curves"]:
                rows.append(
                    dict(
                        case="aletsch",
                        architecture=result["architecture"],
                        seed=result["seed"],
                        index=result["index"],
                        learning_rate=result["correction_lr"],
                        **point,
                    )
                )
    for row in rows:
        row["input"] = (
            "raw"
            if row["architecture"].endswith("_raw")
            else "three_features"
            if row["architecture"].endswith("_3f")
            else "five_features"
            if row["architecture"].endswith("_5f")
            else "scrambled"
        )
    require_cohort(
        rows,
        {
            ("aletsch", i, k, z, t)
            for i in ["raw", "three_features", "five_features", "scrambled"]
            for k in [0, 15, 29]
            for z in range(3)
            for t in range(51)
        },
        ["case", "input", "index", "seed", "step"],
    )
    branches = []
    for _, g in groups(rows, ["case", "input", "index", "seed"]):
        g = sorted(g, key=lambda r: int(r["step"]))
        branches.append(
            dict(
                case="aletsch",
                architecture=g[0]["architecture"],
                learning_rate=number(g[0]["learning_rate"]),
                initial=number(g[0]["speed_median_percent"]),
                first=number(g[1]["speed_median_percent"]),
                final=number(g[-1]["speed_median_percent"]),
            )
        )
    summary = []
    for (c, a, lr), g in groups(branches, ["case", "architecture", "learning_rate"]):
        summary.append(
            dict(
                case=c,
                architecture=a,
                learning_rate=lr,
                cases=len(g),
                median_initial_speed_error_percent=med([r["initial"] for r in g]),
                median_first_speed_error_percent=med([r["first"] for r in g]),
                median_final_speed_error_percent=med([r["final"] for r in g]),
                first_update_increases=sum(r["first"] > r["initial"] for r in g),
            )
        )
    csvout(out / "correction_rate_summary.csv", summary)


def settings(out):
    """CPU exports of SM1.1--SM1.3 from the editable experiment settings."""
    import yaml

    root = Path(__file__).resolve().parent.parent
    spaces = yaml.safe_load((root / "searches/runs/experiment/distributions.yaml").read_text())
    rows = []
    labels = {
        "nbit_init": "Initialization budget $m_0$",
        "nbit": "Correction budget $m$",
        "retrain_freq": "Correction period $\ell$ (time steps)",
        "lr": "Online learning rate",
        "lr_init": "Initial learning rate",
        "nb_layers": "Hidden layers (MLP, CNN)",
        "nb_out_filter": "Hidden width (MLP, CNN)",
        "conv_ker_size": "CNN kernel size",
        "residual": "Residual connections",
        "modes1": "FNO Fourier modes (each direction, tied)",
        "width": "FNO channel width",
    }
    order = [
        "nbit_init",
        "nbit",
        "retrain_freq",
        "lr",
        "lr_init",
        "nb_layers",
        "nb_out_filter",
        "conv_ker_size",
        "residual",
        "modes1",
        "width",
    ]
    lookup = {v["name"]: v for g in spaces.values() for v in g}
    for key in order:
        item = lookup[key]
        rows.append(dict(parameter=labels[key], **item))
    csvout(out / "tableSM1_1_search_settings.csv", rows)
    csvout(
        out / "tableSM1_2_normalization.csv",
        yaml.safe_load((root / "_shared/configs/normalization.yaml").read_text()),
    )
    cohort = yaml.safe_load((root / "input_ablation/runs/experiment/cohort.yaml").read_text())
    rows = []
    for case, shape in [
        ("aletsch", "179 x 244"),
        ("valais", "800 x 600"),
        ("jakobshavn", "350 x 200"),
    ]:
        c = cohort["configurations"][case]
        t = c["time"]
        tr = c["transfer"]
        marine = case == "jakobshavn"
        rows.append(
            dict(
                case=case,
                grid=shape,
                spacing_m=150 if marine else 100,
                type="marine-terminating" if marine else "land-terminating",
                forcing=(
                    "water-depth-dependent calving; coefficient 5 / year"
                    if marine
                    else "ELA "
                    + ", ".join(str(int(row[3])) for row in t["smb"] if row[0] in [200, 300, 500])
                    + " m at years 200, 300, 500"
                ),
                start_year=t["start"],
                end_year=t["end"],
                saved_frames=11 if marine else 31,
                arrhenius=c["physics"]["arrhenius"],
                anchor_start=tr["anchor_start"],
                anchor_interval=tr["anchor_interval"],
                anchor_count=tr["count"],
                increment_years=tr["fine_dt"],
                anchor_lr=tr["anchor_lr"],
                ablation_backbone="FNO (4 blocks, width 24, 7 x 7 modes)"
                if marine
                else "CNN (6 layers, 24 filters)",
                ablation_anchors="1, 5, 9" if marine else "200, 350, 490",
            )
        )
    csvout(out / "tableSM1_3_glacier_settings.csv", rows)


def stationarity(source, out):
    with gzip.open(source / "convergence.csv.gz", "rt") as f:
        rows = list(csv.DictReader(f))
    runs = read(source / "runs.csv")
    counts = {r["case"]: int(r["velocity_solves"]) for r in runs if r["role"] == "search_reference"}
    if set(counts) != set(CASES):
        raise ValueError("Missing canonical reference")
    require_cohort(
        rows,
        {(c, f"step{k:06d}.json") for c, n in counts.items() for k in range(n)},
        ["case", "solve"],
    )
    summary = []
    for (case,), g in groups(rows, ["case"]):
        if any(v["status"] != "converged" for v in g):
            raise ValueError("Unresolved reference solve: " + case)
        grad = [number(v["relative_gradient"]) for v in g]
        change = [
            number(v["refinement_rms" if case == "aletsch" else "velocity_change_rms"]) for v in g
        ]
        if any(v is None or not np.isfinite(v) for v in grad + change):
            raise ValueError("Missing stationarity diagnostics: " + case)
        summary.append(
            dict(
                case=case,
                solves=len(g),
                median_relative_gradient=med(grad),
                maximum_relative_gradient=max(grad),
                median_last_change=None if case == "aletsch" else med(change),
                maximum_last_change=max(change),
            )
        )
    csvout(out / "tableSM2_1_stationarity.csv", summary)
