"""Publication analysis for NEXUS-MI experiment outputs.

This module derives publication tables from generated experiment outputs.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import yaml
from scipy import stats

DATASET_DISPLAY = {"bciciv2a": "BCICIV-2a", "bci42a": "BCICIV-2a", "openbmi": "OpenBMI", "korea": "OpenBMI"}
REGIME_DISPLAY = {"shared_head_only": "SB-PH", "embedding_shared_head_only": "EIB-PH"}
INTERNAL_TO_PAPER_POLICY = {f"A{i}": f"P{i}" for i in range(1, 7)}
COMPONENT_IDS = {"P3_REF", "ONLINE_RANDOM", "ONLINE_PRIORITY", "DOWNLOAD_ONLY", "P5_FULL"}

def _severity_profiles_by_probability() -> dict[tuple[float, float, float], str]:
    from .protocol import sensitivity_profiles
    return {
        (row["online_prob_good"], row["online_prob_med"], row["online_prob_bad"]): row["name"].title()
        for row in sensitivity_profiles()
    }


def _reliability_thresholds() -> dict[str, float]:
    from .protocol import reliability_thresholds_pct
    return reliability_thresholds_pct()
EPS = 1e-9


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml(path: Path, default=None):
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def _float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _nested(d: dict, *keys, default=None):
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _dataset_name(hyper: dict, completion: dict, path: Path) -> str:
    raw = str(_nested(hyper, "dataset", "dataset_name", default=completion.get("dataset", ""))).lower()
    if raw in DATASET_DISPLAY:
        return "bciciv2a" if DATASET_DISPLAY[raw] == "BCICIV-2a" else "openbmi"
    low = str(path).lower()
    if "openbmi" in low or "korea" in low:
        return "openbmi"
    if "bciciv2a" in low or "bci42a" in low:
        return "bciciv2a"
    return raw


def _infer_policy(hyper: dict, start_args: dict, completion: dict, path: Path) -> str:
    explicit = (
        _nested(hyper, "publication_context", "policy_id")
        or hyper.get("policy_id")
        or start_args.get("paper_policy_id")
        or start_args.get("policy_id")
        or completion.get("policy_id")
    )
    if explicit:
        s = str(explicit)
        return INTERNAL_TO_PAPER_POLICY.get(s, s)

    comm = hyper.get("communication", {}) or {}
    if not bool(comm.get("effective", comm.get("requested", True))):
        return "Ideal"
    selection = str(_nested(hyper, "federated_learning", "selection_policy", default=start_args.get("selection_policy", "")))
    buffer_policy = str(_nested(hyper, "federated_learning", "buffer_policy", default=comm.get("buffer_policy", start_args.get("buffer_policy", ""))))
    stale = str(comm.get("stale_policy", start_args.get("stale_policy", "")))
    download = str(comm.get("download_policy", start_args.get("download_policy", "")))
    buffering = bool(comm.get("buffering_enabled", buffer_policy != "none"))

    # Component variants must be distinguished before normal policy inference.
    low_path = str(path).lower()
    if "p5_scheduler_ablation" in low_path or "p5ablation" in low_path:
        if selection == "online_random" and download == "always":
            return "ONLINE_RANDOM"
        if selection == "comm_aware" and download == "always":
            return "ONLINE_PRIORITY"
        if selection == "all" and download == "stale_only":
            return "DOWNLOAD_ONLY"
        if selection == "comm_aware" and download == "stale_only":
            return "P5_FULL"
        return "P3_REF"

    if selection == "all" and (buffer_policy == "none" or not buffering):
        return "P1"
    if selection == "all" and buffer_policy == "fifo" and stale == "accept_all":
        return "P2"
    if selection == "all" and buffer_policy == "fifo" and stale == "drop" and download == "always":
        return "P3"
    if selection == "all" and buffer_policy == "latest" and stale == "drop":
        return "P4"
    if selection == "comm_aware" and buffer_policy == "fifo" and stale == "drop" and download == "stale_only":
        return "P5"
    if selection == "comm_aware" and buffer_policy == "latest" and stale == "drop" and download == "stale_only":
        return "P6"
    return ""


def _infer_severity(hyper: dict, start_args: dict, path: Path) -> str:
    explicit = _nested(hyper, "publication_context", "severity_name") or hyper.get("severity_name") or start_args.get("severity_name")
    if explicit:
        return str(explicit).capitalize()
    comm = hyper.get("communication", {}) or {}
    vals = (
        round(float(comm.get("online_prob_good", start_args.get("online_prob_good", -1))), 2),
        round(float(comm.get("online_prob_med", start_args.get("online_prob_med", -1))), 2),
        round(float(comm.get("online_prob_bad", start_args.get("online_prob_bad", -1))), 2),
    )
    if "severity" in str(path).lower():
        return _severity_profiles_by_probability().get(vals, "")
    return ""


def _infer_scenario(hyper: dict, start_args: dict, completion: dict, path: Path, policy: str, severity: str) -> str:
    explicit = _nested(hyper, "publication_context", "suite") or start_args.get("publication_suite")
    if explicit in {"ideal", "primary", "component", "sensitivity", "robustness"}:
        return str(explicit)
    rep = _int(_nested(hyper, "publication_context", "replicate_id", default=start_args.get("replicate_id", completion.get("replicate_id", 0))))
    low = str(path).lower()
    if rep > 0 or "five_replicates" in low or "replicate" in low:
        return "robustness"
    if policy in COMPONENT_IDS or "p5_scheduler_ablation" in low or "p5ablation" in low:
        return "component"
    if "severity" in low or (severity and policy in {"P3", "P5"} and start_args.get("severity_name")):
        return "sensitivity"
    if policy == "Ideal":
        return "ideal"
    if policy in {f"P{i}" for i in range(1, 7)}:
        return "primary"
    return "unknown"


@dataclass
class RunRecord:
    path: Path
    dataset: str
    regime: str
    policy: str
    scenario: str
    severity: str
    component_role: str
    replicate_id: int
    completed_at: str
    model_seed: int
    trace_seed: int
    scheduler_seed: int
    hyper: dict
    start_args: dict
    system: dict
    result_rows: list[dict[str, str]]

    @property
    def dataset_display(self) -> str:
        return DATASET_DISPLAY.get(self.dataset, self.dataset)

    @property
    def regime_display(self) -> str:
        return REGIME_DISPLAY.get(self.regime, self.regime)


def load_run(path: Path) -> Optional[RunRecord]:
    path = Path(path)
    completion = _read_json(path / "completion_status.json", {}) or {}
    if completion and completion.get("status") != "completed":
        return None
    hyper = _read_yaml(path / "run_hyperparams.yaml", {}) or {}
    if not hyper:
        return None
    start = _read_yaml(path / "run_start_config.yaml", {}) or {}
    start_args = start.get("args", {}) if isinstance(start, dict) else {}
    system = _read_json(path / "system_metrics.json", {}) or {}
    result_rows = _read_csv(path / "results_summary.csv")
    if not result_rows:
        return None
    dataset = _dataset_name(hyper, completion, path)
    regime = str(hyper.get("regime", completion.get("regime", "")))
    policy = _infer_policy(hyper, start_args, completion, path)
    severity = _infer_severity(hyper, start_args, path)
    scenario = _infer_scenario(hyper, start_args, completion, path, policy, severity)
    seeds = hyper.get("seed_provenance", {}) or system.get("seed_provenance", {}) or {}
    runtime = hyper.get("runtime", {}) or {}
    return RunRecord(
        path=path.resolve(),
        dataset=dataset,
        regime=regime,
        policy=policy,
        scenario=scenario,
        severity=severity,
        component_role=str(_nested(hyper, "publication_context", "component_role", default=start_args.get("component_role", "")) or ""),
        replicate_id=_int(_nested(hyper, "publication_context", "replicate_id", default=start_args.get("replicate_id", completion.get("replicate_id", 0)))),
        completed_at=str(completion.get("completed_at", path.stat().st_mtime)),
        model_seed=_int(seeds.get("model_seed", runtime.get("seed", 0))),
        trace_seed=_int(seeds.get("trace_seed", runtime.get("seed", 0))),
        scheduler_seed=_int(seeds.get("scheduler_seed", runtime.get("seed", 0))),
        hyper=hyper,
        start_args=start_args,
        system=system,
        result_rows=result_rows,
    )


def discover_runs(output_root: Path) -> list[RunRecord]:
    root = Path(output_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Experiment output directory does not exist: {root}")
    dirs = {p.parent for p in root.rglob("run_hyperparams.yaml")}
    records = []
    for d in sorted(dirs):
        r = load_run(d)
        if r is not None:
            records.append(r)
    return records


def _completion_sort_key(r: RunRecord):
    try:
        return datetime.fromisoformat(r.completed_at).timestamp()
    except (TypeError, ValueError):
        return r.path.stat().st_mtime


def _latest(records: Iterable[RunRecord], key_fn) -> list[RunRecord]:
    chosen: dict[Any, RunRecord] = {}
    for r in records:
        key = key_fn(r)
        if key not in chosen or _completion_sort_key(r) > _completion_sort_key(chosen[key]):
            chosen[key] = r
    return list(chosen.values())


def _paper_subject_budget_requirements(run: RunRecord) -> tuple[int, tuple[int, ...]]:
    """Return the cohort size and calibration budgets required by the paper protocol."""
    from .protocol import paper_protocol

    cfg = paper_protocol()
    key = "bciciv2a" if run.dataset_display == "BCICIV-2a" else "openbmi" if run.dataset_display == "OpenBMI" else ""
    if not key:
        raise ValueError(f"Unsupported dataset in publication analysis: {run.dataset!r}")
    n_subjects = int(cfg["dataset_defaults"][key]["subjects"])
    budgets = tuple(int(v) for v in cfg["personalization"]["calibration_trials_per_class"])
    return n_subjects, budgets


def subject_budget_means(run: RunRecord) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    """Return paper-protocol subject means across calibration budgets.

    Publication analysis is intentionally strict: every subject must have one
    successful chronological Session-2 result for each paper calibration budget.
    Failing or incomplete runs must not be silently averaged into a publication
    operating point.
    """
    expected_n, expected_budgets = _paper_subject_budget_requirements(run)
    by_sk: dict[tuple[str, int], list[float]] = {}
    failed_rows: list[str] = []
    malformed_rows: list[str] = []

    for idx, row in enumerate(run.result_rows, start=2):  # CSV header is line 1.
        raw_subject = str(row.get("subject", row.get("raw_subject", ""))).strip()
        k_raw = row.get("k")
        ok = str(row.get("ok", "True")).strip().lower() not in {"false", "0", "no"}
        acc = _float(row.get("test_acc"))
        k = _int(k_raw, -1)

        if not raw_subject or k not in expected_budgets:
            malformed_rows.append(f"line {idx}: subject={raw_subject!r}, k={k_raw!r}")
            continue
        subject = raw_subject.zfill(3)
        if not ok or acc is None or not math.isfinite(acc):
            failed_rows.append(f"line {idx}: subject={subject}, k={k}, ok={row.get('ok')!r}, test_acc={row.get('test_acc')!r}")
            continue
        if not (0.0 <= acc <= 1.0):
            malformed_rows.append(f"line {idx}: subject={subject}, k={k}, test_acc={acc!r}")
            continue
        by_sk.setdefault((subject, k), []).append(acc * 100.0)

    if malformed_rows:
        raise ValueError(
            f"Malformed publication result rows in {run.path / 'results_summary.csv'}: "
            + "; ".join(malformed_rows[:5])
            + (f"; ... ({len(malformed_rows)} total)" if len(malformed_rows) > 5 else "")
        )
    if failed_rows:
        raise ValueError(
            f"Failed/incomplete publication result rows in {run.path / 'results_summary.csv'}: "
            + "; ".join(failed_rows[:5])
            + (f"; ... ({len(failed_rows)} total)" if len(failed_rows) > 5 else "")
        )

    observed_subjects = sorted({subject for subject, _ in by_sk})
    expected_subjects = [f"{idx:03d}" for idx in range(1, expected_n + 1)]
    if observed_subjects != expected_subjects:
        missing_subjects = sorted(set(expected_subjects).difference(observed_subjects))
        unexpected_subjects = sorted(set(observed_subjects).difference(expected_subjects))
        raise ValueError(
            f"Publication run {run.path} does not contain the expected {run.dataset_display} cohort. "
            f"Missing subjects: {missing_subjects}; unexpected subjects: {unexpected_subjects}."
        )

    duplicate_keys = {key: len(vals) for key, vals in by_sk.items() if len(vals) != 1}
    if duplicate_keys:
        preview = ", ".join(f"{s}/k={k}:{n}" for (s, k), n in sorted(duplicate_keys.items())[:10])
        raise ValueError(
            f"Publication run {run.path} must contain exactly one result per subject/calibration budget; "
            f"found repeated/missing-style entries: {preview}."
        )

    missing = [
        (subject, k)
        for subject in observed_subjects
        for k in expected_budgets
        if (subject, k) not in by_sk
    ]
    if missing:
        preview = ", ".join(f"{s}/k={k}" for s, k in missing[:10])
        raise ValueError(
            f"Publication run {run.path} is missing {len(missing)} subject/calibration results: {preview}."
        )

    expected_rows = expected_n * len(expected_budgets)
    if len(by_sk) != expected_rows:
        raise ValueError(
            f"Publication run {run.path} contains {len(by_sk)} unique subject/calibration results; "
            f"expected {expected_rows}."
        )

    per_k: dict[int, dict[str, float]] = {k: {} for k in expected_budgets}
    for (subject, k), vals in by_sk.items():
        per_k[k][subject] = float(vals[0])
    per_subject = {
        subject: float(np.mean([per_k[k][subject] for k in expected_budgets]))
        for subject in observed_subjects
    }
    return per_subject, per_k


def _accepted_staleness(run: RunRecord) -> Optional[float]:
    canonical = run.system.get("accepted_update_staleness_summary") or {}
    if canonical.get("count", 0):
        return _float(canonical.get("mean"))
    stale = run.system.get("staleness_summary") or {}
    definition = str(stale.get("definition", "")).lower()
    if stale.get("count", 0) and ("accepted" in definition or "entering aggregation" in definition):
        return _float(stale.get("mean"))

    # Some run formats store pre-admission staleness as staleness_summary.
    # Recover the event-weighted accepted statistic from the saved round trace when needed.
    progress = _read_json(run.path / "comm_progress.json", []) or []
    all_vals: list[float] = []
    dropped: list[float] = []
    for rnd in progress:
        all_vals.extend(float(x) for x in rnd.get("staleness", []) if x is not None)
        dropped.extend(float(x) for x in rnd.get("staleness_dropped", []) if x is not None)
    accepted_count = _int(_nested(run.system, "ongoing", "accepted_updates", default=0))
    if accepted_count > 0 and all_vals:
        accepted_sum = float(sum(all_vals) - sum(dropped))
        return accepted_sum / accepted_count
    return 0.0 if accepted_count == 0 else None


def operating_point(run: RunRecord) -> dict[str, Any]:
    per_subject, per_k = subject_budget_means(run)
    vals = np.array(list(per_subject.values()), dtype=float)
    ongoing = run.system.get("ongoing", {}) or {}
    total = run.system.get("total_communication", {}) or {}
    c2s_bytes = _float(total.get("client_to_server_bytes"), _float(ongoing.get("client_to_server_bytes"), 0.0)) or 0.0
    s2c_bytes = _float(total.get("server_to_client_bytes"), _float(ongoing.get("server_to_client_bytes"), 0.0)) or 0.0
    uploads = _int(ongoing.get("uploads"))
    dropped = _int(ongoing.get("dropped_updates"))
    opportunities = _int(ongoing.get("download_opportunities"))
    avoided = _int(ongoing.get("download_avoided"))
    delay = run.system.get("delay_summary", {}) or {}
    delay_count = _int(delay.get("count"))
    return {
        "dataset": run.dataset_display,
        "regime": run.regime_display,
        "scenario": run.scenario,
        "severity": run.severity,
        "policy": run.policy,
        "n": int(len(vals)),
        "mean_accuracy_pct": float(vals.mean()) if len(vals) else None,
        "sd_accuracy_pct": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "sem_accuracy_pct": float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0,
        "c2s_mb": c2s_bytes / 1_000_000.0,
        "s2c_mb": s2c_bytes / 1_000_000.0,
        "total_mb": (c2s_bytes + s2c_bytes) / 1_000_000.0,
        "rejected_uploads_pct": (100.0 * dropped / uploads) if uploads else 0.0,
        "accepted_staleness_versions": _accepted_staleness(run),
        "buffer_delay_rounds": _float(delay.get("mean_rounds")) if delay_count > 0 else None,
        "avoided_downloads_pct": (100.0 * avoided / opportunities) if opportunities else 0.0,
        "run_dir": str(run.path),
        "model_seed": run.model_seed,
        "trace_seed": run.trace_seed,
        "scheduler_seed": run.scheduler_seed,
        "replicate_id": run.replicate_id,
    }


def calibration_rows(run: RunRecord) -> list[dict[str, Any]]:
    _, per_k = subject_budget_means(run)
    out = []
    for k in sorted(per_k):
        vals = np.array(list(per_k[k].values()), dtype=float)
        out.append({
            "dataset": run.dataset_display,
            "regime": run.regime_display,
            "scenario": run.scenario,
            "severity": run.severity,
            "policy": run.policy,
            "k": k,
            "n": len(vals),
            "mean_accuracy_pct": float(vals.mean()) if len(vals) else None,
            "sd_accuracy_pct": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "sem_accuracy_pct": float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "run_dir": str(run.path),
        })
    return out


def _paired_stats(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    subjects = sorted(set(a).intersection(b))
    diff = np.array([b[s] - a[s] for s in subjects], dtype=float)
    n = len(diff)
    if n == 0:
        return {"n": 0}
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    crit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    ci_low = mean - crit * sem if n > 1 else mean
    ci_high = mean + crit * sem if n > 1 else mean
    if n > 1 and sd > 0:
        t_res = stats.ttest_rel([b[s] for s in subjects], [a[s] for s in subjects], alternative="two-sided")
        t_p = float(t_res.pvalue)
        dz = mean / sd
    else:
        t_p = 1.0 if abs(mean) <= EPS else 0.0
        dz = 0.0 if abs(mean) <= EPS else math.copysign(float("inf"), mean)
    if np.all(np.abs(diff) <= EPS):
        wilcoxon_p = 1.0
    else:
        try:
            wilcoxon_p = float(stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = 1.0
    return {
        "n": n,
        "mean_difference_pp": mean,
        "sd_difference_pp": sd,
        "ci95_low_pp": ci_low,
        "ci95_high_pp": ci_high,
        "paired_effect_size_dz": dz,
        "paired_t_p": t_p,
        "wilcoxon_p": wilcoxon_p,
        "subjects": subjects,
        "differences_pp": diff.tolist(),
    }


def _holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    out = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * pvalues[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out


def _primary_pair_stats(primary: list[RunRecord]) -> list[dict[str, Any]]:
    by_key = {(r.dataset_display, r.regime_display, r.policy): r for r in primary}
    rows = []
    for dataset in ("BCICIV-2a", "OpenBMI"):
        for regime in ("SB-PH", "EIB-PH"):
            p3 = by_key.get((dataset, regime, "P3"))
            p5 = by_key.get((dataset, regime, "P5"))
            if not p3 or not p5:
                continue
            a, _ = subject_budget_means(p3)
            b, _ = subject_budget_means(p5)
            st = _paired_stats(a, b)
            op3, op5 = operating_point(p3), operating_point(p5)
            rows.append({
                "dataset": dataset,
                "regime": regime,
                "n": st["n"],
                "mean_p5_minus_p3_pp": st["mean_difference_pp"],
                "ci95_low_pp": st["ci95_low_pp"],
                "ci95_high_pp": st["ci95_high_pp"],
                "paired_effect_size_dz": st["paired_effect_size_dz"],
                "paired_t_p": st["paired_t_p"],
                "wilcoxon_p": st["wilcoxon_p"],
                "s2c_saved_mb": op3["s2c_mb"] - op5["s2c_mb"],
                "s2c_reduction_pct": 100.0 * (op3["s2c_mb"] - op5["s2c_mb"]) / op3["s2c_mb"] if op3["s2c_mb"] else 0.0,
                "total_traffic_saved_mb": op3["total_mb"] - op5["total_mb"],
            })
    adjusted = _holm([float(r["paired_t_p"]) for r in rows]) if rows else []
    for r, p in zip(rows, adjusted):
        r["holm_adjusted_t_p"] = p
    return rows


def _reliability_rows(ideal: list[RunRecord], primary: list[RunRecord]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ideal_by = {(r.dataset_display, r.regime_display): r for r in ideal}
    rows = []
    subject_rows = []
    for run in primary:
        if run.regime_display != "EIB-PH" or run.policy not in {f"P{i}" for i in range(1, 7)}:
            continue
        ref = ideal_by.get((run.dataset_display, "EIB-PH"))
        if not ref:
            continue
        pol, _ = subject_budget_means(run)
        ide, _ = subject_budget_means(ref)
        subjects = sorted(set(pol).intersection(ide))
        diffs = np.array([pol[s] - ide[s] for s in subjects], dtype=float)
        acc = np.array([pol[s] for s in subjects], dtype=float)
        threshold = _reliability_thresholds()[run.dataset_display]
        dec = diffs <= (-5.0 + EPS)
        inc = diffs >= (5.0 - EPS)
        below = acc < (threshold - EPS)
        rows.append({
            "dataset": run.dataset_display,
            "policy": run.policy,
            "n": len(subjects),
            "mean_accuracy_difference_pp": float(diffs.mean()),
            "median_accuracy_difference_pp": float(np.median(diffs)),
            "minimum_subject_difference_pp": float(diffs.min()),
            "maximum_subject_difference_pp": float(diffs.max()),
            "difference_range_pp": float(diffs.max() - diffs.min()),
            "minimum_subject_accuracy_pct": float(acc.min()),
            "analysis_threshold_pct": threshold,
            "subjects_below_threshold": int(below.sum()),
            "subjects_with_ge_5pp_decrease": int(dec.sum()),
            "subjects_with_ge_5pp_increase": int(inc.sum()),
            "below_threshold_with_ge_5pp_decrease": int(np.logical_and(below, dec).sum()),
        })
        for s in subjects:
            subject_rows.append({
                "dataset": run.dataset_display,
                "regime": "EIB-PH",
                "policy": run.policy,
                "subject": s,
                "ideal_accuracy_pct": ide[s],
                "policy_accuracy_pct": pol[s],
                "difference_vs_ideal_pp": pol[s] - ide[s],
                "analysis_threshold_pct": threshold,
                "below_threshold": pol[s] < (threshold - EPS),
            })
    return rows, subject_rows


def _component_rows(component: list[RunRecord]) -> list[dict[str, Any]]:
    label_order = {
        "P3_REF": (0, "P3 reference"),
        "ONLINE_RANDOM": (1, "Online-random selection"),
        "ONLINE_PRIORITY": (2, "Online-priority selection"),
        "DOWNLOAD_ONLY": (3, "Stale-aware download only"),
        "P5_FULL": (4, "Full P5"),
    }
    by_id = {r.policy: r for r in component}
    if "P3_REF" not in by_id:
        return []
    base = operating_point(by_id["P3_REF"])
    rows = []
    for pid, (order, label) in label_order.items():
        if pid not in by_id:
            continue
        op = operating_point(by_id[pid])
        rows.append({
            "order": order,
            "variant": label,
            "mean_accuracy_pct": op["mean_accuracy_pct"],
            "accuracy_diff_from_p3_pp": op["mean_accuracy_pct"] - base["mean_accuracy_pct"],
            "c2s_mb": op["c2s_mb"],
            "s2c_mb": op["s2c_mb"],
            "s2c_reduction_pct": 100.0 * (base["s2c_mb"] - op["s2c_mb"]) / base["s2c_mb"] if base["s2c_mb"] else 0.0,
            "rejected_uploads_pct": op["rejected_uploads_pct"],
            "accepted_staleness_versions": op["accepted_staleness_versions"],
            "run_dir": op["run_dir"],
        })
    return sorted(rows, key=lambda x: x["order"])


def _sensitivity_rows(sensitivity: list[RunRecord]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ops = []
    by_key = {}
    for r in sensitivity:
        if r.policy not in {"P3", "P5"}:
            continue
        op = operating_point(r)
        op["online_probabilities"] = "/".join(
            f"{float(_nested(r.hyper, 'communication', x, default=r.start_args.get(x, 0))):.2f}"
            for x in ("online_prob_good", "online_prob_med", "online_prob_bad")
        )
        ops.append(op)
        by_key[(r.dataset_display, r.severity, r.policy)] = r
    stats_rows = []
    for dataset in ("BCICIV-2a", "OpenBMI"):
        for severity in ("Mild", "Default", "Severe"):
            p3 = by_key.get((dataset, severity, "P3"))
            p5 = by_key.get((dataset, severity, "P5"))
            if not p3 or not p5:
                continue
            a, _ = subject_budget_means(p3)
            b, _ = subject_budget_means(p5)
            st = _paired_stats(a, b)
            stats_rows.append({
                "dataset": dataset,
                "severity": severity,
                "n": st["n"],
                "mean_p5_minus_p3_pp": st["mean_difference_pp"],
                "ci95_low_pp": st["ci95_low_pp"],
                "ci95_high_pp": st["ci95_high_pp"],
                "paired_t_p": st["paired_t_p"],
            })
    # The sensitivity analysis treats the three availability profiles as one
    # multiplicity family within each dataset.
    for dataset in ("BCICIV-2a", "OpenBMI"):
        indices = [i for i, row in enumerate(stats_rows) if row["dataset"] == dataset]
        adjusted = _holm([float(stats_rows[i]["paired_t_p"]) for i in indices]) if indices else []
        for i, p_adj in zip(indices, adjusted):
            stats_rows[i]["holm_adjusted_t_p"] = p_adj
    return ops, stats_rows


def _safe_spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def _bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for Spearman rho with exact tie re-ranking.

    Resampling creates duplicate observations, so ranks must be recomputed in
    every bootstrap draw. We do that in vectorized batches using
    ``scipy.stats.rankdata(axis=1)`` rather than issuing thousands of individual
    ``spearmanr`` calls.
    """
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(x)
    vals: list[np.ndarray] = []
    remaining = int(n_boot)
    batch_size = min(1000, max(1, int(n_boot)))
    while remaining > 0:
        b = min(batch_size, remaining)
        idx = rng.integers(0, n, size=(b, n))
        xb = x[idx]
        yb = y[idx]
        rx = stats.rankdata(xb, axis=1, method="average")
        ry = stats.rankdata(yb, axis=1, method="average")
        rx = rx - rx.mean(axis=1, keepdims=True)
        ry = ry - ry.mean(axis=1, keepdims=True)
        denom = np.sqrt(np.sum(rx * rx, axis=1) * np.sum(ry * ry, axis=1))
        numer = np.sum(rx * ry, axis=1)
        valid = denom > 0
        if np.any(valid):
            vals.append(numer[valid] / denom[valid])
        remaining -= b
    if not vals:
        return float("nan"), float("nan")
    arr = np.concatenate(vals)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))



def _reconstruct_subject_accepted_staleness(run: RunRecord) -> dict[str, float]:
    """Recover per-subject accepted-update staleness from saved round traces.

    Some run formats retain per-subject counts and round-level communication
    traces but do not persist the per-subject list of accepted
    staleness values.  The communication state is fully determined by those
    traces: local backbone versions, bounded pending-update queues, checkpoint
    retention, stale admission, and download decisions can therefore be replayed
    without EEG data or model weights.

    The replay is validated against the persisted per-subject upload, accept,
    drop, and pre-admission staleness summaries before the reconstructed values
    are returned.  A format mismatch raises ``RuntimeError`` rather than silently
    substituting a different staleness definition.
    """
    progress = _read_json(run.path / "comm_progress.json", []) or []
    comm = run.system.get("per_subject_comm", {}) or {}
    cfg = run.system.get("communication_config", {}) or {}
    if not progress or not comm or not cfg:
        return {}

    subjects = sorted(str(k).zfill(3) for k in comm.keys())
    local_version = {s: 0 for s in subjects}
    buffers: dict[str, list[tuple[int, int]]] = {s: [] for s in subjects}
    all_staleness: dict[str, list[int]] = {s: [] for s in subjects}
    accepted_staleness: dict[str, list[int]] = {s: [] for s in subjects}
    upload_counts = {s: 0 for s in subjects}
    accepted_counts = {s: 0 for s in subjects}
    dropped_counts = {s: 0 for s in subjects}

    buffer_enabled = bool(cfg.get("buffering_enabled", False))
    buffer_policy = str(cfg.get("buffer_policy", "none"))
    buffer_max = max(1, _int(cfg.get("buffer_max_size"), 1))
    selection_policy = str(cfg.get("selection_policy", "all"))
    download_policy = str(cfg.get("download_policy", "always"))
    download_threshold = _int(cfg.get("download_stale_threshold"), _int(cfg.get("stale_threshold"), 0))
    stale_policy = str(cfg.get("stale_policy", "drop"))
    stale_threshold = _int(cfg.get("stale_threshold"), 0)
    retention_margin = _int(cfg.get("checkpoint_retention_margin"), 5)

    retained_versions = {0}
    server_version = 0

    def _sid(value: Any) -> str:
        return str(value).zfill(3)

    def _buffer(subject: str, payload: tuple[int, int]) -> None:
        if not buffer_enabled or buffer_policy == "none":
            return
        queue = buffers[subject]
        if buffer_policy == "latest":
            queue.clear()
            queue.append(payload)
        elif buffer_policy == "fifo":
            while len(queue) >= buffer_max:
                queue.pop(0)
            queue.append(payload)
        else:
            raise RuntimeError(f"Unsupported saved buffer policy {buffer_policy!r} in {run.path}")

    for rnd in progress:
        round_number = _int(rnd.get("round"))
        start_version = _int(rnd.get("server_version_start"), -1)
        if start_version != server_version:
            raise RuntimeError(
                f"Communication replay version mismatch in {run.path} at round {round_number}: "
                f"expected {server_version}, saved {start_version}."
            )

        selected = [_sid(s) for s in rnd.get("selected_subjects", [])]
        selected_set = set(selected)
        online = {_sid(s) for s in rnd.get("available_online_subjects", [])}
        offline = [_sid(s) for s in rnd.get("available_offline_subjects", [])]
        uploaded: list[tuple[str, int, int]] = []

        for subject in selected:
            if subject not in local_version:
                raise RuntimeError(f"Unknown subject {subject!r} in communication trace {run.path}")
            if subject in online:
                lag = server_version - local_version[subject]
                if lag > 0:
                    should_download = download_policy == "always" or (
                        download_policy == "stale_only" and lag > download_threshold
                    )
                    if should_download:
                        local_version[subject] = server_version

                for base_version, produced_round in buffers[subject]:
                    uploaded.append((subject, base_version, produced_round))
                buffers[subject].clear()
                uploaded.append((subject, local_version[subject], round_number))
            else:
                _buffer(subject, (local_version[subject], round_number))

        # Communication-aware policies leave online-but-unselected gateways idle,
        # while unavailable gateways may continue local training and retain the
        # resulting pending update when buffering is enabled.
        if selection_policy == "comm_aware":
            for subject in offline:
                if subject not in selected_set:
                    _buffer(subject, (local_version[subject], round_number))

        accepted_this_round = 0
        for subject, base_version, _produced_round in uploaded:
            upload_counts[subject] += 1
            if base_version not in retained_versions:
                dropped_counts[subject] += 1
                continue

            staleness = server_version - base_version
            all_staleness[subject].append(staleness)
            if stale_policy == "drop" and staleness > stale_threshold:
                dropped_counts[subject] += 1
                continue

            accepted_staleness[subject].append(staleness)
            accepted_counts[subject] += 1
            accepted_this_round += 1

        if accepted_this_round:
            server_version += 1
            retained_versions.add(server_version)
            min_keep = (
                0
                if stale_policy == "accept_all"
                else server_version - stale_threshold - retention_margin
            )
            retained_versions = {v for v in retained_versions if v >= min_keep}

        end_version = _int(rnd.get("server_version_end"), -1)
        if end_version != server_version:
            raise RuntimeError(
                f"Communication replay end-version mismatch in {run.path} at round {round_number}: "
                f"replayed {server_version}, saved {end_version}."
            )

    for subject in subjects:
        saved = comm.get(subject, comm.get(str(int(subject)) if subject.isdigit() else subject, {})) or {}
        expected_uploads = _int(saved.get("upload_events"))
        expected_accepted = _int(saved.get("accepted_events"))
        expected_dropped = _int(saved.get("dropped_events"))
        if upload_counts[subject] != expected_uploads or accepted_counts[subject] != expected_accepted or dropped_counts[subject] != expected_dropped:
            raise RuntimeError(
                f"Communication replay count mismatch for {run.dataset_display} {run.policy} subject {subject}."
            )
        saved_mean = _float(saved.get("staleness_mean"))
        replay_mean = float(np.mean(all_staleness[subject])) if all_staleness[subject] else 0.0
        if saved_mean is not None and not math.isclose(replay_mean, saved_mean, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"Communication replay staleness mismatch for {run.dataset_display} {run.policy} subject {subject}."
            )

    return {
        subject: (float(np.mean(values)) if values else 0.0)
        for subject, values in accepted_staleness.items()
    }


def _association_rows(
    ideal: list[RunRecord], primary: list[RunRecord], bootstrap_samples: int, bootstrap_seed: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compute the descriptive subject-level associations in Supplementary Table S7."""
    warnings = []
    ideal_by = {(r.dataset_display, r.regime_display): r for r in ideal}
    out = []
    variables = [
        ("ideal_link_accuracy", "Ideal-link accuracy"),
        ("online_probability", "Assigned gateway online probability"),
        ("coordinator_rejected_uploads", "Coordinator-rejected uploads (count)"),
        ("accepted_update_staleness", "Mean accepted-update staleness (versions)"),
        ("updates_placed_in_buffer", "Updates placed in buffer (count)"),
        ("avoided_backbone_downloads", "Avoided backbone downloads (count)"),
    ]
    for run in primary:
        if run.regime_display != "EIB-PH" or run.policy not in {"P3", "P5"}:
            continue
        ref = ideal_by.get((run.dataset_display, "EIB-PH"))
        if not ref:
            continue
        pol, _ = subject_budget_means(run)
        ide, _ = subject_budget_means(ref)
        comm = run.system.get("per_subject_comm", {}) or {}
        reconstructed_staleness: dict[str, float] = {}
        if comm and any(_float((c or {}).get("accepted_update_staleness_mean")) is None for c in comm.values()):
            reconstructed_staleness = _reconstruct_subject_accepted_staleness(run)
        subjects = sorted(set(pol).intersection(ide).intersection(str(k).zfill(3) for k in comm.keys()))
        if not subjects:
            # Keys are already report IDs in normal outputs.
            subjects = sorted(set(pol).intersection(ide).intersection(comm.keys()))
        if not subjects:
            continue
        # Subject accuracies are ratios of integer correct-counts, averaged over
        # three calibration budgets. Algebraically equal effects can therefore
        # differ at ~1e-13 after repeated floating-point arithmetic. Spearman
        # ranking must treat those effects as ties rather than impose arbitrary
        # machine-order differences. Twelve decimals is far below the resolution
        # of any accuracy value in these datasets while removing that numerical
        # noise.
        y = np.round(np.array([pol[s] - ide[s] for s in subjects], dtype=float), 12)
        values: dict[str, list[Optional[float]]] = {k: [] for k, _ in variables}
        for s in subjects:
            c = comm.get(s, comm.get(str(int(s)) if s.isdigit() else s, {})) or {}
            values["ideal_link_accuracy"].append(ide[s])
            values["online_probability"].append(_float(c.get("online_probability")))
            values["coordinator_rejected_uploads"].append(_float(c.get("dropped_events"), 0.0))
            accepted = _float(c.get("accepted_update_staleness_mean"))
            if accepted is None:
                accepted = reconstructed_staleness.get(s)
            values["accepted_update_staleness"].append(accepted)
            values["updates_placed_in_buffer"].append(_float(c.get("buffer_events"), 0.0))
            values["avoided_backbone_downloads"].append(_float(c.get("download_avoided_events"), 0.0))
        for vi, (key, label) in enumerate(variables):
            vals = values[key]
            if any(v is None for v in vals):
                warnings.append(
                    f"{run.dataset_display} {run.policy}: {label} unavailable in this run format; correlation omitted."
                )
                rho = lo = hi = float("nan")
                n = 0
            else:
                x = np.array(vals, dtype=float)
                rho = _safe_spearman(x.tolist(), y.tolist())
                lo, hi = _bootstrap_spearman(x, y, bootstrap_samples, bootstrap_seed)
                n = len(x)
            out.append({
                "dataset": run.dataset_display,
                "policy": run.policy,
                "n": n,
                "variable": label,
                "spearman_rho": rho,
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
            })
    return out, sorted(set(warnings))


def _validate_robustness_suite_dir(suite_dir: Path) -> tuple[str, dict[str, Any]] | None:
    """Validate one complete five-replicate robustness child suite.

    Publication analysis must never combine CSVs from different suite executions or
    select a newer partial/diagnostic suite merely because its files have a later
    modification time. A valid child suite contains both regimes, five matched
    P3/P5 replicate pairs per regime, the study seed schedule, and a passing suite
    validation report.
    """
    from .protocol import paper_protocol

    suite_dir = Path(suite_dir)
    required_csvs = (
        "robustness_replicate_pairs.csv",
        "robustness_subject_paired_accuracy.csv",
        "robustness_descriptive_summary.csv",
        "robustness_hierarchical_bootstrap.csv",
        "robustness_pair_validation.csv",
        "robustness_cross_regime_communication_audit.csv",
    )
    if any(not (suite_dir / name).is_file() for name in required_csvs):
        return None
    report_path = suite_dir / "robustness_validation_report.json"
    if not report_path.is_file():
        return None
    report = _read_json(report_path, {}) or {}
    if str(report.get("status", "")).lower() != "passed":
        return None
    if _int(report.get("n_runs")) != 20 or _int(report.get("n_matched_pairs")) != 10:
        return None
    for flag in (
        "all_matched_pair_checks_passed",
        "one_trace_hash_per_replicate",
        "different_trace_hash_across_replicates",
        "fixed_group_assignment_across_replicates",
    ):
        if report.get(flag) is not True:
            return None

    rows = _read_csv(suite_dir / "robustness_replicate_pairs.csv")
    if len(rows) != 10:
        return None
    raw_datasets = {str(r.get("dataset_name", r.get("dataset", ""))).strip().lower() for r in rows}
    if len(raw_datasets) != 1:
        return None
    raw_dataset = next(iter(raw_datasets))
    display_dataset = DATASET_DISPLAY.get(raw_dataset, "")
    if display_dataset not in {"BCICIV-2a", "OpenBMI"}:
        return None

    cfg = paper_protocol()
    robust = cfg["robustness"]
    expected_model = [int(v) for v in robust["model_seeds"]]
    expected_trace = [int(v) for v in robust["availability_trace_seeds"]]
    expected_scheduler = [int(v) for v in robust["scheduler_tie_break_seeds"]]
    expected_group_seed = int(robust["gateway_group_seed"])
    expected_regimes = set(REGIME_DISPLAY)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        regime = str(row.get("regime", ""))
        grouped.setdefault(regime, []).append(row)
    if set(grouped) != expected_regimes:
        return None

    for regime, regime_rows in grouped.items():
        if len(regime_rows) != 5:
            return None
        by_rep: dict[int, dict[str, str]] = {}
        for row in regime_rows:
            rep = _int(row.get("replicate_id"), -1)
            if rep in by_rep or rep not in range(1, 6):
                return None
            by_rep[rep] = row
        if sorted(by_rep) != [1, 2, 3, 4, 5]:
            return None
        for rep in range(1, 6):
            row = by_rep[rep]
            idx = rep - 1
            if _int(row.get("model_seed"), -1) != expected_model[idx]:
                return None
            if _int(row.get("trace_seed"), -1) != expected_trace[idx]:
                return None
            if _int(row.get("scheduler_seed_p5"), -1) != expected_scheduler[idx]:
                return None
            if _int(row.get("group_seed"), -1) != expected_group_seed:
                return None

    n_subjects = int(cfg["dataset_defaults"]["bciciv2a" if display_dataset == "BCICIV-2a" else "openbmi"]["subjects"])
    if any(_int(row.get("n_subjects"), -1) != n_subjects for row in rows):
        return None

    # Companion tables must be non-empty before the suite is accepted.
    for name in required_csvs[1:]:
        if not _read_csv(suite_dir / name):
            return None
    return display_dataset, report


def _robustness_sources(output_root: Path, analysis_dir: Path) -> list[str]:
    """Collect one complete study robustness child suite per dataset.

    ``nexus-mi run robustness`` writes separate child suites for BCICIV-2a and
    OpenBMI. Selection is suite-level, not file-level: only complete passing study
    suites are eligible, and every CSV for a dataset is copied from the same suite.
    """
    copied: list[str] = []
    csv_names = [
        "robustness_replicate_pairs.csv",
        "robustness_subject_paired_accuracy.csv",
        "robustness_descriptive_summary.csv",
        "robustness_hierarchical_bootstrap.csv",
        "robustness_pair_validation.csv",
        "robustness_cross_regime_communication_audit.csv",
    ]
    root = Path(output_root)
    valid_by_dataset: dict[str, list[tuple[Path, dict[str, Any]]]] = {
        "BCICIV-2a": [],
        "OpenBMI": [],
    }
    candidate_dirs = {p.parent for p in root.rglob("robustness_replicate_pairs.csv")}
    for suite_dir in sorted(candidate_dirs):
        valid = _validate_robustness_suite_dir(suite_dir)
        if valid is None:
            continue
        dataset, report = valid
        valid_by_dataset[dataset].append((suite_dir, report))

    selected: dict[str, tuple[Path, dict[str, Any]]] = {}
    for dataset in ("BCICIV-2a", "OpenBMI"):
        candidates = valid_by_dataset[dataset]
        if not candidates:
            continue
        selected[dataset] = max(
            candidates,
            key=lambda item: max(
                (item[0] / "robustness_replicate_pairs.csv").stat().st_mtime,
                (item[0] / "robustness_validation_report.json").stat().st_mtime,
            ),
        )

    for name in csv_names:
        combined: list[dict[str, Any]] = []
        for dataset in ("BCICIV-2a", "OpenBMI"):
            chosen = selected.get(dataset)
            if chosen is None:
                continue
            src = chosen[0] / name
            combined.extend(_read_csv(src))
            copied.append(str(src.resolve()))
        if combined:
            _write_csv(analysis_dir / name, combined)

    if selected:
        payload = []
        for dataset in ("BCICIV-2a", "OpenBMI"):
            chosen = selected.get(dataset)
            if chosen is None:
                continue
            suite_dir, report = chosen
            src = suite_dir / "robustness_validation_report.json"
            payload.append({
                "dataset": dataset,
                "source": str(src.resolve()),
                "report": report,
            })
            copied.append(str(src.resolve()))
        _write_json(analysis_dir / "robustness_validation_reports.json", payload)
    return sorted(set(copied))


def _summary_ci(values: list[float]) -> tuple[float, float, float, float, float, float]:
    """Return mean, sample SD, 95% Student-t CI, min, and max."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"),) * 6
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if arr.size > 1:
        half = float(stats.t.ppf(0.975, arr.size - 1)) * sd / math.sqrt(arr.size)
    else:
        half = 0.0
    return mean, sd, mean - half, mean + half, float(arr.min()), float(arr.max())


def _paper_robustness_tables(analysis_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Build Supplementary Tables S2--S5.

    The robustness suite writes detailed run-level CSVs.  This
    function derives the compact tables reported in the supplement without
    relying on embedded publication values.
    """
    pairs_path = analysis_dir / "robustness_replicate_pairs.csv"
    subjects_path = analysis_dir / "robustness_subject_paired_accuracy.csv"
    boot_path = analysis_dir / "robustness_hierarchical_bootstrap.csv"
    if not (pairs_path.is_file() and subjects_path.is_file() and boot_path.is_file()):
        return {}

    pairs = _read_csv(pairs_path)
    subjects = _read_csv(subjects_path)
    boots = _read_csv(boot_path)
    display_dataset = lambda x: DATASET_DISPLAY.get(str(x).lower(), str(x))
    display_regime = lambda x: REGIME_DISPLAY.get(str(x), str(x))

    boot_by = {
        (display_dataset(r.get("dataset_name", "")), display_regime(r.get("regime", ""))): r
        for r in boots
    }

    # Supplementary Table S2A: replicate-level cohort accuracy.
    s2a: list[dict[str, Any]] = []
    s2b: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in pairs:
        groups.setdefault((display_dataset(r.get("dataset_name", "")), display_regime(r.get("regime", ""))), []).append(r)
    dataset_rank = {"BCICIV-2a": 0, "OpenBMI": 1}
    regime_rank = {"SB-PH": 0, "EIB-PH": 1}
    for (dataset, regime), rows in sorted(groups.items(), key=lambda kv: (dataset_rank.get(kv[0][0], 99), regime_rank.get(kv[0][1], 99))):
        rows = sorted(rows, key=lambda r: (_int(r.get("model_seed")) or 0, _int(r.get("replicate_id")) or 0))
        n_subjects = _int(rows[0].get("n_subjects")) or 0
        for policy, col in (("P3", "p3_mean_accuracy"), ("P5", "p5_mean_accuracy")):
            vals = [100.0 * float(_float(r.get(col), 0.0)) for r in rows]
            mean, sd, lo, hi, mn, mx = _summary_ci(vals)
            s2a.append({
                "dataset": dataset, "regime": regime, "policy": policy, "n": n_subjects,
                "mean_subject_accuracy_across_budgets_pct": mean,
                "sample_sd_across_replicates_pct": sd,
                "student_t_ci95_low_pct": lo, "student_t_ci95_high_pct": hi,
                "observed_range_low_pct": mn, "observed_range_high_pct": mx,
                "matched_replicates": len(rows),
            })
        effects = [float(_float(r.get("p5_minus_p3_accuracy_pp"), 0.0)) for r in rows]
        mean, sd, lo, hi, mn, mx = _summary_ci(effects)
        b = boot_by.get((dataset, regime), {})
        s2b.append({
            "dataset": dataset, "regime": regime, "n": n_subjects,
            "mean_p5_minus_p3_effect_pp": mean,
            "sample_sd_across_replicates_pp": sd,
            "student_t_ci95_low_pp": lo, "student_t_ci95_high_pp": hi,
            "hierarchical_ci95_low_pp": _float(b.get("hierarchical_bootstrap_ci95_low_pp")),
            "hierarchical_ci95_high_pp": _float(b.get("hierarchical_bootstrap_ci95_high_pp")),
            "observed_range_low_pp": mn, "observed_range_high_pp": mx,
            "positive_replicates": int(sum(v > EPS for v in effects)),
            "matched_replicates": len(rows),
        })

    # Supplementary Table S3: subject-level effect heterogeneity across replicates.
    by_subject: dict[tuple[str, str, str], list[float]] = {}
    all_effects: dict[tuple[str, str], list[float]] = {}
    for r in subjects:
        dataset = display_dataset(r.get("dataset_name", "")); regime = display_regime(r.get("regime", ""))
        subject = str(r.get("subject", ""))
        effect = float(_float(r.get("p5_minus_p3_accuracy_pp"), 0.0))
        by_subject.setdefault((dataset, regime, subject), []).append(effect)
        all_effects.setdefault((dataset, regime), []).append(effect)
    s3: list[dict[str, Any]] = []
    for key in sorted(all_effects, key=lambda k: (dataset_rank.get(k[0], 99), regime_rank.get(k[1], 99))):
        dataset, regime = key
        subject_means = [float(np.mean(v)) for (d, g, _), v in by_subject.items() if (d, g) == key]
        full = np.asarray(all_effects[key], dtype=float)
        sm = np.asarray(subject_means, dtype=float)
        s3.append({
            "dataset": dataset, "regime": regime, "n": int(sm.size),
            "median_subject_mean_effect_pp": float(np.median(sm)),
            "subject_mean_range_low_pp": float(sm.min()), "subject_mean_range_high_pp": float(sm.max()),
            "subjects_with_positive_mean_effect": int(np.sum(sm > EPS)),
            "full_subject_by_replicate_range_low_pp": float(full.min()),
            "full_subject_by_replicate_range_high_pp": float(full.max()),
            "effects_le_minus_5pp": int(np.sum(full <= (-5.0 + EPS))),
            "effects_ge_plus_5pp": int(np.sum(full >= (5.0 - EPS))),
            "subject_by_replicate_comparisons": int(full.size),
        })

    # Supplementary Table S4: communication robustness. Communication is
    # regime-invariant at reported precision, so use one regime once per dataset.
    s4: list[dict[str, Any]] = []
    metric_defs = [
        ("C2S traffic", "MB", "p3_client_to_server_mb", "p5_client_to_server_mb", 1.0),
        ("S2C traffic", "MB", "p3_server_to_client_mb", "p5_server_to_client_mb", 1.0),
        ("Total traffic", "MB", "p3_total_mb", "p5_total_mb", 1.0),
        ("Coordinator-rejected uploads", "pp", "p3_rejected_upload_rate", "p5_rejected_upload_rate", 100.0),
        ("Accepted-update staleness", "versions", "p3_accepted_update_staleness", "p5_accepted_update_staleness", 1.0),
        ("Buffered-upload delay", "rounds", "p3_buffered_upload_delay_rounds", "p5_buffered_upload_delay_rounds", 1.0),
        ("Avoided S2C downloads", "pp", "p3_download_avoidance_rate", "p5_download_avoidance_rate", 100.0),
    ]
    for dataset in ("BCICIV-2a", "OpenBMI"):
        rows = groups.get((dataset, "SB-PH"), []) or next((v for (d, _), v in groups.items() if d == dataset), [])
        rows = sorted(rows, key=lambda r: (_int(r.get("model_seed")) or 0, _int(r.get("replicate_id")) or 0))
        for metric, unit, p3col, p5col, scale in metric_defs:
            p3vals = np.asarray([scale * float(_float(r.get(p3col), 0.0)) for r in rows], dtype=float)
            p5vals = np.asarray([scale * float(_float(r.get(p5col), 0.0)) for r in rows], dtype=float)
            diffs = p5vals - p3vals
            p3m, p3sd, *_ = _summary_ci(p3vals.tolist()); p5m, p5sd, *_ = _summary_ci(p5vals.tolist())
            dm, dsd, dlo, dhi, dmn, dmx = _summary_ci(diffs.tolist())
            s4.append({
                "dataset": dataset, "metric": metric, "unit": unit,
                "p3_mean": p3m, "p3_sd": p3sd, "p5_mean": p5m, "p5_sd": p5sd,
                "paired_difference_mean": dm, "paired_difference_sd": dsd,
                "student_t_ci95_low": dlo, "student_t_ci95_high": dhi,
                "observed_range_low": dmn, "observed_range_high": dmx,
                "direction_positive": int(np.sum(diffs > EPS)),
                "direction_negative": int(np.sum(diffs < -EPS)),
                "direction_equal": int(np.sum(np.abs(diffs) <= EPS)),
            })
        reductions = [float(_float(r.get("p5_server_to_client_reduction_percent"), 0.0)) for r in rows]
        dm, dsd, dlo, dhi, dmn, dmx = _summary_ci(reductions)
        s4.append({
            "dataset": dataset, "metric": "S2C traffic reduction", "unit": "%",
            "p3_mean": None, "p3_sd": None, "p5_mean": None, "p5_sd": None,
            "paired_difference_mean": dm, "paired_difference_sd": dsd,
            "student_t_ci95_low": dlo, "student_t_ci95_high": dhi,
            "observed_range_low": dmn, "observed_range_high": dmx,
            "direction_positive": int(sum(v > EPS for v in reductions)),
            "direction_negative": int(sum(v < -EPS for v in reductions)),
            "direction_equal": int(sum(abs(v) <= EPS for v in reductions)),
        })

    # Supplementary Table S5: normalized complete replicate-level results.
    s5: list[dict[str, Any]] = []
    for r in pairs:
        s5.append({
            "dataset": display_dataset(r.get("dataset_name", "")),
            "regime": display_regime(r.get("regime", "")),
            "model_seed": _int(r.get("model_seed")),
            "availability_trace_seed": _int(r.get("trace_seed")),
            "p5_scheduler_tie_break_seed": _int(r.get("scheduler_seed_p5")),
            "p3_mean_subject_accuracy_pct": 100.0 * float(_float(r.get("p3_mean_accuracy"), 0.0)),
            "p5_mean_subject_accuracy_pct": 100.0 * float(_float(r.get("p5_mean_accuracy"), 0.0)),
            "p5_minus_p3_accuracy_difference_pp": float(_float(r.get("p5_minus_p3_accuracy_pp"), 0.0)),
            "s2c_traffic_reduction_pct": float(_float(r.get("p5_server_to_client_reduction_percent"), 0.0)),
        })
    s5.sort(key=lambda r: (dataset_rank.get(r["dataset"], 99), regime_rank.get(r["regime"], 99), r["model_seed"] or 0))
    return {"s2a": s2a, "s2b": s2b, "s3": s3, "s4": s4, "s5": s5}

def analyze_outputs(
    output_root: Path,
    analysis_dir: Path,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 2026,
) -> dict[str, Any]:
    records = discover_runs(output_root)
    analysis_dir = Path(analysis_dir).expanduser().resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Use the latest completed run for a repeated key, but preserve all distinct
    # robustness replicates and sensitivity profiles.
    selected = _latest(
        records,
        lambda r: (r.scenario, r.dataset, r.regime, r.policy, r.severity, r.replicate_id, r.model_seed, r.trace_seed),
    )
    ideal = [r for r in selected if r.scenario == "ideal"]
    primary = [r for r in selected if r.scenario == "primary"]
    component = [r for r in selected if r.scenario == "component"]
    sensitivity = [r for r in selected if r.scenario == "sensitivity"]
    robustness = [r for r in selected if r.scenario == "robustness"]

    all_ops = [operating_point(r) for r in selected if r.scenario in {"ideal", "primary", "component", "sensitivity", "robustness"}]
    _write_csv(analysis_dir / "all_run_operating_points.csv", all_ops)

    publication_cal = []
    for r in ideal + primary:
        publication_cal.extend(calibration_rows(r))
    publication_cal.sort(key=lambda x: (x["dataset"], x["regime"], x["policy"], x["k"]))
    _write_csv(analysis_dir / "figure2_accuracy_by_calibration.csv", publication_cal)

    # Supplementary Table S1 / complete operating points.
    s1 = []
    for r in ideal + primary:
        op = operating_point(r)
        op["link_condition"] = "Ideal link" if r.scenario == "ideal" else "Heterogeneous link"
        s1.append(op)
    policy_rank = {"Ideal": 0, **{f"P{i}": i for i in range(1, 7)}}
    dataset_rank = {"BCICIV-2a": 0, "OpenBMI": 1}
    regime_rank = {"SB-PH": 0, "EIB-PH": 1}
    s1.sort(key=lambda x: (dataset_rank.get(x["dataset"], 99), regime_rank.get(x["regime"], 99), policy_rank.get(x["policy"], 99)))
    _write_csv(analysis_dir / "table_s1_policy_operating_points.csv", s1)

    p35 = _primary_pair_stats(primary)
    _write_csv(analysis_dir / "table_iv_p3_p5_paired_statistics.csv", p35)

    controlled = []
    primary_by = {(r.dataset_display, r.regime_display, r.policy): r for r in primary}
    for dataset in ("BCICIV-2a", "OpenBMI"):
        for regime in ("SB-PH", "EIB-PH"):
            for policy in ("P3", "P5"):
                r = primary_by.get((dataset, regime, policy))
                if r:
                    controlled.append(operating_point(r))
    _write_csv(analysis_dir / "table_iii_p3_p5_operating_points.csv", controlled)

    comp_rows = _component_rows(component)
    _write_csv(analysis_dir / "table_v_component_analysis.csv", comp_rows)

    reliability, reliability_subjects = _reliability_rows(ideal, primary)
    reliability.sort(key=lambda x: (dataset_rank.get(x["dataset"], 99), policy_rank.get(x["policy"], 99)))
    _write_csv(analysis_dir / "table_vi_subject_reliability.csv", reliability)
    _write_csv(analysis_dir / "figure7_subject_reliability.csv", reliability_subjects)

    sens_ops, sens_stats = _sensitivity_rows(sensitivity)
    sens_rank = {"Mild": 0, "Default": 1, "Severe": 2}
    sens_ops.sort(key=lambda x: (dataset_rank.get(x["dataset"], 99), sens_rank.get(x["severity"], 99), policy_rank.get(x["policy"], 99)))
    _write_csv(analysis_dir / "table_s6_sensitivity.csv", sens_ops)
    _write_csv(analysis_dir / "figure6_sensitivity_paired_statistics.csv", sens_stats)

    associations, assoc_warnings = _association_rows(ideal, primary, bootstrap_samples, bootstrap_seed)
    _write_csv(analysis_dir / "table_s7_subject_associations.csv", associations)

    robustness_sources = _robustness_sources(output_root, analysis_dir)
    robustness_tables = _paper_robustness_tables(analysis_dir)
    if robustness_tables:
        _write_csv(analysis_dir / "table_s2a_robustness_cohort_accuracy.csv", robustness_tables["s2a"])
        _write_csv(analysis_dir / "table_s2b_robustness_paired_effects.csv", robustness_tables["s2b"])
        _write_csv(analysis_dir / "table_s3_robustness_subject_heterogeneity.csv", robustness_tables["s3"])
        _write_csv(analysis_dir / "table_s4_robustness_communication.csv", robustness_tables["s4"])
        _write_csv(analysis_dir / "table_s5_robustness_replicates.csv", robustness_tables["s5"])

    summary = {
        "analysis_dir": str(analysis_dir),
        "discovered_completed_runs": len(records),
        "selected_runs": len(selected),
        "scenario_counts": {s: sum(r.scenario == s for r in selected) for s in ("ideal", "primary", "component", "sensitivity", "robustness", "unknown")},
        "association_bootstrap_samples": int(bootstrap_samples),
        "association_bootstrap_seed": int(bootstrap_seed),
        "association_warnings": assoc_warnings,
        "robustness_suite_sources": robustness_sources,
    }

    summary_lines = [
        "# NEXUS-MI generated analysis",
        "",
        f"- Completed run directories discovered: **{len(records)}**",
        f"- Runs selected after de-duplication: **{len(selected)}**",
        f"- Ideal runs: **{len(ideal)}**",
        f"- Primary P1-P6 runs: **{len(primary)}**",
        f"- Component runs: **{len(component)}**",
        f"- Sensitivity runs: **{len(sensitivity)}**",
        f"- Robustness runs: **{len(robustness)}**",
        "",
        "All result tables in this directory were derived directly from experiment outputs.",
    ]
    if assoc_warnings:
        summary_lines += ["", "## Warnings", ""] + [f"- {w}" for w in assoc_warnings]
    (analysis_dir / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return summary
