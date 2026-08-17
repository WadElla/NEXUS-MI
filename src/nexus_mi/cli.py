"""User-facing CLI for NEXUS-MI reproduction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import experiment
from .analysis import analyze_outputs
from .paths import data_root, output_root
from .protocol import paper_protocol, sensitivity_profiles
from .plotting import generate_all
from .preprocessing import inspect_source, prepare_bciciv2a, prepare_openbmi, validate_processed

def _common(dataset: str):
    cfg = paper_protocol()
    d = cfg["dataset_defaults"][dataset]
    fed = cfg["federated_training"]
    pre = cfg["embedding_pretrain"]
    s2 = cfg["session2"]
    pers = cfg["personalization"]
    comm = cfg["communication"]
    probs = comm["availability_probabilities"]
    fracs = comm["availability_group_fractions"]
    calib = ",".join(str(int(v)) for v in pers["calibration_trials_per_class"])
    seed = int(cfg["primary_seed"])
    return [
        str(int(d["dataset_id"])), str(cfg["model"]["name"]), "0", "--all-subjects", "--no-subject-split",
        "--rounds", str(int(fed["rounds"])), "--local-epochs", str(int(fed["local_epochs"])),
        "--lr-local", str(float(fed["learning_rate"])), "--agg-weighting", "uniform",
        "--calib-sizes", calib, "--session2-split-policy", str(s2["split_policy"]),
        "--calib-pool-trials", str(int(d["session2_calibration_pool_trials"])),
        "--fixed-test-trials", str(int(s2["fixed_test_trials_argument"])),
        "--lr-head", str(float(pers["learning_rate"])), "--head-max-epochs", str(int(pers["max_epochs"])),
        "--head-patience", str(int(pers["patience"])), "--batch-size", str(int(fed["batch_size"])),
        "--init-max-epochs", str(int(pre["max_epochs"])), "--init-patience", str(int(pre["patience"])),
        "--lr-init", str(float(pre["learning_rate"])), "--init-best-metric", str(pre["best_metric"]),
        "--session1-val-ratio", str(float(pre["session1_validation_ratio"])),
        "--repeats", str(int(s2["repeats"])),
        "--comm-profile", str(comm["profile"]),
        "--online-prob-good", str(float(probs["high"])),
        "--online-prob-med", str(float(probs["moderate"])),
        "--online-prob-bad", str(float(probs["low"])),
        "--profile-frac-good", str(float(fracs["high"])),
        "--profile-frac-med", str(float(fracs["moderate"])),
        "--max-selected-per-round", str(int(d["max_selected_per_round"])),
        "--buffer-max-size", str(int(comm["buffer_max_size"])),
        "--stale-threshold", str(int(comm["stale_threshold_versions"])),
        "--checkpoint-retention-margin", str(int(comm["checkpoint_retention_margin_versions"])),
        "--download-stale-threshold", str(int(comm["download_threshold_versions"])),
        "--seed", str(seed),
    ]


def run_preset(name: str, dataset: str | None):
    if name == "robustness":
        # The robustness preset fixes the study protocol and executes both datasets.
        robust = paper_protocol()["robustness"]
        argv = _common("bciciv2a") + [
            "--run-suite", "nosplit_multiseed_robustness", "--run-both-datasets",
            "--robustness-replicates", str(int(robust["replicates"])),
            "--robustness-model-seeds", ",".join(str(int(v)) for v in robust["model_seeds"]),
            "--robustness-trace-seeds", ",".join(str(int(v)) for v in robust["availability_trace_seeds"]),
            "--robustness-scheduler-seeds", ",".join(str(int(v)) for v in robust["scheduler_tie_break_seeds"]),
            "--group-seed", str(int(robust["gateway_group_seed"])),
            "--bootstrap-samples", str(int(robust["bootstrap_samples"])),
            "--bootstrap-seed", str(int(robust["bootstrap_seed"])),
        ]
    else:
        if dataset is None:
            raise SystemExit("--dataset is required for this run preset")
        argv = _common(dataset)
        suite = {
            "ideal": "nosplit_main",
            "primary": "nosplit_coordination_ablation",
            "sensitivity": "nosplit_severity_sweep",
            "component": "nosplit_p5_scheduler_ablation",
        }[name]
        if name == "ideal":
            # Match the study's unconstrained-reference runtime metadata as well
            # as its behavior. These values are inert when communication
            # simulation is disabled, but retaining them makes a reproduced
            # run self-describing in the same way as the study runs.
            ideal = paper_protocol()["ideal_link"]
            argv += [
                "--comm-profile", str(ideal["communication_profile"]),
                "--online-prob", str(float(ideal["online_probability"])),
                "--max-selected-per-round", str(int(ideal["max_selected_per_round"])),
                "--buffer-policy", str(ideal["buffer_policy"]),
                "--buffer-max-size", str(int(ideal["buffer_max_size"])),
            ]
        if name in ("primary", "sensitivity", "component"):
            argv += ["--comm-sim"]
        argv += ["--run-suite", suite]
    args = experiment.build_parser().parse_args(argv)
    if name == "sensitivity":
        args.severity_profiles = sensitivity_profiles()
    # Reporting-only provenance. It does not alter training or communication behavior.
    args.publication_suite = name
    experiment.run_suite(args)


def main(argv=None):
    p = argparse.ArgumentParser(prog="nexus-mi", description="NEXUS-MI reproducibility CLI")
    p.add_argument("--data-dir", default=None, help="Root containing bciciv2a/ and openbmi/ (defaults to ./data).")
    p.add_argument("--output-dir", default=None, help="Experiment output root (defaults to ./outputs).")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("inspect-source", help="Check that an extracted official dataset source can be discovered.")
    pi.add_argument("dataset", choices=("bciciv2a", "openbmi"))
    pi.add_argument("--source", required=True)

    pp = sub.add_parser("prepare", help="Preprocess a locally downloaded public dataset.")
    pp.add_argument("dataset", choices=("bciciv2a", "openbmi"))
    pp.add_argument("--source", required=True)

    pv = sub.add_parser("validate-data", help="Validate processed EEG metadata, class counts, file presence, labels, and sampled trial contents.")
    pv.add_argument("dataset", choices=("bciciv2a", "openbmi"))

    pr = sub.add_parser("run", help="Run one NEXUS-MI experiment preset.")
    pr.add_argument("preset", choices=["ideal", "primary", "component", "sensitivity", "robustness"])
    pr.add_argument("--dataset", choices=("bciciv2a", "openbmi"))

    pa = sub.add_parser("analyze", help="Derive publication tables/statistics from generated experiment outputs.")
    pa.add_argument("--input", default=None, help="Experiment output root (defaults to NEXUS_MI_OUTPUT_DIR when set, otherwise ./outputs).")
    pa.add_argument("--out", default=None, help="Analysis directory (defaults to <input>/publication_analysis).")
    pa.add_argument("--bootstrap-samples", type=int, default=10000, help="Subject bootstrap draws for Supplementary S7-style correlations.")
    pa.add_argument("--bootstrap-seed", type=int, default=2026)

    pf = sub.add_parser("figures", help="Regenerate manuscript Figures 2-7 from analyzed experiment outputs.")
    pf.add_argument("--analysis", default=None, help="Analysis directory (defaults to <output>/publication_analysis).")
    pf.add_argument("--out", default=None, help="Figure output directory (defaults to <analysis>/figures).")

    ns = p.parse_args(argv)
    if ns.data_dir:
        os.environ["NEXUS_MI_DATA_DIR"] = str(Path(ns.data_dir).expanduser().resolve())
    if ns.output_dir:
        os.environ["NEXUS_MI_OUTPUT_DIR"] = str(Path(ns.output_dir).expanduser().resolve())

    if ns.command == "inspect-source":
        print(json.dumps(inspect_source(ns.dataset, Path(ns.source)), indent=2))
    elif ns.command == "prepare":
        source = Path(ns.source).expanduser().resolve()
        root = data_root()
        # Fail before processing if the provider files are incomplete/ambiguous.
        print(json.dumps(inspect_source(ns.dataset, source), indent=2))
        prepare_bciciv2a(source, root) if ns.dataset == "bciciv2a" else prepare_openbmi(source, root)
        print(json.dumps(validate_processed(ns.dataset, root), indent=2))
    elif ns.command == "validate-data":
        print(json.dumps(validate_processed(ns.dataset, data_root()), indent=2))
    elif ns.command == "run":
        if ns.preset == "component" and ns.dataset not in (None, "openbmi"):
            raise SystemExit("The paper component analysis is OpenBMI EIB-PH only; use --dataset openbmi.")
        if ns.preset == "robustness" and ns.dataset is not None:
            raise SystemExit("The paper robustness preset always runs both datasets; omit --dataset.")
        run_preset(ns.preset, ns.dataset or ("openbmi" if ns.preset == "component" else None))
    elif ns.command == "analyze":
        inp = Path(ns.input).expanduser().resolve() if ns.input else output_root()
        out = Path(ns.out).expanduser().resolve() if ns.out else inp / "publication_analysis"
        report = analyze_outputs(inp, out, bootstrap_samples=ns.bootstrap_samples, bootstrap_seed=ns.bootstrap_seed)
        print(json.dumps({k: report[k] for k in ("analysis_dir", "discovered_completed_runs", "selected_runs", "scenario_counts")}, indent=2))
    elif ns.command == "figures":
        analysis_dir = Path(ns.analysis).expanduser().resolve() if ns.analysis else output_root() / "publication_analysis"
        out = Path(ns.out).expanduser().resolve() if ns.out else analysis_dir / "figures"
        paths = generate_all(analysis_dir, out)
        print(json.dumps({"figures": [str(p) for p in paths]}, indent=2))


if __name__ == "__main__":
    main()
