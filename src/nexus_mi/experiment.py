#!/usr/bin/env python
# coding: utf-8
"""
NEXUS-MI experiment driver.

This module implements the study-protocol federated personalization
pipeline used for the ideal-link reference, the P1--P6 policy landscape, the
P5 component analysis, the availability-severity study, and the five-realization
P3/P5 robustness evaluation. The implementation preserves the stochastic execution,
communication accounting, Session-1 head retention, Session-2 personalization,
and matched-realization controls required to reproduce the manuscript.

Study-protocol operation
-----------------------------
The ``nexus-mi run`` presets load the study protocol from
the bundled ``paper_protocol.yaml``. The principal controlled policies are:

* P3: all-gateway scheduling, FIFO buffering, stale-drop admission, and
  always-download synchronization.
* P5: communication-aware online-priority scheduling, FIFO buffering,
  stale-drop admission, and version-lag-aware downloading.

The robustness suite uses matched P3/P5 replicate pairs. Within each pair,
policies share the model initialization, availability-group assignment,
subject-by-round availability trace, Session-2 partitions, calibration samples,
and personalization stochastic state. The P5 scheduling tie-break stream is
independently seeded. Across replicates, model and availability-trace seeds vary
while the gateway-group seed remains fixed by the paper protocol.

Stochastic and communication-accounting modes
----------------------------------------------
``legacy_stream`` is the identifier for the process-global stochastic training
stream used by the study; ``per_task_seed`` is an optional diagnostic mode
that reseeds individual training tasks and is not used by publication presets.
Likewise, ``legacy_runtime_metadata`` is the identifier for the raw upload-byte
accounting used by the study. Analysis converts raw bytes to
decimal MB (1 MB = 1,000,000 bytes) and separately reports accepted-update
staleness and buffered-upload delay using the manuscript definitions.

Data portability
----------------
The experiment driver never downloads EEG data implicitly and uses only configured or
repository-relative data paths. Preprocessed EEG is resolved through the
repository data configuration or ``NEXUS_MI_DATA_DIR`` after the user prepares BCICIV-2a or
OpenBMI locally with ``nexus-mi prepare``.
"""

import os
import csv
import json
import copy
import time
import math
import uuid
import random
import argparse
import hashlib
import platform
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

try:
    from . import core as bc
    from .policies import POLICIES
    from .protocol import sensitivity_profiles
except ImportError as exc:
    raise RuntimeError(
        "Could not import NEXUS-MI core helpers."
    ) from exc


REGIMES = [
    "shared_head_only",
    "embedding_shared_head_only",
]
RUN_SUITES = [
    "none",
    "holdout_main",
    "nosplit_main",
    "nosplit_commgrid",
    "nosplit_coordination_ablation",
    "nosplit_severity_sweep",
    "nosplit_multiseed_robustness",
    "nosplit_p5_scheduler_ablation",
    "robustness_preflight",
]

SEVERITY_SWEEP_PROFILES = sensitivity_profiles()

# Run outputs retain A1--A6 internal identifiers and map them explicitly to
# the P1--P6 definitions in nexus_mi.policies.
COORDINATION_ABLATIONS = []
INTERNAL_TO_PAPER_POLICY = {}
for index, (paper_id, policy) in enumerate(POLICIES.items(), start=1):
    internal_id = f"A{index}"
    INTERNAL_TO_PAPER_POLICY[internal_id] = paper_id
    COORDINATION_ABLATIONS.append({
        "id": internal_id,
        "paper_policy_id": paper_id,
        "name": policy.name,
        "label": policy.label,
        "family": policy.family,
        "selection_policy": policy.selection_policy,
        "buffer_policy": policy.buffer_policy,
        "stale_policy": policy.stale_policy,
        "download_policy": policy.download_policy,
    })

SEVERITY_SWEEP_POLICY_IDS = ["A3", "A5"]

def _component_variant(variant_id: str, name: str, label: str, selection: str, download: str, role: str) -> Dict[str, Any]:
    # Component variants hold P3/P5's shared FIFO + stale-drop mechanics fixed
    # and vary only selection/download control, as defined in manuscript Table V.
    p3 = POLICIES["P3"]
    return {
        "id": variant_id,
        "name": name,
        "label": label,
        "family": "p5_component_ablation",
        "selection_policy": selection,
        "buffer_policy": p3.buffer_policy,
        "stale_policy": p3.stale_policy,
        "download_policy": download,
        "component_role": role,
    }


P5_SCHEDULER_ABLATIONS = [
    _component_variant("P3_REF", "p3_fixed_all_fifo_stale_drop", "P3 reference", POLICIES["P3"].selection_policy, POLICIES["P3"].download_policy, "fixed_scheduling_reference"),
    _component_variant("ONLINE_RANDOM", "online_random_fifo_stale_drop", "Online-only random scheduling", "online_random", POLICIES["P3"].download_policy, "online_selection_without_priority"),
    _component_variant("ONLINE_PRIORITY", "online_priority_fifo_stale_drop", "Online-priority scheduling", POLICIES["P5"].selection_policy, POLICIES["P3"].download_policy, "online_selection_with_priority_without_download_control"),
    _component_variant("DOWNLOAD_ONLY", "fixed_all_fifo_stale_drop_stale_download", "Stale-aware download only", POLICIES["P3"].selection_policy, POLICIES["P5"].download_policy, "download_control_without_online_priority_scheduling"),
    _component_variant("P5_FULL", "p5_full_gateway_fifo_stale_drop", "Full P5", POLICIES["P5"].selection_policy, POLICIES["P5"].download_policy, "full_p5_policy"),
]



def ablation_by_id(policy_id: str) -> Dict[str, Any]:
    for ab in COORDINATION_ABLATIONS:
        if ab["id"] == policy_id:
            return dict(ab)
    raise KeyError(f"Unknown ablation id: {policy_id}")


def expected_hetero_online_probability(args) -> float:
    good_frac = float(getattr(args, "profile_frac_good", 0.34))
    med_frac = float(getattr(args, "profile_frac_med", 0.33))
    bad_frac = max(0.0, 1.0 - good_frac - med_frac)
    return (
        good_frac * float(getattr(args, "online_prob_good", 0.95))
        + med_frac * float(getattr(args, "online_prob_med", 0.70))
        + bad_frac * float(getattr(args, "online_prob_bad", 0.40))
    )
DEFAULT_COMMGRID_ONLINE_PROB = 0.60


# =============================================================================
# Basic I/O and JSON helpers
# =============================================================================
def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(obj), f, indent=2)


def save_yaml(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(_json_safe(obj), f, sort_keys=False)


def write_rows_csv(path: str, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def bytes_to_kb(n: float) -> float:
    return float(n) / 1024.0


def bytes_to_mb(n: float) -> float:
    """Convert bytes to decimal megabytes (1 MB = 10^6 bytes)."""
    return float(n) / 1_000_000.0


def bytes_to_mib(n: float) -> float:
    """Convert bytes to binary mebibytes (1 MiB = 2^20 bytes)."""
    return float(n) / (1024.0 * 1024.0)


def timestamp_id() -> str:
    return f'{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}_{uuid.uuid4().hex[:8]}'


# Component-specific RNG offsets preserve the study behavior when the
# corresponding explicit seed is not supplied.  The explicit seed values are
# base seeds; these fixed offsets are then applied to the component RNG.
GROUP_RNG_OFFSET = 13_579
TRACE_RNG_OFFSET = 777
SCHEDULER_RNG_OFFSET = 24_681_357


@dataclass(frozen=True)
class SeedBundle:
    master_seed: int
    model_seed: int
    group_seed: int
    trace_seed: int
    scheduler_seed: int
    split_seed: int
    group_rng_seed: int
    trace_rng_seed: int
    scheduler_rng_base_seed: int
    fallback_compatible: bool


def resolve_seed_bundle(args) -> SeedBundle:
    """Resolve independent RNG bases while retaining the study RNG fallback behavior."""
    master = int(args.seed)
    model = master if getattr(args, "model_seed", None) is None else int(args.model_seed)
    group = master if getattr(args, "group_seed", None) is None else int(args.group_seed)
    trace = master if getattr(args, "trace_seed", None) is None else int(args.trace_seed)
    scheduler = master if getattr(args, "scheduler_seed", None) is None else int(args.scheduler_seed)
    return SeedBundle(
        master_seed=master,
        model_seed=model,
        group_seed=group,
        trace_seed=trace,
        scheduler_seed=scheduler,
        split_seed=int(args.split_seed),
        group_rng_seed=group + GROUP_RNG_OFFSET,
        trace_rng_seed=trace + TRACE_RNG_OFFSET,
        scheduler_rng_base_seed=scheduler + SCHEDULER_RNG_OFFSET,
        fallback_compatible=(
            getattr(args, "model_seed", None) is None
            and getattr(args, "group_seed", None) is None
            and getattr(args, "trace_seed", None) is None
            and getattr(args, "scheduler_seed", None) is None
        ),
    )



def capture_global_rng_state() -> Dict[str, Any]:
    """Capture all global RNG streams that can affect model training."""
    return {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
    }


def restore_global_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG streams after an initialization/pretraining cache hit."""
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)

TRAINING_RNG_MODES = ("legacy_stream", "per_task_seed")


def stable_task_seed(model_seed: int, *parts: Any) -> int:
    """Derive a stable 31-bit seed from the model seed and task identity.

    Python's built-in hash is intentionally not used because it is process
    randomized.  The same subject/round or subject/calibration task therefore
    receives the same stochastic stream in matched P3/P5 runs whenever that
    task occurs under both policies.
    """
    payload = json.dumps(
        [int(model_seed)] + [_json_safe(part) for part in parts],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647


def training_task_seed(args, legacy_seed: int, *task_parts: Any) -> int:
    """Resolve a task seed while preserving the study RNG stream on request."""
    mode = str(getattr(args, "training_rng_mode", "legacy_stream"))
    if mode == "legacy_stream":
        return int(legacy_seed)
    if mode == "per_task_seed":
        return stable_task_seed(resolve_seed_bundle(args).model_seed, *task_parts)
    raise ValueError(f"Unsupported training_rng_mode: {mode}")


def reset_rng_for_training_task(args, seed: int) -> None:
    """Reset global RNGs for dropout/optimizer stochasticity in isolated mode."""
    mode = str(getattr(args, "training_rng_mode", "legacy_stream"))
    if mode == "per_task_seed":
        bc.set_seed(int(seed), deterministic=not bool(getattr(args, "fast_cudnn", False)))
    elif mode != "legacy_stream":
        raise ValueError(f"Unsupported training_rng_mode: {mode}")

def state_sha256(state: Dict[str, torch.Tensor]) -> str:
    """Hash a state dictionary deterministically by key, dtype, shape, and bytes."""
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        value = state[key]
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            h.update(str(tensor.dtype).encode("ascii"))
            h.update(str(tuple(tensor.shape)).encode("ascii"))
            # View as bytes so hashing also works for dtypes that NumPy may not
            # represent directly (for example bfloat16).
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        else:
            h.update(json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def json_sha256(obj: Any) -> str:
    blob = json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_availability_trace(
    subjects: List[str],
    pmap: Dict[str, float],
    group_labels: Dict[str, str],
    n_rounds: int,
    seed_bundle: SeedBundle,
    effective: bool,
) -> Dict[str, Any]:
    """Generate the complete subject-by-round availability realization once."""
    rng = random.Random(int(seed_bundle.trace_rng_seed))
    matrix: List[List[int]] = []
    for _ in range(int(n_rounds)):
        if effective:
            matrix.append([1 if rng.random() < float(pmap[s]) else 0 for s in subjects])
        else:
            matrix.append([1 for _ in subjects])

    core = {
        "schema_version": 1,
        "subject_order": [str(s) for s in subjects],
        "subject_report_order": [bc.subject_to_report_id(s) for s in subjects],
        "rounds": int(n_rounds),
        "effective_communication_simulation": bool(effective),
        "group_seed": int(seed_bundle.group_seed),
        "group_rng_seed": int(seed_bundle.group_rng_seed),
        "trace_seed": int(seed_bundle.trace_seed),
        "trace_rng_seed": int(seed_bundle.trace_rng_seed),
        "group_assignment": {str(s): str(group_labels[s]) for s in subjects},
        "online_probability": {str(s): float(pmap[s]) for s in subjects},
        "availability_matrix": matrix,
    }
    core["group_assignment_hash"] = json_sha256({
        "subject_order": core["subject_order"],
        "group_assignment": core["group_assignment"],
        "online_probability": core["online_probability"],
    })
    core["trace_hash"] = json_sha256({
        "subject_order": core["subject_order"],
        "rounds": core["rounds"],
        "availability_matrix": core["availability_matrix"],
    })
    return core


def load_availability_trace(
    path: str,
    subjects: List[str],
    n_rounds: int,
    expected_pmap: Optional[Dict[str, float]] = None,
    expected_group_labels: Optional[Dict[str, str]] = None,
    expected_seed_bundle: Optional[SeedBundle] = None,
) -> Dict[str, Any]:
    """Load and strictly validate a previously generated availability trace.

    A trace file is not accepted merely because its binary matrix has the right
    shape.  Subject order, group assignment, online probabilities, and both
    stored hashes are checked so a P3/P5 pair cannot silently use the same
    matrix with inconsistent provenance metadata.
    """
    with open(path, "r") as f:
        trace = json.load(f)
    expected_subjects = [str(s) for s in subjects]
    if list(trace.get("subject_order", [])) != expected_subjects:
        raise ValueError(
            "Availability trace subject order does not match the run. "
            f"Expected {expected_subjects}, got {trace.get('subject_order')}."
        )
    if int(trace.get("rounds", -1)) != int(n_rounds):
        raise ValueError(
            f"Availability trace has {trace.get('rounds')} rounds; expected {n_rounds}."
        )

    matrix = trace.get("availability_matrix")
    if not isinstance(matrix, list) or len(matrix) != int(n_rounds):
        raise ValueError("Availability trace matrix is missing or has the wrong number of rounds.")
    clean_matrix: List[List[int]] = []
    for ridx, row in enumerate(matrix, start=1):
        if not isinstance(row, list) or len(row) != len(subjects):
            raise ValueError(f"Invalid availability row length at round {ridx} in {path}.")
        try:
            clean_row = [int(v) for v in row]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-integer availability value at round {ridx} in {path}.") from exc
        if any(v not in (0, 1) for v in clean_row):
            raise ValueError(f"Availability values must be 0/1 at round {ridx} in {path}.")
        clean_matrix.append(clean_row)

    trace_hash = json_sha256({
        "subject_order": expected_subjects,
        "rounds": int(n_rounds),
        "availability_matrix": clean_matrix,
    })
    existing_trace_hash = trace.get("trace_hash")
    if existing_trace_hash is not None and str(existing_trace_hash) != trace_hash:
        raise ValueError(f"Availability trace hash mismatch in {path}.")

    if expected_group_labels is not None:
        expected_groups = {str(s): str(expected_group_labels[s]) for s in subjects}
    else:
        expected_groups = {str(k): str(v) for k, v in (trace.get("group_assignment") or {}).items()}
    if expected_pmap is not None:
        expected_probs = {str(s): float(expected_pmap[s]) for s in subjects}
    else:
        expected_probs = {str(k): float(v) for k, v in (trace.get("online_probability") or {}).items()}
    if set(expected_groups) != set(expected_subjects) or set(expected_probs) != set(expected_subjects):
        raise ValueError(
            "Availability trace must provide a complete group assignment and online probability for every subject."
        )

    loaded_groups = trace.get("group_assignment")
    loaded_probs = trace.get("online_probability")
    if loaded_groups is not None:
        normalized_groups = {str(k): str(v) for k, v in loaded_groups.items()}
        if normalized_groups != expected_groups:
            raise ValueError("Availability trace group assignment does not match the run's group seed/profile.")
    if loaded_probs is not None:
        normalized_probs = {str(k): float(v) for k, v in loaded_probs.items()}
        if normalized_probs != expected_probs:
            raise ValueError("Availability trace online probabilities do not match the run's group seed/profile.")

    group_hash = json_sha256({
        "subject_order": expected_subjects,
        "group_assignment": expected_groups,
        "online_probability": expected_probs,
    })
    existing_group_hash = trace.get("group_assignment_hash")
    if existing_group_hash is not None and str(existing_group_hash) != group_hash:
        raise ValueError(f"Availability group-assignment hash mismatch in {path}.")

    if expected_seed_bundle is not None:
        expected_seed_fields = {
            "group_seed": int(expected_seed_bundle.group_seed),
            "group_rng_seed": int(expected_seed_bundle.group_rng_seed),
            "trace_seed": int(expected_seed_bundle.trace_seed),
            "trace_rng_seed": int(expected_seed_bundle.trace_rng_seed),
        }
        for key, expected_value in expected_seed_fields.items():
            if key in trace and int(trace[key]) != expected_value:
                raise ValueError(
                    f"Availability trace provenance mismatch for {key}: "
                    f"file={trace[key]}, run={expected_value}."
                )
            trace[key] = expected_value

    trace["availability_matrix"] = clean_matrix
    trace["group_assignment"] = expected_groups
    trace["online_probability"] = expected_probs
    trace["trace_hash"] = trace_hash
    trace["group_assignment_hash"] = group_hash
    trace["source_file"] = os.path.abspath(path)
    trace["loaded_from_file"] = True
    return trace


def save_availability_trace_csv(path: str, trace: Dict[str, Any]) -> None:
    subjects = list(trace["subject_order"])
    rows = []
    for round_idx, values in enumerate(trace["availability_matrix"], start=1):
        row = {"round": int(round_idx)}
        row.update({str(subject): int(values[i]) for i, subject in enumerate(subjects)})
        rows.append(row)
    write_rows_csv(path, rows, fieldnames=["round"] + [str(s) for s in subjects])


def make_run_dir(dataset_name: str, exp_name: str, network_name: str, regime: str, protocol: str, comm_label: str) -> str:
    from .paths import output_root
    out_base = os.path.join(str(output_root()), dataset_name, exp_name, regime, protocol, comm_label)
    os.makedirs(out_base, exist_ok=True)
    run_dir = os.path.join(out_base, f"{network_name}_{timestamp_id()}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir



def load_existing_preprocessed_dataset(dataset_name: str) -> "bc.eegDataset":
    """Load locally prepared EEG data from the configured data root.

    Training never downloads or preprocesses EEG implicitly. The processed
    ``rawPython`` folder and its ``dataLabels.csv`` must already exist.
    """
    mode_in_folder = "rawPython"
    from .paths import dataset_dir
    in_data_path = os.path.join(str(dataset_dir(dataset_name)), mode_in_folder)
    in_label_path = os.path.join(in_data_path, "dataLabels.csv")

    if not os.path.isdir(in_data_path):
        raise FileNotFoundError(
            "Processed EEG data folder was not found. Prepare the dataset first with "
            f"`nexus-mi prepare`. Expected folder: {in_data_path}"
        )
    if not os.path.isfile(in_label_path):
        raise FileNotFoundError(
            "Processed EEG label file was not found. Prepare the dataset first with "
            f"`nexus-mi prepare`. Expected file: {in_label_path}"
        )

    print("Data loading in progress", flush=True)
    data = bc.eegDataset(
        dataPath=in_data_path,
        dataLabelsPath=in_label_path,
        preloadData=False,
        transform=None,
    )
    print("Data loading finished", flush=True)
    return data


# =============================================================================
# State / parameter helpers
# =============================================================================
def tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def state_bytes(state: Dict[str, torch.Tensor]) -> int:
    return int(sum(tensor_bytes(v) for v in state.values() if torch.is_tensor(v)))


def metadata_bytes(metadata: Optional[dict]) -> int:
    if not metadata:
        return 0
    blob = json.dumps(_json_safe(metadata), sort_keys=True, separators=(",", ":"))
    return int(len(blob.encode("utf-8")))

COMMUNICATION_BYTE_ACCOUNTING_MODES = ("legacy_runtime_metadata", "protocol_metadata")


def upload_metadata_for_byte_accounting(metadata: Dict[str, Any], args) -> Dict[str, Any]:
    """Return the metadata fields counted as transmitted upload bytes.

    ``legacy_runtime_metadata`` uses the study behavior, which counts a wall-clock
    timestamp whose serialized length can vary across runs. ``protocol_metadata`` is
    an optional diagnostic mode that counts only fields required by the communication
    protocol. Study-protocol presets use ``legacy_runtime_metadata`` for raw-byte
    accounting.
    """
    mode = str(getattr(args, "communication_byte_accounting", "legacy_runtime_metadata"))
    if mode == "legacy_runtime_metadata":
        return dict(metadata)
    if mode == "protocol_metadata":
        required = ("subject_id", "trained_from_version", "produced_round", "perf_score")
        return {key: metadata[key] for key in required if key in metadata}
    raise ValueError(f"Unsupported communication_byte_accounting mode: {mode}")


def clone_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k, v in state.items()}


def backbone_only(full_state: Dict[str, torch.Tensor], backbone_keys: List[str]) -> Dict[str, torch.Tensor]:
    return {k: full_state[k].detach().cpu().clone() for k in backbone_keys if k in full_state}


def head_only(full_state: Dict[str, torch.Tensor], backbone_keys: List[str]) -> Dict[str, torch.Tensor]:
    """Return only the local classifier/head parameters.

    ``backbone_keys`` excludes ``lastLayer.*`` classifier parameters. Keeping
    this helper generic makes the
    personalization initialization robust if a network stores additional
    non-backbone head buffers.
    """
    backbone_key_set = set(backbone_keys)
    return {k: v.detach().cpu().clone() for k, v in full_state.items() if k not in backbone_key_set}


def merge_backbone_with_init_head(backbone_state: Dict[str, torch.Tensor], init_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return bc.merge_backbone_with_local_head(backbone_state, init_state)


def make_personalization_reference_state(
    init_state: Dict[str, torch.Tensor],
    subject_head_state: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Build the full state that supplies the head for Session-2 personalization.

    The final shared/collaborative backbone is merged later in
    evaluate_personalization().  This reference state supplies the non-backbone
    parameters.  When a Session-1 trained subject head is available, it replaces
    the original initialization so personalization starts from Session-1 learning
    rather than from the random/global initial head.
    """
    ref = clone_state(init_state)
    if subject_head_state is not None:
        for k, v in subject_head_state.items():
            ref[k] = v.detach().cpu().clone()
    return ref


def delta_from_backbones(new_backbone: Dict[str, torch.Tensor], base_backbone: Dict[str, torch.Tensor], backbone_keys: List[str]) -> Dict[str, torch.Tensor]:
    return {k: new_backbone[k].detach().cpu() - base_backbone[k].detach().cpu() for k in backbone_keys}


def apply_delta(base_backbone: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor], backbone_keys: List[str]) -> Dict[str, torch.Tensor]:
    return {k: base_backbone[k].detach().cpu() + delta[k].detach().cpu() for k in backbone_keys}


def aggregate_backbones(
    candidate_backbones: Dict[str, Dict[str, torch.Tensor]],
    weights: Dict[str, float],
    backbone_keys: List[str],
) -> Dict[str, torch.Tensor]:
    if not candidate_backbones:
        raise RuntimeError("No candidate backbones supplied to aggregation.")
    subjects = list(candidate_backbones.keys())
    w_sum = float(sum(float(weights.get(s, 0.0)) for s in subjects))
    if w_sum <= 1e-12:
        weights = {s: 1.0 / len(subjects) for s in subjects}
    else:
        weights = {s: float(weights.get(s, 0.0)) / w_sum for s in subjects}

    out = {}
    for k in backbone_keys:
        tensors = [candidate_backbones[s][k].detach().cpu() for s in subjects]
        if tensors[0].dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
            stacked = torch.stack([t.to(torch.int64) for t in tensors], dim=0)
            out[k] = stacked.max(dim=0).values.to(tensors[0].dtype)
            continue
        acc = torch.zeros_like(tensors[0], dtype=torch.float32)
        for s, t in zip(subjects, tensors):
            acc += t.float() * float(weights[s])
        out[k] = acc.to(tensors[0].dtype)
    return out


def compute_weights(
    subjects: List[str],
    weighting: str,
    sample_counts: Dict[str, int],
    perf_scores: Optional[Dict[str, float]],
    perf_metric: str,
    perf_method: str,
    perf_alpha: float,
    perf_eps: float,
    perf_with_samples: bool,
    stale_factors: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if not subjects:
        return {}
    if weighting == "uniform":
        raw = {s: 1.0 for s in subjects}
    elif weighting == "samples":
        raw = {s: float(sample_counts.get(s, 1)) for s in subjects}
    elif weighting == "performance":
        if perf_scores is None:
            raise RuntimeError("performance weighting requested but perf_scores is missing")
        if perf_metric == "val_acc":
            goodness = {s: float(perf_scores.get(s, 0.0)) for s in subjects}
        elif perf_metric == "val_loss":
            goodness = {s: -float(perf_scores.get(s, 0.0)) for s in subjects}
        else:
            raise ValueError("perf_metric must be val_acc or val_loss")
        if perf_method == "softmax":
            mx = max(goodness.values())
            raw = {s: math.exp(float(perf_alpha) * (goodness[s] - mx)) for s in subjects}
        elif perf_method == "inv" and perf_metric == "val_loss":
            raw = {s: 1.0 / (float(perf_scores.get(s, 0.0)) + float(perf_eps)) for s in subjects}
        elif perf_method in ("linear", "inv"):
            raw = {s: max(float(perf_eps), float(perf_scores.get(s, 0.0))) for s in subjects}
        else:
            raise ValueError("Unsupported performance weighting configuration")
        if perf_with_samples:
            raw = {s: raw[s] * float(sample_counts.get(s, 1)) for s in subjects}
    else:
        raise ValueError("agg-weighting must be uniform, samples, or performance")

    if stale_factors:
        raw = {s: raw[s] * float(stale_factors.get(s, 1.0)) for s in subjects}
    total = float(sum(max(0.0, v) for v in raw.values()))
    if total <= 1e-12:
        return {s: 1.0 / len(subjects) for s in subjects}
    return {s: max(0.0, float(raw[s])) / total for s in subjects}


# =============================================================================
# Dataset/session helpers
# =============================================================================
def dataset_name_from_id(dataset_id: int) -> str:
    value = int(dataset_id)
    if value == 0:
        return "bciciv2a"
    if value == 1:
        return "openbmi"
    raise ValueError(f"Unsupported dataset id {dataset_id!r}; expected 0 (BCICIV-2a) or 1 (OpenBMI).")


def parse_int_csv(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_subject_csv(s: Optional[str], all_subjects: List[str]) -> List[str]:
    if not s:
        return []
    out = []
    for token in str(s).split(","):
        token = token.strip()
        if token:
            out.append(bc.resolve_subject_arg(token, all_subjects))
    seen, dedup = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


def all_subjects_from_labels(labels: List[List[str]]) -> List[str]:
    subs = sorted(set(row[3] for row in labels), key=lambda s: int("".join(ch for ch in str(s) if ch.isdigit()) or 0))
    return subs


def subject_session_datasets(full_ds: bc.eegDataset, subject: str, preload: bool) -> Optional[Dict[str, bc.eegDataset]]:
    idx_s1, idx_s2 = bc.split_session_indices(full_ds.labels, subject)
    if len(idx_s1) == 0 or len(idx_s2) == 0:
        return None
    s1 = copy.deepcopy(full_ds)
    s1.createPartialDataset(idx_s1, loadNonLoadedData=bool(preload))
    s2 = copy.deepcopy(full_ds)
    s2.createPartialDataset(idx_s2, loadNonLoadedData=bool(preload))
    return {"s1": s1, "s2": s2}


def build_subject_data(full_ds: bc.eegDataset, subjects: List[str], preload: bool) -> Dict[str, Dict[str, bc.eegDataset]]:
    out = {}
    for s in subjects:
        item = subject_session_datasets(full_ds, s, preload=preload)
        if item is not None:
            out[s] = item
    return out


def make_partial(ds: bc.eegDataset, indices: List[int], preload: bool) -> bc.eegDataset:
    part = copy.deepcopy(ds)
    part.createPartialDataset(list(indices), loadNonLoadedData=bool(preload))
    return part


def combine_datasets(datasets: List[bc.eegDataset], preload: bool) -> bc.eegDataset:
    if not datasets:
        raise RuntimeError("No datasets to combine")
    pooled = copy.deepcopy(datasets[0])
    for ds in datasets[1:]:
        pooled.combineDataset(ds, loadNonLoadedData=bool(preload))
    return pooled


def split_session1_train_val(s1_ds: bc.eegDataset, val_ratio: float, seed: int, preload: bool) -> Tuple[bc.eegDataset, bc.eegDataset]:
    by_cls = bc.indices_by_class(s1_ds.labels)
    rng = random.Random(int(seed))
    tr_idx, va_idx = [], []
    for _, idxs in by_cls.items():
        idxs2 = list(idxs)
        rng.shuffle(idxs2)
        n_val = int(round(len(idxs2) * float(val_ratio)))
        n_val = max(1, min(len(idxs2) - 1, n_val)) if len(idxs2) > 1 else 0
        va_idx.extend(idxs2[:n_val])
        tr_idx.extend(idxs2[n_val:])
    return make_partial(s1_ds, tr_idx, preload), make_partial(s1_ds, va_idx, preload)


def split_session2_fixed_calib_test(
    s2_ds: bc.eegDataset,
    fixed_test_trials: int,
    calib_pool_trials: int,
    preload: bool,
) -> Tuple[bc.eegDataset, bc.eegDataset, Dict[str, Any]]:
    """Chronological Session-2 split for fair calibration-budget evaluation.

    Clean rule used by this script:
      - calibration pool is always explicitly defined by ``calib_pool_trials``;
      - calibration pool = first ``calib_pool_trials`` Session-2 trials;
      - if ``fixed_test_trials > 0``, test set = last ``fixed_test_trials`` trials;
      - if ``fixed_test_trials == 0``, test set = all remaining trials after the calibration pool;
      - if both a front calibration block and a back fixed-test block are requested,
        they must not overlap. Middle trials, if any, are unused.

    Examples:
      --calib-pool-trials 50 --fixed-test-trials 0
          first 50 trials for calibration pool, all remaining trials for test.
      --calib-pool-trials 100 --fixed-test-trials 100
          first 100 trials for calibration pool, last 100 trials for test, middle unused.

    The fixed test set is then reused unchanged for every calibration size k.
    """
    n = int(len(s2_ds.labels))
    if n < 2:
        raise RuntimeError("Session-2 must contain at least two trials to split calibration/test.")

    calib_n = int(calib_pool_trials)
    test_n_arg = int(fixed_test_trials)

    if calib_n <= 0:
        raise ValueError(
            "--calib-pool-trials must be > 0. The calibration pool is intentionally "
            "explicit: first N Session-2 trials are used for calibration."
        )
    if test_n_arg < 0:
        raise ValueError("--fixed-test-trials must be >= 0. Use 0 to test on all remaining trials after calibration.")
    if calib_n >= n:
        raise ValueError(
            f"Invalid Session-2 split: calib_pool_trials={calib_n} but subject has only {n} Session-2 trials. "
            "Calibration must leave at least one trial for testing."
        )

    calib_idx = list(range(0, calib_n))

    if test_n_arg > 0:
        test_n = int(test_n_arg)
        if test_n <= 0:
            raise ValueError("--fixed-test-trials must be positive when explicitly set.")
        if test_n >= n:
            raise ValueError(
                f"Invalid Session-2 split: fixed_test_trials={test_n} but subject has only {n} Session-2 trials."
            )
        if calib_n + test_n > n:
            raise ValueError(
                f"Invalid Session-2 split: first {calib_n} calibration trials and last {test_n} test trials overlap "
                f"because subject has only {n} Session-2 trials. Reduce --calib-pool-trials or --fixed-test-trials."
            )
        test_start = n - test_n
        test_idx = list(range(test_start, n))
        unused_idx = list(range(calib_n, test_start))
        split_policy = "chronological_first_calib_pool_last_fixed_test_middle_unused"
        test_mode = "fixed_last_n_trials"
    else:
        test_idx = list(range(calib_n, n))
        unused_idx = []
        split_policy = "chronological_first_calib_pool_rest_test"
        test_mode = "all_remaining_after_calibration_pool"

    if len(calib_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(
            f"Invalid Session-2 split: n={n}, calib_pool_trials={calib_pool_trials}, "
            f"fixed_test_trials={fixed_test_trials}, calib={len(calib_idx)}, test={len(test_idx)}"
        )

    calib_pool = make_partial(s2_ds, calib_idx, preload=preload)
    test_ds = make_partial(s2_ds, test_idx, preload=preload)
    info = {
        "session2_total_trials": int(n),
        "requested_calib_pool_trials": int(calib_pool_trials),
        "requested_fixed_test_trials": int(fixed_test_trials),
        "calibration_pool_trials": int(len(calib_idx)),
        "fixed_test_trials": int(len(test_idx)),
        "unused_middle_trials": int(len(unused_idx)),
        "calibration_pool_start_index": int(calib_idx[0]),
        "calibration_pool_end_index": int(calib_idx[-1]),
        "test_start_index": int(test_idx[0]),
        "test_end_index": int(test_idx[-1]),
        "calibration_pool_index_hash": json_sha256([int(i) for i in calib_idx]),
        "test_index_hash": json_sha256([int(i) for i in test_idx]),
        "unused_index_hash": json_sha256([int(i) for i in unused_idx]),
        "split_policy": split_policy,
        "test_mode": test_mode,
        "fixed_test_is_same_for_all_k": True,
    }
    return calib_pool, test_ds, info

def first_k_per_class_indices(labels_rows: List[List[str]], k: int) -> List[int]:
    by_cls = bc.indices_by_class(labels_rows)
    selected = []
    for cls in sorted(by_cls.keys()):
        idxs = by_cls[cls]
        if len(idxs) < int(k):
            return []
        selected.extend(idxs[: int(k)])
    return sorted(selected)


def make_chrono_calib_train_val(calib_pool: bc.eegDataset, k_per_class: int, preload: bool) -> Tuple[Optional[bc.eegDataset], Optional[bc.eegDataset], Dict[str, Any]]:
    """Select first k/class chronologically from the fixed calibration pool.

    The selected k/class is the total calibration budget. If k>=2, the last
    selected trial per class becomes validation for early stopping; the rest are
    training. If k==1, validation is disabled.
    """
    selected = first_k_per_class_indices(calib_pool.labels, int(k_per_class))
    if not selected:
        return None, None, {"ok": False, "reason": "insufficient_calibration_pool_per_class", "k_per_class": int(k_per_class)}
    selected_set = set(selected)
    by_cls = bc.indices_by_class(calib_pool.labels)
    train_idx, val_idx = [], []
    for cls in sorted(by_cls.keys()):
        cls_sel = [i for i in by_cls[cls] if i in selected_set][: int(k_per_class)]
        if int(k_per_class) >= 2:
            train_idx.extend(cls_sel[:-1])
            val_idx.extend(cls_sel[-1:])
        else:
            train_idx.extend(cls_sel)
    tr = make_partial(calib_pool, sorted(train_idx), preload=preload)
    va = make_partial(calib_pool, sorted(val_idx), preload=preload) if val_idx else None
    return tr, va, {
        "ok": True,
        "k_per_class": int(k_per_class),
        "selected_total": int(len(selected)),
        "train_total": int(len(train_idx)),
        "val_total": int(len(val_idx)),
        "selected_index_hash_within_calibration_pool": json_sha256([int(i) for i in sorted(selected)]),
        "train_index_hash_within_calibration_pool": json_sha256([int(i) for i in sorted(train_idx)]),
        "val_index_hash_within_calibration_pool": json_sha256([int(i) for i in sorted(val_idx)]),
        "selection_policy": "first_k_per_class_chronological_from_calibration_pool",
        "internal_val_policy": "last_selected_trial_per_class" if val_idx else "none",
    }


def make_random_per_class_calib_train_val_test(
    s2_ds: bc.eegDataset,
    k_per_class: int,
    repeat: int,
    seed: int,
    preload: bool,
) -> Tuple[Optional[bc.eegDataset], Optional[bc.eegDataset], Optional[bc.eegDataset], Dict[str, Any]]:
    """Apply the optional random per-class Session-2 evaluation split.

    For each subject, calibration size k, and repeat, all Session-2 trials are
    grouped by class. Each class list is shuffled with a deterministic RNG; the
    first k trials from each class form the calibration set and the remaining
    trials form the test set. The calibration set is then split into train/val
    using the same one-validation-trial-per-class rule as the diagnostic
    calibration split.

    This function intentionally does not alter data reading or preprocessing. It
    only creates partial datasets from an already constructed Session-2 dataset.
    """
    k = int(k_per_class)
    rep = int(repeat)
    by_cls = bc.indices_by_class(s2_ds.labels)
    class_counts = {str(cls): int(len(idxs)) for cls, idxs in sorted(by_cls.items())}
    if any(len(idxs) < k for idxs in by_cls.values()):
        return None, None, None, {
            "ok": False,
            "reason": "insufficient_session2_trials_per_class",
            "k_per_class": int(k),
            "repeat": int(rep),
            "class_counts": class_counts,
            "selection_policy": "random_k_per_class_from_full_session2_remaining_test",
        }

    split_seed = int(seed) + 17 * int(k) + int(rep)
    rng = random.Random(split_seed)
    calib_idx, test_idx = [], []
    per_class_selected = {}
    per_class_test = {}
    for cls in sorted(by_cls.keys()):
        idxs2 = list(by_cls[cls])
        rng.shuffle(idxs2)
        csel = idxs2[:k]
        ctest = idxs2[k:]
        calib_idx.extend(csel)
        test_idx.extend(ctest)
        per_class_selected[str(cls)] = int(len(csel))
        per_class_test[str(cls)] = int(len(ctest))

    if not calib_idx or not test_idx:
        return None, None, None, {
            "ok": False,
            "reason": "empty_calibration_or_test_after_random_split",
            "k_per_class": int(k),
            "repeat": int(rep),
            "class_counts": class_counts,
            "selected_total": int(len(calib_idx)),
            "test_total": int(len(test_idx)),
            "selection_policy": "random_k_per_class_from_full_session2_remaining_test",
        }

    calib_ds = make_partial(s2_ds, calib_idx, preload=preload)
    test_ds = make_partial(s2_ds, test_idx, preload=preload)
    calib_train, calib_val = bc.make_calib_train_val(calib_ds, k_per_class=k, seed=int(seed) + int(rep) + 99)

    info = {
        "ok": True,
        "session2_split_policy": "random_per_class",
        "split_policy": "random_k_per_class_from_full_session2_remaining_test",
        "selection_policy": "random_k_per_class_from_full_session2_remaining_test",
        "test_policy": "remaining_session2_trials_after_random_k_per_class_calibration",
        "repeat": int(rep),
        "repeat_seed": int(split_seed),
        "internal_val_seed": int(seed) + int(rep) + 99,
        "k_per_class": int(k),
        "session2_total_trials": int(len(s2_ds.labels)),
        "selected_total": int(len(calib_idx)),
        "train_total": int(len(calib_train.labels)) if calib_train is not None else int(len(calib_idx)),
        "val_total": int(len(calib_val.labels)) if calib_val is not None else 0,
        "test_total": int(len(test_idx)),
        "calibration_index_hash": json_sha256([int(i) for i in sorted(calib_idx)]),
        "test_index_hash": json_sha256([int(i) for i in sorted(test_idx)]),
        "calibration_trials_per_class": per_class_selected,
        "test_trials_per_class": per_class_test,
        "class_counts": class_counts,
        "fixed_test_is_same_for_all_k": False,
        "fixed_test_is_same_for_all_repeats": False,
        "calib_pool_trials_used": False,
        "fixed_test_trials_arg_used": False,
        "internal_val_policy": "random_one_validation_trial_per_class" if int(k) >= 2 else "none",
    }
    return calib_train, calib_val, test_ds, info


# =============================================================================
# Training / evaluation helpers
# =============================================================================
def train_local_with_val_earlystop(
    model: nn.Module,
    train_ds: bc.eegDataset,
    val_ds: Optional[bc.eegDataset],
    device: torch.device,
    expects_bands: bool,
    batch_size: int,
    lr: float,
    max_epochs: int,
    patience: int,
    seed: int,
    n_class: int,
    num_workers: int,
    pin_memory: bool,
    train_head: bool,
    best_metric: str,
) -> Dict[str, Any]:
    model = model.to(device)
    if not train_head and hasattr(model, "lastLayer"):
        for p in model.lastLayer.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters in train_local_with_val_earlystop")
    opt = optim.Adam(params, lr=float(lr))
    crit = nn.NLLLoss(reduction="sum")
    tr_loader = bc._make_loader(train_ds, batch_size=int(batch_size), shuffle=True, seed=int(seed), num_workers=int(num_workers), pin_memory=bool(pin_memory))
    va_loader = None
    if val_ds is not None and len(val_ds) > 0:
        va_loader = bc._make_loader(val_ds, batch_size=int(batch_size), shuffle=False, seed=int(seed) + 1, num_workers=int(num_workers), pin_memory=bool(pin_memory))

    maximize = str(best_metric) == "val_acc"
    best_score = -float("inf") if maximize else float("inf")
    best_epoch = 0
    best_state = clone_state(model.state_dict())
    no_improve = 0
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def run(loader, train: bool) -> Dict[str, float]:
        model.train() if train else model.eval()
        total_loss, total_correct, total_n = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for d in loader:
                x = bc.prepare_batch_x(d["data"], expects_bands).to(device, non_blocking=True)
                y = d["label"].long().to(device, non_blocking=True)
                out = model(x)
                loss = crit(out, y)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                pred = out.argmax(dim=1)
                total_loss += float(loss.item())
                total_correct += int((pred == y).sum().item())
                total_n += int(y.size(0))
        return {"loss": total_loss / max(1, total_n), "acc": total_correct / max(1, total_n)}

    for ep in range(1, int(max_epochs) + 1):
        tr = run(tr_loader, train=True)
        hist["train_loss"].append(float(tr["loss"]))
        hist["train_acc"].append(float(tr["acc"]))
        if va_loader is not None:
            va = run(va_loader, train=False)
            hist["val_loss"].append(float(va["loss"]))
            hist["val_acc"].append(float(va["acc"]))
            score = float(va["acc"]) if maximize else float(va["loss"])
        else:
            va = None
            hist["val_loss"].append(None)
            hist["val_acc"].append(None)
            score = float(tr["acc"]) if maximize else float(tr["loss"])
        improved = (score > best_score + 1e-8) if maximize else (score < best_score - 1e-8)
        if improved:
            best_score = score
            best_epoch = ep
            best_state = clone_state(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if va_loader is not None and no_improve >= int(patience):
                break
    model.load_state_dict(best_state, strict=False)
    return {
        "best_state": clone_state(model.state_dict()),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "history": hist,
        "train_best": bc.eval_model(model, train_ds, device, expects_bands, int(batch_size), int(n_class), seed=int(seed) + 7, num_workers=int(num_workers), pin_memory=bool(pin_memory)),
        "val_best": None if val_ds is None else bc.eval_model(model, val_ds, device, expects_bands, int(batch_size), int(n_class), seed=int(seed) + 8, num_workers=int(num_workers), pin_memory=bool(pin_memory)),
    }


def macro_f1_from_cm(cm: Any) -> float:
    arr = np.array(cm, dtype=np.float64)
    f1s = []
    for c in range(arr.shape[0]):
        tp = arr[c, c]
        fp = arr[:, c].sum() - tp
        fn = arr[c, :].sum() - tp
        precision = tp / max(1e-12, tp + fp)
        recall = tp / max(1e-12, tp + fn)
        if precision + recall <= 1e-12:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def enrich_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(metrics)
    if "cm" in out:
        out["macro_f1"] = macro_f1_from_cm(out["cm"])
    return out


# =============================================================================
# Session-2 head-only personalization
# =============================================================================
def evaluate_personalization(
    personalization_mode: str,
    backbone_state: Dict[str, torch.Tensor],
    init_state: Dict[str, torch.Tensor],
    net_ctor,
    model_args: dict,
    calib_train: bc.eegDataset,
    calib_val: Optional[bc.eegDataset],
    test_ds: bc.eegDataset,
    args,
    device: torch.device,
    expects_bands: bool,
    seed: int,
) -> Dict[str, Any]:
    base_state = merge_backbone_with_init_head(backbone_state, init_state)
    model = net_ctor(**model_args)
    model.load_state_dict(base_state, strict=False)
    # Head-only personalization follows the study implementation. Publication
    # presets preserve the study process-global RNG stream; the optional per-task
    # mode reseeds this step explicitly.
    reset_rng_for_training_task(args, int(seed))
    if personalization_mode == "head_only":
        out = bc.head_finetune_with_early_stop(
            model=model,
            calib_train=calib_train,
            calib_val=calib_val,
            test_ds=test_ds,
            device=device,
            expects_bands=expects_bands,
            batch_size=int(args.batch_size),
            lr=float(args.lr_head),
            max_epochs=int(args.head_max_epochs),
            patience=int(args.head_patience),
            seed=int(seed),
            n_class=int(model_args["nClass"]),
            num_workers=int(args.num_workers),
            pin_memory=bool(args.pin_memory),
            stage2=bool(args.head_stage2),
        )
        out["train"] = enrich_metrics(out["train"])
        out["val"] = None if out.get("val") is None else enrich_metrics(out["val"])
        out["test"] = enrich_metrics(out["test"])
        out["personalization_mode"] = "head_only"
        return out
    raise ValueError(f"Unsupported personalization mode: {personalization_mode}")


# =============================================================================
# Regime setup
# =============================================================================
def regime_to_config(regime: str) -> Dict[str, str]:
    """Map the paper regimes to initialization and personalization behavior."""
    if regime == "shared_head_only":
        return {"init_source": "random_global_init", "personalization": "head_only", "embedding_pretrain": "disabled"}
    if regime == "embedding_shared_head_only":
        return {"init_source": "pooled_session1_pretrained_backbone", "personalization": "head_only", "embedding_pretrain": "enabled"}
    raise ValueError(
        f"Unsupported NEXUS-MI regime: {regime}. "
        f"Supported regimes are: {', '.join(REGIMES)}"
    )


def pretrain_pooled_session1_backbone(
    subjects: List[str],
    subj_data: Dict[str, Dict[str, bc.eegDataset]],
    init_state: Dict[str, torch.Tensor],
    backbone_keys: List[str],
    net_ctor,
    model_args: dict,
    args,
    device: torch.device,
    expects_bands: bool,
    preload: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    pooled = combine_datasets([subj_data[s]["s1"] for s in subjects], preload=preload)
    tr, va = split_session1_train_val(pooled, float(args.session1_val_ratio), int(resolve_seed_bundle(args).model_seed) + 901, preload)
    model = net_ctor(**model_args)
    model.load_state_dict(clone_state(init_state), strict=False)
    pretrain_seed = int(resolve_seed_bundle(args).model_seed) + 902
    reset_rng_for_training_task(args, pretrain_seed)
    out = train_local_with_val_earlystop(
        model=model,
        train_ds=tr,
        val_ds=va,
        device=device,
        expects_bands=expects_bands,
        batch_size=int(args.batch_size),
        lr=float(args.lr_init),
        max_epochs=int(args.init_max_epochs),
        patience=int(args.init_patience),
        seed=pretrain_seed,
        n_class=int(model_args["nClass"]),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        train_head=not bool(args.no_local_train_head),
        best_metric=str(args.init_best_metric),
    )
    return backbone_only(out["best_state"], backbone_keys), {
        "enabled": True,
        "source": "pooled_session1_training_subjects",
        "subjects": [bc.subject_to_report_id(s) for s in subjects],
        "pooled_trials": int(len(pooled)),
        "train_trials": int(len(tr)),
        "val_trials": int(len(va)),
        "best_epoch": int(out["best_epoch"]),
        "train_best": enrich_metrics(out["train_best"]),
        "val_best": None if out["val_best"] is None else enrich_metrics(out["val_best"]),
    }


# =============================================================================
# Communication and synchronization
# =============================================================================
@dataclass
class CommunicationConfig:
    requested: bool
    effective: bool
    label: str
    profile: str
    online_prob: float
    online_prob_good: float
    online_prob_med: float
    online_prob_bad: float
    buffering_enabled: bool
    buffer_policy: str
    buffer_max_size: int
    stale_threshold: int
    checkpoint_retention_margin: int
    stale_policy: str
    stale_gamma: float
    download_policy: str
    download_stale_threshold: int


def safe_div(num: float, den: float) -> float:
    den = float(den)
    return float(num) / den if den > 0.0 else 0.0


def build_comm_config(args, force_label: Optional[str] = None) -> CommunicationConfig:
    label = force_label or ("comm_on" if bool(args.comm_sim) else "comm_off")
    requested = bool(args.comm_sim)

    # In this project, comm_on must mean a non-trivial communication-aware run.
    # If the user enables --comm-sim with the default uniform all-online setting,
    # make the link model non-trivial by default. Use comm_off for the all-online
    # reference where bytes are still counted.
    if requested and str(args.comm_profile) == "uniform" and float(args.online_prob) >= 1.0:
        args.online_prob = DEFAULT_COMMGRID_ONLINE_PROB

    # --no-buffering is an alias for --buffer-policy none.
    if bool(getattr(args, "no_buffering", False)):
        args.buffer_policy = "none"

    effective = requested
    buffer_policy = str(getattr(args, "buffer_policy", "fifo"))
    if not effective:
        buffer_policy = "none"
    buffering_enabled = bool(effective and buffer_policy != "none")
    buffer_max_size = max(1, int(getattr(args, "buffer_max_size", 1)))

    return CommunicationConfig(
        requested=requested,
        effective=effective,
        label=label,
        profile=str(args.comm_profile),
        online_prob=float(args.online_prob),
        online_prob_good=float(args.online_prob_good),
        online_prob_med=float(args.online_prob_med),
        online_prob_bad=float(args.online_prob_bad),
        buffering_enabled=buffering_enabled,
        buffer_policy=buffer_policy,
        buffer_max_size=int(buffer_max_size),
        stale_threshold=int(args.stale_threshold),
        checkpoint_retention_margin=int(args.checkpoint_retention_margin),
        stale_policy=str(args.stale_policy),
        stale_gamma=float(args.stale_gamma),
        download_policy=str(getattr(args, "download_policy", "always")),
        download_stale_threshold=int(getattr(args, "download_stale_threshold", args.stale_threshold)),
    )


def online_probability_map(subjects: List[str], args, seed: int) -> Tuple[Dict[str, float], Dict[str, str]]:
    if str(args.comm_profile) == "uniform":
        p = max(0.0, min(1.0, float(args.online_prob)))
        return {s: p for s in subjects}, {s: "uniform" for s in subjects}
    rng = random.Random(int(seed) + 13579)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_good = int(round(n * float(args.profile_frac_good)))
    n_med = int(round(n * float(args.profile_frac_med)))
    n_good = max(0, min(n, n_good))
    n_med = max(0, min(n - n_good, n_med))
    good, med = set(shuffled[:n_good]), set(shuffled[n_good:n_good + n_med])
    pmap, labels = {}, {}
    for s in subjects:
        if s in good:
            pmap[s] = float(args.online_prob_good); labels[s] = "good"
        elif s in med:
            pmap[s] = float(args.online_prob_med); labels[s] = "medium"
        else:
            pmap[s] = float(args.online_prob_bad); labels[s] = "bad"
    return pmap, labels


def init_client_comm(subjects: List[str], init_backbone: Dict[str, torch.Tensor]) -> Dict[str, dict]:
    return {
        s: {
            "local_version": 0,
            "local_backbone": clone_state(init_backbone),
            "session1_head_state": None,
            "session1_head_update_events": 0,
            "session1_head_source_round": None,
            "session1_head_source_version": None,
            "buffer_queue": [],
            "bytes_sent_uplink": 0,
            "bytes_recv_downlink": 0,
            "upload_events": 0,
            "uploaded_from_buffer_events": 0,
            "uploaded_from_fresh_events": 0,
            "download_events": 0,
            "sync_download_events": 0,
            "download_avoided_events": 0,
            "selected_events": 0,
            "unselected_events": 0,
            "selected_online_events": 0,
            "selected_offline_events": 0,
            "offline_deferred_events": 0,
            "last_selected_round": 0,
            "last_upload_round": 0,
            "last_successful_upload_round": 0,
            "last_applied_round": 0,
            "last_uploaded_server_version": 0,
            "last_successful_upload_server_version": 0,
            "last_applied_server_version": 0,
            "buffer_events": 0,
            "buffer_dropped_oldest_events": 0,
            "buffer_overwritten_latest_events": 0,
            "offline_drop_events": 0,
            "offline_unselected_no_buffer_drops": 0,
            "accepted_events": 0,
            "accepted_from_buffer_events": 0,
            "accepted_from_fresh_events": 0,
            "applied_events": 0,
            "applied_from_buffer_events": 0,
            "applied_from_fresh_events": 0,
            "superseded_events": 0,  # Retained field; always zero.
            "dropped_events": 0,
            "delay_rounds_list": [],
            "delay_seconds_list": [],
            "staleness_list": [],
            "staleness_accepted_list": [],
            "staleness_dropped_list": [],
        }
        for s in subjects
    }


def train_payload_for_subject(
    subject: str,
    base_backbone: Dict[str, torch.Tensor],
    base_version: int,
    round_idx: int,
    subj_data: Dict[str, Dict[str, bc.eegDataset]],
    init_state: Dict[str, torch.Tensor],
    backbone_keys: List[str],
    net_ctor,
    model_args: dict,
    args,
    device: torch.device,
    expects_bands: bool,
    seed: int,
) -> Dict[str, Any]:
    full_state = merge_backbone_with_init_head(base_backbone, init_state)
    model = net_ctor(**model_args)
    model.load_state_dict(full_state, strict=False)
    # DataLoader shuffling already receives ``seed``.  In per_task_seed mode,
    # also reset global RNGs so EEGNet dropout is tied to this task rather than
    # to the policy-dependent order/number of preceding local updates.
    reset_rng_for_training_task(args, int(seed))
    stats = bc.train_fixed_epochs(
        model=model,
        train_ds=subj_data[subject]["s1"],
        device=device,
        expects_bands=expects_bands,
        lr=float(args.lr_local),
        batch_size=int(args.batch_size),
        epochs=int(args.local_epochs),
        seed=int(seed),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        train_head=not bool(args.no_local_train_head),
    )
    trained_full_state = model.state_dict()
    new_backbone = backbone_only(trained_full_state, backbone_keys)
    trained_head = head_only(trained_full_state, backbone_keys)
    delta = delta_from_backbones(new_backbone, base_backbone, backbone_keys)
    perf_score = float(stats.get("train_acc", 0.0))
    meta = {
        "subject_id": subject,
        "trained_from_version": int(base_version),
        "produced_round": int(round_idx),
        "produced_wallclock_time": float(time.time()),
    }
    payload = {
        "subject": subject,
        "delta": delta,
        "trained_head": trained_head,
        "metadata": meta,
        "train_stats": stats,
    }
    if str(args.agg_weighting) == "performance":
        # Only performance aggregation transmits performance metadata to the server.
        meta["perf_score"] = perf_score
        payload["perf_score"] = perf_score
    return payload


def record_session1_head_from_payload(subject: str, payload: Dict[str, Any], client_state: Dict[str, dict]) -> None:
    """Store the latest locally trained Session-1 head for later personalization.

    This does not change the communicated payload size because only backbone
    deltas are counted as client-to-server model payloads.  The stored head is a
    local client artifact used to initialize Session-2 personalization.
    """
    head_state = payload.get("trained_head", None)
    if head_state is None:
        return
    client_state[subject]["session1_head_state"] = clone_state(head_state)
    client_state[subject]["session1_head_update_events"] += 1
    meta = payload.get("metadata", {})
    client_state[subject]["session1_head_source_round"] = int(meta.get("produced_round", 0))
    client_state[subject]["session1_head_source_version"] = int(meta.get("trained_from_version", 0))


def client_priority_record(
    s: str,
    client_state: Dict[str, dict],
    server_version: int,
    round_idx: int,
    stale_threshold: int,
    rng: random.Random,
) -> Dict[str, Any]:
    queue = list(client_state[s].get("buffer_queue", []))
    stalenesses, delays = [], []
    for payload in queue:
        meta = payload.get("metadata", {})
        stalenesses.append(int(server_version) - int(meta.get("trained_from_version", server_version)))
        delays.append(int(round_idx) - int(meta.get("produced_round", round_idx)))
    max_staleness = max(stalenesses) if stalenesses else -1
    max_delay = max(delays) if delays else 0
    last_selected = int(client_state[s].get("last_selected_round", 0))
    last_success = int(client_state[s].get("last_successful_upload_round", 0))
    last_success_ver = int(client_state[s].get("last_successful_upload_server_version", 0))
    local_version = int(client_state[s].get("local_version", 0))
    rounds_since_selected = int(round_idx) - last_selected if last_selected > 0 else int(round_idx)
    rounds_since_success = int(round_idx) - last_success if last_success > 0 else int(round_idx)
    version_lag_since_success = max(0, int(server_version) - last_success_ver)
    local_version_lag = max(0, int(server_version) - local_version)
    stale_risk = max_staleness - int(stale_threshold) if queue else -10**9
    return {
        "subject": s,
        "subject_report_id": bc.subject_to_report_id(s),
        "has_buffer": bool(queue),
        "buffer_count": int(len(queue)),
        "max_buffer_staleness": int(max_staleness),
        "max_buffer_delay_rounds": int(max_delay),
        "stale_risk_over_threshold": int(stale_risk) if queue else None,
        "rounds_since_selected": int(rounds_since_selected),
        "rounds_since_successful_upload": int(rounds_since_success),
        "version_lag_since_successful_upload": int(version_lag_since_success),
        "local_version_lag": int(local_version_lag),
        "tie_break": float(rng.random()),
    }


def sort_priority_records(records: List[Dict[str, Any]], policy: str) -> List[Dict[str, Any]]:
    if str(policy) == "comm_aware":
        return sorted(
            records,
            key=lambda x: (
                -int(x["version_lag_since_successful_upload"]),
                -int(x["rounds_since_successful_upload"]),
                -int(x["has_buffer"]),
                -int(x["max_buffer_staleness"]),
                -int(x["max_buffer_delay_rounds"]),
                -int(x["local_version_lag"]),
                float(x["tie_break"]),
                str(x["subject_report_id"]),
            )
        )
    return sorted(
        records,
        key=lambda x: (
            -int(x["has_buffer"]),
            -int(x["max_buffer_staleness"]),
            -int(x["max_buffer_delay_rounds"]),
            -int(x["rounds_since_selected"]),
            float(x["tie_break"]),
            str(x["subject_report_id"]),
        )
    )


def select_subjects_for_round(
    subjects: List[str],
    args,
    client_state: Dict[str, dict],
    server_version: int,
    round_idx: int,
    online_status: Optional[Dict[str, bool]] = None,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Choose upload-eligible clients for the round.

    Policies:
      - all: select every client. Online/offline status is applied after selection.
      - topk: optional diagnostic communication-budget scheduler. It ranks all clients by buffered
        update risk and fairness; online/offline status is applied after selection.
      - online_random: component-analysis scheduler. It samples online/offline first
        and randomly selects currently online gateways up to the round budget.
      - comm_aware: gateway scheduler. It samples online/offline first, selects
        only currently online clients for immediate upload, and prioritizes high
        version lag, long time since accepted upload, pending buffers, and local
        version lag. Offline clients may still create bounded pending payloads in
        the round loop when buffering is enabled, but they are not selected for
        immediate upload.
    """
    subjects = list(subjects)
    n = len(subjects)
    if n == 0:
        return [], [], {"policy": str(args.selection_policy), "effective_policy": "empty", "k": 0}

    policy = str(args.selection_policy)
    k = int(args.max_selected_per_round)
    rng = random.Random(int(resolve_seed_bundle(args).scheduler_rng_base_seed) + 1009 * int(round_idx))
    threshold = int(args.stale_threshold)

    if policy == "all":
        return list(subjects), [], {
            "policy": policy,
            "effective_policy": "all",
            "k_requested": int(k),
            "k_effective": int(n),
            "reason": "all gateways selected before online/offline availability is applied",
        }

    if policy not in ("topk", "online_random", "comm_aware"):
        raise RuntimeError(f"Unsupported selection policy: {policy}")

    if policy == "online_random":
        if online_status is None:
            online_status = {s: True for s in subjects}
        online_candidates = [s for s in subjects if bool(online_status.get(s, True))]
        offline_candidates = [s for s in subjects if not bool(online_status.get(s, True))]
        ranked = list(online_candidates)
        rng.shuffle(ranked)
        if k <= 0 or k >= len(ranked):
            selected = list(ranked)
            effective = "online_random_all_online_candidates"
        else:
            selected = list(ranked[:k])
            effective = "online_random_topk"
        selected_set = set(selected)
        unselected = [s for s in subjects if s not in selected_set]
        summary = {
            "policy": policy, "effective_policy": effective,
            "k_requested": int(k), "k_effective": int(len(selected)),
            "n_online_candidates": int(len(online_candidates)),
            "n_offline_deferred": int(len(offline_candidates)),
            "selected_random_order": [bc.subject_to_report_id(s0) for s0 in selected],
            "unselected_online_subjects": [bc.subject_to_report_id(s0) for s0 in ranked[len(selected):]],
            "offline_deferred_subjects": [bc.subject_to_report_id(s0) for s0 in offline_candidates],
        }
        return selected, unselected, summary

    if policy == "comm_aware":
        if online_status is None:
            online_status = {s: True for s in subjects}
        online_candidates = [s for s in subjects if bool(online_status.get(s, True))]
        offline_candidates = [s for s in subjects if not bool(online_status.get(s, True))]
        candidate_records = [client_priority_record(s, client_state, server_version, round_idx, threshold, rng) for s in online_candidates]
        ranked = sort_priority_records(candidate_records, policy="comm_aware")
        if k <= 0 or k >= len(ranked):
            selected = [r["subject"] for r in ranked]
            effective = "comm_aware_all_online_candidates"
        else:
            selected = [r["subject"] for r in ranked[:k]]
            effective = "comm_aware_online_version_lag_topk"
        selected_set = set(selected)
        unselected = [s for s in subjects if s not in selected_set]
        summary = {
            "policy": policy,
            "effective_policy": effective,
            "k_requested": int(k),
            "k_effective": int(len(selected)),
            "n_online_candidates": int(len(online_candidates)),
            "n_offline_deferred": int(len(offline_candidates)),
            "n_buffered_clients_before_selection": int(sum(1 for r in ranked if r["has_buffer"])),
            "selected_priority": [{kk: vv for kk, vv in r.items() if kk != "subject"} for r in ranked[:len(selected)]],
            "unselected_online_priority": [{kk: vv for kk, vv in r.items() if kk != "subject"} for r in ranked[len(selected):]],
            "offline_deferred_subjects": [bc.subject_to_report_id(s) for s in offline_candidates],
        }
        return selected, unselected, summary

    # Diagnostic top-k policy: rank clients before applying availability.
    if k <= 0 or k >= n:
        return list(subjects), [], {
            "policy": policy,
            "effective_policy": "all",
            "k_requested": int(k),
            "k_effective": int(n),
            "reason": "topk disabled because k<=0 or k>=n_clients",
        }
    records = [client_priority_record(s, client_state, server_version, round_idx, threshold, rng) for s in subjects]
    ranked = sort_priority_records(records, policy="topk")
    selected = [r["subject"] for r in ranked[:k]]
    selected_set = set(selected)
    unselected = [s for s in subjects if s not in selected_set]
    summary = {
        "policy": policy,
        "effective_policy": "topk_buffer_staleness_fairness",
        "k_requested": int(k),
        "k_effective": int(len(selected)),
        "n_buffered_clients_before_selection": int(sum(1 for r in ranked if r["has_buffer"])),
        "selected_priority": [{kk: vv for kk, vv in r.items() if kk != "subject"} for r in ranked[:k]],
        "unselected_priority": [{kk: vv for kk, vv in r.items() if kk != "subject"} for r in ranked[k:]],
    }
    return selected, unselected, summary


def should_download_client(local_version: int, server_version: int, args) -> bool:
    lag = int(server_version) - int(local_version)
    if lag <= 0:
        return False
    if str(getattr(args, "download_policy", "always")) == "always":
        return True
    if str(getattr(args, "download_policy", "always")) == "stale_only":
        return lag > int(getattr(args, "download_stale_threshold", args.stale_threshold))
    raise RuntimeError(f"Unsupported download policy: {getattr(args, 'download_policy', 'always')}")


def buffer_payload_for_subject(
    subject: str,
    payload: Dict[str, Any],
    client_state: Dict[str, dict],
    args,
    comm_cfg: CommunicationConfig,
    ongoing: Dict[str, Any],
    round_info: Dict[str, Any],
    deferred: bool = False,
) -> None:
    """Store or drop an offline payload according to the bounded buffering rules."""
    if not comm_cfg.buffering_enabled or str(comm_cfg.buffer_policy) == "none":
        if deferred:
            client_state[subject]["offline_unselected_no_buffer_drops"] += 1
            ongoing["offline_unselected_no_buffer_drops"] += 1
            round_info["offline_unselected_no_buffer_drops"] += 1
        else:
            client_state[subject]["offline_drop_events"] += 1
            ongoing["offline_selected_no_buffer_drops"] += 1
            round_info["offline_selected_no_buffer_drops"] += 1
        return

    queue = client_state[subject]["buffer_queue"]
    policy = str(comm_cfg.buffer_policy)
    if policy == "latest":
        if len(queue) > 0:
            queue.clear()
            client_state[subject]["buffer_overwritten_latest_events"] += 1
            ongoing["buffer_overwritten_latest"] += 1
            round_info["buffer_overwritten_latest"] += 1
        queue.append(payload)
    elif policy == "fifo":
        max_size = max(1, int(comm_cfg.buffer_max_size))
        while len(queue) >= max_size:
            queue.pop(0)
            client_state[subject]["buffer_dropped_oldest_events"] += 1
            ongoing["buffer_dropped_oldest"] += 1
            round_info["buffer_dropped_oldest"] += 1
        queue.append(payload)
    else:
        raise RuntimeError(f"Unsupported buffer policy: {policy}")

    client_state[subject]["buffer_events"] += 1
    ongoing["buffered_new_payloads"] += 1
    round_info["buffered_new_payloads"] += 1
    if deferred:
        client_state[subject]["offline_deferred_events"] += 1
        ongoing["offline_deferred_buffered_payloads"] += 1
        round_info["offline_deferred_buffered_payloads"] += 1


def add_comm_rates(ongoing: Dict[str, Any], n_clients: int, n_rounds: int, pending_buffer_clients: int) -> Dict[str, float]:
    total_client_rounds = int(n_clients) * int(n_rounds)
    rates = {
        "selection_rate_over_client_rounds": safe_div(ongoing.get("selected_events", 0), total_client_rounds),
        "unselected_rate_over_client_rounds": safe_div(ongoing.get("unselected_events", 0), total_client_rounds),
        "online_availability_rate": safe_div(ongoing.get("online_available_events", 0), total_client_rounds),
        "offline_availability_rate": safe_div(ongoing.get("offline_available_events", 0), total_client_rounds),
        "online_selected_rate": safe_div(ongoing.get("selected_online_events", 0), ongoing.get("selected_events", 0)),
        "offline_selected_rate": safe_div(ongoing.get("selected_offline_events", 0), ongoing.get("selected_events", 0)),
        "selected_online_fraction_of_online_available": safe_div(ongoing.get("selected_online_events", 0), ongoing.get("online_available_events", 0)),
        "upload_rate_over_selected": safe_div(ongoing.get("uploads", 0), ongoing.get("selected_events", 0)),
        "upload_rate_over_online_selected": safe_div(ongoing.get("uploads", 0), ongoing.get("selected_online_events", 0)),
        "fresh_upload_rate_over_selected": safe_div(ongoing.get("uploads_from_fresh", 0), ongoing.get("selected_events", 0)),
        "buffer_upload_fraction_over_uploads": safe_div(ongoing.get("uploads_from_buffer", 0), ongoing.get("uploads", 0)),
        "accept_rate_over_uploads": safe_div(ongoing.get("accepted_updates", 0), ongoing.get("uploads", 0)),
        "drop_rate_over_uploads": safe_div(ongoing.get("dropped_updates", 0), ongoing.get("uploads", 0)),
        "stale_drop_fraction_over_drops": safe_div(ongoing.get("stale_drop_updates", 0), ongoing.get("dropped_updates", 0)),
        "checkpoint_missing_fraction_over_drops": safe_div(ongoing.get("checkpoint_missing_drop_updates", 0), ongoing.get("dropped_updates", 0)),
        "download_rate_over_selected_online": safe_div(ongoing.get("sync_downloads", 0), ongoing.get("selected_online_events", 0)),
        "download_rate_over_opportunities": safe_div(ongoing.get("sync_downloads", 0), ongoing.get("download_opportunities", 0)),
        "download_avoidance_rate_over_download_opportunities": safe_div(ongoing.get("download_avoided", 0), ongoing.get("download_opportunities", 0)),
        "pending_buffer_client_rate_end": safe_div(pending_buffer_clients, n_clients),
    }
    return {k: float(v) for k, v in rates.items()}


def run_federated_backbone_training(
    train_subjects: List[str],
    subj_data: Dict[str, Dict[str, bc.eegDataset]],
    initial_backbone: Dict[str, torch.Tensor],
    init_state: Dict[str, torch.Tensor],
    backbone_keys: List[str],
    net_ctor,
    model_args: dict,
    args,
    device: torch.device,
    expects_bands: bool,
    comm_cfg: CommunicationConfig,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, Any],
    List[dict],
    Dict[str, Dict[str, torch.Tensor]],
    Dict[str, Any],
]:
    seed_bundle = resolve_seed_bundle(args)
    server_backbone = clone_state(initial_backbone)
    server_version = 0
    checkpoint_history = {0: clone_state(server_backbone)}
    client_state = init_client_comm(train_subjects, server_backbone)
    pmap, plabels = online_probability_map(train_subjects, args, seed=int(seed_bundle.group_seed))
    trace_file = str(getattr(args, "availability_trace_file", "") or "").strip()
    if trace_file:
        availability_trace = load_availability_trace(
            trace_file,
            train_subjects,
            int(args.rounds),
            expected_pmap=pmap,
            expected_group_labels=plabels,
            expected_seed_bundle=seed_bundle,
        )
        # A loaded trace defines online/offline states. The group
        # mapping retained in the run still comes from this run's group seed and
        # is validated separately in the paired suite.
    else:
        availability_trace = build_availability_trace(
            train_subjects, pmap, plabels, int(args.rounds), seed_bundle, bool(comm_cfg.effective)
        )
    trace_matrix = availability_trace["availability_matrix"]
    sample_counts = {s: int(len(subj_data[s]["s1"])) for s in train_subjects}

    backbone_download_nbytes = state_bytes(server_backbone) + metadata_bytes({"version": 0})
    # Initial dissemination. Every participating client has version-0 backbone cached.
    total_down = backbone_download_nbytes * len(train_subjects)
    for s in train_subjects:
        client_state[s]["bytes_recv_downlink"] += backbone_download_nbytes
        client_state[s]["download_events"] += 1

    total_up = 0
    progress = []
    ongoing = {
        "selected_events": 0,
        "unselected_events": 0,
        "selected_online_events": 0,
        "selected_offline_events": 0,
        "online_available_events": 0,
        "offline_available_events": 0,
        "downloads": len(train_subjects),
        "initial_downloads": len(train_subjects),
        "sync_downloads": 0,
        "download_opportunities": 0,
        "download_avoided": 0,
        "uploads": 0,
        "uploads_from_fresh": 0,
        "uploads_from_buffer": 0,
        "buffered_new_payloads": 0,
        "buffer_dropped_oldest": 0,
        "buffer_overwritten_latest": 0,
        "offline_deferred_buffered_payloads": 0,
        "offline_selected_no_buffer_drops": 0,
        "offline_unselected_no_buffer_drops": 0,
        "accepted_updates": 0,
        "accepted_from_buffer": 0,
        "accepted_from_fresh": 0,
        "applied_updates": 0,
        "applied_from_buffer": 0,
        "applied_from_fresh": 0,
        "applied_update_bytes": 0,
        "accepted_but_superseded_updates": 0,
        "dropped_updates": 0,
        "dropped_update_bytes": 0,
        "stale_drop_updates": 0,
        "checkpoint_missing_drop_updates": 0,
        "client_to_server_bytes": 0,
        "server_to_client_bytes": total_down,
    }

    for rnd in range(1, int(args.rounds) + 1):
        trace_row = trace_matrix[int(rnd) - 1]
        online_status = {s: bool(int(trace_row[i])) for i, s in enumerate(train_subjects)}
        online_available = [s for s in train_subjects if online_status[s]]
        offline_available = [s for s in train_subjects if not online_status[s]]
        ongoing["online_available_events"] += len(online_available)
        ongoing["offline_available_events"] += len(offline_available)

        selected, unselected, scheduler_info = select_subjects_for_round(
            train_subjects, args, client_state, server_version, rnd, online_status=online_status
        )
        selected_set = set(selected)
        ongoing["selected_events"] += len(selected)
        ongoing["unselected_events"] += len(unselected)
        for _s in selected:
            client_state[_s]["selected_events"] += 1
            client_state[_s]["last_selected_round"] = int(rnd)
        for _s in unselected:
            client_state[_s]["unselected_events"] += 1
        uploaded_payloads: List[Dict[str, Any]] = []
        round_info = {
            "round": int(rnd),
            "server_version_start": int(server_version),
            "n_clients": int(len(train_subjects)),
            "n_selected": int(len(selected)),
            "n_unselected": int(len(unselected)),
            "n_online_available": int(len(online_available)),
            "n_offline_available": int(len(offline_available)),
            "selected_subjects": [bc.subject_to_report_id(s) for s in selected],
            "unselected_subjects": [bc.subject_to_report_id(s) for s in unselected],
            "available_online_subjects": [bc.subject_to_report_id(s) for s in online_available],
            "available_offline_subjects": [bc.subject_to_report_id(s) for s in offline_available],
            "scheduler": scheduler_info,
            "online_subjects": [],
            "offline_subjects": [],
            "downloads": 0,
            "download_opportunities": 0,
            "download_avoided": 0,
            "uploads": 0,
            "uploads_from_fresh": 0,
            "uploads_from_buffer": 0,
            "buffered_new_payloads": 0,
            "buffer_dropped_oldest": 0,
            "buffer_overwritten_latest": 0,
            "offline_deferred_buffered_payloads": 0,
            "offline_selected_no_buffer_drops": 0,
            "offline_unselected_no_buffer_drops": 0,
            "accepted_updates": 0,
            "accepted_from_buffer": 0,
            "accepted_from_fresh": 0,
            "applied_updates": 0,
            "applied_from_buffer": 0,
            "applied_from_fresh": 0,
            "applied_update_bytes": 0,
            "accepted_but_superseded_updates": 0,
            "dropped_updates": 0,
            "dropped_update_bytes": 0,
            "stale_drop_updates": 0,
            "checkpoint_missing_drop_updates": 0,
            "uplink_bytes": 0,
            "downlink_bytes": 0,
            "delay_rounds": [],
            "delay_seconds": [],
            "staleness": [],
            "staleness_accepted": [],
            "staleness_dropped": [],
        }

        for s in selected:
            online = bool(online_status.get(s, True))
            if online:
                ongoing["selected_online_events"] += 1
                client_state[s]["selected_online_events"] += 1
                round_info["online_subjects"].append(bc.subject_to_report_id(s))
                # Synchronize only when the selected download policy says the cached
                # backbone is too stale for fresh local training.
                if int(client_state[s]["local_version"]) != int(server_version):
                    ongoing["download_opportunities"] += 1
                    round_info["download_opportunities"] += 1
                    if should_download_client(int(client_state[s]["local_version"]), int(server_version), args):
                        meta = {"version": int(server_version), "round": int(rnd), "subject_id": s}
                        nbytes = state_bytes(server_backbone) + metadata_bytes(meta)
                        client_state[s]["bytes_recv_downlink"] += nbytes
                        client_state[s]["download_events"] += 1
                        client_state[s]["sync_download_events"] += 1
                        ongoing["downloads"] += 1
                        ongoing["sync_downloads"] += 1
                        ongoing["server_to_client_bytes"] += nbytes
                        round_info["downloads"] += 1
                        round_info["downlink_bytes"] += nbytes
                        client_state[s]["local_backbone"] = clone_state(server_backbone)
                        client_state[s]["local_version"] = int(server_version)
                    else:
                        client_state[s]["download_avoided_events"] += 1
                        ongoing["download_avoided"] += 1
                        round_info["download_avoided"] += 1

                # First upload buffered payloads FIFO. Latest buffering simply has at most one item.
                while client_state[s]["buffer_queue"]:
                    payload = client_state[s]["buffer_queue"].pop(0)
                    payload["upload_source"] = "buffer"
                    uploaded_payloads.append(payload)

                # Then produce and upload a fresh payload from the client's current cached backbone.
                fresh = train_payload_for_subject(
                    s,
                    client_state[s]["local_backbone"],
                    int(client_state[s]["local_version"]),
                    rnd,
                    subj_data,
                    init_state,
                    backbone_keys,
                    net_ctor,
                    model_args,
                    args,
                    device,
                    expects_bands,
                    seed=training_task_seed(
                        args,
                        int(seed_bundle.model_seed) + rnd * 1009 + len(uploaded_payloads),
                        "session1_local", s, int(rnd),
                    ),
                )
                record_session1_head_from_payload(s, fresh, client_state)
                fresh["upload_source"] = "fresh"
                uploaded_payloads.append(fresh)
            else:
                ongoing["selected_offline_events"] += 1
                client_state[s]["selected_offline_events"] += 1
                round_info["offline_subjects"].append(bc.subject_to_report_id(s))
                # Fixed all/top-k policies may select clients that are currently offline.
                # Those clients train from their cached local backbone and then follow
                # the bounded buffer policy.
                payload = train_payload_for_subject(
                    s,
                    client_state[s]["local_backbone"],
                    int(client_state[s]["local_version"]),
                    rnd,
                    subj_data,
                    init_state,
                    backbone_keys,
                    net_ctor,
                    model_args,
                    args,
                    device,
                    expects_bands,
                    seed=training_task_seed(
                        args,
                        int(seed_bundle.model_seed) + rnd * 1009 + len(uploaded_payloads),
                        "session1_local", s, int(rnd),
                    ),
                )
                record_session1_head_from_payload(s, payload, client_state)
                buffer_payload_for_subject(s, payload, client_state, args, comm_cfg, ongoing, round_info, deferred=False)

        # In comm_aware, offline clients are not selected for immediate upload.
        # To model gateway-side local learning during backhaul outages, the script
        # allows unavailable clients to create bounded pending updates when buffering
        # is enabled. With buffer_policy=none, these unavailable opportunities are
        # counted as dropped offline updates without spending time training them.
        if str(args.selection_policy) in ("comm_aware", "online_random"):
            for s in offline_available:
                if s in selected_set:
                    continue
                if comm_cfg.buffering_enabled:
                    payload = train_payload_for_subject(
                        s,
                        client_state[s]["local_backbone"],
                        int(client_state[s]["local_version"]),
                        rnd,
                        subj_data,
                        init_state,
                        backbone_keys,
                        net_ctor,
                        model_args,
                        args,
                        device,
                        expects_bands,
                        seed=training_task_seed(
                            args,
                            int(seed_bundle.model_seed) + rnd * 2003 + len(uploaded_payloads),
                            "session1_local", s, int(rnd),
                        ),
                    )
                    record_session1_head_from_payload(s, payload, client_state)
                    buffer_payload_for_subject(s, payload, client_state, args, comm_cfg, ongoing, round_info, deferred=True)
                else:
                    client_state[s]["offline_unselected_no_buffer_drops"] += 1
                    ongoing["offline_unselected_no_buffer_drops"] += 1
                    round_info["offline_unselected_no_buffer_drops"] += 1

        # Admission: every uploaded payload crosses the wire and is checked independently.
        accepted: List[Dict[str, Any]] = []
        for order, payload in enumerate(uploaded_payloads):
            s = payload["subject"]
            wire_meta = dict(payload["metadata"])
            source = payload.get("upload_source", "fresh")
            accounting_meta = upload_metadata_for_byte_accounting(wire_meta, args)
            nbytes = state_bytes(payload["delta"]) + metadata_bytes(accounting_meta)
            total_up += nbytes
            client_state[s]["bytes_sent_uplink"] += nbytes
            client_state[s]["upload_events"] += 1
            client_state[s]["last_upload_round"] = int(rnd)
            client_state[s]["last_uploaded_server_version"] = int(server_version)
            client_state[s]["uploaded_from_buffer_events" if source == "buffer" else "uploaded_from_fresh_events"] += 1
            ongoing["client_to_server_bytes"] += nbytes
            ongoing["uploads"] += 1
            ongoing["uploads_from_buffer" if source == "buffer" else "uploads_from_fresh"] += 1
            round_info["uploads"] += 1
            round_info["uplink_bytes"] += nbytes
            round_info["uploads_from_buffer" if source == "buffer" else "uploads_from_fresh"] += 1

            trained_from = int(wire_meta["trained_from_version"])
            if trained_from not in checkpoint_history:
                ongoing["dropped_updates"] += 1
                ongoing["checkpoint_missing_drop_updates"] += 1
                ongoing["dropped_update_bytes"] += nbytes
                round_info["dropped_updates"] += 1
                round_info["checkpoint_missing_drop_updates"] += 1
                round_info["dropped_update_bytes"] += nbytes
                client_state[s]["dropped_events"] += 1
                continue

            staleness = int(server_version) - trained_from
            round_info["staleness"].append(staleness)
            client_state[s]["staleness_list"].append(staleness)
            if source == "buffer":
                delay_r = int(rnd) - int(wire_meta["produced_round"])
                delay_s = float(time.time()) - float(wire_meta["produced_wallclock_time"])
                round_info["delay_rounds"].append(delay_r)
                round_info["delay_seconds"].append(delay_s)
                client_state[s]["delay_rounds_list"].append(delay_r)
                client_state[s]["delay_seconds_list"].append(delay_s)

            if staleness > int(args.stale_threshold) and str(args.stale_policy) == "drop":
                ongoing["dropped_updates"] += 1
                ongoing["stale_drop_updates"] += 1
                ongoing["dropped_update_bytes"] += nbytes
                round_info["dropped_updates"] += 1
                round_info["stale_drop_updates"] += 1
                round_info["dropped_update_bytes"] += nbytes
                round_info["staleness_dropped"].append(staleness)
                client_state[s]["dropped_events"] += 1
                client_state[s]["staleness_dropped_list"].append(staleness)
                continue

            stale_factor = 1.0
            if staleness > int(args.stale_threshold) and str(args.stale_policy) == "downweight":
                stale_factor = float(args.stale_gamma) ** float(staleness)
            # stale_policy=accept_all accepts delayed updates with stale_factor=1.0.

            payload["admission"] = {
                "accepted": True,
                "upload_order": int(order),
                "staleness": int(staleness),
                "stale_factor": float(stale_factor),
                "payload_bytes": int(nbytes),
            }
            accepted.append(payload)
            ongoing["accepted_updates"] += 1
            ongoing["accepted_from_buffer" if source == "buffer" else "accepted_from_fresh"] += 1
            round_info["accepted_updates"] += 1
            round_info["accepted_from_buffer" if source == "buffer" else "accepted_from_fresh"] += 1
            client_state[s]["accepted_events"] += 1
            client_state[s]["accepted_from_buffer_events" if source == "buffer" else "accepted_from_fresh_events"] += 1
            client_state[s]["staleness_accepted_list"].append(staleness)
            round_info["staleness_accepted"].append(staleness)
            client_state[s]["last_successful_upload_round"] = int(rnd)
            client_state[s]["last_successful_upload_server_version"] = int(server_version)

        applied_payloads = list(accepted)
        if applied_payloads:
            candidate_backbones: Dict[str, Dict[str, torch.Tensor]] = {}
            stale_factors: Dict[str, float] = {}
            update_sample_counts: Dict[str, int] = {}
            update_perf_scores: Dict[str, float] = {}
            applied_update_keys: List[str] = []
            applied_bytes = 0

            for payload in applied_payloads:
                s = payload["subject"]
                source = payload.get("upload_source", "fresh")
                meta = payload["metadata"]
                trained_from = int(meta["trained_from_version"])
                base = checkpoint_history[trained_from]
                update_key = (
                    f"{s}::v{trained_from}::r{int(meta['produced_round'])}::"
                    f"u{int(payload['admission']['upload_order'])}"
                )
                candidate_backbones[update_key] = apply_delta(base, payload["delta"], backbone_keys)
                stale_factors[update_key] = float(payload["admission"]["stale_factor"])
                update_sample_counts[update_key] = int(sample_counts.get(s, 1))
                if str(args.agg_weighting) == "performance":
                    update_perf_scores[update_key] = float(payload.get("perf_score", meta.get("perf_score", 0.0)))
                applied_update_keys.append(update_key)
                applied_bytes += int(payload["admission"].get("payload_bytes", 0))
                client_state[s]["applied_events"] += 1
                client_state[s]["last_applied_round"] = int(rnd)
                client_state[s]["last_applied_server_version"] = int(server_version)
                client_state[s]["applied_from_buffer_events" if source == "buffer" else "applied_from_fresh_events"] += 1
                ongoing["applied_from_buffer" if source == "buffer" else "applied_from_fresh"] += 1
                round_info["applied_from_buffer" if source == "buffer" else "applied_from_fresh"] += 1

            weights = compute_weights(
                applied_update_keys,
                str(args.agg_weighting),
                update_sample_counts,
                update_perf_scores if str(args.agg_weighting) == "performance" else None,
                str(args.perf_weight_metric),
                str(args.perf_weight_method),
                float(args.perf_weight_alpha),
                float(args.perf_weight_eps),
                bool(args.perf_weight_with_samples),
                stale_factors if str(args.stale_policy) == "downweight" else None,
            )
            server_backbone = aggregate_backbones(candidate_backbones, weights, backbone_keys)
            server_version += 1
            checkpoint_history[server_version] = clone_state(server_backbone)
            if str(args.stale_policy) == "accept_all":
                # For the accept-all delayed-update ablation, keep all retained
                # base checkpoints so delayed payloads are not rejected simply
                # because the base version was pruned.
                min_keep = 0
            else:
                min_keep = (
                    int(server_version)
                    - int(args.stale_threshold)
                    - int(args.checkpoint_retention_margin)
                )
            for v in list(checkpoint_history.keys()):
                if int(v) < min_keep:
                    del checkpoint_history[v]
            ongoing["applied_updates"] += len(applied_payloads)
            ongoing["applied_update_bytes"] += int(applied_bytes)
            round_info["applied_updates"] += len(applied_payloads)
            round_info["applied_update_bytes"] += int(applied_bytes)

        round_info["server_version_end"] = int(server_version)
        round_info["delay_rounds_mean"] = float(np.mean(round_info["delay_rounds"])) if round_info["delay_rounds"] else 0.0
        round_info["delay_rounds_max"] = int(max(round_info["delay_rounds"])) if round_info["delay_rounds"] else 0
        round_info["delay_seconds_mean"] = float(np.mean(round_info["delay_seconds"])) if round_info["delay_seconds"] else 0.0
        round_info["delay_seconds_max"] = float(max(round_info["delay_seconds"])) if round_info["delay_seconds"] else 0.0
        round_info["staleness_all_admission_attempts_mean"] = float(np.mean(round_info["staleness"])) if round_info["staleness"] else 0.0
        round_info["staleness_all_admission_attempts_max"] = int(max(round_info["staleness"])) if round_info["staleness"] else 0
        round_info["accepted_update_staleness_mean"] = float(np.mean(round_info["staleness_accepted"])) if round_info["staleness_accepted"] else 0.0
        round_info["accepted_update_staleness_max"] = int(max(round_info["staleness_accepted"])) if round_info["staleness_accepted"] else 0
        # These names point to the manuscript-defined accepted-update statistic.
        round_info["staleness_mean"] = round_info["accepted_update_staleness_mean"]
        round_info["staleness_max"] = round_info["accepted_update_staleness_max"]
        round_info["online_selected_rate"] = safe_div(round_info["n_selected"] - len(round_info["offline_subjects"]), round_info["n_selected"])
        round_info["offline_selected_rate"] = safe_div(len(round_info["offline_subjects"]), round_info["n_selected"])
        round_info["upload_rate_over_selected"] = safe_div(round_info["uploads"], round_info["n_selected"])
        round_info["accept_rate_over_uploads"] = safe_div(round_info["accepted_updates"], round_info["uploads"])
        round_info["drop_rate_over_uploads"] = safe_div(round_info["dropped_updates"], round_info["uploads"])
        progress.append(round_info)

    per_subject = {}
    all_delay_rounds, all_delay_seconds = [], []
    all_staleness, accepted_staleness, dropped_staleness = [], [], []
    pending_buffer_clients = 0
    for s, st in client_state.items():
        all_delay_rounds.extend(st["delay_rounds_list"])
        all_delay_seconds.extend(st["delay_seconds_list"])
        all_staleness.extend(st["staleness_list"])
        accepted_staleness.extend(st["staleness_accepted_list"])
        dropped_staleness.extend(st["staleness_dropped_list"])
        if len(st["buffer_queue"]) > 0:
            pending_buffer_clients += 1
        per_subject[s] = {
            "subject_report_id": bc.subject_to_report_id(s),
            "comm_profile_label": plabels.get(s, "uniform"),
            "online_probability": float(pmap.get(s, 1.0)),
            "bytes_sent_uplink": int(st["bytes_sent_uplink"]),
            "bytes_recv_downlink": int(st["bytes_recv_downlink"]),
            "upload_events": int(st["upload_events"]),
            "uploaded_from_buffer_events": int(st["uploaded_from_buffer_events"]),
            "uploaded_from_fresh_events": int(st["uploaded_from_fresh_events"]),
            "download_events": int(st["download_events"]),
            "sync_download_events": int(st["sync_download_events"]),
            "download_avoided_events": int(st["download_avoided_events"]),
            "selected_events": int(st["selected_events"]),
            "unselected_events": int(st["unselected_events"]),
            "selected_online_events": int(st["selected_online_events"]),
            "selected_offline_events": int(st["selected_offline_events"]),
            "offline_deferred_events": int(st["offline_deferred_events"]),
            "last_selected_round": int(st["last_selected_round"]),
            "last_upload_round": int(st["last_upload_round"]),
            "last_successful_upload_round": int(st["last_successful_upload_round"]),
            "last_applied_round": int(st["last_applied_round"]),
            "last_uploaded_server_version": int(st["last_uploaded_server_version"]),
            "last_successful_upload_server_version": int(st["last_successful_upload_server_version"]),
            "last_applied_server_version": int(st["last_applied_server_version"]),
            "buffer_events": int(st["buffer_events"]),
            "buffer_dropped_oldest_events": int(st["buffer_dropped_oldest_events"]),
            "buffer_overwritten_latest_events": int(st["buffer_overwritten_latest_events"]),
            "offline_drop_events": int(st["offline_drop_events"]),
            "offline_unselected_no_buffer_drops": int(st["offline_unselected_no_buffer_drops"]),
            "accepted_events": int(st["accepted_events"]),
            "accepted_from_buffer_events": int(st["accepted_from_buffer_events"]),
            "accepted_from_fresh_events": int(st["accepted_from_fresh_events"]),
            "applied_events": int(st["applied_events"]),
            "applied_from_buffer_events": int(st["applied_from_buffer_events"]),
            "applied_from_fresh_events": int(st["applied_from_fresh_events"]),
            "superseded_events": int(st["superseded_events"]),
            "dropped_events": int(st["dropped_events"]),
            "delay_rounds_mean": float(np.mean(st["delay_rounds_list"])) if st["delay_rounds_list"] else 0.0,
            "delay_rounds_max": int(max(st["delay_rounds_list"])) if st["delay_rounds_list"] else 0,
            "staleness_all_admission_attempts_mean": float(np.mean(st["staleness_list"])) if st["staleness_list"] else 0.0,
            "staleness_all_admission_attempts_max": int(max(st["staleness_list"])) if st["staleness_list"] else 0,
            "accepted_update_staleness_mean": float(np.mean(st["staleness_accepted_list"])) if st["staleness_accepted_list"] else 0.0,
            "accepted_update_staleness_max": int(max(st["staleness_accepted_list"])) if st["staleness_accepted_list"] else 0,
            "dropped_update_staleness_mean": float(np.mean(st["staleness_dropped_list"])) if st["staleness_dropped_list"] else 0.0,
            "dropped_update_staleness_max": int(max(st["staleness_dropped_list"])) if st["staleness_dropped_list"] else 0,
            "staleness_mean": float(np.mean(st["staleness_accepted_list"])) if st["staleness_accepted_list"] else 0.0,
            "staleness_max": int(max(st["staleness_accepted_list"])) if st["staleness_accepted_list"] else 0,
            "buffer_pending_at_end": bool(len(st["buffer_queue"]) > 0),
            "buffer_pending_count_at_end": int(len(st["buffer_queue"])),
            "session1_head_available_for_personalization": bool(st.get("session1_head_state") is not None),
            "session1_head_update_events": int(st.get("session1_head_update_events", 0)),
            "session1_head_source_round": st.get("session1_head_source_round"),
            "session1_head_source_version": st.get("session1_head_source_version"),
            "selection_rate_over_rounds": safe_div(st["selected_events"], int(args.rounds)),
            "upload_rate_over_selected": safe_div(st["upload_events"], st["selected_events"]),
            "accept_rate_over_uploads": safe_div(st["accepted_events"], st["upload_events"]),
            "drop_rate_over_uploads": safe_div(st["dropped_events"], st["upload_events"]),
        }

    rates = add_comm_rates(ongoing, len(train_subjects), int(args.rounds), pending_buffer_clients)
    ongoing.update(rates)
    ongoing["total_client_rounds"] = int(len(train_subjects) * int(args.rounds))
    ongoing["pending_buffer_clients_end"] = int(pending_buffer_clients)
    ongoing["pending_buffer_payloads_end"] = int(sum(len(st["buffer_queue"]) for st in client_state.values()))

    comm_config_out = asdict(comm_cfg)
    comm_config_out.update({
        "selection_policy": str(args.selection_policy),
        "max_selected_per_round": int(args.max_selected_per_round),
        "buffer_policy_arg": str(args.buffer_policy),
        "download_policy_arg": str(args.download_policy),
    })


    session1_head_subjects = sorted([s for s, st in client_state.items() if st.get("session1_head_state") is not None])
    session1_head_states = {s: clone_state(client_state[s]["session1_head_state"]) for s in session1_head_subjects}
    session1_head_summary = {
        "policy": "latest_local_session1_trained_head_per_subject",
        "description": "Session-2 personalization merges the final collaborative backbone with each subject's latest locally trained Session-1 classifier head when available; otherwise it falls back to the original initialized head.",
        "available_subject_count": int(len(session1_head_subjects)),
        "available_subjects": [bc.subject_to_report_id(s) for s in session1_head_subjects],
        "fallback_subject_count": int(len(train_subjects) - len(session1_head_subjects)),
        "fallback_subjects": [bc.subject_to_report_id(s) for s in train_subjects if s not in set(session1_head_subjects)],
        "head_training_disabled_warning": bool(args.no_local_train_head),
    }
    metrics = {
        "communication_config": comm_config_out,
        "total_communication": {
            "client_to_server_bytes": int(ongoing["client_to_server_bytes"]),
            "server_to_client_bytes": int(ongoing["server_to_client_bytes"]),
            "total_bytes": int(ongoing["client_to_server_bytes"] + ongoing["server_to_client_bytes"]),
            "client_to_server_mb": bytes_to_mb(ongoing["client_to_server_bytes"]),
            "server_to_client_mb": bytes_to_mb(ongoing["server_to_client_bytes"]),
            "total_mb": bytes_to_mb(ongoing["client_to_server_bytes"] + ongoing["server_to_client_bytes"]),
            "client_to_server_mib": bytes_to_mib(ongoing["client_to_server_bytes"]),
            "server_to_client_mib": bytes_to_mib(ongoing["server_to_client_bytes"]),
            "total_mib": bytes_to_mib(ongoing["client_to_server_bytes"] + ongoing["server_to_client_bytes"]),
            "unit_definition": "decimal MB: 1 MB = 1,000,000 bytes; MiB fields are provided separately for auditability",
        },
        "ongoing": ongoing,
        "communication_rates": rates,
        "session1_head_initialization": session1_head_summary,
        "delay_summary": {
            "count": int(len(all_delay_rounds)),
            "mean_rounds": float(np.mean(all_delay_rounds)) if all_delay_rounds else 0.0,
            "max_rounds": int(max(all_delay_rounds)) if all_delay_rounds else 0,
            "mean_seconds": float(np.mean(all_delay_seconds)) if all_delay_seconds else 0.0,
            "max_seconds": float(max(all_delay_seconds)) if all_delay_seconds else 0.0,
        },
        "staleness_summary": {
            "definition": "event-weighted staleness of uploads accepted by checkpoint and stale-update admission and entering aggregation",
            "count": int(len(accepted_staleness)),
            "mean": float(np.mean(accepted_staleness)) if accepted_staleness else 0.0,
            "max": int(max(accepted_staleness)) if accepted_staleness else 0,
        },
        "accepted_update_staleness_summary": {
            "definition": "same reported statistic as staleness_summary",
            "count": int(len(accepted_staleness)),
            "mean": float(np.mean(accepted_staleness)) if accepted_staleness else 0.0,
            "max": int(max(accepted_staleness)) if accepted_staleness else 0,
        },
        "all_admission_attempt_staleness_summary": {
            "definition": "staleness for uploads whose base checkpoint exists, before stale-drop admission; includes accepted and stale-rejected uploads",
            "count": int(len(all_staleness)),
            "mean": float(np.mean(all_staleness)) if all_staleness else 0.0,
            "max": int(max(all_staleness)) if all_staleness else 0,
        },
        "dropped_update_staleness_summary": {
            "definition": "staleness of uploads rejected specifically by the stale threshold",
            "count": int(len(dropped_staleness)),
            "mean": float(np.mean(dropped_staleness)) if dropped_staleness else 0.0,
            "max": int(max(dropped_staleness)) if dropped_staleness else 0,
        },
        "availability_trace": {k: v for k, v in availability_trace.items() if k != "availability_matrix"},
        "per_subject_comm": per_subject,
        "aggregation_semantics": {
            "accepted_update_policy": "every accepted update is applied; superseding is disabled",
            "candidate_reconstruction": "candidate = checkpoint_history[trained_from_version] + delta",
            "accept_all_policy": "if stale_policy=accept_all, delayed updates are accepted whenever their base checkpoint still exists",
            "drop_policy": "if stale_policy=drop, over-threshold stale updates are rejected",
            "downweight_policy": "if stale_policy=downweight, accepted over-threshold updates receive stale_gamma ** staleness",
            "selection_policy": "all is non-adaptive all-gateway scheduling before availability; topk is a diagnostic pre-availability budgeted selector; online_random samples availability first and randomly selects online gateways; comm_aware samples availability first and selects only online gateways for immediate upload",
            "buffer_policy": "none drops offline payload opportunities; fifo keeps a bounded queue and drops oldest on overflow; latest keeps only the newest pending payload",
            "download_policy": "always downloads whenever local_version differs; stale_only downloads only when server_version - local_version exceeds download_stale_threshold",
        },
        "final_server_version": int(server_version),
        "checkpoint_versions_kept": sorted(int(v) for v in checkpoint_history.keys()),
    }
    return server_backbone, metrics, progress, session1_head_states, availability_trace


# =============================================================================
# Subject splitting and evaluation
# =============================================================================
def choose_train_holdout_subjects(all_subjects: List[str], args, protocol: str, split_iteration: int) -> Tuple[List[str], List[str]]:
    if args.sub:
        one = bc.resolve_subject_arg(args.sub, all_subjects)
        return [one], []
    requested = list(all_subjects) if args.all_subjects else list(all_subjects[:1])
    manual_holdout = parse_subject_csv(args.skip_train_subjects, requested)
    if protocol == "no_split" or bool(args.no_subject_split):
        return requested, []
    if manual_holdout:
        holdout = [s for s in requested if s in set(manual_holdout)]
        train = [s for s in requested if s not in set(holdout)]
        return train, holdout
    rng = random.Random(int(args.split_seed) + int(split_iteration) - 1)
    shuffled = list(requested)
    rng.shuffle(shuffled)
    if int(args.datasetId) == 1 and bool(args.openbmi_protocol):
        n_train = min(int(args.phase1_train_n), len(shuffled))
        train = shuffled[:n_train]
        holdout = shuffled[n_train:n_train + int(args.phase2_holdout_n)]
    else:
        n_hold = min(int(args.bci_holdout_n), max(0, len(shuffled) - 1))
        holdout = shuffled[:n_hold]
        train = shuffled[n_hold:n_hold + int(args.bci_phase1_train_n)] if int(args.bci_phase1_train_n) > 0 else shuffled[n_hold:]
    if not train:
        raise RuntimeError("No training subjects selected")
    return sorted(train), sorted(holdout)


def evaluate_subjects(
    eval_subjects: List[str],
    backbone_state: Dict[str, torch.Tensor],
    init_state: Dict[str, torch.Tensor],
    subj_data: Dict[str, Dict[str, bc.eegDataset]],
    net_ctor,
    model_args: dict,
    args,
    device: torch.device,
    expects_bands: bool,
    preload: bool,
    personalization: str,
    protocol: str,
    subject_head_states: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> Tuple[Dict[str, Any], List[dict]]:
    calib_sizes = parse_int_csv(args.calib_sizes)
    subject_head_states = subject_head_states or {}
    split_policy = str(getattr(args, "session2_split_policy", "chronological"))
    if split_policy not in ("chronological", "random_per_class"):
        raise ValueError(f"Unsupported session2_split_policy: {split_policy}")

    out = {}
    rows = []
    for si, s in enumerate(eval_subjects):
        s2_full = subj_data[s]["s2"]
        subject_head_state = subject_head_states.get(s, None)
        head_init_source = "session1_trained_local_head" if subject_head_state is not None else "global_initial_head_fallback"
        personalization_reference_state = make_personalization_reference_state(init_state, subject_head_state)

        if split_policy == "chronological":
            calib_pool, fixed_test, split_info = split_session2_fixed_calib_test(
                s2_full,
                fixed_test_trials=int(args.fixed_test_trials),
                calib_pool_trials=int(args.calib_pool_trials),
                preload=preload,
            )
            split_info["session2_split_policy"] = "chronological"
            split_info["random_repeats_used"] = False
            repeats_for_subject = 1
        else:
            calib_pool, fixed_test = None, None
            split_info = {
                "session2_split_policy": "random_per_class",
                "split_policy": "random_k_per_class_from_full_session2_remaining_test",
                "session2_total_trials": int(len(s2_full.labels)),
                "requested_repeats": int(args.repeats),
                "calib_pool_trials_arg_ignored": int(args.calib_pool_trials),
                "fixed_test_trials_arg_ignored": int(args.fixed_test_trials),
                "fixed_test_is_same_for_all_k": False,
                "fixed_test_is_same_for_all_repeats": False,
                "description": "For each k and repeat, all Session-2 trials are grouped by class; k trials/class are sampled for calibration and the remaining trials are used for testing.",
            }
            repeats_for_subject = max(1, int(args.repeats))

        subj_res = {
            "subject": s,
            "subject_report_id": bc.subject_to_report_id(s),
            "protocol": protocol,
            "session2_split": split_info,
            "head_initialization_source": head_init_source,
            "k_results": {},
        }

        for k in calib_sizes:
            if split_policy == "chronological":
                tr, va, cinfo = make_chrono_calib_train_val(calib_pool, int(k), preload=preload)
                if tr is None:
                    subj_res["k_results"][str(k)] = {"ok": False, "calibration": cinfo}
                    rows.append({
                        "subject": bc.subject_to_report_id(s), "raw_subject": s, "protocol": protocol,
                        "session2_split_policy": split_policy, "k": int(k), "repeat": 0, "ok": False,
                        "test_acc": None, "test_macro_f1": None, "reason": cinfo.get("reason"),
                    })
                    continue
                res = evaluate_personalization(
                    personalization,
                    backbone_state,
                    personalization_reference_state,
                    net_ctor,
                    model_args,
                    tr,
                    va,
                    fixed_test,
                    args,
                    device,
                    expects_bands,
                    seed=training_task_seed(
                        args,
                        int(resolve_seed_bundle(args).model_seed) + 1000 * (si + 1) + int(k),
                        "session2_personalization", s, int(k), 0,
                    ),
                )
                res["ok"] = True
                res["calibration"] = cinfo
                res["head_initialization_source"] = head_init_source
                res["session2_split_policy"] = split_policy
                res["repeat"] = 0
                subj_res["k_results"][str(k)] = res
                rows.append({
                    "subject": bc.subject_to_report_id(s),
                    "raw_subject": s,
                    "protocol": protocol,
                    "session2_split_policy": split_policy,
                    "k": int(k),
                    "repeat": 0,
                    "ok": True,
                    "personalization": personalization,
                    "head_initialization_source": head_init_source,
                    "calib_train_trials": cinfo["train_total"],
                    "calib_val_trials": cinfo["val_total"],
                    "calib_selected_trials": cinfo.get("selected_total"),
                    "test_trials": split_info["fixed_test_trials"],
                    "fixed_test_trials": split_info["fixed_test_trials"],
                    "test_acc": float(res["test"]["acc"]),
                    "test_macro_f1": float(res["test"].get("macro_f1", 0.0)),
                    "test_loss": float(res["test"]["loss"]),
                    "stage1_best_epoch": res.get("stage1_best_epoch"),
                })
            else:
                repeat_results = []
                accs, f1s = [], []
                for rep in range(repeats_for_subject):
                    tr, va, test_ds, cinfo = make_random_per_class_calib_train_val_test(
                        s2_full,
                        int(k),
                        int(rep),
                        int(resolve_seed_bundle(args).model_seed),
                        preload=preload,
                    )
                    if tr is None or test_ds is None:
                        repeat_results.append({"ok": False, "calibration": cinfo})
                        rows.append({
                            "subject": bc.subject_to_report_id(s), "raw_subject": s, "protocol": protocol,
                            "session2_split_policy": split_policy, "k": int(k), "repeat": int(rep), "ok": False,
                            "test_acc": None, "test_macro_f1": None, "reason": cinfo.get("reason"),
                        })
                        continue
                    res = evaluate_personalization(
                        personalization,
                        backbone_state,
                        personalization_reference_state,
                        net_ctor,
                        model_args,
                        tr,
                        va,
                        test_ds,
                        args,
                        device,
                        expects_bands,
                        seed=training_task_seed(
                            args,
                            int(resolve_seed_bundle(args).model_seed) + 5000 + int(rep) + 31 * int(k),
                            "session2_personalization", s, int(k), int(rep),
                        ),
                    )
                    res["ok"] = True
                    res["calibration"] = cinfo
                    res["head_initialization_source"] = head_init_source
                    res["session2_split_policy"] = split_policy
                    res["repeat"] = int(rep)
                    repeat_results.append(res)
                    accs.append(float(res["test"]["acc"]))
                    f1s.append(float(res["test"].get("macro_f1", 0.0)))
                    rows.append({
                        "subject": bc.subject_to_report_id(s),
                        "raw_subject": s,
                        "protocol": protocol,
                        "session2_split_policy": split_policy,
                        "k": int(k),
                        "repeat": int(rep),
                        "ok": True,
                        "personalization": personalization,
                        "head_initialization_source": head_init_source,
                        "calib_train_trials": cinfo["train_total"],
                        "calib_val_trials": cinfo["val_total"],
                        "calib_selected_trials": cinfo.get("selected_total"),
                        "test_trials": cinfo["test_total"],
                        "fixed_test_trials": cinfo["test_total"],
                        "test_acc": float(res["test"]["acc"]),
                        "test_macro_f1": float(res["test"].get("macro_f1", 0.0)),
                        "test_loss": float(res["test"]["loss"]),
                        "stage1_best_epoch": res.get("stage1_best_epoch"),
                    })
                subj_res["k_results"][str(k)] = {
                    "ok": bool(len(accs) > 0),
                    "session2_split_policy": split_policy,
                    "k_per_class": int(k),
                    "repeats_requested": int(repeats_for_subject),
                    "repeats_completed": int(len(accs)),
                    "mean_acc": float(np.mean(accs)) if accs else None,
                    "std_acc": float(np.std(accs)) if accs else None,
                    "mean_macro_f1": float(np.mean(f1s)) if f1s else None,
                    "std_macro_f1": float(np.std(f1s)) if f1s else None,
                    "head_initialization_source": head_init_source,
                    "repeats": repeat_results,
                }
        out[s] = subj_res
    return out, rows


def build_session2_partition_manifest(eval_results: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact, deterministic proof of Session-2 sample matching.

    Only split provenance and index hashes are retained; no EEG values are
    duplicated.  The resulting hash is compared between P3 and P5.
    """
    subjects = []
    split_keys = [
        "session2_split_policy", "split_policy", "test_mode",
        "session2_total_trials", "calibration_pool_trials", "fixed_test_trials",
        "calibration_pool_index_hash", "test_index_hash", "unused_index_hash",
    ]
    calib_keys = [
        "k_per_class", "repeat", "repeat_seed", "internal_val_seed",
        "selected_total", "train_total", "val_total", "test_total",
        "selected_index_hash_within_calibration_pool",
        "train_index_hash_within_calibration_pool",
        "val_index_hash_within_calibration_pool",
        "calibration_index_hash", "test_index_hash",
    ]
    for raw_subject, subject_result in sorted(eval_results.items(), key=lambda item: str(item[0])):
        split = subject_result.get("session2_split", {}) or {}
        subject_entry: Dict[str, Any] = {
            "raw_subject": str(raw_subject),
            "subject_report_id": str(subject_result.get("subject_report_id", raw_subject)),
            "session2_split": {k: split.get(k) for k in split_keys if k in split},
            "k_partitions": [],
        }
        for k_text, k_result in sorted(
            (subject_result.get("k_results", {}) or {}).items(),
            key=lambda item: int(item[0]),
        ):
            if "repeats" in k_result:
                repeats = k_result.get("repeats", []) or []
                for repeat_result in repeats:
                    calib = (repeat_result or {}).get("calibration", {}) or {}
                    subject_entry["k_partitions"].append({
                        "k": int(k_text),
                        "ok": bool((repeat_result or {}).get("ok", False)),
                        "calibration": {key: calib.get(key) for key in calib_keys if key in calib},
                    })
            else:
                calib = k_result.get("calibration", {}) or {}
                subject_entry["k_partitions"].append({
                    "k": int(k_text),
                    "ok": bool(k_result.get("ok", False)),
                    "calibration": {key: calib.get(key) for key in calib_keys if key in calib},
                })
        subjects.append(subject_entry)
    manifest = {"schema_version": 1, "subjects": subjects}
    manifest["partition_hash"] = json_sha256(manifest)
    return manifest

def summarize_eval_rows(rows: List[dict]) -> Dict[str, Any]:
    summary = {}
    ks = sorted(set(int(r["k"]) for r in rows if r.get("ok")))
    for k in ks:
        sub = [r for r in rows if r.get("ok") and int(r["k"]) == k]
        acc = [float(r["test_acc"]) for r in sub]
        f1 = [float(r["test_macro_f1"]) for r in sub]
        subjects = sorted(set(str(r.get("subject")) for r in sub))
        subject_acc_means = []
        subject_f1_means = []
        for subj in subjects:
            sr = [r for r in sub if str(r.get("subject")) == subj]
            if sr:
                subject_acc_means.append(float(np.mean([float(x["test_acc"]) for x in sr])))
                subject_f1_means.append(float(np.mean([float(x["test_macro_f1"]) for x in sr])))
        repeats = sorted(set(int(r.get("repeat", 0)) for r in sub))
        summary[str(k)] = {
            "n_subjects": int(len(subjects)),
            "n_subject_repeat_rows": int(len(sub)),
            "n_repeats_observed": int(len(repeats)),
            "mean_acc": float(np.mean(acc)) if acc else None,
            "std_acc": float(np.std(acc)) if acc else None,
            "min_acc_worst_user": float(np.min(subject_acc_means)) if subject_acc_means else (float(np.min(acc)) if acc else None),
            "max_acc": float(np.max(acc)) if acc else None,
            "mean_macro_f1": float(np.mean(f1)) if f1 else None,
            "std_macro_f1": float(np.std(f1)) if f1 else None,
            "min_macro_f1_worst_user": float(np.min(subject_f1_means)) if subject_f1_means else (float(np.min(f1)) if f1 else None),
            "mean_subject_acc": float(np.mean(subject_acc_means)) if subject_acc_means else None,
            "std_subject_acc": float(np.std(subject_acc_means)) if subject_acc_means else None,
            "mean_subject_macro_f1": float(np.mean(subject_f1_means)) if subject_f1_means else None,
            "std_subject_macro_f1": float(np.std(subject_f1_means)) if subject_f1_means else None,
        }
    return summary


# =============================================================================
# Reporting
# =============================================================================
def build_hyperparam_report(args, regime: str, protocol: str, comm_cfg: CommunicationConfig, train_subjects: List[str], eval_subjects: List[str]) -> dict:
    rconf = regime_to_config(regime)
    internal_policy_id = str(getattr(args, "policy_id", "") or "")
    paper_policy_id = INTERNAL_TO_PAPER_POLICY.get(internal_policy_id, internal_policy_id)
    return {
        "script": "nexus_mi.experiment",
        "publication_context": {
            "suite": str(getattr(args, "publication_suite", "") or ""),
            "policy_id": paper_policy_id,
            "internal_policy_id": internal_policy_id,
            "policy_name": str(getattr(args, "policy_name", "") or ""),
            "policy_label": str(getattr(args, "policy_label", "") or ""),
            "policy_family": str(getattr(args, "policy_family", "") or ""),
            "component_role": str(getattr(args, "component_role", "") or ""),
            "severity_name": str(getattr(args, "severity_name", "") or ""),
            "severity_description": str(getattr(args, "severity_description", "") or ""),
            "replicate_id": int(getattr(args, "replicate_id", 0)),
        },
        "regime": regime,
        "regime_definition": rconf,
        "protocol": protocol,
        "dataset": {"datasetId": int(args.datasetId), "dataset_name": dataset_name_from_id(args.datasetId), "network": args.network},
        "subjects": {
            "train_subjects": [bc.subject_to_report_id(s) for s in train_subjects],
            "eval_subjects": [bc.subject_to_report_id(s) for s in eval_subjects],
            "no_subject_split": bool(protocol == "no_split"),
        },
        "federated_learning": {
            "rounds": int(args.rounds),
            "local_epochs": int(args.local_epochs),
            "lr_local": float(args.lr_local),
            "agg_weighting": args.agg_weighting,
            "no_local_train_head": bool(args.no_local_train_head),
            "selection_policy": args.selection_policy,
            "selection_policy_definition": "all=non-adaptive all-gateway scheduling before availability; topk=diagnostic pre-availability budgeted selection; online_random=online-first random selection used only for the P5 component analysis; comm_aware=online-first priority scheduling by version lag, upload recency, buffered-update risk and local backbone lag",
            "max_selected_per_round": int(args.max_selected_per_round),
            "download_policy": str(comm_cfg.download_policy),
            "download_stale_threshold": int(comm_cfg.download_stale_threshold),
            "buffer_policy": str(comm_cfg.buffer_policy),
            "buffer_max_size": int(comm_cfg.buffer_max_size),
        },
        "embedding_pretrain": {
            "enabled": rconf["embedding_pretrain"] == "enabled",
            "lr_init": float(args.lr_init),
            "init_max_epochs": int(args.init_max_epochs),
            "init_patience": int(args.init_patience),
            "init_best_metric": args.init_best_metric,
            "session1_val_ratio": float(args.session1_val_ratio),
        },
        "personalization": {
            "mode": rconf["personalization"],
            "calib_sizes": parse_int_csv(args.calib_sizes),
            "lr_head": float(args.lr_head),
            "head_max_epochs": int(args.head_max_epochs),
            "head_patience": int(args.head_patience),
            "head_stage2": bool(args.head_stage2),
            "head_initialization_policy": "latest_session1_trained_local_head_with_global_initial_head_fallback",
            "training_rng_mode": str(getattr(args, "training_rng_mode", "legacy_stream")),
            "dropout_rng_control": "global RNG reset per local/personalization task" if str(getattr(args, "training_rng_mode", "legacy_stream")) == "per_task_seed" else "study process-global RNG stream",
            "backbone_parameters_frozen_during_session2_personalization": True,
        },
        "session2_evaluation": {
            "session2_split_policy": str(getattr(args, "session2_split_policy", "chronological")),
            "fixed_test_trials": int(args.fixed_test_trials),
            "calib_pool_trials": int(args.calib_pool_trials),
            "calib_sizes": parse_int_csv(args.calib_sizes),
            "repeats_requested": int(args.repeats),
            "chronological_selection": "first_k_per_class_from_chronological_calibration_pool",
            "chronological_test": "if fixed_test_trials > 0: last fixed_test_trials; else: all remaining trials after calibration pool",
            "random_per_class_selection": "for each k and repeat: group full Session-2 by class, shuffle each class, use k/class for calibration, and use remaining trials for test",
            "repeats_used": int(args.repeats) if str(getattr(args, "session2_split_policy", "chronological")) == "random_per_class" else 1,
            "no_random_repeats_in_no_split": str(getattr(args, "session2_split_policy", "chronological")) == "chronological",
        },
        "communication": {
            **asdict(comm_cfg),
            "byte_accounting_mode": str(
                getattr(args, "communication_byte_accounting", "legacy_runtime_metadata")
            ),
            "upload_metadata_counted": (
                "subject_id, trained_from_version, produced_round, optional perf_score"
                if str(getattr(args, "communication_byte_accounting", "legacy_runtime_metadata")) == "protocol_metadata"
                else "runtime metadata including produced_wallclock_time"
            ),
        },
        "performance_weighting": {
            "perf_weight_metric": args.perf_weight_metric,
            "perf_weight_method": args.perf_weight_method,
            "perf_weight_alpha": float(args.perf_weight_alpha),
            "perf_weight_eps": float(args.perf_weight_eps),
            "perf_weight_with_samples": bool(args.perf_weight_with_samples),
        },
        "runtime": {
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "pin_memory": bool(args.pin_memory),
            "fast_cudnn": bool(args.fast_cudnn),
            "preload": not bool(args.no_preload),
            "training_rng_mode": str(getattr(args, "training_rng_mode", "legacy_stream")),
        },
    }


def save_reports(
    run_dir: str,
    hyper: dict,
    results: dict,
    eval_rows: List[dict],
    system_metrics: dict,
    comm_progress: List[dict],
    availability_trace: Dict[str, Any],
) -> None:
    save_yaml(os.path.join(run_dir, "run_hyperparams.yaml"), hyper)
    save_json(os.path.join(run_dir, "results.json"), results)
    write_rows_csv(os.path.join(run_dir, "results_summary.csv"), eval_rows)
    save_json(os.path.join(run_dir, "system_metrics.json"), system_metrics)
    save_json(os.path.join(run_dir, "comm_progress.json"), comm_progress)
    save_json(os.path.join(run_dir, "availability_trace.json"), availability_trace)
    save_availability_trace_csv(os.path.join(run_dir, "availability_trace.csv"), availability_trace)

    # Communication round CSV.
    round_rows = []
    for r in comm_progress:
        row = {
            k: v for k, v in r.items()
            if k not in ("delay_rounds", "delay_seconds", "staleness", "staleness_accepted", "staleness_dropped", "scheduler")
        }
        sched = r.get("scheduler", {}) or {}
        row["scheduler_policy"] = sched.get("policy")
        row["scheduler_effective_policy"] = sched.get("effective_policy")
        row["scheduler_k_requested"] = sched.get("k_requested")
        row["scheduler_k_effective"] = sched.get("k_effective")
        row["scheduler_buffered_clients_before_selection"] = sched.get("n_buffered_clients_before_selection", 0)
        row["delay_rounds_count"] = len(r.get("delay_rounds", []))
        row["staleness_count"] = len(r.get("staleness_accepted", []))
        row["staleness_all_admission_attempts_count"] = len(r.get("staleness", []))
        row["staleness_dropped_count"] = len(r.get("staleness_dropped", []))
        round_rows.append(row)
    write_rows_csv(os.path.join(run_dir, "communication_rounds.csv"), round_rows)

    subject_rows = list(system_metrics.get("per_subject_comm", {}).values())
    write_rows_csv(os.path.join(run_dir, "communication_subjects.csv"), subject_rows)
    group_row = {
        "group": "shared",
        "subjects": ";".join(hyper["subjects"]["train_subjects"]),
        **system_metrics.get("ongoing", {}),
        **system_metrics.get("total_communication", {}),
    }
    write_rows_csv(os.path.join(run_dir, "communication_groups.csv"), [group_row])

    communication_report = {
        "definitions": {
            "download": "server-to-client collaborative backbone state only",
            "upload": "client-to-server backbone delta plus metadata only",
            "accepted_updates": "uploaded updates that passed checkpoint and staleness admission",
            "applied_updates": "accepted updates included in aggregation; every accepted update is applied",
            "accepted_but_superseded_updates": "retained field; always zero because every accepted update is applied",
            "topk_selection": "optional diagnostic communication-budget scheduler: buffered/staleness-risk clients first, then least-recently-selected clients; availability is applied after selection",
            "comm_aware_selection": "paper communication-aware scheduler: samples availability first, selects only online clients for immediate upload, and prioritizes version lag, upload lag, pending buffers and local staleness",
            "buffer_policy": "none drops offline update opportunities; fifo keeps a bounded queue and drops oldest on overflow; latest keeps only the newest pending update",
            "stale_policy_accept_all": "accepts delayed updates whenever their trained_from checkpoint is still available",
            "download_policy_stale_only": "avoids a server-to-client download unless the client's local backbone version is more than download_stale_threshold behind the server",
            "session2_split_policy": "chronological uses the paper fixed calibration/test split; random_per_class is an optional diagnostic k-shot split with the remaining Session-2 trials as test",
            "fixed_test_set": "for chronological only: fixed Session-2 test partition, identical across k values; uses last fixed_test_trials when fixed_test_trials>0, otherwise all remaining trials after the calibration pool",
        },
        "hyperparams": hyper,
        "system_metrics": system_metrics,
        "communication_rates": system_metrics.get("communication_rates", {}),
        "session1_head_initialization": system_metrics.get("session1_head_initialization", {}),
    }
    save_json(os.path.join(run_dir, "communication_report.json"), communication_report)

    quick = {
        "overview": {
            "regime": hyper["regime"],
            "protocol": hyper["protocol"],
            "dataset": hyper["dataset"],
            "communication": hyper["communication"],
            "personalization": hyper["personalization"],
        },
        "eval_summary_by_k": results.get("eval_summary_by_k", {}),
        "communication_headline": {
            **system_metrics.get("ongoing", {}),
            **system_metrics.get("total_communication", {}),
            "delay_rounds_mean": system_metrics.get("delay_summary", {}).get("mean_rounds", 0.0),
            "delay_rounds_max": system_metrics.get("delay_summary", {}).get("max_rounds", 0),
            "staleness_mean": system_metrics.get("staleness_summary", {}).get("mean", 0.0),
            "staleness_max": system_metrics.get("staleness_summary", {}).get("max", 0),
        },
    }
    save_json(os.path.join(run_dir, "quick_results_summary.json"), quick)
    quick_rows = []
    for k, v in results.get("eval_summary_by_k", {}).items():
        quick_rows.append({"k": int(k), **v})
    write_rows_csv(os.path.join(run_dir, "quick_results_summary.csv"), quick_rows)
    with open(os.path.join(run_dir, "quick_results_summary.txt"), "w") as f:
        f.write("Quick Results Summary\n")
        f.write("=====================\n")
        f.write(f"Regime: {hyper['regime']}\n")
        f.write(f"Protocol: {hyper['protocol']}\n")
        f.write(f"Communication: {hyper['communication']['label']} requested={hyper['communication']['requested']} effective={hyper['communication']['effective']}\n")
        f.write(f"Personalization: {hyper['personalization']['mode']}\n")
        f.write(f"Head initialization: {hyper['personalization'].get('head_initialization_policy')}\n")
        f.write("\nEvaluation by k\n")
        for k, v in results.get("eval_summary_by_k", {}).items():
            f.write(f"  k={k}: mean_acc={v.get('mean_acc')} std_acc={v.get('std_acc')} worst_acc={v.get('min_acc_worst_user')} mean_macro_f1={v.get('mean_macro_f1')}\n")
        f.write("\nCommunication headline\n")
        for k, v in quick["communication_headline"].items():
            f.write(f"  {k}: {v}\n")


# =============================================================================
# Single run and suites
# =============================================================================
def run_single(
    args,
    regime: str,
    protocol: str,
    comm_label: Optional[str] = None,
    split_iteration: int = 1,
    initialization_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    seed_bundle = resolve_seed_bundle(args)
    bc.set_seed(
        int(seed_bundle.model_seed) + int(split_iteration) - 1,
        deterministic=not bool(args.fast_cudnn),
    )
    dataset_name = dataset_name_from_id(int(args.datasetId))
    cache = initialization_cache if initialization_cache is not None else {}

    if str(getattr(args, "session2_split_policy", "chronological")) == "chronological" and int(args.calib_pool_trials) <= 0:
        raise ValueError("--calib-pool-trials must be > 0 for --session2-split-policy chronological.")
    if int(args.fixed_test_trials) < 0:
        raise ValueError("--fixed-test-trials must be >= 0. Use 0 to test on all remaining Session-2 trials after calibration.")
    if int(args.repeats) <= 0:
        raise ValueError("--repeats must be >= 1. It is used by random_per_class and ignored by chronological.")

    if comm_label == "comm_off":
        args.comm_sim = False
    elif comm_label == "comm_on":
        args.comm_sim = True
    comm_cfg = build_comm_config(args, force_label=comm_label)
    run_dir = make_run_dir(dataset_name, str(args.exp_name), args.network, regime, protocol, comm_cfg.label)
    started_at = datetime.now().isoformat()
    save_yaml(os.path.join(run_dir, "run_start_config.yaml"), {
        "status": "started",
        "started_at": started_at,
        "run_dir": run_dir,
        "dataset": dataset_name,
        "datasetId": int(args.datasetId),
        "network": str(args.network),
        "regime": str(regime),
        "protocol": str(protocol),
        "split_iteration": int(split_iteration),
        "communication": asdict(comm_cfg),
        "seed_provenance": asdict(seed_bundle),
        "args": vars(args),
        "note": "Written before dataset loading/pretraining so an interrupted run remains visible and auditable.",
    })
    print(
        f"[RUN-START] regime={regime} protocol={protocol} comm={comm_cfg.label} "
        f"model_seed={seed_bundle.model_seed} trace_seed={seed_bundle.trace_seed} run_dir={run_dir}",
        flush=True,
    )

    net_ctor = bc.resolve_network_ctor(args.network)
    model_args = bc.get_model_args(int(args.datasetId), args.network)
    expects_bands = False
    device = bc.get_device(int(args.nGPU))
    preload = not bool(args.no_preload)

    full_ds = load_existing_preprocessed_dataset(dataset_name)
    all_subs = all_subjects_from_labels(full_ds.labels)
    train_subjects, holdout_subjects = choose_train_holdout_subjects(all_subs, args, protocol, split_iteration)
    eval_subjects = train_subjects if protocol == "no_split" else holdout_subjects
    if not eval_subjects:
        eval_subjects = train_subjects

    needed_subjects = sorted(set(train_subjects + eval_subjects))
    subj_data = build_subject_data(full_ds, needed_subjects, preload=preload)
    train_subjects = [s for s in train_subjects if s in subj_data]
    eval_subjects = [s for s in eval_subjects if s in subj_data]
    if not train_subjects or not eval_subjects:
        raise RuntimeError("No usable train/eval subjects after session split")

    init_cache_key = "init::" + json_sha256({
        "datasetId": int(args.datasetId),
        "network": str(args.network),
        "model_args": model_args,
        "model_seed": int(seed_bundle.model_seed),
    })
    if init_cache_key in cache:
        cached_init = cache[init_cache_key]
        init_state = clone_state(cached_init["state"])
        restore_global_rng_state(cached_init["post_rng_state"])
        init_cache_hit = True
    else:
        # The run-level global seed was reset immediately before this function's
        # work, so model construction is deterministic for a fixed model seed.
        init_model = net_ctor(**model_args)
        init_state = clone_state(init_model.state_dict())
        cache[init_cache_key] = {
            "state": clone_state(init_state),
            # A cache hit must restore this state; otherwise skipping model
            # construction shifts dropout/global RNG streams relative to an
            # independently executed matched policy run.
            "post_rng_state": capture_global_rng_state(),
        }
        init_cache_hit = False
    backbone_keys = bc.backbone_keys_from_state(init_state)
    init_backbone = backbone_only(init_state, backbone_keys)

    rconf = regime_to_config(regime)
    pretrain_log: Dict[str, Any] = {"enabled": False, "cache_hit": False}
    if rconf["embedding_pretrain"] == "enabled":
        pretrain_cache_key = "eib::" + json_sha256({
            "datasetId": int(args.datasetId),
            "network": str(args.network),
            "model_seed": int(seed_bundle.model_seed),
            "train_subjects": [str(s) for s in train_subjects],
            "session1_val_ratio": float(args.session1_val_ratio),
            "lr_init": float(args.lr_init),
            "init_max_epochs": int(args.init_max_epochs),
            "init_patience": int(args.init_patience),
            "init_best_metric": str(args.init_best_metric),
            "batch_size": int(args.batch_size),
            "no_local_train_head": bool(args.no_local_train_head),
            "training_rng_mode": str(getattr(args, "training_rng_mode", "legacy_stream")),
            "fast_cudnn": bool(args.fast_cudnn),
        })
        if pretrain_cache_key in cache:
            cached = cache[pretrain_cache_key]
            initial_backbone = clone_state(cached["backbone"])
            pretrain_log = copy.deepcopy(cached["log"])
            restore_global_rng_state(cached["post_rng_state"])
            pretrain_log["cache_hit"] = True
        else:
            initial_backbone, pretrain_log = pretrain_pooled_session1_backbone(
                train_subjects,
                subj_data,
                init_state,
                backbone_keys,
                net_ctor,
                model_args,
                args,
                device,
                expects_bands,
                preload,
            )
            pretrain_log["cache_hit"] = False
            cache[pretrain_cache_key] = {
                "backbone": clone_state(initial_backbone),
                "log": copy.deepcopy(pretrain_log),
                # Reusing EIB pretraining must also restore the exact RNG state
                # that a second identical pretraining execution would leave.
                "post_rng_state": capture_global_rng_state(),
            }
    else:
        initial_backbone = clone_state(init_backbone)

    state_hashes = {
        "initial_full_model_sha256": state_sha256(init_state),
        "initial_random_backbone_sha256": state_sha256(init_backbone),
        "collaborative_start_backbone_sha256": state_sha256(initial_backbone),
        "federated_start_rng_state_sha256": json_sha256(capture_global_rng_state()),
        "eib_pretrained_backbone_sha256": state_sha256(initial_backbone)
        if rconf["embedding_pretrain"] == "enabled"
        else None,
        "initialization_cache_hit": bool(init_cache_hit),
        "eib_pretraining_cache_hit": bool(pretrain_log.get("cache_hit", False)),
    }

    final_backbone, system_metrics, comm_progress, session1_head_states, availability_trace = run_federated_backbone_training(
        train_subjects,
        subj_data,
        initial_backbone,
        init_state,
        backbone_keys,
        net_ctor,
        model_args,
        args,
        device,
        expects_bands,
        comm_cfg,
    )
    state_hashes["final_collaborative_backbone_sha256"] = state_sha256(final_backbone)
    if bool(getattr(args, "save_checkpoints", False)):
        checkpoint_bundle = {
            "schema_version": 1,
            "seed_provenance": asdict(seed_bundle),
            "state_hashes": state_hashes,
            "initial_full_model_state": clone_state(init_state),
            "initial_random_backbone_state": clone_state(init_backbone),
            "collaborative_start_backbone_state": clone_state(initial_backbone),
            "eib_pretrained_backbone_state": clone_state(initial_backbone)
            if rconf["embedding_pretrain"] == "enabled"
            else None,
            "final_collaborative_backbone_state": clone_state(final_backbone),
        }
        torch.save(checkpoint_bundle, os.path.join(run_dir, "checkpoint_bundle.pt"))
    system_metrics["seed_provenance"] = asdict(seed_bundle)
    system_metrics["state_hashes"] = state_hashes
    system_metrics["communication_byte_accounting"] = {
        "mode": str(getattr(args, "communication_byte_accounting", "legacy_runtime_metadata")),
        "decimal_mb_definition": "1 MB = 1,000,000 bytes",
        "binary_mib_definition": "1 MiB = 1,048,576 bytes",
        "runtime_wallclock_timestamp_counted_in_upload_bytes": (
            str(getattr(args, "communication_byte_accounting", "legacy_runtime_metadata"))
            == "legacy_runtime_metadata"
        ),
    }
    system_metrics["runtime_environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "device": str(device),
    }

    eval_results, eval_rows = evaluate_subjects(
        eval_subjects,
        final_backbone,
        init_state,
        subj_data,
        net_ctor,
        model_args,
        args,
        device,
        expects_bands,
        preload,
        rconf["personalization"],
        protocol,
        session1_head_states,
    )
    session2_partition_manifest = build_session2_partition_manifest(eval_results)
    results = {
        "regime": regime,
        "protocol": protocol,
        "run_dir": run_dir,
        "train_subjects": [bc.subject_to_report_id(s) for s in train_subjects],
        "eval_subjects": [bc.subject_to_report_id(s) for s in eval_subjects],
        "seed_provenance": asdict(seed_bundle),
        "state_hashes": state_hashes,
        "availability_trace_hash": availability_trace["trace_hash"],
        "group_assignment_hash": availability_trace["group_assignment_hash"],
        "session2_partition_hash": session2_partition_manifest["partition_hash"],
        "session2_partition_manifest": session2_partition_manifest,
        "pretrain_log": pretrain_log,
        "per_subject": eval_results,
        "eval_summary_by_k": summarize_eval_rows(eval_rows),
    }
    hyper = build_hyperparam_report(args, regime, protocol, comm_cfg, train_subjects, eval_subjects)
    hyper["seed_provenance"] = asdict(seed_bundle)
    hyper["state_hashes"] = state_hashes
    hyper["availability_trace_hash"] = availability_trace["trace_hash"]
    hyper["group_assignment_hash"] = availability_trace["group_assignment_hash"]
    hyper["session2_partition_hash"] = session2_partition_manifest["partition_hash"]
    save_reports(run_dir, hyper, results, eval_rows, system_metrics, comm_progress, availability_trace)
    save_json(os.path.join(run_dir, "session2_partition_manifest.json"), session2_partition_manifest)

    completed_at = datetime.now().isoformat()
    completion = {
        "status": "completed",
        "started_at": started_at,
        "completed_at": completed_at,
        "run_dir": run_dir,
        "dataset": dataset_name,
        "regime": regime,
        "policy_id": INTERNAL_TO_PAPER_POLICY.get(str(getattr(args, "policy_id", "")), str(getattr(args, "policy_id", ""))),
        "internal_policy_id": str(getattr(args, "policy_id", "")),
        "publication_suite": str(getattr(args, "publication_suite", "") or ""),
        "replicate_id": int(getattr(args, "replicate_id", 0)),
        "trace_hash": availability_trace["trace_hash"],
        "initial_backbone_hash": state_hashes["collaborative_start_backbone_sha256"],
        "final_backbone_hash": state_hashes["final_collaborative_backbone_sha256"],
    }
    save_json(os.path.join(run_dir, "completion_status.json"), completion)
    print(f"[DONE] regime={regime} protocol={protocol} comm={comm_cfg.label} run_dir={run_dir}")

    run_config = {
        "datasetId": int(args.datasetId),
        "dataset_name": dataset_name,
        "network": str(args.network),
        "session2_split_policy": str(getattr(args, "session2_split_policy", "chronological")),
        "calib_pool_trials": int(args.calib_pool_trials),
        "fixed_test_trials": int(args.fixed_test_trials),
        "calib_sizes": str(args.calib_sizes),
        "repeats": int(args.repeats),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "training_rng_mode": str(getattr(args, "training_rng_mode", "legacy_stream")),
        "communication_byte_accounting": str(
            getattr(args, "communication_byte_accounting", "legacy_runtime_metadata")
        ),
        "seed": int(args.seed),
        "model_seed": int(seed_bundle.model_seed),
        "group_seed": int(seed_bundle.group_seed),
        "trace_seed": int(seed_bundle.trace_seed),
        "scheduler_seed": int(seed_bundle.scheduler_seed),
        "trace_hash": availability_trace["trace_hash"],
        "group_assignment_hash": availability_trace["group_assignment_hash"],
        "session2_partition_hash": session2_partition_manifest["partition_hash"],
        "initial_full_model_sha256": state_hashes["initial_full_model_sha256"],
        "collaborative_start_backbone_sha256": state_hashes["collaborative_start_backbone_sha256"],
        "federated_start_rng_state_sha256": state_hashes["federated_start_rng_state_sha256"],
        "eib_pretrained_backbone_sha256": state_hashes["eib_pretrained_backbone_sha256"],
        "final_collaborative_backbone_sha256": state_hashes["final_collaborative_backbone_sha256"],
        "checkpoint_bundle_saved": bool(getattr(args, "save_checkpoints", False)),
        "replicate_id": int(getattr(args, "replicate_id", 0)),
        "exp_name": str(args.exp_name),
        "comm_profile": str(args.comm_profile),
        "online_prob": float(args.online_prob),
        "online_prob_good": float(args.online_prob_good),
        "online_prob_med": float(args.online_prob_med),
        "online_prob_bad": float(args.online_prob_bad),
        "profile_frac_good": float(args.profile_frac_good),
        "profile_frac_med": float(args.profile_frac_med),
        "max_selected_per_round": int(args.max_selected_per_round),
        "buffer_max_size": int(args.buffer_max_size),
        "stale_threshold": int(args.stale_threshold),
        "checkpoint_retention_margin": int(args.checkpoint_retention_margin),
        "download_stale_threshold": int(args.download_stale_threshold),
        "severity_name": str(getattr(args, "severity_name", "")),
        "severity_description": str(getattr(args, "severity_description", "")),
        "expected_mean_online_prob": float(
            getattr(
                args,
                "expected_mean_online_prob",
                expected_hetero_online_probability(args)
                if str(args.comm_profile) == "hetero"
                else float(args.online_prob),
            )
        ),
        "policy_id": str(getattr(args, "policy_id", "")),
        "policy_name": str(getattr(args, "policy_name", "")),
        "policy_label": str(getattr(args, "policy_label", "")),
        "policy_family": str(getattr(args, "policy_family", "")),
        "component_role": str(getattr(args, "component_role", "")),
    }
    return {
        "run_dir": run_dir,
        "regime": regime,
        "protocol": protocol,
        "comm": comm_cfg.label,
        "summary": results["eval_summary_by_k"],
        "eval_rows": eval_rows,
        "system_metrics": system_metrics,
        "run_config": run_config,
    }


def build_suite_summary_base_row(r: Dict[str, Any], suite: str) -> Dict[str, Any]:
    """Create the run-level part of the expanded suite summary.

    The suite summary is intentionally redundant: it repeats core training,
    split, scheduler, buffering, admission, and communication information in
    every k-row so downstream analysis can be performed directly from
    suite_summary.csv without opening each run directory.
    """
    cfg = r.get("run_config", {}) or {}
    sm = r.get("system_metrics", {}) or {}
    comm_cfg = sm.get("communication_config", {}) or {}
    total = sm.get("total_communication", {}) or {}
    ongoing = sm.get("ongoing", {}) or {}
    rates = sm.get("communication_rates", {}) or {}
    delay = sm.get("delay_summary", {}) or {}
    stale = sm.get("staleness_summary", {}) or {}
    head = sm.get("session1_head_initialization", {}) or {}

    row = {
        "suite": suite,
        "datasetId": cfg.get("datasetId"),
        "dataset_name": cfg.get("dataset_name"),
        "network": cfg.get("network"),
        "regime": r.get("regime"),
        "protocol": r.get("protocol"),
        "comm": r.get("comm"),
        "run_dir": r.get("run_dir"),
        "exp_name": cfg.get("exp_name"),
        "replicate_id": cfg.get("replicate_id"),
        "seed": cfg.get("seed"),
        "model_seed": cfg.get("model_seed"),
        "group_seed": cfg.get("group_seed"),
        "trace_seed": cfg.get("trace_seed"),
        "scheduler_seed": cfg.get("scheduler_seed"),
        "trace_hash": cfg.get("trace_hash"),
        "group_assignment_hash": cfg.get("group_assignment_hash"),
        "session2_partition_hash": cfg.get("session2_partition_hash"),
        "initial_full_model_sha256": cfg.get("initial_full_model_sha256"),
        "collaborative_start_backbone_sha256": cfg.get("collaborative_start_backbone_sha256"),
        "federated_start_rng_state_sha256": cfg.get("federated_start_rng_state_sha256"),
        "eib_pretrained_backbone_sha256": cfg.get("eib_pretrained_backbone_sha256"),
        "final_collaborative_backbone_sha256": cfg.get("final_collaborative_backbone_sha256"),
        "checkpoint_bundle_saved": cfg.get("checkpoint_bundle_saved"),
        "rounds": cfg.get("rounds"),
        "local_epochs": cfg.get("local_epochs"),
        "session2_split_policy": cfg.get("session2_split_policy"),
        "calib_sizes": cfg.get("calib_sizes"),
        "calib_pool_trials": cfg.get("calib_pool_trials"),
        "fixed_test_trials": cfg.get("fixed_test_trials"),
        "repeats": cfg.get("repeats"),
        "comm_profile": cfg.get("comm_profile"),
        "online_prob": cfg.get("online_prob"),
        "online_prob_good": cfg.get("online_prob_good"),
        "online_prob_med": cfg.get("online_prob_med"),
        "online_prob_bad": cfg.get("online_prob_bad"),
        "profile_frac_good": cfg.get("profile_frac_good"),
        "profile_frac_med": cfg.get("profile_frac_med"),
        "selection_policy": comm_cfg.get("selection_policy", ""),
        "max_selected_per_round": comm_cfg.get("max_selected_per_round", cfg.get("max_selected_per_round")),
        "buffer_policy": comm_cfg.get("buffer_policy", ""),
        "buffer_policy_arg": comm_cfg.get("buffer_policy_arg", ""),
        "buffer_max_size": comm_cfg.get("buffer_max_size", cfg.get("buffer_max_size")),
        "buffering_enabled": comm_cfg.get("buffering_enabled"),
        "stale_policy": comm_cfg.get("stale_policy", ""),
        "stale_threshold": comm_cfg.get("stale_threshold", cfg.get("stale_threshold")),
        "stale_gamma": comm_cfg.get("stale_gamma"),
        "download_policy": comm_cfg.get("download_policy", ""),
        "download_stale_threshold": comm_cfg.get("download_stale_threshold", cfg.get("download_stale_threshold")),
        "severity_name": cfg.get("severity_name", ""),
        "severity_description": cfg.get("severity_description", ""),
        "expected_mean_online_prob": cfg.get("expected_mean_online_prob"),
        "policy_id": cfg.get("policy_id", ""),
        "policy_name": cfg.get("policy_name", ""),
        "policy_label": cfg.get("policy_label", ""),
        "policy_family": cfg.get("policy_family", ""),
        "session1_head_available_subject_count": head.get("available_subject_count"),
        "session1_head_fallback_subject_count": head.get("fallback_subject_count"),
        "client_to_server_bytes": total.get("client_to_server_bytes"),
        "server_to_client_bytes": total.get("server_to_client_bytes"),
        "total_bytes": total.get("total_bytes"),
        "client_to_server_mb": total.get("client_to_server_mb"),
        "server_to_client_mb": total.get("server_to_client_mb"),
        "total_mb": total.get("total_mb"),
        "client_to_server_mib": total.get("client_to_server_mib"),
        "server_to_client_mib": total.get("server_to_client_mib"),
        "total_mib": total.get("total_mib"),
        "traffic_unit_definition": total.get("unit_definition"),
        "delay_count": delay.get("count"),
        "delay_rounds_mean": delay.get("mean_rounds"),
        "delay_rounds_max": delay.get("max_rounds"),
        "delay_seconds_mean": delay.get("mean_seconds"),
        "delay_seconds_max": delay.get("max_seconds"),
        "staleness_count": stale.get("count"),
        "staleness_mean": stale.get("mean"),
        "staleness_max": stale.get("max"),
    }

    count_keys = [
        "total_client_rounds", "selected_events", "unselected_events",
        "online_available_events", "offline_available_events",
        "selected_online_events", "selected_offline_events",
        "downloads", "initial_downloads", "sync_downloads", "download_opportunities", "download_avoided",
        "uploads", "uploads_from_fresh", "uploads_from_buffer",
        "buffered_new_payloads", "buffer_dropped_oldest", "buffer_overwritten_latest",
        "offline_deferred_buffered_payloads", "offline_selected_no_buffer_drops", "offline_unselected_no_buffer_drops",
        "accepted_updates", "accepted_from_buffer", "accepted_from_fresh",
        "applied_updates", "applied_from_buffer", "applied_from_fresh", "applied_update_bytes",
        "accepted_but_superseded_updates", "dropped_updates", "dropped_update_bytes",
        "stale_drop_updates", "checkpoint_missing_drop_updates",
        "pending_buffer_clients_end", "pending_buffer_payloads_end",
    ]
    for key in count_keys:
        row[key] = ongoing.get(key)

    for key, val in rates.items():
        row[key] = val
    return row




def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError, OverflowError):
        return None


def _delta(value: Any, baseline: Any) -> Optional[float]:
    v = _to_float_or_none(value)
    b = _to_float_or_none(baseline)
    if v is None or b is None:
        return None
    return v - b


def add_tradeoff_deltas(rows: List[dict], baseline_policy_id: str = "A3") -> List[dict]:
    """Return per-k tradeoff rows with deltas relative to the baseline policy.

    The severity sweep compares A3 against A5. A3 is the fixed-selection FIFO
    stale-drop policy, and A5 is the communication-aware gateway-scheduled FIFO
    stale-drop policy. Deltas are computed as row value minus the matching A3
    value for the same dataset, regime, link severity, and calibration size.
    """
    baseline_by_group = {}
    for row in rows:
        if str(row.get("policy_id", "")) != baseline_policy_id:
            continue
        key = (
            row.get("dataset_name"),
            row.get("regime"),
            row.get("severity_name"),
            row.get("session2_split_policy"),
            row.get("k"),
        )
        baseline_by_group[key] = row

    value_cols = [
        "mean_acc", "mean_macro_f1", "worst_acc", "worst_macro_f1",
        "client_to_server_mb", "server_to_client_mb",
        "drop_rate_over_uploads", "staleness_mean", "delay_rounds_mean",
        "online_selected_rate", "download_avoidance_rate_over_download_opportunities", "buffer_upload_fraction_over_uploads",
        "stale_drop_fraction_over_drops", "checkpoint_missing_fraction_over_drops",
    ]
    out = []
    for row in rows:
        new_row = dict(row)
        key = (
            row.get("dataset_name"),
            row.get("regime"),
            row.get("severity_name"),
            row.get("session2_split_policy"),
            row.get("k"),
        )
        base = baseline_by_group.get(key)
        new_row["delta_baseline_policy_id"] = baseline_policy_id
        new_row["delta_baseline_policy_label"] = base.get("policy_label") if base else ""
        for col in value_cols:
            new_row[f"baseline_{col}"] = base.get(col) if base else None
            new_row[f"delta_{col}_vs_{baseline_policy_id}"] = _delta(row.get(col), base.get(col) if base else None)
        out.append(new_row)
    return out


def aggregate_tradeoff_by_severity(delta_rows: List[dict], baseline_policy_id: str = "A3") -> List[dict]:
    """Aggregate severity-sweep tradeoff rows across calibration sizes.

    This file is intended for compact paper tables. It averages accuracy,
    macro-F1, communication volume, rates, delay, and staleness across the k
    values included in the run, then computes deltas relative to the matching
    A3 aggregate for the same dataset/regime/severity/split.
    """
    group_keys = [
        "suite", "datasetId", "dataset_name", "network", "regime", "protocol", "comm",
        "session2_split_policy", "calib_sizes", "comm_profile", "severity_name",
        "severity_description", "expected_mean_online_prob", "online_prob_good",
        "online_prob_med", "online_prob_bad", "profile_frac_good", "profile_frac_med",
        "policy_id", "policy_name", "policy_label", "policy_family", "selection_policy",
        "max_selected_per_round", "buffer_policy", "buffer_max_size", "stale_policy",
        "stale_threshold", "download_policy", "download_stale_threshold",
    ]
    value_cols = [
        "mean_acc", "mean_macro_f1", "worst_acc", "worst_macro_f1",
        "client_to_server_mb", "server_to_client_mb",
        "drop_rate_over_uploads", "staleness_mean", "delay_rounds_mean",
        "online_selected_rate", "download_avoidance_rate_over_download_opportunities", "buffer_upload_fraction_over_uploads",
        "stale_drop_fraction_over_drops", "checkpoint_missing_fraction_over_drops",
    ]
    grouped: Dict[Tuple[Any, ...], List[dict]] = {}
    for row in delta_rows:
        key = tuple(row.get(k) for k in group_keys)
        grouped.setdefault(key, []).append(row)

    aggs = []
    for key, group in grouped.items():
        first = group[0]
        out = {k: first.get(k) for k in group_keys}
        ks = []
        for r in group:
            try:
                ks.append(int(r.get("k")))
            except (TypeError, ValueError):
                pass
        out["k_values"] = ",".join(str(k) for k in sorted(set(ks)))
        out["n_k_values"] = len(set(ks))
        out["run_dir"] = first.get("run_dir")
        out["exp_name"] = first.get("exp_name")
        for col in value_cols:
            vals = [_to_float_or_none(r.get(col)) for r in group]
            vals = [v for v in vals if v is not None]
            out[f"avg_{col}"] = (sum(vals) / len(vals)) if vals else None
        aggs.append(out)

    base_by_group = {}
    for row in aggs:
        if str(row.get("policy_id", "")) != baseline_policy_id:
            continue
        key = (
            row.get("dataset_name"),
            row.get("regime"),
            row.get("severity_name"),
            row.get("session2_split_policy"),
        )
        base_by_group[key] = row

    for row in aggs:
        key = (
            row.get("dataset_name"),
            row.get("regime"),
            row.get("severity_name"),
            row.get("session2_split_policy"),
        )
        base = base_by_group.get(key)
        row["delta_baseline_policy_id"] = baseline_policy_id
        row["delta_baseline_policy_label"] = base.get("policy_label") if base else ""
        for col in value_cols:
            avg_col = f"avg_{col}"
            row[f"baseline_{avg_col}"] = base.get(avg_col) if base else None
            row[f"delta_{avg_col}_vs_{baseline_policy_id}"] = _delta(row.get(avg_col), base.get(avg_col) if base else None)
    return aggs


def parse_optional_seed_csv(value: Optional[str], expected_count: int, label: str) -> Optional[List[int]]:
    if value is None or str(value).strip() == "":
        return None
    seeds = parse_int_csv(str(value))
    if len(seeds) != int(expected_count):
        raise ValueError(f"--{label} must contain exactly {expected_count} comma-separated integers.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--{label} contains duplicate seeds; robustness replicates must be independently specified.")
    return seeds


def configure_primary_robustness_args(args):
    """Return a copy configured for the manuscript P3/P5 robustness protocol."""
    from .protocol import paper_protocol

    cfg = paper_protocol()
    fed = cfg["federated_training"]
    pre = cfg["embedding_pretrain"]
    s2 = cfg["session2"]
    pers = cfg["personalization"]
    comm = cfg["communication"]
    probs = comm["availability_probabilities"]
    fracs = comm["availability_group_fractions"]
    dataset_key = dataset_name_from_id(int(args.datasetId))
    dataset_cfg = cfg["dataset_defaults"][dataset_key]

    a = copy.deepcopy(args)
    a.run_suite = "none"
    a.all_subjects = True
    a.no_subject_split = True
    a.eval_protocol = "no_split"
    a.comm_sim = True
    a.comm_profile = str(comm["profile"])
    a.no_buffering = False
    a.availability_trace_file = ""
    # Robustness runs retain auditable initialization/final checkpoint bundles.
    a.save_checkpoints = True
    a.training_rng_mode = str(getattr(args, "robustness_rng_mode", "legacy_stream"))
    a.communication_byte_accounting = str(
        getattr(args, "robustness_byte_accounting", "legacy_runtime_metadata")
    )
    if a.training_rng_mode not in TRAINING_RNG_MODES:
        raise ValueError(f"Unsupported robustness RNG mode: {a.training_rng_mode}")
    if a.communication_byte_accounting not in COMMUNICATION_BYTE_ACCOUNTING_MODES:
        raise ValueError(
            f"Unsupported robustness byte-accounting mode: {a.communication_byte_accounting}"
        )
    if bool(getattr(args, "robustness_authoritative", True)):
        if a.training_rng_mode != "legacy_stream":
            raise ValueError(
                "The manuscript robustness study must use --robustness-rng-mode legacy_stream "
                "to preserve the study stochastic training execution."
            )
        if a.communication_byte_accounting != "legacy_runtime_metadata":
            raise ValueError(
                "The manuscript robustness study must use "
                "--robustness-byte-accounting legacy_runtime_metadata to preserve "
                "the study raw-byte accounting."
            )
        if str(a.network).lower() != "eegnet":
            raise ValueError("The manuscript robustness study uses eegNet; do not change the network.")

        a.online_prob_good = float(probs["high"])
        a.online_prob_med = float(probs["moderate"])
        a.online_prob_bad = float(probs["low"])
        a.profile_frac_good = float(fracs["high"])
        a.profile_frac_med = float(fracs["moderate"])
        a.rounds = int(fed["rounds"])
        a.local_epochs = int(fed["local_epochs"])
        a.lr_local = float(fed["learning_rate"])
        a.agg_weighting = "uniform"
        a.no_local_train_head = False
        a.batch_size = int(fed["batch_size"])
        a.session1_val_ratio = float(pre["session1_validation_ratio"])
        a.lr_init = float(pre["learning_rate"])
        a.init_max_epochs = int(pre["max_epochs"])
        a.init_patience = int(pre["patience"])
        a.init_best_metric = str(pre["best_metric"])
        a.lr_head = float(pers["learning_rate"])
        a.head_max_epochs = int(pers["max_epochs"])
        a.head_patience = int(pers["patience"])
        a.head_stage2 = False
        a.buffer_max_size = int(comm["buffer_max_size"])
        a.stale_threshold = int(comm["stale_threshold_versions"])
        a.checkpoint_retention_margin = int(comm["checkpoint_retention_margin_versions"])
        a.stale_gamma = 0.8  # inert under stale-drop admission; retained for run metadata compatibility
        a.download_stale_threshold = int(comm["download_threshold_versions"])
        a.calib_sizes = ",".join(str(int(v)) for v in pers["calibration_trials_per_class"])
        a.session2_split_policy = str(s2["split_policy"])
        a.fixed_test_trials = int(s2["fixed_test_trials_argument"])
        a.repeats = int(s2["repeats"])
        a.fast_cudnn = False
        a.num_workers = 0
        a.calib_pool_trials = int(dataset_cfg["session2_calibration_pool_trials"])
        a.max_selected_per_round = int(dataset_cfg["max_selected_per_round"])
    return a


def robustness_policy(policy_id: str) -> Dict[str, Any]:
    if policy_id not in ("P3", "P5"):
        raise KeyError(f"Unknown robustness policy: {policy_id}")
    policy = POLICIES[policy_id]
    names = {
        "P3": (
            "p3_nonadaptive_fifo_stale_drop_always_download",
            "P3: non-adaptive all-gateway scheduling, FIFO buffering, stale-drop admission, always-download synchronization",
        ),
        "P5": (
            "p5_comm_aware_fifo_stale_drop_lag_download",
            "P5: communication-aware online-priority scheduling, FIFO buffering, stale-drop admission, lag-aware downloads",
        ),
    }
    name, label = names[policy_id]
    return {
        "id": policy_id,
        "name": name,
        "label": label,
        "family": "principal_p3_p5_robustness",
        "selection_policy": policy.selection_policy,
        "buffer_policy": policy.buffer_policy,
        "stale_policy": policy.stale_policy,
        "download_policy": policy.download_policy,
    }


def subject_accuracy_means(eval_rows: List[dict], expected_ks: List[int]) -> Dict[str, float]:
    """Average each subject across the required k values with strict completeness checks."""
    expected = sorted(int(k) for k in expected_ks)
    grouped: Dict[str, Dict[int, List[float]]] = {}
    failed_rows = []
    for row in eval_rows:
        subject = str(row.get("subject"))
        k = int(row.get("k"))
        if not bool(row.get("ok")) or row.get("test_acc") is None:
            failed_rows.append({"subject": subject, "k": k, "reason": row.get("reason")})
            continue
        grouped.setdefault(subject, {}).setdefault(k, []).append(float(row["test_acc"]))
    if failed_rows:
        raise RuntimeError(f"Robustness accuracy output contains failed subject/k rows: {failed_rows[:10]}")
    if not grouped:
        raise RuntimeError("No valid subject-level accuracy rows were produced.")

    out: Dict[str, float] = {}
    for subject, by_k in sorted(grouped.items()):
        observed = sorted(by_k)
        if observed != expected:
            raise RuntimeError(
                f"Subject {subject} has calibration sizes {observed}; expected exactly {expected}."
            )
        # Robustness uses one chronological result per k.  The
        # averaging below still handles diagnostic repeats by averaging within
        # k first, then giving each calibration budget equal weight.
        per_k = [float(np.mean(by_k[k])) for k in expected]
        out[subject] = float(np.mean(per_k))
    return out


def validate_matched_pair(p3: Dict[str, Any], p5: Dict[str, Any]) -> Dict[str, Any]:
    c3, c5 = p3["run_config"], p5["run_config"]
    checks = {
        "same_dataset": c3.get("datasetId") == c5.get("datasetId"),
        "same_regime": p3.get("regime") == p5.get("regime"),
        "same_replicate": c3.get("replicate_id") == c5.get("replicate_id"),
        "same_model_seed": c3.get("model_seed") == c5.get("model_seed"),
        "same_training_rng_mode": c3.get("training_rng_mode") == c5.get("training_rng_mode"),
        "same_communication_byte_accounting": c3.get("communication_byte_accounting") == c5.get("communication_byte_accounting"),
        "same_group_seed": c3.get("group_seed") == c5.get("group_seed"),
        "same_trace_seed": c3.get("trace_seed") == c5.get("trace_seed"),
        "same_trace_hash": c3.get("trace_hash") == c5.get("trace_hash"),
        "same_group_assignment_hash": c3.get("group_assignment_hash") == c5.get("group_assignment_hash"),
        "same_session2_partition_hash": c3.get("session2_partition_hash") == c5.get("session2_partition_hash"),
        "same_initial_full_model_hash": c3.get("initial_full_model_sha256") == c5.get("initial_full_model_sha256"),
        "same_collaborative_start_backbone_hash": c3.get("collaborative_start_backbone_sha256") == c5.get("collaborative_start_backbone_sha256"),
        "same_federated_start_rng_state_hash": c3.get("federated_start_rng_state_sha256") == c5.get("federated_start_rng_state_sha256"),
    }
    if p3.get("regime") == "embedding_shared_head_only":
        checks["same_eib_pretrained_backbone_hash"] = (
            c3.get("eib_pretrained_backbone_sha256") == c5.get("eib_pretrained_backbone_sha256")
        )
    failed = [key for key, passed in checks.items() if not bool(passed)]
    if failed:
        raise RuntimeError(
            "Matched P3/P5 validation failed for "
            f"replicate={c3.get('replicate_id')} regime={p3.get('regime')}: {failed}"
        )
    return {
        "replicate_id": c3.get("replicate_id"),
        "regime": p3.get("regime"),
        "trace_hash": c3.get("trace_hash"),
        "group_assignment_hash": c3.get("group_assignment_hash"),
        "session2_partition_hash": c3.get("session2_partition_hash"),
        "initial_backbone_hash": c3.get("collaborative_start_backbone_sha256"),
        **checks,
        "all_checks_passed": True,
    }


def validate_cross_regime_communication(runs: List[Dict[str, Any]]) -> List[dict]:
    """Audit communication decisions for fixed replicate/policy across regimes."""
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for run in runs:
        cfg = run["run_config"]
        grouped.setdefault((int(cfg.get("replicate_id", 0)), str(cfg.get("policy_id"))), []).append(run)
    audit_rows = []
    decision_keys = [
        "selected_events", "unselected_events", "online_available_events", "offline_available_events",
        "selected_online_events", "selected_offline_events", "downloads", "sync_downloads",
        "download_opportunities", "download_avoided", "uploads", "uploads_from_fresh",
        "uploads_from_buffer", "accepted_updates", "dropped_updates", "stale_drop_updates",
        "checkpoint_missing_drop_updates", "applied_updates",
    ]
    for (replicate_id, policy_id), group in sorted(grouped.items()):
        if len(group) != 2:
            continue
        left, right = sorted(group, key=lambda r: r["regime"])
        lo = left["system_metrics"]["ongoing"]
        ro = right["system_metrics"]["ongoing"]
        mismatches = {key: [lo.get(key), ro.get(key)] for key in decision_keys if lo.get(key) != ro.get(key)}
        left_total = left["system_metrics"]["total_communication"]
        right_total = right["system_metrics"]["total_communication"]
        s2c_equal = left_total.get("server_to_client_bytes") == right_total.get("server_to_client_bytes")
        rates_equal = left["system_metrics"].get("communication_rates") == right["system_metrics"].get("communication_rates")
        stale_a = left["system_metrics"].get("staleness_summary", {}) or {}
        stale_b = right["system_metrics"].get("staleness_summary", {}) or {}
        staleness_equal = all(stale_a.get(k) == stale_b.get(k) for k in ("count", "mean", "max"))
        delay_a = left["system_metrics"].get("delay_summary", {}) or {}
        delay_b = right["system_metrics"].get("delay_summary", {}) or {}
        delay_rounds_equal = all(delay_a.get(k) == delay_b.get(k) for k in ("count", "mean_rounds", "max_rounds"))
        trace_equal = left["run_config"].get("trace_hash") == right["run_config"].get("trace_hash")
        group_equal = left["run_config"].get("group_assignment_hash") == right["run_config"].get("group_assignment_hash")
        byte_mode_a = str(left["run_config"].get("communication_byte_accounting", "legacy_runtime_metadata"))
        byte_mode_b = str(right["run_config"].get("communication_byte_accounting", "legacy_runtime_metadata"))
        byte_mode_equal = byte_mode_a == byte_mode_b
        c2s_equal = left_total.get("client_to_server_bytes") == right_total.get("client_to_server_bytes")
        row = {
            "replicate_id": replicate_id,
            "policy_id": policy_id,
            "regime_a": left["regime"],
            "regime_b": right["regime"],
            "decision_counts_match": len(mismatches) == 0,
            "communication_rates_match": bool(rates_equal),
            "accepted_staleness_match": bool(staleness_equal),
            "buffered_delay_rounds_match": bool(delay_rounds_equal),
            "trace_hash_match": bool(trace_equal),
            "group_assignment_hash_match": bool(group_equal),
            "server_to_client_bytes_match": bool(s2c_equal),
            "communication_byte_accounting_mode_a": byte_mode_a,
            "communication_byte_accounting_mode_b": byte_mode_b,
            "communication_byte_accounting_mode_match": bool(byte_mode_equal),
            "client_to_server_bytes_match": bool(c2s_equal),
            "client_to_server_byte_difference": int(right_total.get("client_to_server_bytes", 0)) - int(left_total.get("client_to_server_bytes", 0)),
            "decision_mismatches": mismatches,
        }
        failed = []
        if mismatches:
            failed.append("decision_counts")
        if not rates_equal:
            failed.append("communication_rates")
        if not staleness_equal:
            failed.append("accepted_staleness")
        if not delay_rounds_equal:
            failed.append("buffered_delay_rounds")
        if not trace_equal:
            failed.append("trace_hash")
        if not group_equal:
            failed.append("group_assignment_hash")
        if not s2c_equal:
            failed.append("server_to_client_bytes")
        if not byte_mode_equal:
            failed.append("communication_byte_accounting_mode")
        if byte_mode_a == "protocol_metadata" and not c2s_equal:
            failed.append("client_to_server_bytes")
        if failed:
            raise RuntimeError(
                f"Communication audit failed for replicate={replicate_id}, policy={policy_id}: "
                f"failed={failed}, decision_mismatches={mismatches}"
            )
        audit_rows.append(row)
    return audit_rows


def build_robustness_pair_outputs(runs: List[Dict[str, Any]]) -> Tuple[List[dict], List[dict], List[dict]]:
    by_key: Dict[Tuple[str, int, str], Dict[str, Dict[str, Any]]] = {}
    for run in runs:
        cfg = run["run_config"]
        key = (str(cfg.get("dataset_name")), int(cfg.get("replicate_id", 0)), str(run["regime"]))
        by_key.setdefault(key, {})[str(cfg.get("policy_id"))] = run

    pair_rows: List[dict] = []
    subject_rows: List[dict] = []
    validation_rows: List[dict] = []
    for (dataset_name, replicate_id, regime), policies in sorted(by_key.items()):
        if set(policies.keys()) != {"P3", "P5"}:
            raise RuntimeError(f"Incomplete robustness pair for {dataset_name}, replicate={replicate_id}, regime={regime}")
        p3, p5 = policies["P3"], policies["P5"]
        validation_rows.append(validate_matched_pair(p3, p5))
        expected_ks = parse_int_csv(str(p3["run_config"].get("calib_sizes", "15,20,30")))
        if expected_ks != parse_int_csv(str(p5["run_config"].get("calib_sizes", "15,20,30"))):
            raise RuntimeError(f"P3/P5 calibration-size settings differ for replicate={replicate_id}, regime={regime}")
        a3 = subject_accuracy_means(p3["eval_rows"], expected_ks)
        a5 = subject_accuracy_means(p5["eval_rows"], expected_ks)
        if set(a3) != set(a5):
            raise RuntimeError(f"P3/P5 subject sets differ for replicate={replicate_id}, regime={regime}")
        subjects = sorted(a3)
        for subject in subjects:
            subject_rows.append({
                "dataset_name": dataset_name,
                "replicate_id": replicate_id,
                "regime": regime,
                "subject": subject,
                "model_seed": p3["run_config"]["model_seed"],
                "trace_seed": p3["run_config"]["trace_seed"],
                "scheduler_seed": p5["run_config"]["scheduler_seed"],
                "p3_subject_mean_accuracy": a3[subject],
                "p5_subject_mean_accuracy": a5[subject],
                "p5_minus_p3_accuracy_fraction": a5[subject] - a3[subject],
                "p5_minus_p3_accuracy_pp": 100.0 * (a5[subject] - a3[subject]),
            })
        cohort3 = float(np.mean([a3[s] for s in subjects]))
        cohort5 = float(np.mean([a5[s] for s in subjects]))
        sm3, sm5 = p3["system_metrics"], p5["system_metrics"]
        t3, t5 = sm3["total_communication"], sm5["total_communication"]
        o3, o5 = sm3["ongoing"], sm5["ongoing"]
        r3, r5 = sm3["communication_rates"], sm5["communication_rates"]
        s3, s5 = sm3["staleness_summary"], sm5["staleness_summary"]
        d3, d5 = sm3["delay_summary"], sm5["delay_summary"]
        s2c3 = float(t3["server_to_client_mb"])
        s2c5 = float(t5["server_to_client_mb"])
        pair_rows.append({
            "dataset_name": dataset_name,
            "replicate_id": replicate_id,
            "regime": regime,
            "n_subjects": len(subjects),
            "model_seed": p3["run_config"]["model_seed"],
            "group_seed": p3["run_config"]["group_seed"],
            "trace_seed": p3["run_config"]["trace_seed"],
            "scheduler_seed_p5": p5["run_config"]["scheduler_seed"],
            "trace_hash": p3["run_config"]["trace_hash"],
            "group_assignment_hash": p3["run_config"]["group_assignment_hash"],
            "session2_partition_hash": p3["run_config"]["session2_partition_hash"],
            "initial_backbone_hash": p3["run_config"]["collaborative_start_backbone_sha256"],
            "p3_mean_accuracy": cohort3,
            "p5_mean_accuracy": cohort5,
            "p5_minus_p3_accuracy_fraction": cohort5 - cohort3,
            "p5_minus_p3_accuracy_pp": 100.0 * (cohort5 - cohort3),
            "p5_accuracy_higher": bool(cohort5 > cohort3),
            "p3_client_to_server_mb": float(t3["client_to_server_mb"]),
            "p5_client_to_server_mb": float(t5["client_to_server_mb"]),
            "p5_minus_p3_client_to_server_mb": float(t5["client_to_server_mb"]) - float(t3["client_to_server_mb"]),
            "p3_server_to_client_mb": s2c3,
            "p5_server_to_client_mb": s2c5,
            "p5_minus_p3_server_to_client_mb": s2c5 - s2c3,
            "p5_server_to_client_reduction_percent": 100.0 * (s2c3 - s2c5) / s2c3 if s2c3 > 0 else None,
            "p3_total_mb": float(t3["total_mb"]),
            "p5_total_mb": float(t5["total_mb"]),
            "p5_minus_p3_total_mb": float(t5["total_mb"]) - float(t3["total_mb"]),
            "p3_rejected_upload_rate": float(r3["drop_rate_over_uploads"]),
            "p5_rejected_upload_rate": float(r5["drop_rate_over_uploads"]),
            "p5_minus_p3_rejected_upload_rate_pp": 100.0 * (float(r5["drop_rate_over_uploads"]) - float(r3["drop_rate_over_uploads"])),
            "p3_accepted_update_staleness": float(s3["mean"]),
            "p5_accepted_update_staleness": float(s5["mean"]),
            "p5_minus_p3_accepted_update_staleness": float(s5["mean"]) - float(s3["mean"]),
            "p3_buffered_upload_delay_rounds": float(d3["mean_rounds"]),
            "p5_buffered_upload_delay_rounds": float(d5["mean_rounds"]),
            "p5_minus_p3_buffered_upload_delay_rounds": float(d5["mean_rounds"]) - float(d3["mean_rounds"]),
            "p3_download_avoidance_rate": float(r3["download_avoidance_rate_over_download_opportunities"]),
            "p5_download_avoidance_rate": float(r5["download_avoidance_rate_over_download_opportunities"]),
            "p5_minus_p3_download_avoidance_pp": 100.0 * (
                float(r5["download_avoidance_rate_over_download_opportunities"])
                - float(r3["download_avoidance_rate_over_download_opportunities"])
            ),
            "p3_uploads": int(o3["uploads"]),
            "p5_uploads": int(o5["uploads"]),
            "p3_dropped_updates": int(o3["dropped_updates"]),
            "p5_dropped_updates": int(o5["dropped_updates"]),
            "p3_run_dir": p3["run_dir"],
            "p5_run_dir": p5["run_dir"],
        })
    return pair_rows, subject_rows, validation_rows


def t_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    if len(values) < 2:
        return None, None
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    from scipy.stats import t as student_t

    critical = float(student_t.ppf((1.0 + confidence) / 2.0, df=len(values) - 1))
    half = critical * sd / math.sqrt(len(values))
    return mean - half, mean + half


def _descriptive_stats(values: List[float], prefix: str) -> Dict[str, Any]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {
            f"mean_{prefix}": None,
            f"sd_{prefix}": None,
            f"median_{prefix}": None,
            f"min_{prefix}": None,
            f"max_{prefix}": None,
        }
    return {
        f"mean_{prefix}": float(np.mean(vals)),
        f"sd_{prefix}": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        f"median_{prefix}": float(np.median(vals)),
        f"min_{prefix}": float(np.min(vals)),
        f"max_{prefix}": float(np.max(vals)),
    }


def summarize_robustness_pairs(pair_rows: List[dict]) -> List[dict]:
    """Create descriptive replicate-level summaries without pseudo-replication."""
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for row in pair_rows:
        grouped.setdefault((str(row["dataset_name"]), str(row["regime"])), []).append(row)

    metrics = {
        "p5_minus_p3_accuracy_pp": "p5_minus_p3_accuracy_pp",
        "p5_server_to_client_reduction_percent": "s2c_reduction_percent",
        "p5_minus_p3_client_to_server_mb": "p5_minus_p3_client_to_server_mb",
        "p5_minus_p3_total_mb": "p5_minus_p3_total_mb",
        "p5_minus_p3_rejected_upload_rate_pp": "p5_minus_p3_rejected_upload_rate_pp",
        "p5_minus_p3_accepted_update_staleness": "p5_minus_p3_accepted_update_staleness",
        "p5_minus_p3_buffered_upload_delay_rounds": "p5_minus_p3_buffered_upload_delay_rounds",
        "p5_minus_p3_download_avoidance_pp": "p5_minus_p3_download_avoidance_pp",
    }

    out = []
    for (dataset_name, regime), rows in sorted(grouped.items()):
        acc = [float(r["p5_minus_p3_accuracy_pp"]) for r in rows]
        lo, hi = t_confidence_interval(acc)
        summary: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "regime": regime,
            "n_replicate_pairs": len(rows),
            "ci95_low_p5_minus_p3_accuracy_pp": lo,
            "ci95_high_p5_minus_p3_accuracy_pp": hi,
            "positive_replicates": int(sum(v > 0 for v in acc)),
            "zero_replicates": int(sum(v == 0 for v in acc)),
            "negative_replicates": int(sum(v < 0 for v in acc)),
            "positive_replicate_percent": 100.0 * sum(v > 0 for v in acc) / len(acc),
            "note": (
                "Descriptive replicate-level summary only. Do not treat repeated subject-by-seed rows as "
                "independent observations; repeated-measures inference must use the paired subject file."
            ),
        }
        for source_col, prefix in metrics.items():
            summary.update(_descriptive_stats([r.get(source_col) for r in rows], prefix))
        out.append(summary)
    return out


def hierarchical_bootstrap_accuracy(
    subject_rows: List[dict],
    n_bootstrap: int = 10_000,
    seed: int = 2026,
) -> List[dict]:
    """Crossed hierarchical bootstrap for the paired P5-minus-P3 accuracy effect.

    Replicate pairs and subjects are resampled independently with replacement.
    The P3/P5 difference is never broken: each sampled cell is the already paired
    subject-level contrast averaged across k.  This respects the crossed design
    in which the same subjects recur under independent model/trace replicates.
    """
    if int(n_bootstrap) < 100:
        raise ValueError("--bootstrap-samples must be at least 100.")
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for row in subject_rows:
        grouped.setdefault((str(row["dataset_name"]), str(row["regime"])), []).append(row)

    outputs: List[dict] = []
    for group_index, ((dataset_name, regime), rows) in enumerate(sorted(grouped.items())):
        by_rep: Dict[int, Dict[str, float]] = {}
        for row in rows:
            rep = int(row["replicate_id"])
            subject = str(row["subject"])
            value = float(row["p5_minus_p3_accuracy_pp"])
            if subject in by_rep.setdefault(rep, {}):
                raise RuntimeError(
                    f"Duplicate subject contrast for dataset={dataset_name}, regime={regime}, "
                    f"replicate={rep}, subject={subject}"
                )
            by_rep[rep][subject] = value
        replicates = sorted(by_rep)
        if len(replicates) < 2:
            raise RuntimeError(
                f"Hierarchical bootstrap requires at least two replicate pairs for {dataset_name}/{regime}."
            )
        subject_sets = [set(by_rep[rep]) for rep in replicates]
        if any(sset != subject_sets[0] for sset in subject_sets[1:]):
            raise RuntimeError(
                f"Subject sets differ across replicates for {dataset_name}/{regime}; "
                "the crossed bootstrap cannot be applied safely."
            )
        subjects = sorted(subject_sets[0])
        if not subjects:
            raise RuntimeError(f"No paired subject contrasts for {dataset_name}/{regime}.")

        matrix = np.asarray(
            [[by_rep[rep][subject] for subject in subjects] for rep in replicates],
            dtype=np.float64,
        )
        observed = float(matrix.mean())
        rng = np.random.default_rng(int(seed) + 100_003 * int(group_index))
        boot = np.empty(int(n_bootstrap), dtype=np.float64)
        for b in range(int(n_bootstrap)):
            rep_idx = rng.integers(0, len(replicates), size=len(replicates))
            subject_idx = rng.integers(0, len(subjects), size=len(subjects))
            sampled = matrix[np.ix_(rep_idx, subject_idx)]
            boot[b] = float(sampled.mean())
        low, high = np.percentile(boot, [2.5, 97.5])
        outputs.append({
            "dataset_name": dataset_name,
            "regime": regime,
            "n_replicate_pairs": int(len(replicates)),
            "n_subjects": int(len(subjects)),
            "bootstrap_samples": int(n_bootstrap),
            "bootstrap_seed": int(seed) + 100_003 * int(group_index),
            "observed_mean_p5_minus_p3_accuracy_pp": observed,
            "hierarchical_bootstrap_ci95_low_pp": float(low),
            "hierarchical_bootstrap_ci95_high_pp": float(high),
            "bootstrap_probability_effect_gt_zero": float(np.mean(boot > 0.0)),
            "bootstrap_probability_effect_lt_zero": float(np.mean(boot < 0.0)),
            "bootstrap_probability_effect_eq_zero": float(np.mean(boot == 0.0)),
            "method": (
                "Crossed hierarchical bootstrap: resample matched replicate pairs and subjects "
                "independently with replacement; retain paired P5-minus-P3 subject contrasts "
                "and equal weighting across k within each subject."
            ),
            "interpretation_guard": (
                "The interval estimates the mean policy contrast across the evaluated subject cohort "
                "and independent joint model/trace realizations. It does not separate model-seed "
                "variance from trace-seed variance."
            ),
        })
    return outputs


def write_robustness_outputs(
    suite_dir: str,
    runs: List[Dict[str, Any]],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 2026,
) -> None:
    pair_rows, subject_rows, validation_rows = build_robustness_pair_outputs(runs)
    cross_regime_rows = validate_cross_regime_communication(runs)
    descriptive_rows = summarize_robustness_pairs(pair_rows)
    bootstrap_rows = hierarchical_bootstrap_accuracy(
        subject_rows,
        n_bootstrap=int(bootstrap_samples),
        seed=int(bootstrap_seed),
    )

    trace_hashes_by_rep: Dict[int, set] = {}
    group_hashes = set()
    for run in runs:
        cfg = run["run_config"]
        trace_hashes_by_rep.setdefault(int(cfg["replicate_id"]), set()).add(str(cfg["trace_hash"]))
        group_hashes.add(str(cfg["group_assignment_hash"]))
    per_rep_single_trace = all(len(v) == 1 for v in trace_hashes_by_rep.values())
    independent_hashes = [next(iter(trace_hashes_by_rep[r])) for r in sorted(trace_hashes_by_rep)]
    independent_traces = len(set(independent_hashes)) == len(independent_hashes)
    fixed_groups = len(group_hashes) == 1
    if not per_rep_single_trace or not independent_traces or not fixed_groups:
        raise RuntimeError(
            "Robustness trace validation failed: "
            f"per_rep_single_trace={per_rep_single_trace}, independent_traces={independent_traces}, fixed_groups={fixed_groups}"
        )

    write_rows_csv(os.path.join(suite_dir, "robustness_replicate_pairs.csv"), pair_rows)
    write_rows_csv(os.path.join(suite_dir, "robustness_subject_paired_accuracy.csv"), subject_rows)
    write_rows_csv(os.path.join(suite_dir, "robustness_descriptive_summary.csv"), descriptive_rows)
    write_rows_csv(os.path.join(suite_dir, "robustness_hierarchical_bootstrap.csv"), bootstrap_rows)
    write_rows_csv(os.path.join(suite_dir, "robustness_pair_validation.csv"), validation_rows)
    write_rows_csv(os.path.join(suite_dir, "robustness_cross_regime_communication_audit.csv"), cross_regime_rows)
    all_pair_checks = all(bool(r.get("all_checks_passed")) for r in validation_rows)
    validation_status = (
        "passed"
        if all_pair_checks and per_rep_single_trace and independent_traces and fixed_groups
        else "failed"
    )
    run_modes = sorted({
        str(run["run_config"].get("training_rng_mode", "legacy_stream"))
        for run in runs
    })
    byte_modes = sorted({
        str(run["run_config"].get("communication_byte_accounting", "legacy_runtime_metadata"))
        for run in runs
    })
    save_json(os.path.join(suite_dir, "robustness_validation_report.json"), {
        "status": validation_status,
        "n_runs": int(len(runs)),
        "n_matched_pairs": int(len(validation_rows)),
        "training_rng_modes": run_modes,
        "communication_byte_accounting_modes": byte_modes,
        "hierarchical_bootstrap": bootstrap_rows,
        "all_matched_pair_checks_passed": all_pair_checks,
        "one_trace_hash_per_replicate": per_rep_single_trace,
        "different_trace_hash_across_replicates": independent_traces,
        "fixed_group_assignment_across_replicates": fixed_groups,
        "trace_hashes_by_replicate": {str(k): sorted(v) for k, v in trace_hashes_by_rep.items()},
        "group_assignment_hashes": sorted(group_hashes),
        "communication_audit_rows": cross_regime_rows,
        "unit_audit": {
            "traffic": "decimal MB computed directly from raw bytes using 1 MB = 1,000,000 bytes",
            "accepted_update_staleness": "event-weighted over admitted uploads only",
            "buffered_upload_delay": "event-weighted over buffered uploads that pass checkpoint availability and reach stale admission",
            "rejected_upload_rate": "coordinator-dropped transmitted uploads divided by all transmitted uploads",
            "download_avoidance": "avoided synchronization downloads divided by download opportunities",
        },
    })




def run_robustness_preflight(args) -> str:
    """Validate seed separation and trace determinism without model training."""
    base = configure_primary_robustness_args(args)
    dataset_name = dataset_name_from_id(int(base.datasetId))
    full_ds = load_existing_preprocessed_dataset(dataset_name)
    subjects = all_subjects_from_labels(full_ds.labels)
    if not subjects:
        raise RuntimeError("Preflight could not resolve any subjects from the processed labels.")

    net_ctor = bc.resolve_network_ctor(base.network)
    model_args = bc.get_model_args(int(base.datasetId), base.network)
    model_seed_a = int(base.seed)
    model_seed_b = int(base.seed) + 1
    trace_seed_a = int(base.seed) + 10_000
    trace_seed_b = int(base.seed) + 10_001
    scheduler_seed = int(base.seed) + 20_000
    group_seed = int(base.group_seed) if base.group_seed is not None else int(base.seed)

    def initial_hash(model_seed: int) -> str:
        bc.set_seed(int(model_seed), deterministic=not bool(base.fast_cudnn))
        return state_sha256(clone_state(net_ctor(**model_args).state_dict()))

    init_a1 = initial_hash(model_seed_a)
    init_a2 = initial_hash(model_seed_a)
    init_b = initial_hash(model_seed_b)

    group_args = copy.deepcopy(base)
    group_args.group_seed = group_seed
    pmap, labels = online_probability_map(subjects, group_args, seed=group_seed)

    def bundle(model_seed: int, trace_seed: int) -> SeedBundle:
        a = copy.deepcopy(base)
        a.model_seed = int(model_seed)
        a.group_seed = int(group_seed)
        a.trace_seed = int(trace_seed)
        a.scheduler_seed = int(scheduler_seed)
        return resolve_seed_bundle(a)

    trace_a1 = build_availability_trace(subjects, pmap, labels, int(base.rounds), bundle(model_seed_a, trace_seed_a), True)
    trace_a2 = build_availability_trace(subjects, pmap, labels, int(base.rounds), bundle(model_seed_a, trace_seed_a), True)
    trace_b = build_availability_trace(subjects, pmap, labels, int(base.rounds), bundle(model_seed_a, trace_seed_b), True)
    trace_same_under_model_change = build_availability_trace(
        subjects, pmap, labels, int(base.rounds), bundle(model_seed_b, trace_seed_a), True
    )

    checks = {
        "same_model_seed_repeats_initial_hash": init_a1 == init_a2,
        "different_model_seed_changes_initial_hash": init_a1 != init_b,
        "same_trace_seed_repeats_trace_hash": trace_a1["trace_hash"] == trace_a2["trace_hash"],
        "different_trace_seed_changes_trace_hash": trace_a1["trace_hash"] != trace_b["trace_hash"],
        "model_seed_does_not_change_trace": trace_a1["trace_hash"] == trace_same_under_model_change["trace_hash"],
        "fixed_group_hash_across_traces": trace_a1["group_assignment_hash"] == trace_b["group_assignment_hash"],
        "trace_subject_order_complete": trace_a1["subject_order"] == [str(s) for s in subjects],
        "trace_round_count_complete": len(trace_a1["availability_matrix"]) == int(base.rounds),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if failed:
        raise RuntimeError(f"Robustness preflight failed: {failed}")

    from .paths import output_root
    preflight_dir = os.path.join(
        str(output_root()),
        dataset_name,
        str(args.exp_name),
        f"robustness_preflight_{timestamp_id()}",
    )
    os.makedirs(preflight_dir, exist_ok=True)
    save_json(os.path.join(preflight_dir, "trace_a.json"), trace_a1)
    save_json(os.path.join(preflight_dir, "trace_b.json"), trace_b)
    save_availability_trace_csv(os.path.join(preflight_dir, "trace_a.csv"), trace_a1)
    save_availability_trace_csv(os.path.join(preflight_dir, "trace_b.csv"), trace_b)
    report = {
        "status": "passed",
        "dataset": dataset_name,
        "network": str(base.network),
        "subject_count": len(subjects),
        "rounds": int(base.rounds),
        "model_seed_a": model_seed_a,
        "model_seed_b": model_seed_b,
        "trace_seed_a": trace_seed_a,
        "trace_seed_b": trace_seed_b,
        "group_seed": group_seed,
        "scheduler_seed": scheduler_seed,
        "initial_model_hash_a": init_a1,
        "initial_model_hash_b": init_b,
        "trace_hash_a": trace_a1["trace_hash"],
        "trace_hash_b": trace_b["trace_hash"],
        "group_assignment_hash": trace_a1["group_assignment_hash"],
        "checks": checks,
        "note": "This preflight validates RNG separation and provenance only; it does not replace the seed-2026 full reproduction run.",
    }
    save_json(os.path.join(preflight_dir, "robustness_preflight_report.json"), report)
    print(f"[ROBUSTNESS PREFLIGHT PASSED] {preflight_dir}")
    return preflight_dir

def run_suite(args) -> None:
    suite = str(args.run_suite)
    if bool(getattr(args, "run_both_datasets", False)):
        if suite not in ("nosplit_multiseed_robustness", "robustness_preflight"):
            raise ValueError(
                "--run-both-datasets is supported only for robustness_preflight or "
                "nosplit_multiseed_robustness."
            )
        for dataset_id, dataset_label in ((0, "BCICIV-2a"), (1, "OpenBMI")):
            child = copy.deepcopy(args)
            child.datasetId = int(dataset_id)
            child.run_both_datasets = False
            print(
                f"[BOTH-DATASETS] starting {dataset_label} (datasetId={dataset_id}) ",
                f"suite={suite}",
                flush=True,
            )
            run_suite(child)
        return
    if suite == "robustness_preflight":
        run_robustness_preflight(args)
        return
    if suite == "none":
        protocol = "no_split" if (bool(args.no_subject_split) or str(args.eval_protocol) == "no_split") else "holdout"
        if protocol == "no_split":
            args.no_subject_split = True
        run_single(args, args.regime, protocol, None, int(args.split_iteration))
        return
    runs = []
    if suite == "holdout_main":
        for it in range(1, int(args.num_split_iterations) + 1):
            for regime in REGIMES:
                a = copy.deepcopy(args)
                a.run_suite = "none"
                runs.append(run_single(a, regime, "holdout", None, it))
    elif suite == "nosplit_main":
        for regime in REGIMES:
            a = copy.deepcopy(args)
            a.run_suite = "none"
            a.no_subject_split = True
            runs.append(run_single(a, regime, "no_split", None, 1))
    elif suite == "nosplit_commgrid":
        for regime in REGIMES:
            for comm in ("comm_off", "comm_on"):
                a = copy.deepcopy(args)
                a.run_suite = "none"
                a.no_subject_split = True
                runs.append(run_single(a, regime, "no_split", comm, 1))
    elif suite == "nosplit_coordination_ablation":
        # Primary P1--P6 suite: keeps the no-split protocol and the two
        # head-only regimes, then compares non-adaptive all-gateway scheduling against the
        # communication-aware gateway scheduler and adds buffering/staleness
        # ablations. The public ``nexus-mi`` commands apply the fixed study
        # presets; the low-level parser also retains diagnostic configuration
        # options that are not used by the publication presets.
        for regime in REGIMES:
            for ab in COORDINATION_ABLATIONS:
                a = copy.deepcopy(args)
                a.run_suite = "none"
                a.no_subject_split = True
                a.comm_sim = True
                a.selection_policy = ab["selection_policy"]
                a.buffer_policy = ab["buffer_policy"]
                a.no_buffering = (ab["buffer_policy"] == "none")
                a.stale_policy = ab["stale_policy"]
                a.download_policy = ab["download_policy"]
                a.policy_id = ab["id"]
                a.policy_name = ab["name"]
                a.policy_label = ab["label"]
                a.policy_family = ab["family"]
                a.severity_name = str(getattr(args, "severity_name", ""))
                a.severity_description = str(getattr(args, "severity_description", ""))
                a.expected_mean_online_prob = expected_hetero_online_probability(a) if str(a.comm_profile) == "hetero" else float(a.online_prob)
                a.exp_name = f"{args.exp_name}_{ab['name']}"
                runs.append(run_single(a, regime, "no_split", "comm_on", 1))
    elif suite == "nosplit_severity_sweep":
        # Availability-sensitivity suite used in the study. It runs only EIB-PH
        # and compares P3 with P5 across mild/default/severe heterogeneous-link
        # availability profiles. These policies retain the A3/A5 run identifiers
        # used by the primary suite so their saved output schema remains consistent.
        # Both use FIFO buffering
        # and stale-drop admission, so the sweep
        # isolates the effect of communication-aware gateway scheduling and stale-aware
        # downloads under increasing communication severity.
        regime = "embedding_shared_head_only"
        selected_ablation_ids = list(SEVERITY_SWEEP_POLICY_IDS)
        severity_profiles = list(getattr(args, "severity_profiles", SEVERITY_SWEEP_PROFILES))
        for sev in severity_profiles:
            for policy_id in selected_ablation_ids:
                ab = ablation_by_id(policy_id)
                a = copy.deepcopy(args)
                a.run_suite = "none"
                a.no_subject_split = True
                a.comm_sim = True
                a.comm_profile = "hetero"
                a.online_prob_good = float(sev["online_prob_good"])
                a.online_prob_med = float(sev["online_prob_med"])
                a.online_prob_bad = float(sev["online_prob_bad"])
                a.selection_policy = ab["selection_policy"]
                a.buffer_policy = ab["buffer_policy"]
                a.no_buffering = (ab["buffer_policy"] == "none")
                a.stale_policy = ab["stale_policy"]
                a.download_policy = ab["download_policy"]
                a.policy_id = ab["id"]
                a.policy_name = ab["name"]
                a.policy_label = ab["label"]
                a.policy_family = ab["family"]
                a.severity_name = str(sev["name"])
                a.severity_description = (
                    f"heterogeneous-link severity {sev['name']}: "
                    f"high/moderate/low online probabilities "
                    f"{sev['online_prob_good']:.2f}/{sev['online_prob_med']:.2f}/{sev['online_prob_bad']:.2f}"
                )
                a.expected_mean_online_prob = expected_hetero_online_probability(a)
                a.exp_name = f"{args.exp_name}_severity_{sev['name']}_{ab['name']}"
                runs.append(run_single(a, regime, "no_split", "comm_on", 1))
    elif suite == "nosplit_p5_scheduler_ablation":
        regime = "embedding_shared_head_only"
        if int(args.datasetId) != 1:
            raise ValueError("The publication component analysis is defined for OpenBMI (datasetId=1).")
        for variant in P5_SCHEDULER_ABLATIONS:
            a = copy.deepcopy(args)
            a.run_suite = "none"; a.no_subject_split = True; a.comm_sim = True; a.comm_profile = "hetero"
            a.selection_policy = variant["selection_policy"]; a.buffer_policy = variant["buffer_policy"]
            a.no_buffering = False; a.stale_policy = variant["stale_policy"]; a.download_policy = variant["download_policy"]
            a.policy_id = variant["id"]; a.policy_name = variant["name"]; a.policy_label = variant["label"]
            a.policy_family = variant["family"]; a.component_role = variant["component_role"]
            a.severity_name = "default"
            a.severity_description = "default heterogeneous-link profile for P5 component analysis"
            a.expected_mean_online_prob = expected_hetero_online_probability(a)
            a.exp_name = f"{args.exp_name}_p5ablation_{variant['name']}"
            runs.append(run_single(a, regime, "no_split", "comm_on", 1))
    elif suite == "nosplit_multiseed_robustness":
        base = configure_primary_robustness_args(args)
        n_rep = int(args.robustness_replicates)
        if n_rep <= 0:
            raise ValueError("--robustness-replicates must be >= 1.")
        if str(getattr(args, "availability_trace_file", "") or "").strip():
            raise ValueError(
                "The multi-replicate suite generates and saves one independent trace per replicate. "
                "Do not combine it with --availability-trace-file; use single-run mode for a supplied trace."
            )
        model_seeds = parse_optional_seed_csv(args.robustness_model_seeds, n_rep, "robustness-model-seeds")
        trace_seeds = parse_optional_seed_csv(args.robustness_trace_seeds, n_rep, "robustness-trace-seeds")
        scheduler_seeds = parse_optional_seed_csv(args.robustness_scheduler_seeds, n_rep, "robustness-scheduler-seeds")
        if model_seeds is None:
            model_seeds = [int(args.seed) + i for i in range(n_rep)]
        if trace_seeds is None:
            trace_seeds = [int(args.seed) + 10_000 + i for i in range(n_rep)]
        if scheduler_seeds is None:
            scheduler_seeds = [int(args.seed) + 20_000 + i for i in range(n_rep)]
        fixed_group_seed = int(args.group_seed) if args.group_seed is not None else int(args.seed)
        initialization_cache: Dict[str, Any] = {}
        for rep_index in range(n_rep):
            replicate_id = rep_index + 1
            for regime in REGIMES:
                for policy_id in ("P3", "P5"):
                    policy = robustness_policy(policy_id)
                    a = copy.deepcopy(base)
                    a.model_seed = int(model_seeds[rep_index])
                    a.group_seed = int(fixed_group_seed)
                    a.trace_seed = int(trace_seeds[rep_index])
                    a.scheduler_seed = int(scheduler_seeds[rep_index])
                    a.replicate_id = int(replicate_id)
                    a.selection_policy = policy["selection_policy"]
                    a.buffer_policy = policy["buffer_policy"]
                    a.no_buffering = False
                    a.stale_policy = policy["stale_policy"]
                    a.download_policy = policy["download_policy"]
                    a.policy_id = policy["id"]
                    a.policy_name = policy["name"]
                    a.policy_label = policy["label"]
                    a.policy_family = policy["family"]
                    a.severity_name = "default"
                    a.severity_description = (
                        "default heterogeneous-link profile: high/moderate/low online probabilities "
                        f"{a.online_prob_good:.2f}/{a.online_prob_med:.2f}/{a.online_prob_bad:.2f}"
                    )
                    a.expected_mean_online_prob = expected_hetero_online_probability(a)
                    a.exp_name = (
                        f"{args.exp_name}_rep{replicate_id:02d}_m{a.model_seed}_t{a.trace_seed}_"
                        f"{regime}_{policy_id.lower()}"
                    )
                    run = run_single(
                        a,
                        regime,
                        "no_split",
                        "comm_on",
                        1,
                        initialization_cache=initialization_cache,
                    )
                    runs.append(run)
                # Fail immediately if a matched pair was not actually matched.
                validate_matched_pair(runs[-2], runs[-1])
    else:
        raise ValueError(f"Unknown run_suite: {suite}")

    # Save expanded suite summaries next to the first run's exp folder.
    if runs:
        # run_dir layout:
        # output/<dataset>/<exp_name>/<regime>/<protocol>/<comm>/<network_timestamp>/
        # Individual suite runs append policy/severity names to exp_name. Store
        # suite-level summaries under the base args.exp_name folder so the 6- or
        # 12-run suite has one obvious analysis directory instead of being nested
        # under the first ablation's run folder.
        dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(runs[0]["run_dir"])))))
        exp_dir = os.path.join(dataset_dir, str(args.exp_name))
        suite_dir = os.path.join(exp_dir, f"suite_{suite}_{timestamp_id()}")
        os.makedirs(suite_dir, exist_ok=True)

        rows = []
        run_rows = []
        for r in runs:
            base = build_suite_summary_base_row(r, suite)
            run_rows.append(dict(base))
            for k, v in r["summary"].items():
                row = dict(base)
                row.update({
                    "k": int(k),
                    "n_subjects": v.get("n_subjects"),
                    "n_subject_repeat_rows": v.get("n_subject_repeat_rows"),
                    "n_repeats_observed": v.get("n_repeats_observed"),
                    "mean_acc": v.get("mean_acc"),
                    "std_acc": v.get("std_acc"),
                    "worst_acc": v.get("min_acc_worst_user"),
                    "max_acc": v.get("max_acc"),
                    "mean_macro_f1": v.get("mean_macro_f1"),
                    "std_macro_f1": v.get("std_macro_f1"),
                    "worst_macro_f1": v.get("min_macro_f1_worst_user"),
                    "mean_subject_acc": v.get("mean_subject_acc"),
                    "std_subject_acc": v.get("std_subject_acc"),
                    "mean_subject_macro_f1": v.get("mean_subject_macro_f1"),
                    "std_subject_macro_f1": v.get("std_subject_macro_f1"),
                })
                rows.append(row)

        write_rows_csv(os.path.join(suite_dir, "suite_summary.csv"), rows)
        write_rows_csv(os.path.join(suite_dir, "suite_communication_summary.csv"), run_rows)

        if suite == "nosplit_severity_sweep":
            tradeoff_rows = add_tradeoff_deltas(rows, baseline_policy_id="A3")
            tradeoff_aggregate_rows = aggregate_tradeoff_by_severity(tradeoff_rows, baseline_policy_id="A3")
            write_rows_csv(os.path.join(suite_dir, "suite_tradeoff_summary.csv"), tradeoff_rows)
            write_rows_csv(os.path.join(suite_dir, "suite_tradeoff_summary_by_severity.csv"), tradeoff_aggregate_rows)
        if suite == "nosplit_multiseed_robustness":
            write_robustness_outputs(
                suite_dir,
                runs,
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_seed=int(args.bootstrap_seed),
            )

        save_json(os.path.join(suite_dir, "suite_runs.json"), runs)
        save_json(os.path.join(suite_dir, "suite_summary_definitions.json"), {
            "suite_summary_csv": "one row per run and calibration size k; includes evaluation metrics plus full run-level communication counters/rates",
            "suite_communication_summary_csv": "one row per run; includes the same communication counters/rates without repeating them for every k",
            "session2_split_policy": {
                "chronological": "paper split: first calib_pool_trials Session-2 trials define the calibration pool; the test set is the last fixed_test_trials trials or all remaining trials if fixed_test_trials=0",
                "random_per_class": "diagnostic split: for each subject, k, and repeat, randomly select k Session-2 trials per class for calibration and evaluate on the remaining Session-2 trials",
            },
            "communication_count_columns": "selected/unselected/online/offline/download/upload/buffer/accepted/applied/drop counters are copied from system_metrics['ongoing']",
            "communication_rate_columns": "all rate columns are copied from system_metrics['communication_rates']",
            "delay_staleness_columns": "delay columns use buffered-upload delay; staleness columns use admitted-update staleness from system_metrics delay_summary and staleness_summary",
            "policy_ids": {
                "A1": "P1: non-adaptive all-gateway scheduling, no buffering",
                "A2": "P2: non-adaptive all-gateway scheduling, FIFO buffering, accept-if-base-retained delayed admission",
                "A3": "P3: non-adaptive all-gateway scheduling, FIFO buffering, stale-drop admission",
                "A4": "P4: non-adaptive all-gateway scheduling, latest-pending buffering, stale-drop admission",
                "A5": "P5: communication-aware gateway scheduling, FIFO buffering, stale-drop admission",
                "A6": "P6: communication-aware gateway scheduling, latest-pending buffering, stale-drop admission",
            },
            "severity_sweep": {
                "meaning": "Availability-sensitivity experiment. Runs embedding_shared_head_only only. Compares P3 and P5 under mild/default/severe heterogeneous-link profiles (stored internally as A3/A5 for compatibility with the primary suite).",
                "baseline_for_delta": "A3; all deltas in suite_tradeoff_summary.csv and suite_tradeoff_summary_by_severity.csv are row minus matching A3 for the same dataset/regime/severity/split/k, or averaged across k in the aggregate file.",
                "profiles": list(getattr(args, "severity_profiles", SEVERITY_SWEEP_PROFILES)),
                "note": "high/moderate/low are client link-profile groups within one heterogeneous-link condition; mild/default/severe are the overall severity levels.",
            },
        })
        print(f"[SUITE DONE] {suite_dir}")


# =============================================================================
# Parser
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "NEXUS-MI communication-aware EEG FL experiment driver with independent model/group/trace/scheduler seeds, "
            "saved availability traces and hashes, decimal-MB and accepted-update-staleness reporting, "
            "and a five-replicate matched P3/P5 robustness suite"
        )
    )
    p.add_argument("datasetId", type=int, nargs="?", default=0, help='0=BCICIV-2a, 1=OpenBMI')
    p.add_argument("network", type=str, nargs="?", default="eegNet", help="eegNet (paper protocol)")
    p.add_argument("nGPU", type=int, nargs="?", default=0, help="GPU id")
    p.add_argument("--sub", type=str, default=None)
    p.add_argument("--all-subjects", action="store_true")
    p.add_argument("--openbmi-protocol", action="store_true")
    p.add_argument("--phase1-train-n", type=int, default=40)
    p.add_argument("--phase2-holdout-n", type=int, default=14)
    p.add_argument("--bci-phase1-train-n", type=int, default=6)
    p.add_argument("--bci-holdout-n", type=int, default=3)
    p.add_argument("--split-iteration", type=int, default=1)
    p.add_argument("--num-split-iterations", type=int, default=10)
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--skip-train-subjects", type=str, default="")
    p.add_argument("--no-subject-split", action="store_true")
    p.add_argument("--eval-protocol", type=str, default="holdout", choices=["holdout", "no_split"],
                   help="Protocol for single-run mode. Use no_split to train/evaluate the same subjects; holdout to evaluate held-out subjects.")

    p.add_argument("--regime", type=str, default="shared_head_only", choices=REGIMES)
    p.add_argument("--session1-val-ratio", type=float, default=0.2)
    p.add_argument("--init-max-epochs", type=int, default=1500)
    p.add_argument("--init-patience", type=int, default=200)
    p.add_argument("--init-best-metric", type=str, default="val_acc", choices=["val_acc", "val_loss"])
    p.add_argument("--lr-init", type=float, default=1e-3)

    p.add_argument("--rounds", type=int, default=120)
    p.add_argument("--local-epochs", type=int, default=50)
    p.add_argument("--lr-local", type=float, default=1e-3)
    p.add_argument("--agg-weighting", type=str, default="uniform", choices=["uniform", "samples", "performance"])
    p.add_argument("--perf-weight-metric", type=str, default="val_acc", choices=["val_acc", "val_loss"])
    p.add_argument("--perf-weight-method", type=str, default="softmax", choices=["softmax", "inv", "linear"])
    p.add_argument("--perf-weight-alpha", type=float, default=5.0)
    p.add_argument("--perf-weight-eps", type=float, default=1e-8)
    p.add_argument("--perf-weight-with-samples", action="store_true")
    p.add_argument("--no-local-train-head", action="store_true")

    p.add_argument("--comm-sim", action="store_true")
    p.add_argument("--comm-profile", type=str, default="uniform", choices=["uniform", "hetero"])
    p.add_argument("--online-prob", type=float, default=1.0)
    p.add_argument("--online-prob-good", type=float, default=0.95)
    p.add_argument("--online-prob-med", type=float, default=0.70)
    p.add_argument("--online-prob-bad", type=float, default=0.40)
    p.add_argument("--profile-frac-good", type=float, default=0.34)
    p.add_argument("--profile-frac-med", type=float, default=0.33)
    p.add_argument("--selection-policy", type=str, default="all", choices=["all", "topk", "online_random", "comm_aware"],
                   help="Gateway upload scheduling policy: all=non-adaptive all-gateway baseline, topk=diagnostic budgeted scheduler, online_random=online-only random scheduler, comm_aware=online-first priority gateway scheduler.")
    p.add_argument("--max-selected-per-round", type=int, default=0,
                   help="Maximum gateways selected per round for topk/comm_aware. 0 means no budget cap.")
    p.add_argument("--buffer-policy", type=str, default="fifo", choices=["none", "fifo", "latest"],
                   help="Bounded offline buffering policy. fifo drops oldest when full; latest overwrites older pending payloads.")
    p.add_argument("--buffer-max-size", type=int, default=1,
                   help="Maximum pending payloads per client for fifo buffering. latest always keeps one payload.")
    p.add_argument("--no-buffering", action="store_true",
                   help="Alias for --buffer-policy none.")
    p.add_argument("--stale-threshold", type=int, default=2)
    p.add_argument("--checkpoint-retention-margin", type=int, default=5)
    p.add_argument("--stale-policy", type=str, default="drop", choices=["accept_all", "drop", "downweight"],
                   help="Delayed-update admission policy after checkpoint validation.")
    p.add_argument("--stale-gamma", type=float, default=0.8)
    p.add_argument("--download-policy", type=str, default="always", choices=["always", "stale_only"],
                   help="Server-to-client synchronization policy for selected online clients.")
    p.add_argument("--download-stale-threshold", type=int, default=1,
                   help="For --download-policy stale_only, download only when server_version - local_version exceeds this value.")

    p.add_argument("--calib-sizes", type=str, default="15,20,30")
    p.add_argument("--session2-split-policy", type=str, default="chronological", choices=["chronological", "random_per_class"],
                   help="Session-2 personalization/evaluation split. chronological is the paper protocol; random_per_class is an optional diagnostic split with remaining Session-2 trials as test.")
    p.add_argument("--calib-pool-trials", type=int, default=100, help="For --session2-split-policy chronological: number of Session-2 trials used for the front calibration pool.")
    p.add_argument("--fixed-test-trials", type=int, default=0, help="For --session2-split-policy chronological: number of Session-2 trials used for testing from the end. If 0, all remaining trials after the calibration pool are used as test. Ignored by random_per_class.")
    p.add_argument("--repeats", type=int, default=1, help="For random_per_class: number of repeated random k-per-class splits. For chronological, the deterministic split uses one evaluation regardless of this value.")
    p.add_argument("--lr-head", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--head-patience", type=int, default=100)
    p.add_argument("--head-max-epochs", type=int, default=750)
    p.add_argument("--head-stage2", action="store_true")

    p.add_argument("--calib-target-acc", type=float, default=0.60)
    p.add_argument("--no-preload", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--fast-cudnn", action="store_true")
    p.add_argument(
        "--communication-byte-accounting",
        type=str,
        default="legacy_runtime_metadata",
        choices=COMMUNICATION_BYTE_ACCOUNTING_MODES,
        help="Upload-byte accounting for single runs. protocol_metadata excludes the non-protocol wall-clock timestamp; legacy_runtime_metadata uses the study raw-byte accounting.",
    )
    p.add_argument(
        "--training-rng-mode",
        type=str,
        default="legacy_stream",
        choices=TRAINING_RNG_MODES,
        help="Stochastic-training control for single runs: legacy_stream uses the study process-global dropout stream; per_task_seed resets Python/NumPy/PyTorch RNGs for each local and personalization task.",
    )
    p.add_argument("--seed", type=int, default=2026, help="Master seed used when component-specific seeds are omitted.")
    p.add_argument("--model-seed", type=int, default=None, help="Controls model initialization and the stochastic training path, including DataLoader ordering, EIB pretraining, local Adam training, and Session-2 head personalization.")
    p.add_argument("--group-seed", type=int, default=None, help="Controls assignment of subjects to high/moderate/low availability groups.")
    p.add_argument("--trace-seed", type=int, default=None, help="Controls the complete subject-by-round Bernoulli availability trace.")
    p.add_argument("--scheduler-seed", type=int, default=None, help="Controls deterministic round-specific P5 scheduler tie-breaking.")
    p.add_argument("--availability-trace-file", type=str, default="", help="Optional saved availability_trace.json to load in single-run mode. Subject order and round count are validated strictly.")
    p.add_argument("--robustness-replicates", type=int, default=5, help="Number of matched P3/P5 seed/trace pairs for nosplit_multiseed_robustness.")
    p.add_argument("--robustness-model-seeds", type=str, default="", help="Optional comma-separated model seeds; length must equal --robustness-replicates.")
    p.add_argument("--robustness-trace-seeds", type=str, default="", help="Optional comma-separated trace seeds; length must equal --robustness-replicates.")
    p.add_argument("--robustness-scheduler-seeds", type=str, default="", help="Optional comma-separated scheduler seeds; length must equal --robustness-replicates.")
    p.add_argument("--bootstrap-samples", type=int, default=10000, help="Number of crossed hierarchical bootstrap resamples for the paired subject-level accuracy contrast.")
    p.add_argument("--bootstrap-seed", type=int, default=2026, help="Deterministic seed for the crossed hierarchical bootstrap analysis.")
    p.add_argument(
        "--robustness-byte-accounting",
        type=str,
        default="legacy_runtime_metadata",
        choices=COMMUNICATION_BYTE_ACCOUNTING_MODES,
        help="Upload-byte accounting for robustness runs. legacy_runtime_metadata uses the study raw-byte semantics; protocol_metadata is diagnostic only.",
    )
    p.add_argument(
        "--robustness-rng-mode",
        type=str,
        default="legacy_stream",
        choices=TRAINING_RNG_MODES,
        help="Training RNG mode for robustness runs. legacy_stream uses the study stochastic execution; per_task_seed is diagnostic only.",
    )
    p.add_argument(
        "--robustness-authoritative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the manuscript's fixed 120-round, 50-local-epoch, k={15,20,30}, default-link P3/P5 protocol. Disable only for short code-path smoke tests.",
    )
    p.add_argument(
        "--run-both-datasets",
        action="store_true",
        help="Execute the robustness preflight or full multi-seed suite sequentially for BCICIV-2a and OpenBMI in one command (40 runs for five replicates).",
    )
    p.add_argument("--exp-name", type=str, default="nexus_mi")
    p.add_argument("--run-suite", type=str, default="none", choices=RUN_SUITES)
    p.add_argument("--save-checkpoints", action="store_true", help="Save initial, collaborative-start, EIB (when applicable), and final backbone states in checkpoint_bundle.pt. Enabled automatically by the robustness suite.")
    return p


def main():
    args = build_parser().parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()