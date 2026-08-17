"""Dataset preparation for the two public NEXUS-MI datasets.

The repository never ships EEG recordings. Users obtain source files from the
original dataset providers and run this module locally. Source discovery is
recursive so the ``--source`` argument may point at an extracted download root
rather than a particular directory layout.
"""
from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.io import loadmat, savemat

BCICIV2A_SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
OPENBMI_CHANNEL_INDICES = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]


def _require(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def _find_unique(source: Path, names: Sequence[str], what: str) -> Path:
    """Find exactly one file matching one of ``names`` below ``source``.

    Exact files at the source root are preferred. Recursive discovery makes the
    package tolerant of provider-created wrapper directories while still
    failing visibly on ambiguous extractions.
    """
    source = Path(source).expanduser().resolve()
    _require(source, "dataset source directory")
    if not source.is_dir():
        raise NotADirectoryError(f"Dataset source must be an extracted directory: {source}")

    for name in names:
        direct = source / name
        if direct.is_file():
            return direct

    matches = []
    wanted = set(names)
    for p in source.rglob("*"):
        if p.is_file() and p.name in wanted:
            matches.append(p.resolve())
    matches = sorted(set(matches))
    if not matches:
        joined = ", ".join(names)
        raise FileNotFoundError(
            f"Could not find {what} below {source}. Expected one of: {joined}. "
            "Extract the official dataset archive(s) before running preparation."
        )
    if len(matches) > 1:
        listing = "\n  - ".join(str(x) for x in matches[:12])
        extra = "" if len(matches) <= 12 else f"\n  ... and {len(matches)-12} more"
        raise RuntimeError(
            f"Ambiguous {what}: found {len(matches)} matching files below {source}:\n  - {listing}{extra}\n"
            "Point --source at a directory containing a single extracted copy of the dataset."
        )
    return matches[0]


def _find_optional_unique(source: Path, names: Sequence[str], what: str) -> Path | None:
    """Find zero or one matching file, failing only when multiple copies are found."""
    try:
        return _find_unique(source, names, what)
    except FileNotFoundError:
        return None


def _bciciv2a_source_files(source: Path) -> list[tuple[str, Path, Path | None]]:
    """Resolve the official BCICIV-2a recordings and available label files.

    The official competition GDF archive contains all 18 recordings. Training
    recordings (``A??T.gdf``) carry the motor-imagery cue labels in their GDF
    annotations, so a separate ``A??T.mat`` file is optional. Evaluation
    recordings (``A??E.gdf``) use hidden/unknown cue annotations and therefore
    require the nine ``A??E.mat`` files from the competition's true-label
    archive. Existing extractions that also contain ``A??T.mat``
    are also accepted and use those MAT labels directly.
    """
    rows = []
    for suffix in ("T", "E"):
        for i in range(1, 10):
            stem = f"A{i:02d}{suffix}"
            gdf = _find_unique(source, [f"{stem}.gdf"], f"BCICIV-2a {stem} GDF")
            if suffix == "E":
                lab = _find_unique(source, [f"{stem}.mat"], f"BCICIV-2a {stem} evaluation true-label MAT")
            else:
                lab = _find_optional_unique(source, [f"{stem}.mat"], f"BCICIV-2a {stem} training-label MAT")
            rows.append((stem, gdf, lab))
    return rows


def _openbmi_source_file(source: Path, session: int, subject: int) -> Path:
    """Locate one OpenBMI MI file for a session/subject pair.

    The original GigaDB filenames are unambiguous and preferred. The OpenBMI
    toolbox session/subject layout with ``EEG_MI.mat`` files is also accepted.
    """
    source = Path(source).expanduser().resolve()
    official = f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
    try:
        return _find_unique(source, [official], f"OpenBMI session {session}, subject {subject}")
    except FileNotFoundError:
        pass

    candidates = [
        source / f"session{session}" / f"s{subject}" / "EEG_MI.mat",
        source / f"session{session}" / f"s{subject:02d}" / "EEG_MI.mat",
        source / f"session{session}" / f"subj{subject:02d}" / "EEG_MI.mat",
        source / f"session{session}" / f"subject{subject:02d}" / "EEG_MI.mat",
    ]
    existing = [p.resolve() for p in candidates if p.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise RuntimeError(
            f"Ambiguous OpenBMI session {session}, subject {subject}: "
            + ", ".join(str(p) for p in existing)
        )

    # Alternate source layout: find EEG_MI.mat files whose path contains an
    # unmistakable session and subject component. Do not guess from arbitrary
    # parent directory names.
    session_tokens = {f"session{session}", f"session{session:02d}", f"sess{session}", f"sess{session:02d}"}
    subject_tokens = {f"s{subject}", f"s{subject:02d}", f"subj{subject}", f"subj{subject:02d}", f"subject{subject}", f"subject{subject:02d}"}
    matches = []
    for p in source.rglob("EEG_MI.mat"):
        parts = {part.lower() for part in p.parts}
        if parts.intersection(session_tokens) and parts.intersection(subject_tokens):
            matches.append(p.resolve())
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find OpenBMI session {session}, subject {subject} below {source}. "
            f"Expected the official filename {official} or the OpenBMI-toolbox layout "
            f"session{session}/s{subject}/EEG_MI.mat."
        )
    raise RuntimeError(
        f"Ambiguous OpenBMI session {session}, subject {subject}: found {len(matches)} EEG_MI.mat files. "
        "Point --source at a single extracted dataset copy."
    )


def inspect_source(dataset: str, source: Path) -> dict:
    """Validate source-file discovery without preprocessing EEG data."""
    ds = dataset.lower()
    source = Path(source).expanduser().resolve()
    if ds == "bciciv2a":
        rows = _bciciv2a_source_files(source)
        gdfs = [g for _, g, _ in rows]
        labels = [m for _, _, m in rows if m is not None]
        eval_labels = [m for stem, _, m in rows if stem.endswith("E") and m is not None]
        train_labels = [m for stem, _, m in rows if stem.endswith("T") and m is not None]
        return {
            "dataset": ds,
            "source": str(source),
            "recording_files": len(gdfs),
            "evaluation_true_label_files": len(eval_labels),
            "optional_training_label_files": len(train_labels),
            "files_found": len(gdfs) + len(labels),
            "training_label_source": "MAT files when present; otherwise GDF class-cue annotations",
            "ok": len(gdfs) == 18 and len(eval_labels) == 9,
        }
    if ds == "openbmi":
        files = [_openbmi_source_file(source, session, subject) for session in (1, 2) for subject in range(1, 55)]
        unique = {str(p) for p in files}
        return {
            "dataset": ds,
            "source": str(source),
            "session_subject_files": len(files),
            "unique_files": len(unique),
            "ok": len(files) == 108 and len(unique) == 108,
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def _bciciv2a_training_labels_from_events(events: np.ndarray, event_labels: dict[str, int]) -> np.ndarray:
    """Recover 0-based training classes from official BCICIV-2a GDF cues."""
    cue_names = ["769", "770", "771", "772"]
    cue_code_to_class = {event_labels[name]: cls for cls, name in enumerate(cue_names) if name in event_labels}
    if len(cue_code_to_class) != 4:
        raise RuntimeError(
            f"BCICIV-2a GDF annotations do not expose all four class cues {cue_names}. "
            f"Found annotation keys: {sorted(event_labels)}"
        )
    y = np.array(
        [cue_code_to_class[int(code)] for code in np.asarray(events)[:, 2] if int(code) in cue_code_to_class],
        dtype=int,
    )
    return y


def parse_bciciv2a_file(gdf_path: Path, label_path: Path | None, chans=tuple(range(22))):
    import mne

    fs = 250
    offset = 2
    raw = mne.io.read_raw_gdf(str(gdf_path), stim_channel="auto", preload=True, verbose=False)
    events, event_labels = mne.events_from_annotations(raw, verbose=False)
    preferred = ["768", "Start of Trial, Trigger at t=0s", "Start of Trial", "start of trial", "Start"]
    start_code = next((event_labels[k] for k in preferred if k in event_labels), None)
    if start_code is None:
        raise RuntimeError(
            f"Could not identify the BCICIV-2a trial-start annotation (768) in {gdf_path}. "
            f"Available annotations: {sorted(event_labels)}"
        )
    eeg = raw.get_data()[list(chans), :]
    starts = [e for e in events[:, [0, 2]].tolist() if e[1] == start_code]
    interval = np.arange(0, 4 * fs) + offset * fs
    x = np.stack([eeg[:, interval + e[0]] for e in starts], axis=2) * 1e6
    if x.shape[-1] != 288:
        raise RuntimeError(f"Expected 288 trials in {gdf_path}, found {x.shape[-1]}")
    if label_path is not None:
        y = loadmat(label_path)["classlabel"].squeeze() - 1
        if len(y) != 288:
            raise RuntimeError(f"Expected 288 labels in {label_path}, found {len(y)}")
    else:
        # In the official BCICIV-2a training GDFs, cue annotation descriptions
        # 769/770/771/772 correspond to left hand/right hand/feet/tongue.
        # MNE maps those descriptions to internal event ids, so recover the
        # semantic class from the description rather than assuming numeric MNE ids.
        try:
            y = _bciciv2a_training_labels_from_events(events, event_labels)
        except RuntimeError as exc:
            raise RuntimeError(f"Training labels were not supplied for {gdf_path}. {exc}") from exc
        if len(y) != 288:
            raise RuntimeError(
                f"Expected 288 class-cue annotations in training GDF {gdf_path}, found {len(y)}. "
                "If your provider supplies separate training label MAT files, place them beside or below --source."
            )
    return {"x": x, "y": y, "c": np.array(raw.info["ch_names"])[list(chans)].tolist(), "s": fs}


def prepare_bciciv2a(source: Path, data_root: Path) -> Path:
    source_rows = _bciciv2a_source_files(source)
    by_stem = {stem: (gdf, lab) for stem, gdf, lab in source_rows}
    rawmat = data_root / "bciciv2a" / "rawMat"
    rawmat.mkdir(parents=True, exist_ok=True)
    for session_tag, suffix in [("s", "T"), ("se", "E")]:
        for i in range(1, 10):
            stem = f"A{i:02d}{suffix}"
            gdf, lab = by_stem[stem]
            out = rawmat / f"{session_tag}{i:03d}.mat"
            if not out.exists():
                savemat(out, parse_bciciv2a_file(gdf, lab))
    return mat_to_python(rawmat, data_root / "bciciv2a" / "rawPython")


def parse_openbmi_file(path: Path, chans=OPENBMI_CHANNEL_INDICES, downsample_factor: int = 4):
    """Mirror the OpenBMI preprocessing used for NEXUS-MI."""
    import resampy

    d = loadmat(path)
    x = np.concatenate((d["EEG_MI_train"][0, 0]["smt"], d["EEG_MI_test"][0, 0]["smt"]), axis=1).astype(np.float32)
    y = np.concatenate(
        (d["EEG_MI_train"][0, 0]["y_dec"].squeeze(), d["EEG_MI_test"][0, 0]["y_dec"].squeeze()), axis=0
    ).astype(int) - 1
    c = np.array([m.item() for m in d["EEG_MI_train"][0, 0]["chan"].squeeze().tolist()])
    fs = int(d["EEG_MI_train"][0, 0]["fs"].squeeze().item())
    x = x[:, :, np.array(chans)]
    c = c[np.array(chans)]
    if downsample_factor != 1:
        x_new = np.zeros((int(x.shape[0] / downsample_factor), x.shape[1], x.shape[2]), np.float32)
        for i in range(x.shape[2]):
            x_new[:, :, i] = resampy.resample(x[:, :, i], fs, fs / downsample_factor, axis=0)
        x = x_new
        fs = fs / downsample_factor
    x = np.transpose(x, (2, 0, 1))
    if x.shape[0] != len(chans):
        raise RuntimeError(f"OpenBMI channel selection failed for {path}: {x.shape}")
    if len(y) != 200:
        raise RuntimeError(f"Expected 200 MI trials in {path}, found {len(y)}")
    return {"x": x, "y": y, "c": c, "s": fs}


def prepare_openbmi(source: Path, data_root: Path) -> Path:
    rawmat = data_root / "openbmi" / "rawMat"
    rawmat.mkdir(parents=True, exist_ok=True)
    for session in (1, 2):
        tag = "s" if session == 1 else "se"
        for i in range(1, 55):
            inp = _openbmi_source_file(source, session, i)
            out = rawmat / f"{tag}{i:03d}.mat"
            if not out.exists():
                savemat(out, parse_openbmi_file(inp))
    return mat_to_python(rawmat, data_root / "openbmi" / "rawPython")


def mat_to_python(dataset_path: Path, save_path: Path) -> Path:
    save_path.mkdir(parents=True, exist_ok=True)
    rows = [["id", "relativeFilePath", "label", "subject", "session"]]
    idx = 0
    for f in sorted(dataset_path.glob("*.mat")):
        d = loadmat(f, verify_compressed_data_integrity=False)
        eeg = np.transpose(d["x"], (2, 0, 1)).astype("float32")
        labels = d["y"]
        subject = f"{int(f.name[-7:-4]):03d}"
        session = 1 if f.name[1] == "e" else 0
        if len(labels) == 1:
            labels = np.transpose(labels)
        for j, label in enumerate(labels):
            lab = int(np.asarray(label).reshape(-1)[0])
            item = {"id": idx, "data": eeg[j, :, :], "label": lab}
            rel = f"{idx:05d}.dat"
            with open(save_path / rel, "wb") as fp:
                pickle.dump(item, fp)
            rows.append([idx, rel, lab, subject, session])
            idx += 1
    with open(save_path / "dataLabels.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(save_path / "dataInfo.csv", "w", newline="") as f:
        csv.writer(f).writerows([["fs", 250], ["chanName", "See source dataset"]])
    return save_path


def validate_processed(dataset: str, data_root: Path) -> dict:
    """Validate processed EEG metadata, cardinality, labels, and sampled trials.

    The validator is intentionally stricter than the training loader so that a
    malformed or incomplete preprocessing result fails before an expensive
    experiment is launched. Every metadata row and referenced file is checked;
    trial contents are inspected deterministically for the first and last trial
    of each subject/session pair.
    """
    ds = dataset.lower()
    if ds not in {"bciciv2a", "openbmi"}:
        raise ValueError(f"Unsupported dataset: {dataset}")

    expected_subjects = 9 if ds == "bciciv2a" else 54
    expected_trials_per_session = 288 if ds == "bciciv2a" else 200
    expected_channels = 22 if ds == "bciciv2a" else 20
    expected_classes = 4 if ds == "bciciv2a" else 2
    expected_trials_per_class = expected_trials_per_session // expected_classes
    required_columns = ["id", "relativeFilePath", "label", "subject", "session"]

    folder = Path(data_root).expanduser().resolve() / ds / "rawPython"
    labels_path = folder / "dataLabels.csv"
    info_path = folder / "dataInfo.csv"
    _require(labels_path, "processed dataLabels.csv")
    _require(info_path, "processed dataInfo.csv")

    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    metadata_errors: list[str] = []
    if fieldnames != required_columns:
        metadata_errors.append(
            f"dataLabels.csv columns are {fieldnames!r}; expected {required_columns!r}"
        )

    by_subject_session: dict[tuple[str, str], int] = {}
    class_counts: dict[tuple[str, str], dict[int, int]] = {}
    rows_by_pair: dict[tuple[str, str], list[dict]] = {}
    missing_file_count = 0
    missing_files_sample: list[str] = []
    seen_ids: set[int] = set()
    seen_paths: set[str] = set()
    duplicate_ids: list[int] = []
    duplicate_paths: list[str] = []
    invalid_rows: list[str] = []

    for row_index, row in enumerate(rows, start=2):
        try:
            row_id = int(row.get("id", ""))
            rel = str(row.get("relativeFilePath", "")).strip()
            label = int(row.get("label", ""))
            subject = str(row.get("subject", "")).strip().zfill(3)
            session = str(row.get("session", "")).strip()
        except (TypeError, ValueError) as exc:
            invalid_rows.append(f"row {row_index}: unparseable metadata ({exc})")
            continue

        if not rel:
            invalid_rows.append(f"row {row_index}: empty relativeFilePath")
            continue
        if label not in range(expected_classes):
            invalid_rows.append(
                f"row {row_index}: label {label} outside expected range 0..{expected_classes - 1}"
            )
        if subject not in {f"{i:03d}" for i in range(1, expected_subjects + 1)}:
            invalid_rows.append(f"row {row_index}: unexpected subject {subject!r}")
        if session not in {"0", "1"}:
            invalid_rows.append(f"row {row_index}: unexpected session {session!r}")

        if row_id in seen_ids:
            duplicate_ids.append(row_id)
        seen_ids.add(row_id)
        if rel in seen_paths:
            duplicate_paths.append(rel)
        seen_paths.add(rel)

        key = (subject, session)
        by_subject_session[key] = by_subject_session.get(key, 0) + 1
        class_counts.setdefault(key, {})[label] = class_counts.setdefault(key, {}).get(label, 0) + 1
        normalized = {
            "id": row_id,
            "relativeFilePath": rel,
            "label": label,
            "subject": subject,
            "session": session,
        }
        rows_by_pair.setdefault(key, []).append(normalized)

        data_file = folder / rel
        if not data_file.is_file():
            missing_file_count += 1
            if len(missing_files_sample) < 10:
                missing_files_sample.append(str(data_file))

    expected_rows = expected_subjects * 2 * expected_trials_per_session
    expected_pairs = {
        (f"{i:03d}", str(session))
        for i in range(1, expected_subjects + 1)
        for session in (0, 1)
    }
    actual_pairs = set(by_subject_session)
    missing_pairs = sorted(expected_pairs - actual_pairs)
    unexpected_pairs = sorted(actual_pairs - expected_pairs)
    bad_counts = {
        f"{subject}/session{session}": count
        for (subject, session), count in sorted(by_subject_session.items())
        if count != expected_trials_per_session
    }
    bad_class_counts: dict[str, dict[int, int]] = {}
    for pair in sorted(expected_pairs & actual_pairs):
        observed = class_counts.get(pair, {})
        if any(observed.get(cls, 0) != expected_trials_per_class for cls in range(expected_classes)):
            bad_class_counts[f"{pair[0]}/session{pair[1]}"] = {
                cls: int(observed.get(cls, 0)) for cls in range(expected_classes)
            }

    # Confirm the sampling metadata written by preprocessing.
    info_rows: dict[str, str] = {}
    with open(info_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                info_rows[str(row[0]).strip()] = str(row[1]).strip()
    sampling_hz = None
    try:
        sampling_hz = float(info_rows.get("fs", ""))
    except ValueError:
        metadata_errors.append(f"dataInfo.csv has invalid fs value {info_rows.get('fs')!r}")
    if sampling_hz != 250.0:
        metadata_errors.append(f"dataInfo.csv fs is {sampling_hz!r}; expected 250")

    sampled_file_errors: list[str] = []
    sampled_shapes: set[tuple[int, ...]] = set()
    sampled_trials_checked = 0
    if missing_file_count == 0:
        for pair in sorted(expected_pairs & actual_pairs):
            pair_rows = sorted(rows_by_pair[pair], key=lambda row: row["id"])
            sample_rows = pair_rows[:1]
            if len(pair_rows) > 1:
                sample_rows += pair_rows[-1:]
            for row in sample_rows:
                trial_path = folder / row["relativeFilePath"]
                try:
                    with open(trial_path, "rb") as fp:
                        item = pickle.load(fp)
                except Exception as exc:
                    sampled_file_errors.append(f"{trial_path}: could not load ({exc})")
                    continue
                sampled_trials_checked += 1
                if not isinstance(item, dict) or "data" not in item or "label" not in item:
                    sampled_file_errors.append(f"{trial_path}: expected a mapping containing data and label")
                    continue
                try:
                    stored_label = int(item["label"])
                except (TypeError, ValueError):
                    sampled_file_errors.append(f"{trial_path}: stored label is not an integer")
                    continue
                if stored_label != row["label"]:
                    sampled_file_errors.append(
                        f"{trial_path}: stored label {stored_label} != CSV label {row['label']}"
                    )
                if "id" in item:
                    try:
                        if int(item["id"]) != row["id"]:
                            sampled_file_errors.append(
                                f"{trial_path}: stored id {item['id']!r} != CSV id {row['id']}"
                            )
                    except (TypeError, ValueError):
                        sampled_file_errors.append(f"{trial_path}: stored id is not an integer")
                arr = np.asarray(item["data"])
                sampled_shapes.add(tuple(int(v) for v in arr.shape))
                if arr.shape != (expected_channels, 1000):
                    sampled_file_errors.append(
                        f"{trial_path}: shape {tuple(arr.shape)} != {(expected_channels, 1000)}"
                    )
                if not np.issubdtype(arr.dtype, np.number):
                    sampled_file_errors.append(f"{trial_path}: data dtype {arr.dtype} is not numeric")
                elif not np.isfinite(arr).all():
                    sampled_file_errors.append(f"{trial_path}: data contains NaN or infinite values")

    subjects = sorted({subject for subject, _ in actual_pairs})
    sessions = sorted({session for _, session in actual_pairs})
    ok = (
        fieldnames == required_columns
        and len(rows) == expected_rows
        and len(subjects) == expected_subjects
        and sessions == ["0", "1"]
        and not metadata_errors
        and not invalid_rows
        and not duplicate_ids
        and not duplicate_paths
        and not bad_counts
        and not bad_class_counts
        and not missing_pairs
        and not unexpected_pairs
        and missing_file_count == 0
        and not sampled_file_errors
        and sampled_trials_checked == len(expected_pairs) * 2
    )
    result = {
        "dataset": ds,
        "folder": str(folder),
        "rows": len(rows),
        "subjects": len(subjects),
        "sessions": sessions,
        "expected_rows": expected_rows,
        "expected_trials_per_subject_session": expected_trials_per_session,
        "expected_trials_per_class_per_session": expected_trials_per_class,
        "expected_label_values": list(range(expected_classes)),
        "metadata_columns": fieldnames,
        "metadata_errors": metadata_errors,
        "invalid_rows_sample": invalid_rows[:10],
        "duplicate_ids_sample": sorted(set(duplicate_ids))[:10],
        "duplicate_paths_sample": sorted(set(duplicate_paths))[:10],
        "bad_subject_session_counts": bad_counts,
        "bad_class_counts": bad_class_counts,
        "missing_subject_sessions": [f"{s}/session{se}" for s, se in missing_pairs],
        "unexpected_subject_sessions": [f"{s}/session{se}" for s, se in unexpected_pairs],
        "missing_trial_file_count": missing_file_count,
        "missing_trial_files_sample": missing_files_sample,
        "sampling_hz": sampling_hz,
        "sampled_trials_checked": sampled_trials_checked,
        "sampled_trial_shapes": [list(shape) for shape in sorted(sampled_shapes)],
        "expected_trial_shape": [expected_channels, 1000],
        "sampled_file_errors": sampled_file_errors[:10],
        "ok": ok,
    }
    if not ok:
        raise RuntimeError(f"Processed-data validation failed: {result}")
    return result

