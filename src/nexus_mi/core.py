"""Shared data, model, training, and personalization primitives for NEXUS-MI."""
from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from .eeg_dataset import eegDataset
from . import models


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False if deterministic else True


def resolve_network_ctor(network_name: str):
    if network_name.lower() != "eegnet":
        raise RuntimeError(
            f"NEXUS-MI paper experiments use EEGNet only; got {network_name!r}."
        )
    return models.eegNet


def get_model_args(datasetId: int, network_name: str) -> dict:
    if network_name.lower() != "eegnet":
        raise RuntimeError("NEXUS-MI experiments use EEGNet only.")
    dataset_id = int(datasetId)
    if dataset_id == 0:
        return {
            'nChan': 22,
            'nTime': 1000,
            'nClass': 4,
            'dropoutP': 0.5,
            'F1': 8,
            'D': 2,
            'C1': 125,
        }
    if dataset_id == 1:
        return {
            'nChan': 20,
            'nTime': 1000,
            'nClass': 2,
            'dropoutP': 0.5,
            'F1': 8,
            'D': 2,
            'C1': 125,
        }
    raise ValueError(f"Unsupported datasetId {datasetId!r}; expected 0 (BCICIV-2a) or 1 (OpenBMI).")


def prepare_batch_x(x: torch.Tensor, expects_bands: bool) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x).float()
    if expects_bands:
        raise RuntimeError("NEXUS-MI uses raw EEG with EEGNet, not filter-bank input.")
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4 and x.shape[1] == 1:
        return x
    raise ValueError(f'Expected raw input (B,Chan,Time) but got {tuple(x.shape)}')


def backbone_keys_from_state(state: Dict[str, torch.Tensor]) -> List[str]:
    return [k for k in state.keys() if not k.startswith('lastLayer.')]


def merge_backbone_with_local_head(global_backbone: Dict[str, torch.Tensor], local_full_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    merged = {}
    for k, v in local_full_state.items():
        merged[k] = global_backbone[k] if k in global_backbone else v
    for k, v in global_backbone.items():
        if k not in merged:
            merged[k] = v
    return merged


def freeze_all_but_last_layer(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if not hasattr(model, 'lastLayer'):
        raise RuntimeError('Model has no attribute lastLayer (needed for head-only tuning).')
    for p in model.lastLayer.parameters():
        p.requires_grad = True


def subject_to_report_id(subj: str) -> str:
    s = str(subj).strip()
    digits = ''.join(ch for ch in s if ch.isdigit())
    if digits:
        return f'{int(digits):03d}'
    return s


def resolve_subject_arg(sub_arg: str, all_subjects: List[str]) -> str:
    if sub_arg in all_subjects:
        return sub_arg
    s = str(sub_arg).strip()
    digits = ''.join(ch for ch in s if ch.isdigit())
    if digits:
        target = int(digits)
        cands = []
        for cand in all_subjects:
            cd = ''.join(ch for ch in str(cand) if ch.isdigit())
            if cd and int(cd) == target:
                cands.append(cand)
        if len(cands) == 1:
            return cands[0]
    raise RuntimeError(f'Unknown subject {sub_arg!r}. Available: {all_subjects}')


def get_device(nGPU: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f'cuda:{int(nGPU)}')
    return torch.device('cpu')


def split_session_indices(labels: List[List[str]], subj: str) -> Tuple[List[int], List[int]]:
    idx_ses1 = [i for i, row in enumerate(labels) if len(row) > 4 and row[3] == subj and row[4] == '0']
    idx_ses2 = [i for i, row in enumerate(labels) if len(row) > 4 and row[3] == subj and row[4] == '1']
    return idx_ses1, idx_ses2


def indices_by_class(labels_rows: List[List[str]]) -> Dict[int, List[int]]:
    m: Dict[int, List[int]] = {}
    for i, row in enumerate(labels_rows):
        y = int(row[2])
        m.setdefault(y, []).append(i)
    return m


def make_calib_train_val(calib_ds: eegDataset, k_per_class: int, seed: int) -> Tuple[eegDataset, Optional[eegDataset]]:
    if k_per_class <= 1:
        return calib_ds, None
    rng = random.Random(seed)
    by_cls = indices_by_class(calib_ds.labels)
    train_idx, val_idx = [], []
    for _, idxs in by_cls.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        val_idx.append(idxs[0])
        train_idx.extend(idxs[1:])
    if not train_idx or not val_idx:
        return calib_ds, None
    train_ds = copy.deepcopy(calib_ds)
    train_ds.createPartialDataset(train_idx)
    val_ds = copy.deepcopy(calib_ds)
    val_ds.createPartialDataset(val_idx)
    return train_ds, val_ds


def _make_loader(
    ds: eegDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> torch.utils.data.DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=(bool(num_workers) and int(num_workers) > 0),
    )


def train_fixed_epochs(
    model: torch.nn.Module,
    train_ds: eegDataset,
    device: torch.device,
    expects_bands: bool,
    lr: float,
    batch_size: int,
    epochs: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    train_head: bool = True,
) -> Dict[str, float]:
    model = model.to(device)
    if not train_head and hasattr(model, 'lastLayer'):
        for p in model.lastLayer.parameters():
            p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr)
    crit = torch.nn.NLLLoss(reduction='sum')
    loader = _make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    last_loss = 0.0
    last_acc = 0.0
    for _ in range(int(epochs)):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for d in loader:
            x = prepare_batch_x(d['data'], expects_bands).to(
                device, non_blocking=True
            )
            y = d['label'].long().to(device, non_blocking=True)
            out = model(x)
            loss = crit(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        last_loss = total_loss / max(1, total)
        last_acc = correct / max(1, total)

    return {'train_loss': float(last_loss), 'train_acc': float(last_acc)}


def eval_model(
    model: torch.nn.Module,
    ds: eegDataset,
    device: torch.device,
    expects_bands: bool,
    batch_size: int,
    n_class: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, object]:
    model = model.to(device)
    model.eval()
    crit = torch.nn.NLLLoss(reduction='sum')
    loader = _make_loader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    total_loss = 0.0
    total = 0
    correct = 0
    cm = np.zeros((n_class, n_class), dtype=int)

    with torch.no_grad():
        for d in loader:
            x = prepare_batch_x(d['data'], expects_bands).to(
                device, non_blocking=True
            )
            y = d['label'].long().to(device, non_blocking=True)
            out = model(x)
            loss = crit(out, y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            for yt, yp in zip(
                y.detach().cpu().numpy(), pred.detach().cpu().numpy()
            ):
                if 0 <= yt < n_class and 0 <= yp < n_class:
                    cm[int(yt), int(yp)] += 1
            correct += (pred == y).sum().item()
            total += y.size(0)

    return {
        'loss': float(total_loss / max(1, total)),
        'acc': float(correct / max(1, total)),
        'cm': cm,
    }


def head_finetune_with_early_stop(
    model: torch.nn.Module,
    calib_train: eegDataset,
    calib_val: Optional[eegDataset],
    test_ds: eegDataset,
    device: torch.device,
    expects_bands: bool,
    batch_size: int,
    lr: float,
    max_epochs: int,
    patience: int,
    seed: int,
    n_class: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    stage2: bool = False,
    stage2_max_epochs: int = 600,
) -> Dict[str, object]:
    freeze_all_but_last_layer(model)
    model = model.to(device)
    crit = torch.nn.NLLLoss(reduction='sum')
    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    train_loader = _make_loader(
        calib_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = (
        None
        if calib_val is None
        else _make_loader(
            calib_val,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val_inacc = float('inf')
    best_epoch = 0
    no_improve = 0
    train_loss_at_earlystop = None

    def run_one_epoch(loader, train: bool):
        model.train() if train else model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.set_grad_enabled(train):
            for d in loader:
                x = prepare_batch_x(d['data'], expects_bands).to(
                    device, non_blocking=True
                )
                y = d['label'].long().to(device, non_blocking=True)
                out = model(x)
                loss = crit(out, y)
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return total_loss / max(1, total), correct / max(1, total)

    for ep in range(int(max_epochs)):
        tr_loss, _ = run_one_epoch(train_loader, train=True)
        if val_loader is None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep + 1
            continue

        _, va_acc = run_one_epoch(val_loader, train=False)
        va_inacc = 1.0 - va_acc
        if va_inacc < best_val_inacc - 1e-6:
            best_val_inacc = va_inacc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep + 1
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= int(patience):
            train_loss_at_earlystop = float(tr_loss)
            break

    model.load_state_dict(best_state)
    if stage2 and val_loader is not None and train_loss_at_earlystop is not None:
        combined = copy.deepcopy(calib_train)
        combined.combineDataset(calib_val)
        comb_loader = _make_loader(
            combined,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + 99,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for _ in range(int(stage2_max_epochs)):
            run_one_epoch(comb_loader, train=True)
            va2_loss, _ = run_one_epoch(val_loader, train=False)
            if float(va2_loss) < float(train_loss_at_earlystop):
                break

    train_metrics = eval_model(
        model,
        calib_train,
        device,
        expects_bands,
        batch_size,
        n_class,
        seed=seed + 7,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_metrics = (
        None
        if calib_val is None
        else eval_model(
            model,
            calib_val,
            device,
            expects_bands,
            batch_size,
            n_class,
            seed=seed + 8,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    )
    test_metrics = eval_model(
        model,
        test_ds,
        device,
        expects_bands,
        batch_size,
        n_class,
        seed=seed + 9,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return {
        'stage1_best_epoch': int(best_epoch),
        'stage1_best_val_inacc': (
            None if val_loader is None else float(best_val_inacc)
        ),
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'stage2_enabled': bool(stage2 and val_loader is not None),
    }
