"""Dataset loader for trial-wise NEXUS-MI EEG files."""
from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Iterable

from torch.utils.data import Dataset


class eegDataset(Dataset):
    """Load preprocessed EEG trials described by ``dataLabels.csv``.

    Each row in the label file identifies one pickled trial.  The required
    columns are ``id``, relative file name, and integer class label; subject and
    session columns may follow.  Trial files contain a mapping with at least
    ``data`` and ``label`` entries.

    The class name is retained for compatibility with the experiment code.
    """

    def __init__(
        self,
        dataPath,
        dataLabelsPath,
        transform=None,
        preloadData: bool = False,
    ) -> None:
        self.labels: list[list] = []
        self.data: list[dict] = []
        self.dataPath = dataPath
        self.dataLabelsPath = dataLabelsPath
        self.preloadData = bool(preloadData)
        self.transform = transform

        with Path(dataLabelsPath).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=",")
            next(reader, None)  # header
            self.labels = [list(row) for row in reader]

        for row in self.labels:
            row[2] = int(row[2])

        if self.preloadData:
            self.data = [self._load_trial(row) for row in self.labels]

    def _trial_path(self, row: list) -> Path:
        return Path(self.dataPath) / str(row[1])

    def _load_trial(self, row: list) -> dict:
        with self._trial_path(row).open("rb") as handle:
            trial = pickle.load(handle)
        if self.transform is not None:
            trial = self.transform(trial)
        return trial

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        trial = self.data[idx] if self.preloadData else self._load_trial(self.labels[idx])
        return {"data": trial["data"], "label": trial["label"]}

    def createPartialDataset(self, idx: Iterable[int], loadNonLoadedData: bool = False) -> None:
        indices = list(idx)
        self.labels = [self.labels[i] for i in indices]
        if self.preloadData:
            self.data = [self.data[i] for i in indices]
        elif loadNonLoadedData:
            self.data = [self._load_trial(row) for row in self.labels]
            self.preloadData = True

    def combineDataset(self, otherDataset, loadNonLoadedData: bool = False) -> None:
        self.labels.extend(otherDataset.labels)
        if self.preloadData or loadNonLoadedData:
            self.data = [self._load_trial(row) for row in self.labels]
            self.preloadData = True

    def changeTransform(self, newTransform) -> None:
        self.transform = newTransform
        if self.preloadData:
            self.data = [self._load_trial(row) for row in self.labels]
