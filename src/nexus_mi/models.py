"""Neural-network definitions used by NEXUS-MI.

The experiments use the EEGNet architecture introduced by Lawhern et al.
(2018).  This module provides the PyTorch implementation and the exact
configuration used by the NEXUS-MI study experiments.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    """2-D convolution with an optional max-norm constraint on its filters."""

    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.doWeightNorm:
            with torch.no_grad():
                self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm))
        return super().forward(x)


class eegNet(nn.Module):
    """EEGNet variant used in the NEXUS-MI experiments.

    The study configuration uses 1,000 samples per trial,
    ``F1=8``, depth multiplier ``D=2``, ``F2=16``, temporal kernel length
    ``C1=125``, and dropout 0.5.  ``nChan`` and ``nClass`` are dataset-specific.

    The attribute names and module nesting are kept stable because the
    experiment driver separates ``firstBlocks`` (shared backbone) from
    ``lastLayer`` (subject-specific classifier head).
    """

    def __init__(
        self,
        nChan: int,
        nTime: int,
        nClass: int = 2,
        dropoutP: float = 0.25,
        F1: int = 8,
        D: int = 2,
        C1: int = 125,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.F1 = int(F1)
        self.D = int(D)
        self.F2 = self.F1 * self.D
        self.nTime = int(nTime)
        self.nClass = int(nClass)
        self.nChan = int(nChan)
        self.C1 = int(C1)

        temporal_spatial = nn.Sequential(
            nn.Conv2d(
                1,
                self.F1,
                kernel_size=(1, self.C1),
                padding=(0, self.C1 // 2),
                bias=False,
            ),
            nn.BatchNorm2d(self.F1),
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                kernel_size=(self.nChan, 1),
                groups=self.F1,
                bias=False,
                max_norm=1,
            ),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=4),
            nn.Dropout(p=dropoutP),
        )

        separable_temporal = nn.Sequential(
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                kernel_size=(1, 22),
                padding=(0, 11),
                groups=self.F1 * self.D,
                bias=False,
            ),
            nn.Conv2d(self.F1 * self.D, self.F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=8),
            nn.Dropout(p=dropoutP),
        )

        self.firstBlocks = nn.Sequential(temporal_spatial, separable_temporal)

        # Match the experiment's constructor semantics: infer the classifier
        # kernel from a dummy forward pass before the final layer is created.
        # The random dummy tensor is intentional because it preserves the RNG
        # consumption used by the study experiment initialization.
        dummy = torch.rand(1, 1, self.nChan, self.nTime)
        self.firstBlocks.eval()
        feature_shape = self.firstBlocks(dummy).shape[2:]
        self.fSize = feature_shape

        self.lastLayer = nn.Sequential(
            nn.Conv2d(self.F2, self.nClass, kernel_size=(1, self.fSize[1])),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.firstBlocks(x)
        x = self.lastLayer(x)
        return x.squeeze(3).squeeze(2)


EEGNet = eegNet

__all__ = ["Conv2dWithConstraint", "eegNet", "EEGNet"]
