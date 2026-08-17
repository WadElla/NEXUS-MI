#!/usr/bin/env bash
set -euo pipefail

# Full NEXUS-MI paper reproduction after both public datasets have been prepared.
# Large EEG data may live outside the repository via NEXUS_MI_DATA_DIR.
# Experiment outputs may likewise be redirected with NEXUS_MI_OUTPUT_DIR.

nexus-mi validate-data bciciv2a
nexus-mi validate-data openbmi

nexus-mi run ideal --dataset bciciv2a
nexus-mi run ideal --dataset openbmi

nexus-mi run primary --dataset bciciv2a
nexus-mi run primary --dataset openbmi

nexus-mi run component --dataset openbmi

nexus-mi run sensitivity --dataset bciciv2a
nexus-mi run sensitivity --dataset openbmi

nexus-mi run robustness

nexus-mi analyze
nexus-mi figures
