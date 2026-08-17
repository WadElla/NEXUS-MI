# Paper experiment map

The command-line interface maps directly onto the NEXUS-MI experimental design.

| Command | Paper experiment | Expected completed runs |
|---|---|---:|
| `nexus-mi run ideal --dataset bciciv2a/openbmi` | Ideal-link SB-PH and EIB-PH reference | 4 total |
| `nexus-mi run primary --dataset bciciv2a/openbmi` | P1–P6 default heterogeneous-link landscape | 24 total |
| `nexus-mi run component --dataset openbmi` | OpenBMI EIB-PH P5 component analysis | 5 |
| `nexus-mi run sensitivity --dataset bciciv2a/openbmi` | EIB-PH P3/P5 mild/default/severe availability | 12 total |
| `nexus-mi run robustness` | Five matched P3/P5 realizations, both datasets/regimes | 40 |

The complete paper workload therefore contains **85 completed experiment runs**.

For Session-2 personalization, the shared backbone is held fixed and only the subject-specific classifier head is optimized from the limited calibration data.

## Per-run outputs

Each completed run is self-describing. The run directory records the full configuration and the reported measurements needed for later aggregation, including:

- `run_hyperparams.yaml`: dataset, regime, communication policy, training parameters, and seed/provenance metadata;
- `results_summary.csv`: subject-by-calibration-budget held-out accuracy rows;
- `system_metrics.json`: raw communication bytes, upload admission, accepted staleness, buffering delay, download behavior, and runtime metadata;
- availability-trace/provenance files for matched heterogeneous-link and robustness runs;
- completion status and suite summaries used by `nexus-mi analyze`.

Experiment outputs are created only under the configured local output root and are excluded from Git by default.

## Publication analysis

After the required suites finish, derive the publication tables directly from the generated run directories:

```bash
nexus-mi analyze
```

This creates `outputs/publication_analysis/` containing, where the corresponding runs are available:

- complete ideal/P1–P6 operating points;
- per-calibration-budget accuracy data;
- controlled P3/P5 operating points, communication savings, and paired Student-t/Wilcoxon statistics with Holm adjustment;
- P5 component outcomes;
- subject-level reliability relative to the matched ideal-link reference;
- severity-sensitivity outcomes and paired effects, with Holm adjustment across the three profiles within each dataset;
- subject-level association analysis using 10,000 bootstrap resamples;
- robustness tables derived from the generated robustness outputs:
  - `table_s2a_robustness_cohort_accuracy.csv`;
  - `table_s2b_robustness_paired_effects.csv`;
  - `table_s3_robustness_subject_heterogeneity.csv`;
  - `table_s4_robustness_communication.csv`;
  - `table_s5_robustness_replicates.csv`;

The analysis reads experiment outputs only. `nexus-mi figures` regenerates manuscript Figures 2–7 as vector PDFs directly from the analyzed output tables; plotted publication values are not embedded in the plotting code.

## One-command full reproduction

Once both datasets are downloaded and preprocessed, the full sequence can be launched with:

```bash
./scripts/reproduce_all.sh
```

This is intentionally expensive: it executes the full 85-run protocol before publication analysis and figure generation.

## Fixed communication parameters

The paper presets use FIFO capacity 3 where enabled, stale-update threshold 2 backbone versions, checkpoint-retention margin 5 versions, stale-aware download threshold 1 version, and per-round selection budgets 6 (BCICIV-2a) and 40 (OpenBMI).
