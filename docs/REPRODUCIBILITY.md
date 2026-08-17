# Reproducibility protocol

The public study presets load their run hyperparameters from [`../src/nexus_mi/paper_protocol.yaml`](../src/nexus_mi/paper_protocol.yaml) rather than relying on the low-level parser defaults. The EEGNet architecture is fixed in the runtime and uses the study configuration for channels/classes and `F1/D/F2/C1/dropout`.

The paper uses the **subject as the statistical unit**. For aggregate policy comparisons, each subject is first averaged equally across Session-2 calibration budgets `k={15,20,30}`.

## Session-2 personalization

Session-2 personalization holds the shared backbone fixed and optimizes only EEGNet's subject-specific classifier head (`lastLayer`) using the limited calibration set.

## Primary realization

The complete P1–P6 landscape, P5 component analysis, availability-severity analysis, and subject-level analyses use seed 2026 with matched partitions, model initialization, gateway-group assignment, and realized availability traces for controlled policy comparisons.

## Five-realization P3/P5 robustness

The principal P3/P5 comparison uses:

- model seeds: 2026–2030;
- availability-trace seeds: 12026–12030;
- P5 scheduler tie-break seeds: 22026–22030;
- gateway-group seed: fixed at 2026;
- crossed hierarchical bootstrap: 10,000 draws, preserving the within-subject P3/P5 pairing.

The experiment driver saves the availability traces and seed/partition metadata needed to identify each matched realization.

## Communication units and definitions

Communication is reported in **decimal MB (1 MB = 10^6 bytes)**. Raw byte counters are retained and MiB is reported separately where applicable.

The paper's accepted-update staleness statistic is event-weighted over updates that pass coordinator admission and enter aggregation. The experiment driver reports that definition directly.

The study's raw upload-byte accounting includes the serialized runtime timestamp stored with an update. Its textual serialization can differ by a few bytes between executions, even when learning and communication decisions are identical. This has no effect on training and is far below the precision of the reported decimal-MB values.

## Publication analysis

Run:

```bash
nexus-mi analyze
nexus-mi figures
```

`analyze` computes the publication tables and statistics directly from generated experiment outputs, including Supplementary Tables S2–S7. `figures` regenerates manuscript Figures 2–7 from the analyzed tables.

See [`ENVIRONMENT.md`](ENVIRONMENT.md) for the saved paper-run software environment.
