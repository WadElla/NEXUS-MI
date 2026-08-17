# Dataset setup

EEG recordings are **not included** in this repository. NEXUS-MI uses two public benchmark datasets and processes files downloaded by the user on the user's own machine. The runtime reads only user-supplied dataset locations and the configured local data directory.

Large data can live outside the Git checkout:

```bash
export NEXUS_MI_DATA_DIR=/path/to/nexus-mi-data
```

## BCICIV-2a

Provider: **BCI Competition IV, Data Set 2a**.

- Competition page and download terms: `https://www.bbci.de/competition/iv/`
- Evaluation-set true labels: `https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip`

Download and extract the Data Set 2a GDF archive from the official competition site, then extract the evaluation true-label archive into the same source tree (subdirectories are fine). Follow the provider's current access and use terms.

For the official competition layout, NEXUS-MI expects:

- 18 recordings: `A01T.gdf` ... `A09T.gdf` and `A01E.gdf` ... `A09E.gdf`;
- 9 evaluation label files: `A01E.mat` ... `A09E.mat`.

The training GDFs contain the four motor-imagery cue annotations (`769`–`772`), so separate `A01T.mat` ... `A09T.mat` files are **not required**. If an existing extraction also contains those nine training-label MAT files, NEXUS-MI accepts them as an alternative source of training labels.

Check the extraction before preprocessing:

```bash
nexus-mi inspect-source bciciv2a --source /path/to/extracted/BCICIV2a
```

Then preprocess:

```bash
nexus-mi --data-dir /path/to/nexus-mi-data \
  prepare bciciv2a --source /path/to/extracted/BCICIV2a
```

The experiment-ready dataset is written to:

```text
/path/to/nexus-mi-data/bciciv2a/rawPython/
```

The preparation path reproduces the study representation: 22 EEG channels, four-second epochs, 250 Hz, and 288 trials per subject/session.

## OpenBMI

Provider: **GigaDB, Supporting data for “EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy.”**

- Dataset record: `https://gigadb.org/dataset/100542`

OpenBMI is very large, so it is intentionally not mirrored here. For NEXUS-MI, download the **motor-imagery (`EEG_MI`) MATLAB files** for all 54 subjects in both sessions and extract/store them locally. Other OpenBMI paradigms are not required by this project.

NEXUS-MI accepts the provider filenames recursively:

```text
sess01_subj01_EEG_MI.mat
...
sess01_subj54_EEG_MI.mat
sess02_subj01_EEG_MI.mat
...
sess02_subj54_EEG_MI.mat
```

It also accepts the OpenBMI-toolbox directory organization:

```text
OpenBMI-source/
  session1/s1/EEG_MI.mat
  ...
  session1/s54/EEG_MI.mat
  session2/s1/EEG_MI.mat
  ...
  session2/s54/EEG_MI.mat
```

Check source discovery first:

```bash
nexus-mi inspect-source openbmi --source /path/to/extracted/OpenBMI
```

Then preprocess:

```bash
nexus-mi --data-dir /path/to/nexus-mi-data \
  prepare openbmi --source /path/to/extracted/OpenBMI
```

For OpenBMI, NEXUS-MI uses the selected 20-channel motor-imagery montage from the study. The zero-based source-channel indices are `7,32,8,9,33,10,34,12,35,13,36,14,37,17,38,18,39,19,40,20`; they are also recorded in `src/nexus_mi/paper_protocol.yaml`. Source 1000-Hz data are downsampled to 250 Hz with `resampy` (declared as a dependency). The experiment-ready data are written to:

```text
/path/to/nexus-mi-data/openbmi/rawPython/
```

## Validate processed data

Before launching expensive training, validate both datasets:

```bash
nexus-mi --data-dir /path/to/nexus-mi-data validate-data bciciv2a
nexus-mi --data-dir /path/to/nexus-mi-data validate-data openbmi
```

Validation checks the metadata schema, total trial counts, subject/session coverage, per-class counts, uniqueness of trial IDs/paths, the presence of every stored trial file, 250-Hz metadata, and deterministic first/last trial samples from every subject/session pair. Sampled trial files are checked for the expected `channels x 1000 samples` shape, finite numeric data, and agreement between stored IDs/labels and `dataLabels.csv`.

## Data are never committed

The repository `.gitignore` excludes the processed dataset folders and common raw EEG/checkpoint formats. A normal reproduction flow is:

```text
official provider files
        ↓
local NEXUS-MI preprocessing
        ↓
local processed EEG
        ↓
experiments
        ↓
local outputs
```


If you switch to a different source extraction after preprocessing has already been completed, remove the corresponding local `<data-dir>/<dataset>/rawMat/` and `rawPython/` directories before running `prepare` again. This prevents previously generated intermediate files from being reused with a different source tree.

## Dataset citations

If you use the datasets through this repository, please cite the original dataset publications. If you also use the NEXUS-MI framework, implementation, experimental protocol, or analysis methodology, please cite the NEXUS-MI paper as well:

- **BCICIV-2a:** M. Tangermann et al., “Review of the BCI Competition IV,” *Frontiers in Neuroscience*, vol. 6, article 55, 2012.
- **OpenBMI:** M.-H. Lee et al., “EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy,” *GigaScience*, vol. 8, no. 5, giz002, 2019.

Users remain responsible for the original providers' access terms and dataset conditions.
