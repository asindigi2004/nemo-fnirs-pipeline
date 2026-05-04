# NEMO fNIRS — Frontal Valence Sensitivity Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Seed 42](https://img.shields.io/badge/seed-42-green.svg)]()
[![DSAA 2026](https://img.shields.io/badge/venue-DSAA%202026-orange.svg)]()

Replication and forward-modelling pipeline for the paper:

> **Frontal fNIRS Sensitivity Validation for Emotional Valence Processing**  
> Archita Sindigi · BITS Pilani, K.K. Birla Goa Campus · DSAA 2026  
> Supervised by Dr. Manideepa Mukherjee

---

## ⚠️ Caveats (read before interpreting results)

| Caveat | Detail |
|--------|--------|
| No MRI co-registration | Nominal `artinis-brite23` positions only |
| Head→MNI approximation | Okamoto 2004 analytic formula, ~8 mm SD error |
| Median MNI snap = 35.3 mm | Exceeds Metz 2022 25 mm threshold — channel-level Spearman is **exploratory** |
| Region labels | Anterior / Frontopolar / Dorsal are cap-geometry labels, **not neuroanatomical regions** |
| Regional hierarchy p-value | 3-region exact match has p = 1/6 ≈ 0.17 under a uniform null — supporting, not confirming |

---

## Key results

| Result | Value |
|--------|-------|
| Valence FDR-significant channels | **8 / 17** (q < 0.05, BH) |
| Arousal FDR-significant channels | **0 / 17** (specificity check) |
| Directional consistency | **16 / 17** positive for PV > NV |
| Regional hierarchy match | **YES** (Frontopolar > Anterior > Dorsal) |
| Channel Spearman ρ | 0.390 (n = 8, exploratory) |
| Median MNI snap | 35.3 mm |

---

## Data setup

This pipeline requires two external datasets that are **not included** in this repository. Download them separately:

### 1. NEMO dataset
1. Go to [osf.io/pd9rv](https://osf.io/pd9rv)
2. Download and extract the archive
3. Locate the file at `empe_csv/epochs.csv`

### 2. fOLD toolbox
1. Clone or download [github.com/gabri470/fOLD-public](https://github.com/gabri470/fOLD-public)
2. You need the file at `Supplementary/10-10.xls`, sheet `3_Chn`

Both datasets have their own licences — please consult them before use.

---

## Installation

```bash
git clone https://github.com/asindigi2004/nemo-fnirs-pipeline.git
cd nemo-fnirs-pipeline
pip install -r requirements.txt
```

Python 3.9+ required. Tested on Python 3.10 and 3.11.

---

## Usage

```bash
python nemo_pipeline.py \
    --fold-dir   /path/to/fOLD-public-master/Supplementary \
    --epochs-csv /path/to/NEMO/empe_csv/epochs.csv \
    --output-dir ./outputs \
    --seed       42
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--fold-dir` | Yes | — | Path to `fOLD-public-master/Supplementary/` (contains `10-10.xls`) |
| `--epochs-csv` | Yes | — | Path to NEMO epochs CSV (`empe_csv/epochs.csv`) |
| `--output-dir` | No | `./outputs` | Directory for figures and `results.json` |
| `--seed` | No | `42` | Random seed for permutation test. Paper results used seed 42. |

---

## Outputs

Running the pipeline produces:

```
outputs/
├── results.json          # All numeric results (see structure below)
└── figures/
    └── MAIN_FIGURE.png   # 8-panel summary figure
```

### results.json structure

```json
{
  "pipeline_version": "DSAA_2026",
  "random_seed": 42,
  "caveats": { ... },
  "empirical": {
    "n_subjects": 31,
    "n_valid_channels": 17,
    "n_sig_fdr_bh": 8,
    "n_sig_arousal_fdr": 0,
    "channel_results": {
      "S1_D1": {"t": 4.394, "p_raw": 0.0001, "p_fdr": 0.0021, "sig_fdr": true, "region": "Anterior"},
      ...
    }
  },
  "forward_model": {
    "regional_hierarchy": {
      "fold_order": ["Frontopolar", "Anterior", "Dorsal"],
      "empirical_order": ["Frontopolar", "Anterior", "Dorsal"],
      "match": true
    },
    "channel_spearman": {"r": 0.39, "p": 0.339, "p_perm": 0.353, "status": "EXPLORATORY"},
    ...
  }
}
```

An example `outputs/example_results.json` is committed to this repository — use it to verify your run matches the paper without needing to download the data first.

---

## Reproducing paper results

```bash
python nemo_pipeline.py \
    --fold-dir   /path/to/fOLD-public-master/Supplementary \
    --epochs-csv /path/to/NEMO/empe_csv/epochs.csv \
    --output-dir ./outputs \
    --seed 42
```

Expected key outputs (all verified against paper):

```
8/17 FDR-significant valence channels
0/17 arousal channels
Spearman r = 0.390
Median MNI snap = 35.3 mm
Regional hierarchy match: YES
```

---

## Running the tests

No external data needed. The smoke test runs entirely on synthetic data:

```bash
pip install pytest
python -m pytest tests/smoke_test.py -v
```

Expected output: **29 passed, 4 skipped** on the original pipeline;
**33 passed** on the refactored version.

The 4 skipped tests cover `mirror_detector_d8()`, which is only present
in the refactored pipeline. All core statistical functions are covered
in both versions.

---

## Pipeline structure

| Part | Function | Description |
|------|----------|-------------|
| 1 | `load_fold_table()` | Load and validate fOLD Brodmann table |
| 2 | `compute_mni_midpoints()` | Brite-24 optode midpoints → MNI via Okamoto 2004 |
| 3 | `load_ifg_sensitivity()` | fOLD IFG (BA44+45+47) specificity per channel |
| 4 | `load_epochs()` | Load and validate NEMO epochs CSV |
| 5 | `run_valence_contrast()` | PV−NV and HA−LA contrasts + FDR correction |
| 6 | `run_forward_model()` | Spearman, regional hierarchy, LOO, null ROIs |
| 7 | `generate_figures()` | 8-panel summary figure |
| 8 | `save_results()` | Serialise all results to `results.json` |

---

## References

- Spapé et al. (2024). NEMO: A database for emotion analysis using fNIRS. *IEEE Trans. Affective Computing*, 15(3), 1166–1178. https://doi.org/10.1109/TAFFC.2023.3315971
- Zimeo Morais et al. (2018). fOLD: A toolbox for optode localization. *Neurophotonics*, 5(3), 035003. https://doi.org/10.1117/1.NPh.5.3.035003
- Okamoto et al. (2004). 3D probabilistic anatomical cranio-cerebral correlation. *NeuroImage*, 21(1), 99–111. https://doi.org/10.1016/j.neuroimage.2003.08.026
- Metz et al. (2022). Validation of fNIRS measurements of prefrontal cortex activation. *Frontiers in Human Neuroscience*, 16, 875820. https://doi.org/10.3389/fnhum.2022.875820
- Benjamini & Hochberg (1995). Controlling the false discovery rate. *JRSS-B*, 57(1), 289–300.

---

## Licence

MIT — see `LICENSE`. The NEMO dataset and fOLD toolbox have their own separate licences; this repository does not redistribute either.
