# NEMO fNIRS — Frontal Sensitivity Validation Pipeline

**DSAA 2026 submission** · Archita Sindigi, BITS Pilani Goa
Supervised by Dr. Manideepa Mukherjee

Replication and forward sensitivity analysis of emotional valence processing using the [NEMO fNIRS dataset](https://osf.io/pd9rv) (Spapé et al. 2024).

---

## What this pipeline does

**Primary result:** Group-level HbO valence contrast (positive > negative) across 31 subjects, 17 valid frontal channels, with Benjamini-Hochberg FDR correction and an arousal specificity check.

**Secondary result (exploratory):** Forward sensitivity validation using the [fOLD](https://github.com/gabri470/fOLD-public) Monte Carlo atlas. IFG ROI = BA44 + BA45 + BA47. Regional hierarchy (Frontopolar > Anterior > Dorsal) compared between fOLD predictions and empirical |t|-values.

**Key result:** 8/17 channels FDR-significant (q < 0.05) for valence; 0/17 for arousal — confirming valence specificity.

---

## Caveats (read before citing)

- No individual MRI co-registration. Optode positions use the artinis-brite23 generic template.
- Head-to-MNI conversion via Okamoto et al. (2004) analytic approximation (~8 mm SD error).
- Median MNI snap distance = 35.3 mm, which **exceeds** the 25 mm Metz et al. (2022) threshold. Channel-level Spearman results are therefore **exploratory** — the regional hierarchy match is the primary forward model result.
- Region labels (Anterior, Frontopolar, Dorsal) reflect cap geometry per Spapé et al. (2024) Fig. 4, **not** verified neuroanatomical regions.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/nemo-fnirs-pipeline.git
cd nemo-fnirs-pipeline
pip install -r requirements.txt
```

Requires Python 3.9+. Tested on 3.10 and 3.11.

---

## Data setup (two downloads required — not included in this repo)

### 1. NEMO dataset
Download from OSF: **https://osf.io/pd9rv**

You need: `empe_csv/epochs.csv` (Task A — emotional perception, semicolon-delimited).

```
data/
└── epochs.csv   ← place it here, or pass the full path via --epochs-csv
```

### 2. fOLD toolbox
Download from GitHub: **https://github.com/gabri470/fOLD-public**

You need: `Supplementary/10-10.xls` (sheet `3_Chn`).

```
fOLD-public-master/
└── Supplementary/
    └── 10-10.xls   ← pass the Supplementary/ folder via --fold-dir
```

> **Note:** Neither dataset is redistributed here. Both have their own licences — cite the original papers (see below).

---

## Usage

```bash
python nemo_pipeline.py \
    --fold-dir   ./fOLD-public-master/Supplementary \
    --epochs-csv ./data/epochs.csv
```

All arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--fold-dir` | Yes | — | Path to fOLD Supplementary/ folder |
| `--epochs-csv` | Yes | — | Path to NEMO epochs.csv |
| `--output-dir` | No | `./outputs` | Where to save figures and results.json |
| `--seed` | No | `42` | Random seed for permutation test |

### Reproducing paper results exactly

```bash
python nemo_pipeline.py \
    --fold-dir   ./fOLD-public-master/Supplementary \
    --epochs-csv ./data/epochs.csv \
    --output-dir ./outputs \
    --seed 42
```

The permutation p-value (Part 6a) uses `--seed 42`. All other results are deterministic.

---

## Outputs

After running, `outputs/` contains:

```
outputs/
├── figures/
│   └── MAIN_FIGURE.png    8-panel summary figure
└── results.json           all numeric results (machine-readable)
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
    "channel_results": { "S1_D1": { "t": 4.394, "p_fdr": 0.0021, ... }, ... }
  },
  "forward_model": {
    "regional_hierarchy": { "fold_order": [...], "empirical_order": [...], "match": true },
    "channel_spearman": { "r": 0.39, "p": 0.339, "status": "EXPLORATORY" },
    ...
  }
}
```

---

## Pipeline structure

| Part | Description |
|---|---|
| 1 | Load fOLD Brodmann specificity table |
| 2 | Brite-24 optode midpoints → approximate MNI (Okamoto 2004) |
| 3 | IFG sensitivity per channel (BA44 + BA45 + BA47) |
| 4 | Load and filter NEMO epochs (0–12 s, bad epoch exclusion) |
| 5 | Valence contrast (PV-NV) and arousal specificity check, FDR-BH |
| 6 | Forward sensitivity: Spearman, regional hierarchy, LOO, null ROI |
| 7 | 8-panel figure |
| 8 | results.json |

---

## Dependencies

See `requirements.txt`. Key versions:

- `scipy >= 1.11` — required for `false_discovery_control`. Older versions fall back to a manual Benjamini-Hochberg implementation (results are identical).
- `mne >= 1.0` — for `artinis-brite23` standard montage.

---

## Citing this work

If you use this pipeline, please cite:

**This pipeline:**
> Sindigi, A. (2026). *NEMO fNIRS frontal sensitivity validation pipeline* [Software]. GitHub. https://github.com/asindigi2004/nemo-fnirs-pipeline

**NEMO dataset (required):**
> Spapé, M., Mäkelä, K., & Ruotsalo, T. (2024). NEMO: A database for emotion analysis using functional near-infrared spectroscopy. *IEEE Transactions on Affective Computing*, 15(3), 1166–1178. https://doi.org/10.1109/TAFFC.2023.3315971

**fOLD forward model (required):**
> Zimeo Morais, G. A., Balardin, J. B., & Sato, J. R. (2018). fOLD: A toolbox for optode localization and cortical region-of-interest detection in functional near-infrared spectroscopy. *Neurophotonics*, 5(3), 035003. https://doi.org/10.1117/1.NPh.5.3.035003

**MNI approximation method:**
> Okamoto, M., et al. (2004). 3D probabilistic anatomical cranio-cerebral correlation via the international 10-20 system. *NeuroImage*, 21(1), 99–111. https://doi.org/10.1016/j.neuroimage.2003.08.026

**FDR correction:**
> Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.

---

## Licence

MIT — see [LICENSE](LICENSE).

Note: the NEMO dataset and fOLD toolbox each have their own licences. This repo contains only the analysis code.
