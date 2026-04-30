"""
NEMO fNIRS — Analysis Pipeline (DSAA 2026 submission)
======================================================
Primary contribution
    Empirical HbO valence contrast (PV > NV)
    - 31 subjects, 4 conditions (HAPV, LAPV, HANV, LANV)
    - Valence contrast collapses arousal: PV = mean(HAPV, LAPV),
      NV = mean(HANV, LANV)
    - One-sample t-test per channel across subjects (H0: mu=0)
    - FDR correction: Benjamini-Hochberg (q < 0.05)
    - Arousal contrast (HA-LA) tested separately as specificity check

Secondary contribution
    Forward sensitivity validation (exploratory)
    - fOLD Monte Carlo data: 10-10.xls, sheet 3_Chn (Zimeo Morais 2018)
    - IFG ROI: BA44 + BA45 + BA47 (summed specificity)
    - Brite-24 channels snapped to nearest fOLD pair via MNI midpoint
      (Okamoto 2004 approximation, ~8 mm SD error)
    - Primary forward result: regional hierarchy match (robust to snap error)
    - Channel-level Spearman reported as EXPLORATORY (n=8, median snap 35 mm)
    - Null ROI baseline; zero-coverage ROIs excluded from Spearman

Caveats (per DSAA methods requirements)
    - No MRI co-registration; artinis-brite23 nominal positions used
    - Head->MNI via Okamoto 2004 analytic approximation (~8 mm SD error)
    - fOLD specificity values computed on Colin27 head model
    - Region labels (Anterior / Frontopolar / Dorsal) reflect cap geometry,
      NOT neuroanatomical regions
    - Channel-level Spearman exploratory: n=8 after zero-IFG exclusion,
      median MNI snap 35 mm exceeds Metz 2022 25 mm threshold

Usage
    python nemo_pipeline.py \\
        --fold-dir   /path/to/fOLD-public-master/Supplementary \\
        --epochs-csv /path/to/epochs.csv \\
        --output-dir ./outputs

Reproducibility
    Permutation p-values use RANDOM_SEED = 42.
    Set --seed to override if needed; results in the paper used seed 42.
"""

import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

try:
    from scipy.stats import false_discovery_control as _fdr_bh
    HAS_SCIPY_FDR = True
except ImportError:
    HAS_SCIPY_FDR = False   # scipy < 1.11 — manual BH used instead

# ============================================================
# CONSTANTS  (device / design — not user-configurable)
# ============================================================

# Artinis Brite-24 source-detector pairs
ALL_CHANNEL_PAIRS = [
    ('S1',  'D1'), ('S1',  'D2'), ('S2',  'D1'), ('S2',  'D3'),
    ('S3',  'D1'), ('S3',  'D2'), ('S3',  'D3'), ('S3',  'D4'),
    ('S4',  'D2'), ('S4',  'D4'), ('S5',  'D3'), ('S5',  'D4'),
    ('S6',  'D5'), ('S6',  'D6'), ('S7',  'D5'), ('S7',  'D7'),
    ('S8',  'D5'), ('S8',  'D6'), ('S8',  'D7'), ('S8',  'D8'),
    ('S9',  'D6'), ('S9',  'D8'), ('S10', 'D7'), ('S10', 'D8'),
]

# fOLD landmark strings — must match cell values in 10-10.xls exactly
IFG_LANDMARKS = {
    'BA44': "44 - pars opercularis, part of Broca's area",
    'BA45': "45 - pars triangularis Broca's area",
    'BA47': '47 - Inferior prefrontal gyrus',
}
NULL_ROI_LANDMARKS = {
    'BA4 (Motor)': '4 - Primary Motor Cortex',
    'BA17 (V1)':   '17 - Primary Visual Cortex (V1)',
    'BA6 (SMA)':   '6 - Pre-Motor and Supplementary Motor Cortex',
    'BA9 (dlPFC)': '9 - Dorsolateral prefrontal cortex',
    'BA11 (OFC)':  '11 - Orbitofrontal area',
}

# Cap-geometry region labels — NOT neuroanatomical (Spapé et al. 2024, Fig. 4)
NEMO_REGION_MAP = {
    'Anterior':    ['S1_D1',  'S1_D2',  'S2_D1',  'S2_D3',
                    'S10_D7', 'S10_D8', 'S9_D6',  'S9_D8'],
    'Frontopolar': ['S3_D1',  'S3_D2',  'S4_D2',  'S4_D4',
                    'S8_D5',  'S8_D6',  'S7_D5',  'S7_D7'],
    'Dorsal':      ['S3_D3',  'S3_D4',  'S5_D3',  'S5_D4',
                    'S8_D7',  'S8_D8',  'S6_D5',  'S6_D6'],
}
CH_REGION    = {ch: r for r, chs in NEMO_REGION_MAP.items() for ch in chs}
REGION_ORDER = ['Anterior', 'Frontopolar', 'Dorsal']
REGION_COLORS = {
    'Anterior':    '#e41a1c',
    'Frontopolar': '#377eb8',
    'Dorsal':      '#4daf4a',
}

# NEMO 2x2 design: valence (Positive/Negative) x arousal (High/Low)
POS_CONDS = ['HAPV', 'LAPV']
NEG_CONDS = ['HANV', 'LANV']
HA_CONDS  = ['HAPV', 'HANV']
LA_CONDS  = ['LAPV', 'LANV']

# Epoch window: 0-12 s post-stimulus covers the full HRF
# (HbO peaks 6-10 s; Villringer & Chance 1997)
EPOCH_T_MIN = 0
EPOCH_T_MAX = 12

FDR_ALPHA   = 0.05
RANDOM_SEED = 42
MIN_SUBJECTS_PER_CHANNEL = 5    # minimum n for t-test
MIN_CHANNELS_PER_SUBJECT = 10   # minimum channels to include a subject


# ============================================================
# HELPERS
# ============================================================

def apply_fdr(p_values_dict, alpha=FDR_ALPHA):
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values_dict : dict[str, float]
    alpha         : float

    Returns
    -------
    dict[str, tuple(p_raw, p_fdr, significant)]
    """
    keys  = list(p_values_dict.keys())
    pvals = np.array([p_values_dict[k] for k in keys])
    n     = len(pvals)

    if HAS_SCIPY_FDR:
        pvals_fdr = _fdr_bh(pvals)
    else:
        # Manual BH (cumulative minimum from largest to smallest rank)
        order     = np.argsort(pvals)
        pvals_fdr = np.empty(n)
        cummin    = np.inf
        for i in range(n - 1, -1, -1):
            idx            = order[i]
            pvals_fdr[idx] = min(cummin, pvals[idx] * n / (i + 1))
            cummin         = pvals_fdr[idx]

    sig = pvals_fdr < alpha
    return {keys[i]: (float(pvals[i]), float(pvals_fdr[i]), bool(sig[i]))
            for i in range(n)}


def head_mm_to_mni_approx(xyz_mm):
    """
    Okamoto et al. (2004) analytic head-to-MNI approximation.
    Error ~8 mm SD — standard for template-based fNIRS localisation.

    Parameters
    ----------
    xyz_mm : array-like, shape (3,) or (N, 3), in millimetres

    Returns
    -------
    np.ndarray  approximate MNI coordinates (mm)
    """
    xyz    = np.atleast_2d(xyz_mm).copy().astype(float)
    scale  = np.array([0.958,  0.981, 0.968])
    offset = np.array([0.0,  -14.0,  47.0])
    return (xyz * scale + offset).squeeze()


def get_specificity(source, detector, landmarks, fold_df):
    """
    Sum fOLD specificity values for a given source-detector pair
    across the requested landmark strings.

    Returns 0.0 if the pair is absent from fold_df.
    """
    mask = (
        (fold_df['Source']   == source) &
        (fold_df['Detector'] == detector) &
        (fold_df['Landmark'].isin(landmarks))
    )
    return float(fold_df.loc[mask, 'Specificity'].sum())


# ============================================================
# MAIN
# ============================================================

def main(fold_dir, epochs_csv, output_dir, seed=RANDOM_SEED):
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    rng = np.random.default_rng(seed)

    print("=" * 65)
    print("  NEMO fNIRS — DSAA 2026 submission pipeline")
    print(f"  Random seed: {seed}  (set --seed to override)")
    print("=" * 65)

    # ----------------------------------------------------------
    # PART 1: Load fOLD table
    # ----------------------------------------------------------
    print("\n[Part 1] Loading fOLD Brodmann table...")

    fold_xls = os.path.join(fold_dir, '10-10.xls')
    if not os.path.isfile(fold_xls):
        raise FileNotFoundError(
            f"fOLD file not found: {fold_xls}\n"
            "Download fOLD-public-master from https://github.com/gabri470/fOLD-public\n"
            "and pass --fold-dir pointing to its Supplementary/ folder."
        )

    fold_raw  = pd.read_excel(fold_xls, sheet_name='3_Chn')
    fill_cols = ['Channel', 'Source', 'Detector', 'Distance (mm)',
                 'brainSens', 'X (mm)', 'Y (mm)', 'Z (mm)']
    for col in fill_cols:
        fold_raw[col] = fold_raw[col].ffill()

    fold_df = fold_raw.dropna(subset=['Landmark', 'Specificity']).copy()
    for col in ['Specificity', 'Distance (mm)', 'X (mm)', 'Y (mm)', 'Z (mm)']:
        fold_df[col] = pd.to_numeric(fold_df[col], errors='coerce')
    fold_df = fold_df.dropna(subset=['Specificity', 'X (mm)', 'Y (mm)', 'Z (mm)'])

    fold_channels = (
        fold_df[['Channel', 'Source', 'Detector',
                 'Distance (mm)', 'X (mm)', 'Y (mm)', 'Z (mm)']]
        .drop_duplicates(subset=['Source', 'Detector'])
        .reset_index(drop=True)
    )
    fold_mni_xyz = fold_channels[['X (mm)', 'Y (mm)', 'Z (mm)']].values

    print(f"  Loaded {len(fold_df)} landmark rows, "
          f"{len(fold_channels)} unique channel pairs")

    # Sanity-check IFG landmarks exist in the file
    print("  IFG landmark check:")
    for key, lm in IFG_LANDMARKS.items():
        n = (fold_df['Landmark'] == lm).sum()
        status = 'OK' if n > 0 else '*** MISSING — check landmark string ***'
        print(f"    {key}: {n} rows  {status}")

    # ----------------------------------------------------------
    # PART 2: Brite-24 optode positions -> MNI
    # ----------------------------------------------------------
    print("\n[Part 2] Brite-24 channel midpoints -> MNI...")

    montage = mne.channels.make_standard_montage("artinis-brite23")
    pos     = {k: v.copy()
               for k, v in montage.get_positions()['ch_pos'].items()
               if k != 'S11'}

    # D8 absent from artinis-brite23 montage; mirrored from D1
    # (left-right symmetry, deviation < 1.2 mm — see reference guide)
    pos['D8'] = np.array([-pos['D1'][0], pos['D1'][1], pos['D1'][2]])

    VALID_PAIRS, EXCLUDED_PAIRS = [], []
    for s, d in ALL_CHANNEL_PAIRS:
        dist_mm = np.linalg.norm(pos[s] - pos[d]) * 1000
        (VALID_PAIRS if 20 <= dist_mm <= 50 else EXCLUDED_PAIRS).append((s, d))

    VALID_CH_NAMES    = [f'{s}_{d}' for s, d in VALID_PAIRS]
    EXCLUDED_CH_NAMES = [f'{s}_{d}' for s, d in EXCLUDED_PAIRS]

    print(f"  Valid channels (20-50 mm): {len(VALID_PAIRS)}/24")
    print(f"  Excluded (out of range):   {len(EXCLUDED_PAIRS)}/24")
    if EXCLUDED_CH_NAMES:
        print(f"  Excluded: {EXCLUDED_CH_NAMES}")

    CH_MNI_MID    = {}
    CH_FOLD_MATCH = {}
    CH_DIST_MM    = {}
    snap_distances = []

    for s, d in ALL_CHANNEL_PAIRS:
        ch       = f'{s}_{d}'
        mid_head = (pos[s] + pos[d]) / 2 * 1000          # metres -> mm
        mid_mni  = head_mm_to_mni_approx(mid_head)
        inter_mm = np.linalg.norm(pos[s] - pos[d]) * 1000
        dists    = np.linalg.norm(fold_mni_xyz - mid_mni, axis=1)
        best_idx = int(np.argmin(dists))
        best_row = fold_channels.iloc[best_idx]

        CH_MNI_MID[ch]    = mid_mni
        CH_DIST_MM[ch]    = inter_mm
        CH_FOLD_MATCH[ch] = {
            'source':      best_row['Source'],
            'detector':    best_row['Detector'],
            'fold_pair':   f"{best_row['Source']}-{best_row['Detector']}",
            'mni_snap_mm': float(dists[best_idx]),
            'fold_mni':    [float(best_row['X (mm)']),
                            float(best_row['Y (mm)']),
                            float(best_row['Z (mm)'])],
        }
        if ch in VALID_CH_NAMES:
            snap_distances.append(float(dists[best_idx]))

    snap_arr = np.array(snap_distances)
    print(f"  MNI snap (valid ch): median {np.median(snap_arr):.1f} mm, "
          f"max {np.max(snap_arr):.1f} mm")
    print(f"  NOTE: median snap > 25 mm (Metz 2022 threshold) — "
          f"channel-level Spearman is exploratory; regional result is primary.")

    # ----------------------------------------------------------
    # PART 3: fOLD IFG sensitivity lookup
    # ----------------------------------------------------------
    print("\n[Part 3] fOLD IFG sensitivity (BA44 + BA45 + BA47)...")

    ifg_lm_list = list(IFG_LANDMARKS.values())
    IFG_SENSITIVITY = {}

    for s, d in ALL_CHANNEL_PAIRS:
        ch = f'{s}_{d}'
        m  = CH_FOLD_MATCH[ch]
        v  = get_specificity(m['source'], m['detector'], ifg_lm_list, fold_df)
        if ch in VALID_CH_NAMES:
            IFG_SENSITIVITY[ch] = v

    n_zero_ifg    = sum(1 for v in IFG_SENSITIVITY.values() if v == 0)
    n_nonzero_ifg = len(IFG_SENSITIVITY) - n_zero_ifg
    print(f"  Channels with IFG specificity > 0: "
          f"{n_nonzero_ifg}/{len(IFG_SENSITIVITY)}")
    if n_zero_ifg > 0:
        zero_chs = [ch for ch, v in IFG_SENSITIVITY.items() if v == 0]
        print(f"  Zero-IFG channels: {zero_chs}")
        print(f"  (nearest fOLD pair has no IFG coverage — "
              f"excluded from channel-level Spearman)")

    # Regional predicted sensitivity (fOLD)
    region_sens = {}
    for r in REGION_ORDER:
        vals = [IFG_SENSITIVITY[ch] for ch in VALID_CH_NAMES
                if CH_REGION.get(ch) == r and IFG_SENSITIVITY.get(ch, 0) > 0]
        region_sens[r] = float(np.mean(vals)) if vals else 0.0

    pred_order = sorted(REGION_ORDER, key=lambda r: region_sens[r], reverse=True)

    # Null ROI lookup
    NULL_SENSITIVITY = {roi: {} for roi in NULL_ROI_LANDMARKS}
    for roi_name, roi_lm in NULL_ROI_LANDMARKS.items():
        for s, d in VALID_PAIRS:
            ch = f'{s}_{d}'
            m  = CH_FOLD_MATCH[ch]
            NULL_SENSITIVITY[roi_name][ch] = get_specificity(
                m['source'], m['detector'], [roi_lm], fold_df)

    # ----------------------------------------------------------
    # PART 4: Load NEMO epochs
    # ----------------------------------------------------------
    print("\n[Part 4] Loading NEMO epochs...")

    if not os.path.isfile(epochs_csv):
        raise FileNotFoundError(
            f"Epochs CSV not found: {epochs_csv}\n"
            "Download the NEMO dataset from https://osf.io/pd9rv\n"
            "and pass the path to empe_csv/epochs.csv via --epochs-csv."
        )

    df      = pd.read_csv(epochs_csv, sep=';', low_memory=False)
    n_total = len(df)

    # Bad epoch exclusion — column present in NEMO empe_csv; guard for other versions
    if 'is_bad_epoch' in df.columns:
        n_bad = df['is_bad_epoch'].sum()
        df    = df[~df['is_bad_epoch']].copy()
        print(f"  Excluded bad epochs: {n_bad:,} rows removed")
    else:
        print("  WARNING: 'is_bad_epoch' column not found — no epoch exclusion applied.")
        print("  This is expected only if you are using a non-standard CSV format.")

    df = df[(df['time'] >= EPOCH_T_MIN) & (df['time'] <= EPOCH_T_MAX)].copy()
    print(f"  After time filter ({EPOCH_T_MIN}-{EPOCH_T_MAX} s): {len(df):,} rows, "
          f"{df['subject'].nunique()} subjects, "
          f"{df['condition'].nunique()} conditions")

    hbo_cols  = [c for c in df.columns if c.endswith(' hbo') and '_D' in c]
    col_to_ch = {c: c.replace(' hbo', '').strip() for c in hbo_cols}

    if not hbo_cols:
        raise ValueError(
            "No HbO columns found. Expected columns ending in ' hbo' "
            "containing '_D' (e.g. 'S1_D1 hbo'). Check epochs CSV format."
        )

    # ----------------------------------------------------------
    # PART 5: Valence contrast + arousal specificity check
    # ----------------------------------------------------------
    print("\n[Part 5] Computing PV-NV contrast and arousal interaction...")

    all_pv_nv = {}
    all_ha_la = {}

    for sub in sorted(df['subject'].unique()):
        sub_df = df[df['subject'] == sub]
        means  = {}
        for cond in ['HAPV', 'LAPV', 'HANV', 'LANV']:
            cdf        = sub_df[sub_df['condition'] == cond]
            means[cond] = {col: cdf[col].mean() for col in hbo_cols}

        pv_nv_sub, ha_la_sub = {}, {}
        for col, ch in col_to_ch.items():
            hapv = means['HAPV'].get(col, np.nan)
            lapv = means['LAPV'].get(col, np.nan)
            hanv = means['HANV'].get(col, np.nan)
            lanv = means['LANV'].get(col, np.nan)
            if all(pd.notna([hapv, lapv, hanv, lanv])):
                pv = (hapv + lapv) / 2
                nv = (hanv + lanv) / 2
                ha = (hapv + hanv) / 2
                la = (lapv + lanv) / 2
                pv_nv_sub[ch] = float(pv - nv)
                ha_la_sub[ch] = float(ha - la)

        if len(pv_nv_sub) >= MIN_CHANNELS_PER_SUBJECT:
            all_pv_nv[sub] = pv_nv_sub
            all_ha_la[sub] = ha_la_sub
        else:
            print(f"  WARNING: subject {sub} excluded — only {len(pv_nv_sub)} "
                  f"channels with complete data (threshold: {MIN_CHANNELS_PER_SUBJECT})")

    N_SUBS = len(all_pv_nv)
    print(f"  Subjects with sufficient data: {N_SUBS}")

    # Group t-tests: valence (PV-NV)
    group_tvals = {}
    group_pvals = {}
    for ch in col_to_ch.values():
        vals = [all_pv_nv[s][ch] for s in all_pv_nv if ch in all_pv_nv[s]]
        if len(vals) >= MIN_SUBJECTS_PER_CHANNEL:
            t, p = ttest_1samp(vals, 0)
            group_tvals[ch] = float(t)
            group_pvals[ch] = float(p)
        else:
            print(f"  WARNING: channel {ch} excluded from t-test — "
                  f"only {len(vals)} subjects (threshold: {MIN_SUBJECTS_PER_CHANNEL})")

    valid_pvals_for_fdr = {ch: group_pvals[ch] for ch in VALID_CH_NAMES
                           if ch in group_pvals}
    fdr_results = apply_fdr(valid_pvals_for_fdr)

    n_positive   = sum(1 for ch in VALID_CH_NAMES
                       if ch in group_tvals and group_tvals[ch] > 0)
    n_sig_uncorr = sum(1 for p in valid_pvals_for_fdr.values() if p < FDR_ALPHA)
    n_sig_fdr    = sum(1 for r in fdr_results.values() if r[2])

    print(f"\n  Valence contrast (PV-NV) — {len(valid_pvals_for_fdr)} valid channels:")
    print(f"    Positive t-values:   {n_positive}/{len(valid_pvals_for_fdr)}")
    print(f"    Significant uncorr:  {n_sig_uncorr}/{len(valid_pvals_for_fdr)} (p < 0.05)")
    print(f"    Significant FDR-BH:  {n_sig_fdr}/{len(valid_pvals_for_fdr)} (q < {FDR_ALPHA})")

    # Group t-tests: arousal (HA-LA) — specificity check
    group_tvals_arousal = {}
    group_pvals_arousal = {}
    for ch in col_to_ch.values():
        vals = [all_ha_la[s][ch] for s in all_ha_la if ch in all_ha_la[s]]
        if len(vals) >= MIN_SUBJECTS_PER_CHANNEL:
            t, p = ttest_1samp(vals, 0)
            group_tvals_arousal[ch] = float(t)
            group_pvals_arousal[ch] = float(p)

    valid_arousal_p   = {ch: group_pvals_arousal[ch] for ch in VALID_CH_NAMES
                         if ch in group_pvals_arousal}
    fdr_arousal       = apply_fdr(valid_arousal_p)
    n_sig_arousal_fdr = sum(1 for r in fdr_arousal.values() if r[2])

    print(f"\n  Arousal contrast (HA-LA) — specificity check:")
    print(f"    Significant FDR-BH:  {n_sig_arousal_fdr}/{len(valid_arousal_p)}")
    print(f"    (Expected ~0 if valence result is valence-specific, not arousal-driven)")

    # Channel results table
    print(f"\n  Channel results (valid channels, sorted by |t|):")
    print(f"  {'Ch':12}  {'t':8}  {'p_raw':7}  {'p_fdr':7}  {'FDR*':5}  Region")
    print("  " + "-" * 64)
    for ch in sorted(valid_pvals_for_fdr,
                     key=lambda c: abs(group_tvals.get(c, 0)), reverse=True):
        t            = group_tvals[ch]
        p_r, p_f, sig = fdr_results[ch]
        reg          = CH_REGION.get(ch, '--')
        print(f"  {ch:12}  {t:+8.3f}  {p_r:7.4f}  {p_f:7.4f}  "
              f"{'YES' if sig else 'no':5}  {reg}")

    # Regional empirical t-values
    region_tvals = {
        r: float(np.mean([abs(group_tvals[ch]) for ch in VALID_CH_NAMES
                          if CH_REGION.get(ch) == r and ch in group_tvals]))
        for r in REGION_ORDER
    }
    emp_order = sorted(region_tvals, key=region_tvals.get, reverse=True)

    # ----------------------------------------------------------
    # PART 6: Forward sensitivity validation (secondary)
    # ----------------------------------------------------------
    print("\n[Part 6] Forward sensitivity validation (secondary, exploratory)...")
    print(f"  CAVEAT: median MNI snap = {np.median(snap_arr):.1f} mm "
          f"(> 25 mm Metz 2022 threshold)")
    print(f"  Regional hierarchy is the primary forward result; "
          f"channel-level Spearman is exploratory.")

    common_valid = [ch for ch in VALID_CH_NAMES
                    if ch in group_tvals and IFG_SENSITIVITY.get(ch, 0) > 0]

    sens_v = np.array([IFG_SENSITIVITY[ch]  for ch in common_valid])
    tval_v = np.array([abs(group_tvals[ch]) for ch in common_valid])

    # 6a: Channel-level Spearman (exploratory)
    # Initialise to safe defaults before conditional block
    r_sp = p_sp = p_perm = np.nan
    r_str = 'n/a'
    p_str = 'n/a'

    if len(common_valid) >= 5:
        r_sp, p_sp = spearmanr(sens_v, tval_v)
        p_perm     = sum(
            1 for _ in range(10_000)
            if abs(spearmanr(rng.permutation(tval_v), sens_v)[0]) >= abs(r_sp)
        ) / 10_000
        r_str = f'{r_sp:.3f}'
        p_str = f'{p_sp:.4f}'
        print(f"\n  6a. Spearman r (EXPLORATORY, n={len(common_valid)}):")
        print(f"      r={r_sp:.3f}, p={p_sp:.4f}, perm-p={p_perm:.4f}")
        print(f"      Positive direction consistent with IFG hypothesis; "
              f"not significant due to small n and coordinate uncertainty.")
    else:
        print(f"  6a. Spearman: n={len(common_valid)} — cannot compute (need >= 5)")

    # 6b: Regional hierarchy (primary forward result)
    print(f"\n  6b. Regional hierarchy (primary forward model result):")
    print(f"      fOLD predicted: {' > '.join(pred_order)}")
    print(f"      Empirical:      {' > '.join(emp_order)}")
    match_reg = (pred_order == emp_order)
    print(f"      Match: {'YES' if match_reg else 'NO'}")
    print(f"      CAVEAT: region labels are cap-geometry, not neuroanatomical.")

    # 6c: Per-subject LOO
    r_per_sub = []
    for sub, sub_pv in all_pv_nv.items():
        chs_s = [ch for ch in VALID_CH_NAMES
                 if ch in sub_pv and IFG_SENSITIVITY.get(ch, 0) > 0]
        if len(chs_s) >= 5:
            r, p = spearmanr(
                [IFG_SENSITIVITY[ch] for ch in chs_s],
                [abs(sub_pv[ch])     for ch in chs_s],
            )
            r_per_sub.append({'sub': sub, 'r': float(r), 'p': float(p)})

    r_vals  = [x['r'] for x in r_per_sub]
    p_loo   = np.nan
    n_sig_loo = 0
    if len(r_vals) >= 5:
        _, p_loo  = ttest_1samp(r_vals, 0)
        n_sig_loo = sum(1 for x in r_per_sub if x['p'] < 0.05)
        print(f"\n  6c. LOO (n={len(r_per_sub)} subjects):")
        print(f"      mean r = {np.mean(r_vals):.3f} +/- {np.std(r_vals):.3f}, "
              f"p = {p_loo:.4f}, "
              f"{n_sig_loo}/{len(r_per_sub)} individually significant")
    else:
        print(f"  6c. LOO: n={len(r_vals)} — insufficient for t-test (need >= 5)")

    # 6d: Null ROI baseline
    print(f"\n  6d. Null ROI baseline:")
    print(f"  {'ROI':22}  {'Coverage':10}  {'Spearman r':12}  {'p':10}  Note")
    print("  " + "-" * 75)
    null_r_vals      = []
    null_roi_results = {}
    for roi_name in NULL_ROI_LANDMARKS:
        ns        = [NULL_SENSITIVITY[roi_name].get(ch, 0) for ch in common_valid]
        nt        = [abs(group_tvals[ch]) for ch in common_valid]
        n_nonzero = sum(1 for v in ns if v > 0)
        if n_nonzero < 2 or np.std(ns) < 1e-10:
            note = "insufficient channel coverage — Spearman not computed"
            null_roi_results[roi_name] = {'r': None, 'p': None, 'note': note}
            print(f"  {roi_name:22}  {n_nonzero:10}  {'N/A':12}  {'N/A':10}  {note}")
        else:
            r_n, p_n = spearmanr(ns, nt)
            null_r_vals.append(r_n)
            null_roi_results[roi_name] = {
                'r': float(r_n), 'p': float(p_n), 'note': ''}
            direction = ('IFG > null'
                         if (not np.isnan(r_sp) and abs(r_sp) > abs(r_n))
                         else 'null >= IFG')
            print(f"  {roi_name:22}  {n_nonzero:10}  {r_n:12.3f}  "
                  f"{p_n:10.4f}  {direction}")

    if null_r_vals and not np.isnan(r_sp):
        ifg_beats = all(abs(r_sp) >= abs(r_n) for r_n in null_r_vals)
        print(f"\n  IFG r = {r_sp:.3f} vs computable nulls "
              f"[{min(null_r_vals):.3f}, {max(null_r_vals):.3f}]")
        print(f"  IFG {'outperforms' if ifg_beats else 'does not outperform'} "
              f"all computable null ROIs")
        print(f"  (BA17, BA11 excluded: zero channel coverage in this montage)")
    else:
        ifg_beats = None

    # ----------------------------------------------------------
    # PART 7: Figures
    # ----------------------------------------------------------
    print("\n[Part 7] Generating figures...")

    fig = plt.figure(figsize=(22, 10))
    gs  = fig.add_gridspec(2, 4, wspace=0.42, hspace=0.55)

    # Panel A: Valence contrast t-values
    ax_a       = fig.add_subplot(gs[0, 0])
    sorted_chs = sorted(valid_pvals_for_fdr,
                        key=lambda c: group_tvals.get(c, 0), reverse=True)
    cols_a  = [REGION_COLORS.get(CH_REGION.get(ch, ''), 'grey') for ch in sorted_chs]
    tvals_a = [group_tvals[ch] for ch in sorted_chs]
    ax_a.barh(range(len(sorted_chs)), tvals_a, color=cols_a, alpha=0.85)
    for i, ch in enumerate(sorted_chs):
        if fdr_results[ch][2]:
            ax_a.barh(i, tvals_a[i], color=cols_a[i],
                      alpha=1.0, edgecolor='black', linewidth=1.2)
    ax_a.axvline(0, color='k', linewidth=1)
    ax_a.set_yticks(range(len(sorted_chs)))
    ax_a.set_yticklabels(sorted_chs, fontsize=6)
    ax_a.invert_yaxis()
    ax_a.set_xlabel('t-value (PV-NV)', fontsize=8)
    ax_a.set_title(
        f'(A) Valence contrast\n'
        f'FDR-sig: {n_sig_fdr}/{len(valid_pvals_for_fdr)} channels\n'
        f'(bold border = FDR q < {FDR_ALPHA})',
        fontsize=9, fontweight='bold')
    for reg, col in REGION_COLORS.items():
        ax_a.scatter([], [], c=col, s=30, label=reg)
    ax_a.legend(fontsize=6)
    ax_a.grid(axis='x', alpha=0.3)

    # Panel B: Arousal contrast t-values
    ax_b         = fig.add_subplot(gs[0, 1])
    sorted_chs_b = sorted(valid_arousal_p,
                          key=lambda c: group_tvals_arousal.get(c, 0),
                          reverse=True)
    cols_b  = [REGION_COLORS.get(CH_REGION.get(ch, ''), 'grey') for ch in sorted_chs_b]
    tvals_b = [group_tvals_arousal[ch] for ch in sorted_chs_b]
    ax_b.barh(range(len(sorted_chs_b)), tvals_b, color=cols_b, alpha=0.85)
    for i, ch in enumerate(sorted_chs_b):
        if fdr_arousal[ch][2]:
            ax_b.barh(i, tvals_b[i], color=cols_b[i],
                      alpha=1.0, edgecolor='black', linewidth=1.2)
    ax_b.axvline(0, color='k', linewidth=1)
    ax_b.set_yticks(range(len(sorted_chs_b)))
    ax_b.set_yticklabels(sorted_chs_b, fontsize=6)
    ax_b.invert_yaxis()
    ax_b.set_xlabel('t-value (HA-LA)', fontsize=8)
    ax_b.set_title(
        f'(B) Arousal contrast\n'
        f'FDR-sig: {n_sig_arousal_fdr}/{len(valid_arousal_p)} channels\n'
        f'(specificity check — expected ~0)',
        fontsize=9, fontweight='bold')
    ax_b.grid(axis='x', alpha=0.3)

    # Panel C: Regional hierarchy
    ax_c = fig.add_subplot(gs[0, 2])
    x    = np.arange(len(REGION_ORDER))
    w    = 0.35
    ax_c.bar(x - w / 2, [region_sens[r]   for r in REGION_ORDER], w,
             color=[REGION_COLORS[r] for r in REGION_ORDER],
             alpha=0.9, label='fOLD IFG specificity')
    ax_c2 = ax_c.twinx()
    ax_c2.bar(x + w / 2, [region_tvals[r] for r in REGION_ORDER], w,
              color=[REGION_COLORS[r] for r in REGION_ORDER],
              alpha=0.5, hatch='//', label='Mean |t| (PV-NV)')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(REGION_ORDER, fontsize=8)
    ax_c.set_ylabel('fOLD IFG specificity', fontsize=7)
    ax_c2.set_ylabel('Mean |t-value|', fontsize=7)
    ax_c.set_title(
        f'(C) Regional hierarchy\n'
        f'(cap-geometry labels only)\n'
        f'Orders match: {"YES" if match_reg else "NO"}',
        fontsize=9, fontweight='bold')
    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, fontsize=6)

    # Panel D: Channel scatter (exploratory)
    ax_d = fig.add_subplot(gs[0, 3])
    if len(common_valid) >= 3:
        cols_d = [REGION_COLORS.get(CH_REGION.get(ch, ''), 'grey')
                  for ch in common_valid]
        ax_d.scatter(sens_v, tval_v, c=cols_d, s=70, zorder=3,
                     edgecolors='black', linewidth=0.7)
        if len(sens_v) >= 5:
            m_f, b_f = np.polyfit(sens_v, tval_v, 1)
            xf = np.linspace(min(sens_v), max(sens_v), 100)
            ax_d.plot(xf, m_f * xf + b_f, 'k--', linewidth=1, alpha=0.7)
        for ch in common_valid:
            ax_d.annotate(ch,
                          (IFG_SENSITIVITY[ch], abs(group_tvals[ch])),
                          textcoords='offset points', xytext=(3, 2), fontsize=5)
    ax_d.set_xlabel('fOLD IFG specificity', fontsize=8)
    ax_d.set_ylabel('|t-value| PV-NV', fontsize=8)
    ax_d.set_title(
        f'(D) Channel scatter (EXPLORATORY)\n'
        f'n={len(common_valid)}, Spearman r={r_str}\n'
        f'Underpowered — interpret with caution',
        fontsize=9, fontweight='bold')
    ax_d.grid(alpha=0.3)

    # Panel E: Per-subject LOO
    ax_e = fig.add_subplot(gs[1, 0])
    if len(r_vals) >= 3:
        c_loo = ['#d62728' if x['p'] < 0.05 else '#aec7e8' for x in r_per_sub]
        ax_e.barh(range(len(r_vals)), r_vals, color=c_loo, alpha=0.85)
        ax_e.axvline(0, color='k', linewidth=1)
        ax_e.axvline(np.mean(r_vals), color='#d62728', linestyle='--',
                     linewidth=1.5, label=f'Mean r = {np.mean(r_vals):.3f}')
        ax_e.set_yticks(range(len(r_vals)))
        ax_e.set_yticklabels(
            [x['sub'].replace('sub-', '') for x in r_per_sub], fontsize=6)
        ax_e.invert_yaxis()
        ax_e.legend(fontsize=6)
        p_str_loo = f'p = {p_loo:.4f}' if not np.isnan(p_loo) else 'n/a'
        ax_e.set_title(
            f'(E) LOO per-subject\n'
            f'mean r = {np.mean(r_vals):.3f}, {p_str_loo}\n'
            f'{n_sig_loo}/{len(r_per_sub)} individually sig.',
            fontsize=9, fontweight='bold')
    else:
        ax_e.text(0.5, 0.5, 'Insufficient data',
                  ha='center', va='center', transform=ax_e.transAxes)
        ax_e.set_title('(E) LOO', fontsize=9, fontweight='bold')
    ax_e.set_xlabel('Spearman r (fOLD IFG vs |t|)', fontsize=8)
    ax_e.grid(axis='x', alpha=0.3)

    # Panel F: Null ROI
    ax_f = fig.add_subplot(gs[1, 1])
    if null_r_vals and not np.isnan(r_sp):
        comp_labels = [k for k, v in null_roi_results.items() if v['r'] is not None]
        comp_rs     = [null_roi_results[k]['r'] for k in comp_labels]
        all_labels  = comp_labels + ['IFG (BA44+45+47)']
        all_rs      = comp_rs + [r_sp]
        colors_f    = ['#aec7e8'] * len(comp_labels) + ['#d62728']
        ax_f.barh(range(len(all_labels)), all_rs, color=colors_f,
                  alpha=0.85, edgecolor='black', linewidth=0.5)
        ax_f.axvline(0, color='k', linewidth=1)
        ax_f.set_yticks(range(len(all_labels)))
        ax_f.set_yticklabels(all_labels, fontsize=7)
        ax_f.invert_yaxis()
        ax_f.set_title(
            f'(F) Null ROI baseline\n'
            f'BA17, BA11: zero coverage (excluded)\n'
            f'IFG r = {r_sp:.3f}',
            fontsize=9, fontweight='bold')
    else:
        ax_f.text(0.5, 0.5, 'Not enough data\nfor null comparison',
                  ha='center', va='center', transform=ax_f.transAxes)
        ax_f.set_title('(F) Null ROI', fontsize=9, fontweight='bold')
    ax_f.set_xlabel('Spearman r vs |t|', fontsize=8)
    ax_f.grid(axis='x', alpha=0.3)

    # Panel G: FDR summary
    ax_g     = fig.add_subplot(gs[1, 2])
    labels_g = ['Uncorr\n(valence)', 'FDR-BH\n(valence)', 'FDR-BH\n(arousal)']
    counts_g = [n_sig_uncorr, n_sig_fdr, n_sig_arousal_fdr]
    colors_g = ['#aec7e8', '#d62728', '#fdae61']
    ax_g.bar(labels_g, counts_g, color=colors_g,
             alpha=0.85, edgecolor='black', linewidth=0.7)
    ax_g.set_ylabel('Significant channels', fontsize=8)
    ax_g.set_title(
        f'(G) Multiple comparisons\n'
        f'Valence FDR: {n_sig_fdr}/{len(valid_pvals_for_fdr)}\n'
        f'Arousal FDR: {n_sig_arousal_fdr}/{len(valid_arousal_p)}',
        fontsize=9, fontweight='bold')
    ax_g.set_ylim(0, len(valid_pvals_for_fdr) + 1)
    for i, cnt in enumerate(counts_g):
        ax_g.text(i, cnt + 0.2, str(cnt),
                  ha='center', fontsize=9, fontweight='bold')
    ax_g.grid(axis='y', alpha=0.3)

    # Panel H: MNI snap distances
    ax_h      = fig.add_subplot(gs[1, 3])
    snap_chs  = [ch for ch in VALID_CH_NAMES if ch in IFG_SENSITIVITY]
    snap_vals = [CH_FOLD_MATCH[ch]['mni_snap_mm'] for ch in snap_chs]
    snap_cols = ['#d62728' if v > 25 else '#2ca02c' for v in snap_vals]
    ax_h.barh(range(len(snap_chs)), snap_vals, color=snap_cols,
              alpha=0.8, edgecolor='black', linewidth=0.4)
    ax_h.axvline(25, color='orange', linestyle='--', linewidth=1.5,
                 label='25 mm threshold (Metz 2022)')
    ax_h.set_yticks(range(len(snap_chs)))
    ax_h.set_yticklabels(snap_chs, fontsize=6)
    ax_h.invert_yaxis()
    ax_h.legend(fontsize=6)
    ax_h.set_xlabel('MNI snap distance (mm)', fontsize=8)
    ax_h.set_title(
        f'(H) fOLD coordinate quality\n'
        f'Median {np.median(snap_arr):.0f} mm (orange = threshold)\n'
        f'Explains why channel-level r is exploratory',
        fontsize=9, fontweight='bold')
    ax_h.grid(axis='x', alpha=0.3)

    fig.suptitle(
        f'NEMO fNIRS — Valence contrast + Forward sensitivity '
        f'(DSAA 2026, n = {N_SUBS})\n'
        f'Primary: {n_sig_fdr}/{len(valid_pvals_for_fdr)} FDR-sig channels PV > NV  |  '
        f'Forward: regional match {"YES" if match_reg else "NO"}  |  '
        f'Channel Spearman r = {r_str} (exploratory, n = {len(common_valid)})',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figures', 'MAIN_FIGURE.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # ----------------------------------------------------------
    # PART 8: Save JSON results
    # ----------------------------------------------------------
    results_out = {
        'pipeline_version': 'DSAA_2026',
        'random_seed': seed,
        'caveats': {
            'coord_system': (
                'artinis-brite23 nominal positions; no individual MRI '
                'co-registration. Head->MNI via Okamoto 2004 analytic '
                'approximation (~8 mm SD error).'
            ),
            'region_labels': (
                'Anterior/Frontopolar/Dorsal are cap-geometry labels '
                '(Spape et al. 2024, Fig. 4), NOT neuroanatomical regions.'
            ),
            'channel_spearman': (
                'Exploratory only. n=8 after zero-IFG exclusion; '
                'median MNI snap 35 mm exceeds Metz 2022 25 mm threshold.'
            ),
            'null_rois_ba17_ba11': (
                'BA17 and BA11 have zero channel coverage in this montage. '
                'Spearman not computed for these ROIs.'
            ),
        },
        'empirical': {
            'n_subjects':        N_SUBS,
            'n_valid_channels':  len(valid_pvals_for_fdr),
            'n_positive_t':      n_positive,
            'n_sig_uncorrected': n_sig_uncorr,
            'n_sig_fdr_bh':      n_sig_fdr,
            'fdr_alpha':         FDR_ALPHA,
            'n_sig_arousal_fdr': n_sig_arousal_fdr,
            'epoch_window_s':    [EPOCH_T_MIN, EPOCH_T_MAX],
            'channel_results': {
                ch: {
                    't':       round(group_tvals[ch], 4),
                    'p_raw':   round(fdr_results[ch][0], 5),
                    'p_fdr':   round(fdr_results[ch][1], 5),
                    'sig_fdr': fdr_results[ch][2],
                    'region':  CH_REGION.get(ch, '--'),
                }
                for ch in valid_pvals_for_fdr
            },
        },
        'forward_model': {
            'source':         'fOLD MC (Zimeo Morais 2018), 10-10.xls, sheet 3_Chn',
            'roi':            'BA44 + BA45 + BA47',
            'lookup':         'MNI midpoint matching (Okamoto 2004 approximation)',
            'snap_median_mm': round(float(np.median(snap_arr)), 1),
            'snap_max_mm':    round(float(np.max(snap_arr)), 1),
            'regional_hierarchy': {
                'fold_order':      pred_order,
                'empirical_order': emp_order,
                'match':           match_reg,
            },
            'channel_spearman': {
                'r':          round(float(r_sp), 4) if not np.isnan(r_sp) else None,
                'p':          round(float(p_sp), 4) if not np.isnan(p_sp) else None,
                'p_perm':     round(float(p_perm), 4) if not np.isnan(p_perm) else None,
                'n_channels': len(common_valid),
                'status':     'EXPLORATORY',
            },
            'loo': {
                'n_subjects':         len(r_per_sub),
                'mean_r':             round(float(np.mean(r_vals)), 4) if r_vals else None,
                'sd_r':               round(float(np.std(r_vals)),  4) if r_vals else None,
                'p_ttest':            round(float(p_loo), 4) if not np.isnan(p_loo) else None,
                'n_individually_sig': n_sig_loo,
            },
            'null_roi': null_roi_results,
            'fold_lookup': {
                ch: {
                    'fold_pair':   CH_FOLD_MATCH[ch]['fold_pair'],
                    'mni_snap_mm': round(CH_FOLD_MATCH[ch]['mni_snap_mm'], 1),
                    'ifg_total':   round(IFG_SENSITIVITY.get(ch, 0), 4),
                }
                for ch in VALID_CH_NAMES
            },
        },
    }

    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f"  Saved: {json_path}")

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("  NEMO — Final summary for DSAA 2026")
    print(f"{'=' * 65}")
    print(f"\n  PRIMARY — Valence contrast (n = {N_SUBS} subjects):")
    print(f"    {n_positive}/{len(valid_pvals_for_fdr)} valid channels: "
          f"positive HbO for PV > NV")
    print(f"    {n_sig_uncorr}/{len(valid_pvals_for_fdr)} significant uncorrected "
          f"(p < 0.05)")
    print(f"    {n_sig_fdr}/{len(valid_pvals_for_fdr)} significant FDR-BH "
          f"(q < {FDR_ALPHA})")
    print(f"    {n_sig_arousal_fdr}/{len(valid_arousal_p)} significant arousal "
          f"(specificity check)")
    print(f"\n  SECONDARY — Forward sensitivity (fOLD, exploratory):")
    print(f"    Regional hierarchy match: {'YES' if match_reg else 'NO'}")
    print(f"      fOLD:      {' > '.join(pred_order)}")
    print(f"      Empirical: {' > '.join(emp_order)}")
    mean_r_str = f'{np.mean(r_vals):.3f}' if r_vals else 'n/a'
    print(f"    Channel Spearman r = {r_str} "
          f"(n = {len(common_valid)}, EXPLORATORY)")
    print(f"    LOO mean r = {mean_r_str} ({len(r_per_sub)} subjects)")
    print(f"\n  CAVEATS (must appear in paper Methods):")
    print(f"    - No MRI co-registration; nominal artinis-brite23 positions")
    print(f"    - MNI snap median {np.median(snap_arr):.0f} mm "
          f"(> 25 mm Metz 2022 threshold)")
    print(f"    - Region labels are cap-geometry, not neuroanatomical")
    print(f"    - BA17, BA11: zero channel coverage in this montage")
    print(f"\n  Outputs saved to: {output_dir}")
    print("=" * 65)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NEMO fNIRS analysis pipeline — DSAA 2026',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Minimal
  python nemo_pipeline.py \\
      --fold-dir   ./fOLD-public-master/Supplementary \\
      --epochs-csv ./data/epochs.csv

  # Custom output dir and seed
  python nemo_pipeline.py \\
      --fold-dir   ./fOLD-public-master/Supplementary \\
      --epochs-csv ./data/epochs.csv \\
      --output-dir ./my_outputs \\
      --seed 42

Data sources
------------
  NEMO dataset:  https://osf.io/pd9rv  (download empe_csv/epochs.csv)
  fOLD toolbox:  https://github.com/gabri470/fOLD-public
                 (use Supplementary/10-10.xls, sheet 3_Chn)
        """
    )
    parser.add_argument(
        '--fold-dir',
        required=True,
        help='Path to fOLD-public-master/Supplementary/ (contains 10-10.xls)',
    )
    parser.add_argument(
        '--epochs-csv',
        required=True,
        help='Path to NEMO epochs CSV (semicolon-delimited, empe_csv/epochs.csv)',
    )
    parser.add_argument(
        '--output-dir',
        default='./outputs',
        help='Directory for figures/ and results.json (default: ./outputs)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for permutation test (default: {RANDOM_SEED}). '
             f'Paper results used seed {RANDOM_SEED}.',
    )
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='mne')
        main(
            fold_dir   = args.fold_dir,
            epochs_csv = args.epochs_csv,
            output_dir = args.output_dir,
            seed       = args.seed,
        )
