"""
tests/smoke_test.py
===================
Smoke test for nemo_pipeline.py — runs entirely on synthetic data.

No external downloads required. Does NOT need the fOLD XLS file or
the real NEMO epochs CSV. Tests the core statistical functions using
a minimal synthetic dataset.

Compatible with both the original monolithic pipeline and the
refactored version (which exposes run_valence_contrast as a
standalone function). Refactored-only tests are skipped cleanly
on the original pipeline.

Run with:
    python -m pytest tests/smoke_test.py -v

Or directly:
    python tests/smoke_test.py
"""

import sys
import os
import math
import tempfile

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nemo_pipeline import (
    apply_fdr,
    head_mm_to_mni_approx,
    get_specificity,
    FDR_ALPHA,
    MIN_SUBJECTS_PER_CHANNEL,
    MIN_CHANNELS_PER_SUBJECT,
    REGION_ORDER,
    CH_REGION,
)

# Sub-functions present only in the refactored pipeline
try:
    from nemo_pipeline import run_valence_contrast, load_epochs, mirror_detector_d8
    HAS_SUBFUNCTIONS = True
except ImportError:
    run_valence_contrast = load_epochs = mirror_detector_d8 = None
    HAS_SUBFUNCTIONS = False


# ============================================================
# SYNTHETIC DATA
# ============================================================

SYNTH_SEED       = 42
SYNTH_N_SUBJECTS = 5
SYNTH_CONDITIONS = ['HAPV', 'LAPV', 'HANV', 'LANV']
SYNTH_CHANNELS   = [
    'S1_D1',  'S3_D1',  'S8_D6',  'S3_D2',  'S4_D2',
    'S9_D6',  'S6_D5',  'S8_D5',  'S9_D8',  'S5_D4',
    'S8_D8',  'S5_D3',  'S2_D1',  'S8_D7',  'S10_D7',
    'S7_D5',  'S10_D8',
]
SYNTH_TIME     = np.arange(0, 12.1, 0.1)
SYNTH_SUBJECTS = [f'sub-{100 + i}' for i in range(1, SYNTH_N_SUBJECTS + 1)]


def make_synthetic_epochs_df(seed=SYNTH_SEED):
    """
    Minimal synthetic epochs DataFrame in NEMO semicolon-CSV format.
    HAPV/LAPV conditions have a +0.1 shift relative to HANV/LANV so
    the valence t-test has a detectable positive signal.
    """
    rng  = np.random.default_rng(seed)
    rows = []
    for sub in SYNTH_SUBJECTS:
        for cond in SYNTH_CONDITIONS:
            shift = 0.1 if cond in ('HAPV', 'LAPV') else 0.0
            for t in SYNTH_TIME:
                row = {
                    'subject':      sub,
                    'condition':    cond,
                    'time':         round(float(t), 1),
                    'is_bad_epoch': False,
                }
                for ch in SYNTH_CHANNELS:
                    row[f'{ch} hbo'] = float(rng.normal(shift, 0.5))
                rows.append(row)
    return pd.DataFrame(rows)


def save_synthetic_csv(df, path):
    df.to_csv(path, sep=';', index=False)


def _compute_contrast_manually(df):
    """
    Inline re-implementation of the Part 5 logic for use when
    run_valence_contrast is not importable (original pipeline).
    Returns a dict with the same keys as the fixture below.
    """
    hbo_cols  = [c for c in df.columns if c.endswith(' hbo') and '_D' in c]
    col_to_ch = {c: c.replace(' hbo', '').strip() for c in hbo_cols}
    df = df[(df['time'] >= 0) & (df['time'] <= 12)].copy()

    all_pv_nv, all_ha_la = {}, {}
    for sub in sorted(df['subject'].unique()):
        sub_df = df[df['subject'] == sub]
        means  = {cond: {col: sub_df[sub_df['condition'] == cond][col].mean()
                         for col in hbo_cols}
                  for cond in SYNTH_CONDITIONS}
        pv_nv_sub, ha_la_sub = {}, {}
        for col, ch in col_to_ch.items():
            vals = [means[c].get(col, np.nan) for c in SYNTH_CONDITIONS]
            if all(pd.notna(vals)):
                hapv, lapv, hanv, lanv = (means[c][col] for c in SYNTH_CONDITIONS)
                pv_nv_sub[ch] = float((hapv + lapv) / 2 - (hanv + lanv) / 2)
                ha_la_sub[ch] = float((hapv + hanv) / 2 - (lapv + lanv) / 2)
        if len(pv_nv_sub) >= MIN_CHANNELS_PER_SUBJECT:
            all_pv_nv[sub] = pv_nv_sub
            all_ha_la[sub] = ha_la_sub

    group_tvals, group_pvals = {}, {}
    for ch in SYNTH_CHANNELS:
        vals = [all_pv_nv[s][ch] for s in all_pv_nv if ch in all_pv_nv[s]]
        if len(vals) >= MIN_SUBJECTS_PER_CHANNEL:
            t, p = ttest_1samp(vals, 0)
            group_tvals[ch], group_pvals[ch] = float(t), float(p)

    valid_pvals = {ch: group_pvals[ch] for ch in SYNTH_CHANNELS if ch in group_pvals}
    fdr_results = apply_fdr(valid_pvals)

    arousal_tvals, arousal_pvals = {}, {}
    for ch in SYNTH_CHANNELS:
        vals = [all_ha_la[s][ch] for s in all_ha_la if ch in all_ha_la[s]]
        if len(vals) >= MIN_SUBJECTS_PER_CHANNEL:
            t, p = ttest_1samp(vals, 0)
            arousal_tvals[ch], arousal_pvals[ch] = float(t), float(p)

    fdr_arousal       = apply_fdr({ch: arousal_pvals[ch] for ch in SYNTH_CHANNELS
                                   if ch in arousal_pvals})
    n_sig_arousal_fdr = sum(1 for r in fdr_arousal.values() if r[2])

    region_tvals = {r: float(np.mean([abs(group_tvals[ch]) for ch in SYNTH_CHANNELS
                                       if CH_REGION.get(ch) == r and ch in group_tvals]))
                    for r in REGION_ORDER}
    emp_order = sorted(region_tvals, key=region_tvals.get, reverse=True)

    return dict(
        all_pv_nv         = all_pv_nv,
        n_subs            = len(all_pv_nv),
        group_tvals       = group_tvals,
        fdr_results       = fdr_results,
        valid_pvals       = valid_pvals,
        fdr_arousal       = fdr_arousal,
        region_tvals      = region_tvals,
        emp_order         = emp_order,
        n_positive        = sum(1 for t in group_tvals.values() if t > 0),
        n_sig_uncorr      = sum(1 for p in valid_pvals.values() if p < FDR_ALPHA),
        n_sig_fdr         = sum(1 for r in fdr_results.values() if r[2]),
        n_sig_arousal_fdr = n_sig_arousal_fdr,
    )


# ============================================================
# TESTS: apply_fdr
# ============================================================

class TestApplyFdr:

    def test_returns_all_keys(self):
        pvals = {'ch1': 0.01, 'ch2': 0.5, 'ch3': 0.001}
        assert set(apply_fdr(pvals).keys()) == set(pvals.keys())

    def test_tuple_structure(self):
        for _, (p_raw, p_fdr, sig) in apply_fdr({'a': 0.01, 'b': 0.5}).items():
            assert isinstance(p_raw, float)
            assert isinstance(p_fdr, float)
            assert isinstance(sig,   bool)

    def test_p_raw_preserved(self):
        pvals = {'ch1': 0.01, 'ch2': 0.5, 'ch3': 0.03}
        for k in pvals:
            assert math.isclose(apply_fdr(pvals)[k][0], pvals[k], rel_tol=1e-9)

    def test_fdr_values_in_unit_interval(self):
        pvals = {f'ch{i}': v for i, v in enumerate([0.001, 0.01, 0.05, 0.2, 0.8])}
        for _, (_, p_fdr, _) in apply_fdr(pvals).items():
            assert 0.0 <= p_fdr <= 1.0

    def test_near_zero_all_significant(self):
        pvals = {f'ch{i}': 1e-6 for i in range(10)}
        assert all(r[2] for r in apply_fdr(pvals).values())

    def test_large_p_not_significant(self):
        pvals = {f'ch{i}': 0.9 for i in range(10)}
        assert not any(r[2] for r in apply_fdr(pvals).values())

    def test_fdr_non_decreasing(self):
        pvals      = {'a': 0.001, 'b': 0.01, 'c': 0.05, 'd': 0.3}
        result     = apply_fdr(pvals)
        sorted_fdr = [result[k][1] for k in sorted(pvals, key=pvals.get)]
        for i in range(len(sorted_fdr) - 1):
            assert sorted_fdr[i] <= sorted_fdr[i + 1] + 1e-10

    def test_single_channel(self):
        result = apply_fdr({'only': 0.03})
        p_raw, p_fdr, _ = result['only']
        assert math.isclose(p_raw, 0.03)
        assert math.isclose(p_fdr, 0.03, rel_tol=1e-6)


# ============================================================
# TESTS: head_mm_to_mni_approx
# ============================================================

class TestHeadMniApprox:

    def test_shape_1d(self):
        assert head_mm_to_mni_approx([10.0, 20.0, 30.0]).shape == (3,)

    def test_shape_2d(self):
        assert head_mm_to_mni_approx([[10., 20., 30.], [5., 15., 25.]]).shape == (2, 3)

    def test_origin(self):
        np.testing.assert_allclose(
            head_mm_to_mni_approx([0., 0., 0.]),
            [0., -14., 47.], atol=1e-6)

    def test_finite(self):
        assert np.all(np.isfinite(head_mm_to_mni_approx([50., -30., 80.])))


# ============================================================
# TESTS: mirror_detector_d8  (refactored only)
# ============================================================

@pytest.mark.skipif(not HAS_SUBFUNCTIONS,
                    reason="mirror_detector_d8 only in refactored pipeline")
class TestMirrorDetectorD8:

    def _pos(self, x=0.05, y=0.02, z=0.10):
        return {'D1': np.array([x, y, z])}

    def test_x_negated(self):
        d8, _ = mirror_detector_d8(self._pos(x=0.05))
        assert math.isclose(d8[0], -0.05, rel_tol=1e-9)

    def test_y_z_unchanged(self):
        d8, _ = mirror_detector_d8(self._pos(y=0.02, z=0.10))
        assert math.isclose(d8[1], 0.02, rel_tol=1e-9)
        assert math.isclose(d8[2], 0.10, rel_tol=1e-9)

    def test_deviation_non_negative_float(self):
        _, dev = mirror_detector_d8(self._pos())
        assert isinstance(dev, float) and dev >= 0.0

    def test_output_shape(self):
        d8, _ = mirror_detector_d8(self._pos())
        assert d8.shape == (3,)


# ============================================================
# TESTS: valence contrast integration
# ============================================================

class TestValenceContrastIntegration:

    @pytest.fixture(scope='class')
    def R(self):
        """Build synthetic data and run contrast. Class-scoped so built once."""
        df_synth = make_synthetic_epochs_df()

        if HAS_SUBFUNCTIONS:
            with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.csv', delete=False) as f:
                tmp = f.name
            save_synthetic_csv(df_synth, tmp)
            try:
                df, hbo_cols, col_to_ch = load_epochs(tmp)
                out = run_valence_contrast(df, hbo_cols, col_to_ch, SYNTH_CHANNELS)
                (all_pv_nv, all_ha_la, n_subs, group_tvals, fdr_results,
                 _, fdr_arousal, valid_pvals, _,
                 region_tvals, emp_order,
                 n_positive, n_sig_uncorr, n_sig_fdr, n_sig_arousal_fdr) = out
            finally:
                os.unlink(tmp)
            return dict(
                all_pv_nv=all_pv_nv, n_subs=n_subs,
                group_tvals=group_tvals, fdr_results=fdr_results,
                valid_pvals=valid_pvals, fdr_arousal=fdr_arousal,
                region_tvals=region_tvals, emp_order=emp_order,
                n_positive=n_positive, n_sig_uncorr=n_sig_uncorr,
                n_sig_fdr=n_sig_fdr, n_sig_arousal_fdr=n_sig_arousal_fdr,
            )
        else:
            df_f = df_synth[~df_synth['is_bad_epoch']].copy()
            return _compute_contrast_manually(df_f)

    def test_n_subs(self, R):
        assert R['n_subs'] == SYNTH_N_SUBJECTS

    def test_channel_keys_complete(self, R):
        assert set(R['valid_pvals'].keys()) == set(SYNTH_CHANNELS)
        assert set(R['fdr_results'].keys()) == set(SYNTH_CHANNELS)

    def test_t_values_finite_floats(self, R):
        for ch, t in R['group_tvals'].items():
            assert isinstance(t, float) and math.isfinite(t), \
                f"t for {ch}: {t}"

    def test_fdr_structure(self, R):
        for ch, (p_raw, p_fdr, sig) in R['fdr_results'].items():
            assert isinstance(sig, bool)
            assert 0.0 <= p_raw <= 1.0
            assert 0.0 <= p_fdr <= 1.0

    def test_fdr_leq_uncorrected(self, R):
        assert R['n_sig_fdr'] <= R['n_sig_uncorr']

    def test_n_positive_in_range(self, R):
        assert 0 <= R['n_positive'] <= len(SYNTH_CHANNELS)

    def test_arousal_non_negative(self, R):
        assert R['n_sig_arousal_fdr'] >= 0

    def test_three_regions(self, R):
        assert set(R['region_tvals'].keys()) == {'Anterior', 'Frontopolar', 'Dorsal'}

    def test_emp_order_permutation(self, R):
        assert sorted(R['emp_order']) == sorted(['Anterior', 'Frontopolar', 'Dorsal'])

    def test_contrast_values_finite(self, R):
        for sub, ch_vals in R['all_pv_nv'].items():
            for ch, val in ch_vals.items():
                assert math.isfinite(val), f"Non-finite: {sub}/{ch}"

    def test_majority_positive_t(self, R):
        """Synthetic PV>NV shift of +0.1 should produce majority positive t."""
        assert R['n_positive'] > len(SYNTH_CHANNELS) // 2, \
            f"Expected majority positive t-values, got {R['n_positive']}"


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))
