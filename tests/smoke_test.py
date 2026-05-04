"""
tests/smoke_test.py
===================
Smoke test for nemo_pipeline.py — runs entirely on synthetic data.
No external downloads required.

Run with:
    python -m pytest tests/smoke_test.py -v
Or directly:
    python tests/smoke_test.py
"""

import sys, os, math, tempfile
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nemo_pipeline import (
    apply_fdr, head_mm_to_mni_approx, get_specificity,
    FDR_ALPHA, MIN_SUBJECTS_PER_CHANNEL, MIN_CHANNELS_PER_SUBJECT,
    REGION_ORDER, CH_REGION,
)

try:
    from nemo_pipeline import run_valence_contrast, load_epochs, mirror_detector_d8
    HAS_SUBFUNCTIONS = True
except ImportError:
    run_valence_contrast = load_epochs = mirror_detector_d8 = None
    HAS_SUBFUNCTIONS = False

# ── Synthetic data config ────────────────────────────────────
SYNTH_SEED       = 42
SYNTH_N_SUBJECTS = 5
SYNTH_CONDITIONS = ['HAPV', 'LAPV', 'HANV', 'LANV']
SYNTH_CHANNELS   = [
    'S1_D1','S3_D1','S8_D6','S3_D2','S4_D2','S9_D6','S6_D5','S8_D5',
    'S9_D8','S5_D4','S8_D8','S5_D3','S2_D1','S8_D7','S10_D7','S7_D5','S10_D8',
]
SYNTH_TIME     = np.arange(0, 12.1, 0.1)
SYNTH_SUBJECTS = [f'sub-{100+i}' for i in range(1, SYNTH_N_SUBJECTS+1)]

def make_synthetic_epochs_df(seed=SYNTH_SEED):
    rng, rows = np.random.default_rng(seed), []
    for sub in SYNTH_SUBJECTS:
        for cond in SYNTH_CONDITIONS:
            shift = 0.1 if cond in ('HAPV','LAPV') else 0.0
            for t in SYNTH_TIME:
                row = {'subject': sub, 'condition': cond,
                       'time': round(float(t),1), 'is_bad_epoch': False}
                for ch in SYNTH_CHANNELS:
                    row[f'{ch} hbo'] = float(rng.normal(shift, 0.5))
                rows.append(row)
    return pd.DataFrame(rows)

def compute_contrast_manually(df):
    """Fallback contrast logic for monolithic pipeline version."""
    hbo_cols  = [c for c in df.columns if c.endswith(' hbo') and '_D' in c]
    col_to_ch = {c: c.replace(' hbo','').strip() for c in hbo_cols}
    df = df[(df['time'] >= 0) & (df['time'] <= 12)].copy()
    all_pv_nv, all_ha_la = {}, {}
    for sub in sorted(df['subject'].unique()):
        sub_df = df[df['subject'] == sub]
        means  = {cond: {col: sub_df[sub_df['condition']==cond][col].mean()
                         for col in hbo_cols} for cond in SYNTH_CONDITIONS}
        pv_nv_sub, ha_la_sub = {}, {}
        for col, ch in col_to_ch.items():
            v = [means[c].get(col, np.nan) for c in SYNTH_CONDITIONS]
            if all(pd.notna(v)):
                pv_nv_sub[ch] = float((v[0]+v[1])/2 - (v[2]+v[3])/2)
                ha_la_sub[ch] = float((v[0]+v[2])/2 - (v[1]+v[3])/2)
        if len(pv_nv_sub) >= MIN_CHANNELS_PER_SUBJECT:
            all_pv_nv[sub] = pv_nv_sub
            all_ha_la[sub] = ha_la_sub
    group_tvals, group_pvals = {}, {}
    for ch in SYNTH_CHANNELS:
        vals = [all_pv_nv[s][ch] for s in all_pv_nv if ch in all_pv_nv[s]]
        if len(vals) >= MIN_SUBJECTS_PER_CHANNEL:
            t, p = ttest_1samp(vals, 0)
            group_tvals[ch] = float(t); group_pvals[ch] = float(p)
    valid_pvals = {ch: group_pvals[ch] for ch in SYNTH_CHANNELS if ch in group_pvals}
    fdr_results = apply_fdr(valid_pvals)
    n_positive  = sum(1 for ch in SYNTH_CHANNELS if group_tvals.get(ch,0) > 0)
    n_uncorr    = sum(1 for p in valid_pvals.values() if p < FDR_ALPHA)
    n_fdr       = sum(1 for r in fdr_results.values() if r[2])
    group_tvals_aro, group_pvals_aro = {}, {}
    for ch in SYNTH_CHANNELS:
        vals = [all_ha_la[s][ch] for s in all_ha_la if ch in all_ha_la[s]]
        if len(vals) >= MIN_SUBJECTS_PER_CHANNEL:
            t, p = ttest_1samp(vals, 0)
            group_tvals_aro[ch] = float(t); group_pvals_aro[ch] = float(p)
    valid_aro_p = {ch: group_pvals_aro[ch] for ch in SYNTH_CHANNELS if ch in group_pvals_aro}
    fdr_arousal = apply_fdr(valid_aro_p)
    n_aro_fdr   = sum(1 for r in fdr_arousal.values() if r[2])
    region_tvals = {r: float(np.mean([abs(group_tvals[ch]) for ch in SYNTH_CHANNELS
                    if CH_REGION.get(ch)==r and ch in group_tvals])) for r in REGION_ORDER}
    emp_order = sorted(region_tvals, key=region_tvals.get, reverse=True)
    return dict(all_pv_nv=all_pv_nv, n_subs=len(all_pv_nv), group_tvals=group_tvals,
                fdr_results=fdr_results, valid_pvals_for_fdr=valid_pvals,
                fdr_arousal=fdr_arousal, region_tvals=region_tvals, emp_order=emp_order,
                n_positive=n_positive, n_sig_uncorr=n_uncorr,
                n_sig_fdr=n_fdr, n_sig_arousal_fdr=n_aro_fdr)


# ── apply_fdr tests ──────────────────────────────────────────
class TestApplyFdr:
    def test_returns_all_keys(self):
        p = {'a':0.01,'b':0.5,'c':0.001}
        assert set(apply_fdr(p).keys()) == set(p.keys())

    def test_tuple_structure(self):
        for k,(p_raw,p_fdr,sig) in apply_fdr({'a':0.01,'b':0.5}).items():
            assert isinstance(p_raw,float) and isinstance(p_fdr,float) and isinstance(sig,bool)

    def test_p_raw_preserved(self):
        p = {'a':0.01,'b':0.5,'c':0.03}
        for k in p:
            assert math.isclose(apply_fdr(p)[k][0], p[k], rel_tol=1e-9)

    def test_fdr_in_0_1(self):
        p = {f'ch{i}':v for i,v in enumerate([0.001,0.01,0.05,0.2,0.8])}
        for k,(_,p_fdr,__) in apply_fdr(p).items():
            assert 0.0 <= p_fdr <= 1.0

    def test_near_zero_all_significant(self):
        p = {f'ch{i}':1e-6 for i in range(10)}
        assert all(r[2] for r in apply_fdr(p).values())

    def test_large_p_not_significant(self):
        p = {f'ch{i}':0.9 for i in range(10)}
        assert not any(r[2] for r in apply_fdr(p).values())

    def test_non_decreasing_after_sort(self):
        p = {'a':0.001,'b':0.01,'c':0.05,'d':0.3}
        r = apply_fdr(p)
        fdrs = [r[k][1] for k in sorted(p, key=p.get)]
        for i in range(len(fdrs)-1):
            assert fdrs[i] <= fdrs[i+1] + 1e-10

    def test_single_channel(self):
        r = apply_fdr({'only':0.03})
        assert math.isclose(r['only'][0], 0.03)
        assert math.isclose(r['only'][1], 0.03, rel_tol=1e-6)


# ── head_mm_to_mni_approx tests ──────────────────────────────
class TestHeadMniApprox:
    def test_shape_1d(self):
        assert head_mm_to_mni_approx([10.,20.,30.]).shape == (3,)

    def test_shape_2d(self):
        assert head_mm_to_mni_approx([[10.,20.,30.],[5.,15.,25.]]).shape == (2,3)

    def test_origin_to_offset(self):
        np.testing.assert_allclose(head_mm_to_mni_approx([0.,0.,0.]),
                                   [0.,-14.,47.], atol=1e-6)

    def test_finite_output(self):
        assert np.all(np.isfinite(head_mm_to_mni_approx([50.,-30.,80.])))


# ── mirror_detector_d8 tests (refactored only) ───────────────
@pytest.mark.skipif(not HAS_SUBFUNCTIONS,
                    reason="mirror_detector_d8 only in refactored pipeline")
class TestMirrorDetectorD8:
    def _pos(self, x=0.05, y=0.02, z=0.10):
        return {'D1': np.array([x,y,z])}

    def test_x_negated(self):
        d8,_ = mirror_detector_d8(self._pos(x=0.05))
        assert math.isclose(d8[0], -0.05, rel_tol=1e-9)

    def test_y_z_unchanged(self):
        d8,_ = mirror_detector_d8(self._pos())
        assert math.isclose(d8[1], 0.02, rel_tol=1e-9)
        assert math.isclose(d8[2], 0.10, rel_tol=1e-9)

    def test_deviation_non_negative_float(self):
        _,dev = mirror_detector_d8(self._pos())
        assert isinstance(dev, float) and dev >= 0.0

    def test_output_shape(self):
        d8,_ = mirror_detector_d8(self._pos())
        assert d8.shape == (3,)


# ── valence contrast integration ─────────────────────────────
class TestValenceContrastIntegration:

    @pytest.fixture(scope='class')
    def cr(self):
        df_synth = make_synthetic_epochs_df()
        if HAS_SUBFUNCTIONS:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                tmp = f.name
            df_synth.to_csv(tmp, sep=';', index=False)
            try:
                df, hbo_cols, col_to_ch = load_epochs(tmp)
                out = run_valence_contrast(df, hbo_cols, col_to_ch, SYNTH_CHANNELS)
                (all_pv_nv, all_ha_la, n_subs, group_tvals, fdr_results,
                 _, fdr_arousal, valid_pvals, _,
                 region_tvals, emp_order,
                 n_positive, n_uncorr, n_fdr, n_aro_fdr) = out
            finally:
                os.unlink(tmp)
            return dict(all_pv_nv=all_pv_nv, n_subs=n_subs, group_tvals=group_tvals,
                        fdr_results=fdr_results, valid_pvals_for_fdr=valid_pvals,
                        fdr_arousal=fdr_arousal, region_tvals=region_tvals,
                        emp_order=emp_order, n_positive=n_positive,
                        n_sig_uncorr=n_uncorr, n_sig_fdr=n_fdr, n_sig_arousal_fdr=n_aro_fdr)
        else:
            df_f = df_synth[~df_synth['is_bad_epoch']].copy()
            return compute_contrast_manually(df_f)

    def test_n_subs(self, cr):
        assert cr['n_subs'] == SYNTH_N_SUBJECTS

    def test_channel_keys_complete(self, cr):
        assert set(cr['valid_pvals_for_fdr'].keys()) == set(SYNTH_CHANNELS)
        assert set(cr['fdr_results'].keys())         == set(SYNTH_CHANNELS)

    def test_t_values_finite_floats(self, cr):
        for ch,t in cr['group_tvals'].items():
            assert isinstance(t, float) and math.isfinite(t)

    def test_fdr_structure(self, cr):
        for ch,(p_raw,p_fdr,sig) in cr['fdr_results'].items():
            assert isinstance(sig, bool)
            assert 0.0 <= p_raw <= 1.0
            assert 0.0 <= p_fdr <= 1.0

    def test_fdr_leq_uncorrected(self, cr):
        assert cr['n_sig_fdr'] <= cr['n_sig_uncorr']

    def test_n_positive_in_range(self, cr):
        assert 0 <= cr['n_positive'] <= len(SYNTH_CHANNELS)

    def test_arousal_fdr_non_negative(self, cr):
        assert cr['n_sig_arousal_fdr'] >= 0

    def test_three_regions(self, cr):
        assert set(cr['region_tvals'].keys()) == {'Anterior','Frontopolar','Dorsal'}

    def test_emp_order_is_permutation(self, cr):
        assert sorted(cr['emp_order']) == sorted(['Anterior','Frontopolar','Dorsal'])

    def test_contrast_values_finite(self, cr):
        for sub,ch_vals in cr['all_pv_nv'].items():
            for ch,val in ch_vals.items():
                assert math.isfinite(val), f"Non-finite: {sub}/{ch}"

    def test_majority_positive_t(self, cr):
        assert cr['n_positive'] > len(SYNTH_CHANNELS) // 2


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))


# ============================================================
# TESTS: BH fallback logic regression (Comment 5)
# ============================================================

class TestBHFallback:
    """
    Regression test for the manual Benjamini-Hochberg fallback.

    The pipeline has a manual BH implementation for scipy < 1.11,
    guarded by HAS_SCIPY_FDR. This test monkey-patches the flag to
    False, forcing the fallback path regardless of installed scipy
    version, then verifies its output matches the native implementation.

    This ensures the fallback is never silently broken by a refactor.
    """

    # Known p-values with a mix of significant and non-significant results
    TEST_PVALS = {
        'ch0': 0.001,
        'ch1': 0.008,
        'ch2': 0.039,
        'ch3': 0.041,
        'ch4': 0.189,
        'ch5': 0.257,
        'ch6': 0.530,
        'ch7': 0.740,
    }

    def _run_with_flag(self, flag_value):
        """Run apply_fdr with HAS_SCIPY_FDR forced to flag_value."""
        import nemo_pipeline as npm
        original = npm.HAS_SCIPY_FDR
        npm.HAS_SCIPY_FDR = flag_value
        try:
            result = npm.apply_fdr(self.TEST_PVALS.copy())
        finally:
            npm.HAS_SCIPY_FDR = original
        return result

    def test_fallback_adjusted_pvalues_match_native(self):
        """Manual BH adjusted p-values must match scipy within 1e-6."""
        native   = self._run_with_flag(True)
        fallback = self._run_with_flag(False)
        for ch in self.TEST_PVALS:
            assert abs(native[ch][1] - fallback[ch][1]) < 1e-6, (
                f"BH fallback p_fdr mismatch for {ch}: "
                f"native={native[ch][1]:.8f}, fallback={fallback[ch][1]:.8f}"
            )

    def test_fallback_significance_decisions_match_native(self):
        """Significance flags (True/False) must be identical between paths."""
        native   = self._run_with_flag(True)
        fallback = self._run_with_flag(False)
        for ch in self.TEST_PVALS:
            assert native[ch][2] == fallback[ch][2], (
                f"Significance mismatch for {ch}: "
                f"native={native[ch][2]}, fallback={fallback[ch][2]}"
            )

    def test_fallback_raw_pvalues_preserved(self):
        """Raw p-values must be unchanged in both paths."""
        fallback = self._run_with_flag(False)
        for ch, p in self.TEST_PVALS.items():
            assert math.isclose(fallback[ch][0], p, rel_tol=1e-9), (
                f"Raw p-value corrupted for {ch}: "
                f"expected={p}, got={fallback[ch][0]}"
            )

    def test_fallback_fdr_values_non_decreasing(self):
        """BH-adjusted p-values must be non-decreasing when sorted by raw p."""
        fallback   = self._run_with_flag(False)
        sorted_fdr = [fallback[k][1]
                      for k in sorted(self.TEST_PVALS, key=self.TEST_PVALS.get)]
        for i in range(len(sorted_fdr) - 1):
            assert sorted_fdr[i] <= sorted_fdr[i + 1] + 1e-10, (
                f"BH fallback not monotone at index {i}: "
                f"{sorted_fdr[i]:.6f} > {sorted_fdr[i+1]:.6f}"
            )

    def test_fallback_all_significant_when_all_tiny(self):
        """All near-zero p-values should survive FDR in both paths."""
        tiny = {f'ch{i}': 1e-8 for i in range(10)}
        import nemo_pipeline as npm
        original = npm.HAS_SCIPY_FDR
        npm.HAS_SCIPY_FDR = False
        try:
            result = npm.apply_fdr(tiny)
        finally:
            npm.HAS_SCIPY_FDR = original
        assert all(r[2] for r in result.values()), \
            "BH fallback: all near-zero p-values should be significant"

    def test_fallback_none_significant_when_all_large(self):
        """Large p-values should not survive FDR in fallback path."""
        large = {f'ch{i}': 0.9 for i in range(10)}
        import nemo_pipeline as npm
        original = npm.HAS_SCIPY_FDR
        npm.HAS_SCIPY_FDR = False
        try:
            result = npm.apply_fdr(large)
        finally:
            npm.HAS_SCIPY_FDR = original
        assert not any(r[2] for r in result.values()), \
            "BH fallback: large p-values should not be significant"