"""Tests for the Axis-3 CQR conditional-interval evaluation.

These verify the core claim the script is meant to support: under
level-dependent (input-conditional) heteroscedasticity, a conformalized
level-conditional Gaussian head (CQR) beats a *static* per-(node, horizon)
split-conformal baseline at matched marginal coverage -- narrower intervals and
flatter conditional coverage -- while both retain valid marginal coverage.

Runnable via ``python -m pytest tests/test_cqr_conditional_intervals.py`` or
directly with ``python tests/test_cqr_conditional_intervals.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cqr_conditional_intervals import (  # noqa: E402
    conformal_quantile,
    make_synthetic_mu_y,
    paired_bootstrap_verdict,
    run_all_arms,
)


def _result(alpha: float = 0.10):
    mu_v, y_v = make_synthetic_mu_y(800, 12, 40, seed=1)
    mu_t, y_t = make_synthetic_mu_y(800, 12, 40, seed=2)
    return run_all_arms(mu_v, y_v, mu_t, y_t, alpha=alpha, n_level_bins=5, cqr_group="nodeh")


def test_cqr_narrower_at_matched_coverage():
    res = _result()
    mc = res["matched_coverage"]
    # CQR must be strictly narrower than static split-conformal once val
    # coverage is held equal.
    assert mc["cqr_level"]["mpiw"] < mc["conformal_pernodeh"]["mpiw"]
    assert mc["mpiw_reduction_pct"] > 5.0


def test_cqr_flatter_conditional_coverage():
    res = _result()
    target = res["nominal_coverage"]
    cc = res["conditional_coverage"]

    def level_spread(d):
        bins = [v for k, v in d.items() if k.startswith("level_bin_")]
        return max(bins) - min(bins)

    # The whole point: static conformal mis-covers across congestion levels;
    # the level-conditional head keeps coverage roughly constant.
    assert level_spread(cc["cqr_level"]) < level_spread(cc["conformal_pernodeh"])
    # And its worst congested-level bin is far closer to nominal than conformal's.
    conf_worst = min(v for k, v in cc["conformal_pernodeh"].items() if k.startswith("level_bin_"))
    cqr_worst = min(v for k, v in cc["cqr_level"].items() if k.startswith("level_bin_"))
    assert abs(cqr_worst - target) < abs(conf_worst - target)


def test_cqr_paired_verdict_better():
    res = _result()
    v = res["verdict_cqr_vs_conformal"]
    # CQR significantly better (lower Winkler) than conformal, CI strictly < 0.
    assert v["verdict"] == "BETTER"
    assert v["ci_high"] < 0.0


def test_arms_present_and_marginally_valid():
    res = _result()
    arms = res["arms"]
    for name in (
        "trivial_global_gaussian",
        "conformal_pernodeh",
        "gaussian_pernodeh",
        "cqr_level",
    ):
        assert name in arms
    # Split-conformal must be a sane marginal interval (coverage in a believable
    # band around nominal).
    assert 0.7 < arms["conformal_pernodeh"]["picp"] < 0.98


def test_conformal_quantile_finite_sample_rank():
    # n=9, alpha=0.1 -> rank=ceil(10*0.9)=9 -> the max.
    s = np.arange(1.0, 10.0)
    assert conformal_quantile(s, 0.10) == 9.0
    # alpha large enough that rank collapses to the median order statistic.
    assert conformal_quantile(s, 0.5) == 5.0


def test_paired_bootstrap_detects_no_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    b = a + rng.normal(0, 1e-9, 500)  # essentially identical
    v = paired_bootstrap_verdict(a, b, n_boot=500, seed=0)
    assert v["verdict"] == "TIE"


if __name__ == "__main__":
    test_cqr_narrower_at_matched_coverage()
    test_cqr_flatter_conditional_coverage()
    test_cqr_paired_verdict_better()
    test_arms_present_and_marginally_valid()
    test_conformal_quantile_finite_sample_rank()
    test_paired_bootstrap_detects_no_difference()
    print("all CQR tests passed")
