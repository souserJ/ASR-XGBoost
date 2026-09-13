"""Regression checks for experimental validity, not ASR superiority."""
import numpy as np
import pytest

import asr_ademp as sim


def small_data():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(100, 3))
    truth = sim.expit(X[:, 0])
    return dict(X=X, y=(rng.random(100)<truth).astype(float),
                p_true=truth, p_obs=truth, R=np.ones((10, 10)), land=np.ones((10, 10), bool))


def small_args():
    return sim.parse_args(["--num-round", "4", "--early-stop", "2", "--threads", "1",
                           "--inner-folds", "2", "--lam-grid", "0,1", "--min-block-score", "2"])


def test_outer_labels_cannot_change_predictions_or_tuning():
    data = small_data()
    train, test = np.arange(80), np.arange(80, 100)
    labels = sim.grid_regions(10, 2).ravel()
    first = sim.evaluate_fold(data, train, test, labels, small_args(), 7)
    changed = dict(data, y=data["y"].copy())
    changed["y"][test] = 1-changed["y"][test]
    second = sim.evaluate_fold(changed, train, test, labels, small_args(), 7)
    for method in sim.METHODS:
        np.testing.assert_array_equal(first[0][method], second[0][method])
    assert first[2] == second[2]
    np.testing.assert_allclose([r["selected"] for r in first[1]], [r["selected"] for r in second[1]])


def test_inner_fit_and_early_stop_exclude_scoring_labels(monkeypatch):
    data, args = small_data(), small_args()
    train = np.arange(80)
    scores = sim.make_folds(train, args.inner_folds, 13)
    calls = []
    real_fit = sim.fit_predict
    def spy(data, fit, stop, *a, **kw):
        calls.append((fit.copy(), stop.copy()))
        return real_fit(data, fit, stop, *a, **kw)
    monkeypatch.setattr(sim, "fit_predict", spy)
    sim.tune_lambdas(data, train, np.zeros(100, int), args, 13)
    assert len(calls) == args.inner_folds*len(args.lam_grid)
    for fi, score in enumerate(scores):
        for fit, stop in calls[fi*2:(fi+1)*2]:
            assert not np.intersect1d(score, np.r_[fit, stop]).size
            assert not np.intersect1d(fit, stop).size


def test_zero_lambda_exactly_matches_ce():
    data, args = small_data(), small_args()
    fit, stop = sim.fit_stop_split(np.arange(80), 4)
    ce, rounds = sim.fit_predict(data, fit, stop, args, 4)
    zero, rounds_zero = sim.fit_predict(data, fit, stop, args, 4,
                                       soft=np.zeros(100), delta=np.ones(100), lam=np.zeros(100))
    np.testing.assert_array_equal(ce, zero)
    assert rounds == rounds_zero


def test_predict_uses_best_iteration(monkeypatch):
    seen = {}
    class Booster:
        best_iteration = 2
        def predict(self, data, **kwargs):
            seen.update(kwargs)
            return np.zeros(data.num_row())
    def train(*args, **kwargs):
        return Booster()
    monkeypatch.setattr(sim.xgb, "train", train)
    data = small_data()
    sim.fit_predict(data, np.arange(60), np.arange(60, 80), small_args(), 1)
    assert seen["iteration_range"] == (0, 3)


def test_spatial_groups_are_disjoint_and_all_pixels_covered():
    groups = sim.grid_regions(11, 4).ravel()
    idx = np.arange(121)
    folds = sim.make_folds(idx, 3, 3, groups)
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), idx)
    for i, fold in enumerate(folds):
        for other in folds[i+1:]:
            assert not set(groups[fold]) & set(groups[other])
    assert groups.max() == 15


def test_soft_reference_has_no_wraparound_and_excludes_ocean():
    p = np.zeros((4, 4))
    p[-1, -1] = 1
    land = np.ones((4, 4), bool)
    soft = sim.soft_reference(p.ravel(), np.ones_like(p), land, 2).reshape(4, 4)
    assert soft[0, 0] == 0
    assert soft[2, 2] > 0
    land[-1, -1] = False
    soft = sim.soft_reference(p.ravel(), np.ones_like(p), land, 2).reshape(4, 4)
    assert soft[2, 2] == 0


def test_mcse_and_paired_differences_do_not_pool_scenarios():
    values = [1., 2., 3.]
    stats = sim.summary_stats(values)
    assert stats["n"] == 3
    assert stats["mean"] == 2
    assert stats["mcse"] == pytest.approx(1/np.sqrt(3))
    rows = []
    for scenario, offset in (("a", 0), ("b", 100)):
        for rep in range(3):
            for method in sim.METHODS:
                value = offset+rep+(rep if method == "ASR" else 0)
                rows.append(dict(scenario=scenario, partition="p", validation="random", rep=rep,
                                 method=method, metric="brier", value=value))
    summary, pairs = sim.summarize(rows)
    assert len(summary) == 8
    for row in pairs:
        assert row["n"] == 3
        assert row["mean"] == 1
        assert row["mcse"] == pytest.approx(1/np.sqrt(3))
    assert np.isnan(sim.summary_stats([1])["mcse"])


def test_auc_ties_and_single_class_are_handled():
    data = small_data()
    metrics = sim.prediction_metrics(data, np.arange(100), np.full(100, .5))
    assert metrics["auc"] == .5
    data["y"][:] = 0
    assert np.isnan(sim.prediction_metrics(data, np.arange(100), np.full(100, .5))["auc"])


def test_clipped_noise_marginal_probability():
    rng = np.random.default_rng(8)
    for mu, sigma in ((.02, .35), (.5, .2), (.98, .35), (.5, 0)):
        numerical = np.clip(mu+rng.normal(0, sigma, 200000), .001, .999).mean()
        assert float(sim.clipped_normal_mean(mu, sigma)) == pytest.approx(numerical, abs=.003)


def test_seed_streams_do_not_depend_on_loop_order():
    keys = ["g_clean", "g_mid", "g_detection"]
    original = {k: sim.seed_for(42, k, 0, "landscape") for k in keys}
    reordered = {k: sim.seed_for(42, k, 0, "landscape") for k in reversed(keys)}
    assert original == reordered
    assert len(set(original.values())) == len(keys)
