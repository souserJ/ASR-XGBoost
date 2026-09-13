#!/usr/bin/env python3
"""ADEMP simulation runner **with a land/sea-geometry (coast) factor**.

Copied from asr_ademp.py; the only added factor is coast ∈ {vertical, horizontal,
archipelago, none}. Landscape generation, losses, gating and partitions are unchanged
(they come from asr_demo_coast.py, itself a copy of asr_demo_modified.py).
Conditions become scenario × coast × partition × protocol × repetition.

Coast pairing (`--coast-pairing`):
  * `shared` (default) - the same landscape/DGP/observation/sampling random streams are
    reused across coast levels within a (scenario, repetition): a common-random-numbers
    (blocked) comparison in which the only intended difference is the coastline, so any
    coast effect is not confounded with field randomness. coast=vertical + shared exactly
    reproduces the asr_ademp.py pilot.
  * `independent` - each coast level draws its own streams (fully independent landscapes).

Outer held-out labels never enter fitting, early stopping or lambda selection.
Each inner fold builds its own CE reference using only that fold's fit subset.
All four methods use identical folds, Brier early stopping and training budgets.
"""
import argparse
import csv
import hashlib
import inspect
import json
import math
from itertools import count
from pathlib import Path
import platform
import time

import numpy as np
from scipy.special import expit, ndtr
from scipy.stats import rankdata, t as student_t
import xgboost as xgb

import asr_demo_coast as legacy

VERSION = "ademp-coast-1"
METHODS = ("CE", "SR1", "SRcv", "ASR")


def seed_for(root, *keys):
    """Order-independent random streams; adding/reordering scenarios changes none."""
    digest = hashlib.sha256(json.dumps([root, *keys]).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def grid_regions(n, k=3):
    yy, xx = np.mgrid[:n, :n]
    return (yy * k // n) * k + xx * k // n


def clipped_normal_mean(mu, sigma, lo=0.001, hi=0.999):
    """E[clip(mu + Normal(0,sigma),lo,hi)]: marginal Bernoulli probability."""
    mu, sigma = np.broadcast_arrays(mu, sigma)
    safe = np.where(sigma > 0, sigma, 1.)
    a, b = (lo - mu) / safe, (hi - mu) / safe
    phi = lambda z: np.exp(-z*z/2) / np.sqrt(2*np.pi)
    value = lo*ndtr(a) + mu*(ndtr(b)-ndtr(a)) + safe*(phi(a)-phi(b)) + hi*ndtr(-b)
    return np.where(sigma > 0, value, np.clip(mu, lo, hi))


def generate_data(args, scenario, coast, rep):
    prefix = (scenario, coast, rep) if args.coast_pairing == "independent" else (scenario, rep)
    seeds = {key: seed_for(args.seed, *prefix, key)
             for key in ("landscape", "dgp_regions", "observation", "sample")}
    rng = lambda key: np.random.default_rng(seeds[key])
    cfg = dict(legacy.GENERATIONS[scenario])
    cfg["coast"] = coast
    cfg["ocean"] = coast != "none"
    x1, x2, x3, resistance, truth, ocean = legacy.gen_landscape(args.grid, rng("landscape"), cfg)
    land = np.ones(truth.shape, bool) if ocean is None else ~ocean
    # Generate these regions once, independently of the method's partition.
    dgp = legacy.default_blocks(args.grid, rng("dgp_regions"), args.dgp_blocks, land)
    if cfg.get("frag", 0):
        step = (x1 > 0).astype(float) + (x2 > 0).astype(float) - 1
        truth = expit(cfg.get("logit_scale", 1.) * (
            legacy.response_logit(x1, x2, x3) + cfg["frag"]*step*(dgp % 2 == 0)))
    obs_rng = rng("observation")
    detection = np.ones_like(truth)
    if cfg.get("mode") == "lgcp":
        z = legacy.matern_field(args.grid, obs_rng, cfg.get("len_scale", 18.))
        intensity = cfg.get("nu", .8) * np.exp(.6*z) * truth
        p_obs = -np.expm1(-intensity)
    elif cfg.get("observation_model") == "detection":
        yy, xx = np.mgrid[:args.grid, :args.grid]
        low, high = cfg.get("detect_min", .35), cfg.get("detect_max", .95)
        detection = low + (high-low)*(.5 + .5*np.sin(xx/args.grid*3*np.pi)*np.cos(yy/args.grid*2*np.pi))
        p_obs = np.clip(truth*detection, .001, .999)
    else:
        sigma = cfg.get("noise_base", .08) + cfg.get("noise_diff", .10)*(dgp % 2)
        # Same marginal label distribution as the original iid probability noise.
        p_obs = clipped_normal_mean(truth, sigma)
    y = (obs_rng.random(truth.shape) < p_obs).astype(float).ravel()
    land_idx = np.flatnonzero(land.ravel())
    sample = np.sort(rng("sample").choice(land_idx, int(args.sample*len(land_idx)), replace=False))
    if len(sample) < 30:
        raise ValueError("Fewer than 30 sampled land pixels; increase --grid or --sample")
    return dict(X=np.column_stack([x1.ravel(), x2.ravel(), x3.ravel()]), y=y,
                p_true=truth.ravel(), p_obs=p_obs.ravel(), detection=detection.ravel(),
                R=resistance, land=land, dgp=dgp, sample=sample, seeds=seeds)


def model_regions(data, args, scenario, coast, rep, partition):
    prefix = (scenario, coast, rep) if args.coast_pairing == "independent" else (scenario, rep)
    rng = np.random.default_rng(seed_for(args.seed, *prefix, "model_regions", partition))
    if partition == "p_grid":
        labels = np.where(data["land"], grid_regions(args.grid), -1)
    else:
        labels = legacy.make_partition(partition, args.grid, rng, data["R"],
                                       args.nblocks, data["land"])
    # Even grid/resistance regions cannot join the two land masses across ocean.
    if np.any(~data["land"]):
        labels = legacy.split_blocks_at_barriers(labels, ~data["land"])
    return labels.ravel()


def make_folds(indices, nfold, seed, groups=None):
    """Label-independent splits, with complete spatial groups held out together."""
    rng = np.random.default_rng(seed)
    if groups is None:
        if len(indices) < nfold:
            raise ValueError("Not enough observations for folds")
        return [np.sort(f) for f in np.array_split(rng.permutation(indices), nfold)]
    unique = np.unique(groups[indices])
    if len(unique) < nfold:
        raise ValueError("Not enough occupied spatial groups for folds")
    # Greedy balance group sizes; randomized order breaks equal-size ties.
    unique = rng.permutation(unique)
    unique = sorted(unique, key=lambda g: -np.sum(groups[indices] == g))
    bins, sizes = [[] for _ in range(nfold)], np.zeros(nfold, int)
    for group in unique:
        dest = int(np.argmin(sizes))
        pts = indices[groups[indices] == group]
        bins[dest].extend(pts.tolist())
        sizes[dest] += len(pts)
    return [np.asarray(sorted(b), int) for b in bins]


def soft_reference(p, resistance, land, beta):
    """Finite 8-neighbour grid: no np.roll wraparound or ocean neighbours."""
    n = len(resistance)
    p = p.reshape(n, n)
    num, den = np.zeros_like(p), np.zeros_like(p)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            dst = (slice(max(0, -dy), min(n, n-dy)), slice(max(0, -dx), min(n, n-dx)))
            src = (slice(max(0, dy), min(n, n+dy)), slice(max(0, dx), min(n, n+dx)))
            valid = land[dst] & land[src]
            w = np.exp(-beta*(resistance[dst]+resistance[src])/2)*valid
            num[dst] += w*p[src]
            den[dst] += w
    return np.divide(num, den, out=p.copy(), where=den > 0).ravel()


def fit_predict(data, fit, stop, args, seed, soft=None, delta=None, lam=0.):
    """Early stopping uses a dedicated training-side split, never the score fold."""
    params = dict(objective="binary:logistic", eta=.1, max_depth=4,
                  subsample=.8, colsample_bytree=.8, seed=seed, nthread=args.threads,
                  tree_method="hist", base_score=.5, disable_default_eval_metric=1)
    custom = soft is not None and np.any(np.asarray(lam) != 0)
    def metric(pred, dmatrix):
        p = expit(pred) if custom else pred
        return "brier", float(np.mean((p-dmatrix.get_label())**2))
    kw = dict(num_boost_round=args.num_round, early_stopping_rounds=args.early_stop,
              evals=[(xgb.DMatrix(data["X"][stop], label=data["y"][stop]), "stop")],
              maximize=False, verbose_eval=False)
    kw["custom_metric" if "custom_metric" in inspect.signature(xgb.train).parameters else "feval"] = metric
    if custom:
        strength = lam if np.ndim(lam) == 0 else np.asarray(lam)[fit]
        kw["obj"] = legacy.make_asr_objective(delta[fit], soft[fit], strength)
    bst = xgb.train(params, xgb.DMatrix(data["X"][fit], label=data["y"][fit]), **kw)
    rounds = bst.best_iteration + 1
    pred = expit(bst.predict(xgb.DMatrix(data["X"]), output_margin=True,
                            iteration_range=(0, rounds)))
    return pred, rounds


def fit_stop_split(indices, seed):
    if len(indices) < 10:
        raise ValueError("Too few training observations for separate early stopping")
    perm = np.random.default_rng(seed).permutation(indices)
    nstop = max(2, int(.15*len(perm)))
    return perm[nstop:], perm[:nstop]


def choose_lambda(losses, lam_grid, rule):
    """Rows are score folds, columns are candidate lambdas; fold SE is heuristic."""
    means = np.mean(losses, axis=0)
    se = np.std(losses, axis=0, ddof=1)/np.sqrt(len(losses)) if len(losses)>1 else np.zeros(len(lam_grid))
    best = int(np.argmin(means))
    threshold = means[best] + (se[best] if rule == "1se" else 0.)
    chosen = min(l for l, m in zip(lam_grid, means) if m <= threshold + 1e-15)
    return float(chosen), means, se


def tune_lambdas(data, train, labels, args, seed, groups=None):
    folds = make_folds(train, args.inner_folds, seed, groups)
    block_ids = np.unique(labels[train])
    records, global_losses = [], []
    losses = {int(b): [] for b in block_ids}
    counts = {int(b): 0 for b in block_ids}
    for fold_id, score in enumerate(folds):
        inner_train = np.setdiff1d(train, score)
        fold_seed = seed_for(seed, "inner", fold_id)
        fit, stop = fit_stop_split(inner_train, fold_seed)
        reference, _ = fit_predict(data, fit, stop, args, fold_seed)
        soft = soft_reference(reference, data["R"], data["land"], args.beta)
        delta = legacy.gate_frozen(reference)
        errors = []
        for lam in args.lam_grid:
            pred = reference if lam == 0 else fit_predict(data, fit, stop, args, fold_seed, soft, delta, lam)[0]
            errors.append((pred[score]-data["y"][score])**2)
        errors = np.asarray(errors).T
        global_losses.append(errors.mean(axis=0))
        for block in block_ids:
            mask = labels[score] == block
            if mask.any():
                losses[int(block)].append(errors[mask].mean(axis=0))
                counts[int(block)] += int(mask.sum())
    global_lam, means, se = choose_lambda(np.asarray(global_losses), args.lam_grid, args.rule)
    def record(block, n, folds_used, chosen, avg, stderr, fallback):
        for lam, value, uncertainty in zip(args.lam_grid, avg, stderr):
            records.append(dict(block=block, n_score=n, n_folds=folds_used, candidate=lam,
                                brier=value, fold_se=uncertainty, selected=chosen, fallback=fallback))
    record("global", len(train), len(folds), global_lam, means, se, False)
    lam_map = np.full(len(labels), global_lam)
    for block in block_ids:
        b = int(block)
        fallback = len(losses[b]) < 2 or counts[b] < args.min_block_score
        if fallback:
            chosen, avg, stderr = global_lam, np.full(len(args.lam_grid), np.nan), np.full(len(args.lam_grid), np.nan)
        else:
            chosen, avg, stderr = choose_lambda(np.asarray(losses[b]), args.lam_grid, args.rule)
        lam_map[labels == b] = chosen
        record(b, counts[b], len(losses[b]), chosen, avg, stderr, fallback)
    return global_lam, lam_map, records


def evaluate_fold(data, train, test, labels, args, seed, groups=None):
    # The test parameter is used only to subset predictions after all fitting.
    global_lam, lam_map, curves = tune_lambdas(data, train, labels, args, seed, groups)
    fit, stop = fit_stop_split(train, seed_for(seed, "final_stop"))
    reference, rounds_ce = fit_predict(data, fit, stop, args, seed)
    soft = soft_reference(reference, data["R"], data["land"], args.beta)
    delta = legacy.gate_frozen(reference)
    preds, rounds = {"CE": reference[test]}, {"CE": rounds_ce}
    for method, lam in (("SR1", 1.), ("SRcv", global_lam), ("ASR", lam_map)):
        p, rounds[method] = fit_predict(data, fit, stop, args, seed, soft, delta, lam)
        preds[method] = p[test]
    diag = dict(global_lambda=global_lam, lambda_mean=float(lam_map[train].mean()),
                lambda_sd=float(lam_map[train].std()), lambda_zero_fraction=float(np.mean(lam_map[train] == 0)),
                delta_mean=float(delta[fit].mean()), soft_shift_mae=float(np.mean(abs(soft[fit]-reference[fit]))),
                n_train=len(train), n_fit=len(fit), n_stop=len(stop), n_test=len(test),
                fallback_blocks=sum(c["fallback"] for c in curves)/len(args.lam_grid))
    diag.update({"rounds_"+m: rounds[m] for m in METHODS})
    return preds, curves, diag


def prediction_metrics(data, idx, pred):
    y, truth, obs = data["y"][idx], data["p_true"][idx], data["p_obs"][idx]
    positive = y == 1
    npos, nneg = int(positive.sum()), int((~positive).sum())
    auc = ((rankdata(pred)[positive].sum()-npos*(npos+1)/2)/(npos*nneg)
           if npos and nneg else np.nan)
    return dict(auc=float(auc), brier=float(np.mean((pred-y)**2)),
                logloss=legacy.logloss_score(y, pred),
                recall=float(np.mean(pred[positive] >= .5)) if positive.any() else np.nan,
                mse_true=float(np.mean((pred-truth)**2)), bias_true=float(np.mean(pred-truth)),
                mse_obs=float(np.mean((pred-obs)**2)),
                expected_brier=float(np.mean((pred-obs)**2 + obs*(1-obs))))


def summary_stats(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    n = len(a)
    mean = float(a.mean()) if n else np.nan
    sd = float(a.std(ddof=1)) if n > 1 else np.nan
    mcse = sd/np.sqrt(n) if n > 1 else np.nan
    half = float(student_t.ppf(.975, n-1)*mcse) if n > 1 else np.nan
    return dict(n=n, mean=mean, sd=sd, mcse=mcse, mc_ci_low=mean-half, mc_ci_high=mean+half)


def summarize(rows):
    """Only independent repetitions within the same DGP/partition/protocol pool."""
    summary, paired = [], []
    conditions = sorted({(r["scenario"], r["coast"], r["partition"], r["validation"]) for r in rows})
    for condition in conditions:
        subset = [r for r in rows if (r["scenario"], r["coast"], r["partition"], r["validation"]) == condition]
        base = dict(zip(("scenario", "coast", "partition", "validation"), condition))
        for metric in sorted({r["metric"] for r in subset}):
            by_method = {m: {r["rep"]: r["value"] for r in subset if r["metric"] == metric and r["method"] == m}
                         for m in METHODS}
            for method in METHODS:
                summary.append(dict(base, method=method, metric=metric,
                                    **summary_stats(list(by_method[method].values()))))
            for comparator in ("CE", "SR1", "SRcv"):
                common = sorted(by_method["ASR"].keys() & by_method[comparator].keys())
                differences = [by_method["ASR"][r]-by_method[comparator][r] for r in common]
                paired.append(dict(base, comparison="ASR-"+comparator, metric=metric,
                                   **summary_stats(differences)))
    return summary, paired


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def unique_out_dir(path):
    """Never overwrite a previous run: if the directory exists, append -1, -2, ... instead."""
    if not path.exists():
        return path
    for index in count(1):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            print(f"{path} already exists, writing to {candidate} instead", flush=True)
            return candidate
    raise RuntimeError("unreachable")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gens", default="g_clean,g_mid,g_hard,g_nonstationary,g_detection")
    p.add_argument("--parts", default="p_voronoi,p_grid,p_resist")
    p.add_argument("--validation", default="random,spatial")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--grid", type=int, default=60)
    p.add_argument("--sample", type=float, default=.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nblocks", type=int, default=8)
    p.add_argument("--dgp-blocks", type=int, default=8)
    p.add_argument("--outer-folds", type=int, default=3)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--lam-grid", default="0,0.5,1,2,3.5")
    p.add_argument("--rule", choices=("min", "1se"), default="min")
    p.add_argument("--min-block-score", type=int, default=30)
    p.add_argument("--num-round", type=int, default=300)
    p.add_argument("--early-stop", type=int, default=30)
    p.add_argument("--beta", type=float, default=2.)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--coast", default="vertical,horizontal,archipelago,none")
    p.add_argument("--coast-pairing", choices=("shared", "independent"), default="shared")
    p.add_argument("--out", type=Path, default=Path(__file__).parent/"results"/time.strftime("coast_%Y%m%d_%H%M%S"))
    a = p.parse_args(argv)
    a.gens, a.parts, a.validation, a.coast = ([x.strip() for x in s.split(",")] for s in (a.gens, a.parts, a.validation, a.coast))
    a.lam_grid = sorted(set(float(v) for v in a.lam_grid.split(",")))
    for value, valid in ((a.gens, legacy.GENERATIONS), (a.parts, legacy.PARTITIONS),
                         (a.validation, ("random", "spatial")), (a.coast, legacy.COASTS)):
        if not set(value) <= set(valid) or len(value) != len(set(value)):
            p.error(f"Unknown or duplicate values: {value}")
    if not 0 < a.sample <= 1 or a.grid < 10 or min(a.reps, a.nblocks, a.dgp_blocks, a.num_round, a.early_stop, a.threads, a.min_block_score) < 1:
        p.error("Invalid grid, sample, repeat count or training parameters")
    if min(a.outer_folds, a.inner_folds) < 2:
        p.error("Both fold counts must be >= 2")
    if not a.lam_grid or not all(np.isfinite(a.lam_grid)) or a.lam_grid[0] != 0 or not np.isfinite(a.beta) or a.beta < 0:
        p.error("Lambda grid must be finite, nonnegative and include zero; beta must be finite and nonnegative")
    return a


def main(argv=None):
    args = parse_args(argv)
    args.out = unique_out_dir(args.out)
    args.out.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config["out"] = str(args.out.resolve())
    sources = [Path(__file__), Path(legacy.__file__)]
    manifest = dict(version=VERSION, config=config, python=platform.python_version(), numpy=np.__version__,
                    xgboost=xgb.__version__, scipy=__import__("scipy").__version__, gstools=legacy.gs.__version__,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    targets="p_obs: observed occurrence; p_true: latent suitability (diagnostic under detection/LGCP)",
                    uncertainty="MCSE across independent repetitions; outer folds pooled per repetition, not replicates")
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    rows, curves, diagnostics, failures, seeds = [], [], [], [], []
    start = time.monotonic()
    for scenario in args.gens:
        for coast in args.coast:
            for rep in range(args.reps):
                try:
                    data = generate_data(args, scenario, coast, rep)
                except Exception as exc:
                    failures.append(dict(scenario=scenario, coast=coast, rep=rep, partition="all", validation="all", error=repr(exc)))
                    write_csv(args.out/"failures.csv", failures)
                    print(f"FAIL {scenario}/{coast} rep={rep}: {exc}", flush=True)
                    continue
                seeds.append(dict(scenario=scenario, coast=coast, rep=rep,
                                  land_pixels=int(data["land"].sum()), **data["seeds"]))
                np.savez_compressed(args.out/f"truth_{scenario}_{coast}_{rep}.npz", p_true=data["p_true"], p_obs=data["p_obs"],
                                    detection=data["detection"], y=data["y"], sample=data["sample"],
                                    land=data["land"], dgp=data["dgp"])
                for partition in args.parts:
                    for protocol in args.validation:
                        context = dict(scenario=scenario, coast=coast, partition=partition, validation=protocol, rep=rep)
                        try:
                            labels = model_regions(data, args, scenario, coast, rep, partition)
                            # Fixed evaluation grid, unrelated to method or noise regions.
                            groups = grid_regions(args.grid, 4).ravel() if protocol == "spatial" else None
                            outer_seed = seed_for(args.seed, *((scenario, coast, rep) if args.coast_pairing == "independent" else (scenario, rep)), protocol, "outer")
                            folds = make_folds(data["sample"], args.outer_folds, outer_seed, groups)
                            preds = {m: np.full(len(data["y"]), np.nan) for m in METHODS}
                            fold_map = np.full(len(data["y"]), -1, int)
                            fold_curves, fold_diags = [], []
                            for fi, test in enumerate(folds):
                                train = np.setdiff1d(data["sample"], test)
                                fold_seed = seed_for(outer_seed, fi)
                                result, cr, diag = evaluate_fold(data, train, test, labels, args, fold_seed, groups)
                                fold_map[test] = fi
                                for m in METHODS:
                                    preds[m][test] = result[m]
                                fold_curves.extend(dict(context, fold=fi, **r) for r in cr)
                                fold_diags.append(dict(context, fold=fi, seed=fold_seed, **diag))
                            if not all(np.isfinite(preds[m][data["sample"]]).all() for m in METHODS):
                                raise ValueError("Missing or non-finite outer predictions")
                            for method in METHODS:
                                for metric, value in prediction_metrics(data, data["sample"], preds[method][data["sample"]]).items():
                                    rows.append(dict(context, method=method, metric=metric, value=value))
                            curves.extend(fold_curves)
                            diagnostics.extend(fold_diags)
                            np.savez_compressed(args.out/f"pred_{scenario}_{coast}_{rep}_{partition}_{protocol}.npz",
                                                sample=data["sample"], fold=fold_map, model_regions=labels, **preds)
                            print(f"OK {scenario}/{coast} rep={rep} {partition} {protocol} ({time.monotonic()-start:.1f}s)", flush=True)
                        except Exception as exc:
                            failures.append(dict(scenario=scenario, coast=coast, rep=rep, partition=partition, validation=protocol, error=repr(exc)))
                            print(f"FAIL {context}: {exc}", flush=True)
                        summary, paired = summarize(rows)
                        for filename, records in (("replicates.csv", rows), ("summary.csv", summary),
                                                  ("paired.csv", paired), ("lambda_curves.csv", curves),
                                                  ("diagnostics.csv", diagnostics), ("failures.csv", failures), ("seeds.csv", seeds)):
                            write_csv(args.out/filename, records)
    print(f"Results: {args.out.resolve()} | failed conditions: {len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
