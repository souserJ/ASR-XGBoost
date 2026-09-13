#!/usr/bin/env python3
"""Reproduce the simulation figure used in the paper.

The figure is a single-repetition illustration of the ADEMP study reported in
the manuscript (`asr_ademp.py`).  It is qualitative, and every number quoted in
the paper comes from the aggregated tables written by `asr_ademp.py`.

Panels
------
Top row, the true suitability field, the CE prediction and the ASR prediction.
Bottom row, the ASR - CE difference and a calibration curve for both methods
(the curve is drawn wider than a map panel).

Predictions are drawn at the outer held-out pixels, which is what the ADEMP
protocol produces.  The ocean is shaded grey and land carrying no prediction is
left white, so grey never stands for a missing value.

Two steps are needed
--------------------
Step 1, generate the underlying run (one scenario, one repetition, the Voronoi
partition, random folds, and all land pixels sampled so that the predicted
surface is complete).  This takes a few seconds on a laptop:

    python asr_ademp.py --gens g_mid --parts p_voronoi --validation random \\
        --reps 1 --sample 1.0 --out results/figure_pvoronoi

Random folds are used for the illustration because the pooled held-out map of
the spatial-fold protocol stitches together three differently trained models,
which leaves visible seams along the fold boundaries.  The reported study uses
both protocols; only the illustration is drawn with random folds.

Step 2, draw the figure:

    python make_sim_figure.py --run results/figure_pvoronoi

Repeated runs never overwrite each other.  When the output directory already
exists, ``asr_ademp.py`` appends ``-1``, ``-2`` and so on, and this script then
picks the most recent candidate that predicts every land pixel, so a stale or
partially sampled run is not read by accident.

Defaults reproduce the panel arrangement of the manuscript.  Use
``--partition p_grid`` only if the run was made with the regular grid
partition, since the per-block panel then shows square blocks, which is correct
but less informative as an illustration.

Requires numpy and matplotlib.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TITLE_FS = 13.0
LABEL_FS = 12.0
TICK_FS = 11.0
LEGEND_FS = 11.0
OCEAN_RGB = (0.78, 0.78, 0.78)
RISK_CMAP = plt.get_cmap("viridis").copy()
RISK_CMAP.set_bad((0.0, 0.0, 0.0, 0.0))
DIFF_CMAP = plt.get_cmap("coolwarm").copy()
DIFF_CMAP.set_bad((0.0, 0.0, 0.0, 0.0))


def read_block_lambda(path, scenario, partition, validation, rep, fold):
    """Per-block selected strength for one scenario, partition, protocol, repetition and fold.

    The first row recorded for a block carries its selected value; the candidate
    rows of that block share the same ``selected`` entry.
    """
    chosen = {}
    with open(path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (row["scenario"], row["partition"], row["validation"],
                    int(row["rep"]), int(row["fold"])) != (scenario, partition, validation, rep, fold):
                continue
            if row["block"] == "global" or row["block"] in chosen:
                continue
            try:
                chosen[int(row["block"])] = float(row["selected"])
            except ValueError:
                continue
    return chosen


def run_files(directory, scenario, rep, partition, validation):
    """Paths and presence of the three inputs needed by the figure."""
    files = (directory / f"truth_{scenario}_{rep}.npz",
             directory / f"pred_{scenario}_{rep}_{partition}_{validation}.npz",
             directory / "lambda_curves.csv")
    return files, all(f.exists() for f in files)


def land_coverage(directory, scenario, rep, partition, validation, n):
    """Fraction of land pixels carrying an outer held-out prediction."""
    truth = np.load(directory / f"truth_{scenario}_{rep}.npz")
    pred = np.load(directory / f"pred_{scenario}_{rep}_{partition}_{validation}.npz")
    land = truth["land"].reshape(n, n).astype(bool).ravel()
    return float((~np.isnan(pred["CE"]))[land].mean())


def resolve_run(base, scenario, rep, partition, validation, n, limit=50):
    """Pick the run directory to read.

    `asr_ademp.py` never overwrites an existing output directory, so a repeated
    command writes to `<name>-1`, `<name>-2` and so on.  Existing candidates are
    searched, preferring one whose predictions cover all land pixels and, among
    those, the most recent.
    """
    candidates = [base] + [base.with_name(f"{base.name}-{i}") for i in range(1, limit + 1)]
    usable = [d for d in candidates if run_files(d, scenario, rep, partition, validation)[1]]
    if not usable:
        return None
    usable.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for directory in usable:
        if land_coverage(directory, scenario, rep, partition, validation, n) >= 0.999:
            return directory
    return usable[0]


def calibration(pred, y, bins=10, min_count=5):
    """Observed frequency against mean predicted probability, over equal-width bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (pred >= lo) & (pred < hi)
        if sel.sum() >= min_count:
            xs.append(float(pred[sel].mean()))
            ys.append(float(y[sel].mean()))
    return np.asarray(xs), np.asarray(ys)


def draw_map(figure, ax, values, land, n, cmap, vmin=None, vmax=None, title="",
             colorbar=True, colorbar_label=None):
    """Draw one raster panel over a land/ocean background.

    The ocean is filled grey and land without a prediction stays white, because
    the data layer is transparent wherever its value is missing.
    """
    background = np.ones((n, n, 3))
    background[~land] = OCEAN_RGB
    ax.imshow(background, interpolation="nearest")
    data = np.where(land, np.asarray(values, float).reshape(n, n), np.nan)
    image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        bar = figure.colorbar(image, ax=ax, fraction=0.040, pad=0.015)
        bar.ax.tick_params(labelsize=TICK_FS)
        if colorbar_label:
            bar.set_label(colorbar_label, fontsize=LABEL_FS)
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("results/figure_pvoronoi"),
                        help="ADEMP output directory holding truth_*.npz, pred_*.npz and lambda_curves.csv")
    parser.add_argument("--scenario", default="g_mid")
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--partition", default="p_voronoi")
    parser.add_argument("--validation", default="random")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--grid", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("figures/ademp_sim_demo.png"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--allow-partial", action="store_true",
                        help="draw the map even if the run predicts only part of the land surface")
    args = parser.parse_args(argv)

    n = args.grid
    base = args.run
    args.run = resolve_run(base, args.scenario, args.rep, args.partition, args.validation, n)
    if args.run is None:
        sys.exit(f"No run with the required files found at {base} or {base}-1, {base}-2, ...\n"
                 "Generate one first, for example:\n"
                 f"  python asr_ademp.py --gens {args.scenario} --parts {args.partition} "
                 f"--validation {args.validation} --reps {args.rep + 1} --sample 1.0 --out {base}")
    if args.run != base:
        print(f"using {args.run} (from {base})")
    truth_path = args.run / f"truth_{args.scenario}_{args.rep}.npz"
    pred_path = args.run / f"pred_{args.scenario}_{args.rep}_{args.partition}_{args.validation}.npz"
    curves_path = args.run / "lambda_curves.csv"
    truth = np.load(truth_path)
    pred = np.load(pred_path)
    lam = read_block_lambda(curves_path, args.scenario, args.partition, args.validation,
                            args.rep, args.fold)

    land = truth["land"].reshape(n, n).astype(bool)
    regions = pred["model_regions"].reshape(n, n)
    held = ~np.isnan(pred["CE"])
    if not held.any():
        sys.exit("No held-out predictions in " + str(pred_path))
    coverage = float(held[land.ravel()].mean())
    if coverage < 0.999 and not args.allow_partial:
        sys.exit(f"The run in {args.run} predicts only {coverage:.0%} of the land pixels, so most of the map would be blank.\n"
                 "Regenerate the run with --sample 1.0, for example:\n"
                 f"  python asr_ademp.py --gens {args.scenario} --parts {args.partition} "
                 f"--validation {args.validation} --reps 1 --sample 1.0 --out {args.run}\n"
                 "Pass --allow-partial to draw the sparse map anyway.")

    ce, asr = pred["CE"], pred["ASR"]
    difference = asr - ce
    y = truth["y"][held]

    lam_map = np.full(n * n, np.nan)
    for block, value in lam.items():
        lam_map[regions.ravel() == block] = value

    figure = plt.figure(figsize=(11.5, 7.8))
    # The top row holds three maps; only the last carries a colourbar, so the
    # maps can sit close together.  The bottom row is positioned explicitly
    # after the layout is fixed, so that its map matches the top-row maps and
    # the calibration panel comes out a little wider; the gridspec slots below
    # only provide the axes.
    outer = figure.add_gridspec(2, 1, hspace=0.21, height_ratios=(1.0, 1.0))
    top = outer[0].subgridspec(1, 3, wspace=0.04)
    bottom = outer[1].subgridspec(1, 6, wspace=1.50)
    axes = [[figure.add_subplot(top[0]), figure.add_subplot(top[1]), figure.add_subplot(top[2])],
            [figure.add_subplot(bottom[1:3]), figure.add_subplot(bottom[3:5])]]

    draw_map(figure, axes[0][0], truth["p_true"], land, n, RISK_CMAP,
             0.05, 0.95, "True suitability", colorbar=False)
    draw_map(figure, axes[0][1], ce, land, n, RISK_CMAP, 0.05, 0.95, "CE prediction",
             colorbar=False)
    draw_map(figure, axes[0][2], asr, land, n, RISK_CMAP, 0.05, 0.95, "ASR prediction",
             colorbar_label="Predicted risk")

    ax = axes[1][1]
    for values, label, colour in ((ce, "CE", "#1f77b4"), (asr, "ASR", "#d62728")):
        xs, ys = calibration(values[held], y)
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.3, color=colour, label=label)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="0.5", label="Perfect")
    ax.set_xlabel("Mean predicted probability", fontsize=LABEL_FS)
    ax.set_ylabel("Observed frequency", fontsize=LABEL_FS)
    ax.set_title("Calibration", fontsize=TITLE_FS)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="upper left")
    ax.tick_params(labelsize=TICK_FS)

    limit = float(np.nanmax(np.abs(difference[held]))) or 0.1
    diff_image = draw_map(figure, axes[1][0], difference, land, n, DIFF_CMAP,
                          -limit, limit, r"ASR $-$ CE")

    figure.subplots_adjust(left=0.05, right=0.97, top=0.955, bottom=0.075)

    # The bottom-left map has to match the top-row maps, and the calibration
    # panel is stretched a little wider so its curves are easier to read.  The
    # gridspec slot for the map comes out smaller than a top map because the
    # colourbar is carved out of it and `imshow` then shrinks the box to a
    # square, so the three bottom panels are placed here by hand instead.
    ref = axes[0][0].get_position()
    side_w, side_h = ref.width, ref.height
    slot = axes[1][1].get_position()
    bar = diff_image.colorbar.ax
    bar_slot = bar.get_position()
    bar_gap = 0.006
    map_x0 = bar_slot.x0 - bar_gap - side_w
    axes[1][0].set_position([map_x0, slot.y0, side_w, side_h])
    bar.set_position([map_x0 + side_w + bar_gap, slot.y0, bar_slot.width, side_h])
    axes[1][1].set_position([slot.x0, slot.y0, slot.width * 1.2, side_h])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
