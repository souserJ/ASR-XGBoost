#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR-XGBoost 最小演示版 (Minimal Demo) —— 多情形模拟研究
========================================================
自适应空间正则化 XGBoost 核心思想演示 + 分块自适应 λ 选择 + 模拟研究聚合。

本脚本为"小而真"的公开演示代码（全部合成数据，无任何真实疾病数据）。

数据生成（每环都有标准出处）:
  - 环境变量场: GSTools 的 Matern 协方差高斯随机场（地统计标准做法）
  - 阻力面:     地形山脊 + 噪声（Zeller et al. 2012 阻力面构建惯例）
  - 适宜性:     协变量响应函数（virtual species 模拟惯例,
                Hirzel et al. 2001; Meynard & Kaplan 2012; Leroy et al. 2016）
  - 观测标签:   ① 伯努利出现采样 + 分块检测/报告异质性噪声（漏报的代理,
                呼应 presence-only 数据局限）;
                ② 点过程模式: 非齐次泊松/LGCP —— 像元事件数 ~ Poisson(λ(s)),
                出现 = 至少 1 个事件, λ(s) = ν·e^{0.6Z(s)}·p_true。

方法（与论文口径一致）:
  - ASR 损失:      L = L_CE + λ_i·δ(p̂)·(p − ṗ)² ,  冻结门控 δ(p̂) = 1 − 2|p̂ − 0.5|
  - 软标签:        ṗ_i = Σ_j w_ij·p̂_j / Σ_j w_ij ,  w_ij = exp(−β·(R_i+R_j)/2)
  - 分块自适应 λ:  每块在块内 k 折交叉验证上按 Brier 均值网格搜索 λ*，
                    λ 候选范围 0~3.5（默认 0,0.5,...,3.5；λ=0 即无正则化等价 CE，
                    与论文敏感性分析一致）；
                    小区域按质心最近邻合并（对应行政分块 strategy B 小国合并）
  - 划分:          7:2:1 训练/验证/测试（与论文协议一致）；测试集不参与任何选择
  - 评估:          测试集 AUC/Brier/LogLoss + 空间指标（Moran's I/ContED/Iso ratio，
                   与论文口径一致） + final 式空间 CV（3×3 网格块折，
                   块不跨折；每折重训 CE 并重算软标签/门控，无泄漏；
                   ASR 在 CV 内用训练侧预选 λ*，与测试集口径一致）

模拟研究（--study）:
  默认三档难度生成（g_clean 简单 / g_mid 中等 / g_hard 困难，--gens 可换），
  多种分块方式 × 多个种子重复，每次模拟计算精度，
  最后求全部模拟的平均精度（mean±std），同时报告测试集与空间 CV 两套指标；
  逐模拟缓存（study_cache.pkl，key 含全部参数；--no-cache 强制重跑）。

运行:
  python asr_demo.py                          # 单次详细演示（300×300, 16 块）
  python asr_demo.py --study                  # 模拟研究：三档难度 × 3 分块 × 2 种子（逐模拟缓存）
  python asr_demo.py --blocks regions.npy     # 用 draw_regions.py 自己圈的区域
依赖:  numpy, xgboost, matplotlib, gstools  (见 requirements.txt)

引用:  使用本代码请引用论文（作者待补）。
"""
import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
import gstools as gs

# ==========================================================================
# 1. 模拟数据生成
# ==========================================================================
def matern_field(n, rng, len_scale, nu=1.5):
    """GSTools Matern 协方差高斯随机场（标准化），地统计标准生成。"""
    model = gs.Matern(dim=2, len_scale=len_scale, nu=nu, var=1.0)
    srf = gs.SRF(model, seed=int(rng.integers(0, 2 ** 31 - 1)))
    yy, xx = np.mgrid[0:n, 0:n]
    f = np.asarray(srf((xx, yy))).reshape(n, n)   # GSTools 传 meshgrid 返回扁平场，统一 reshape
    return (f - f.mean()) / f.std()


def response_logit(X1, X2, X3):
    """适宜性 logit：主效应 + 交互项（产生非纯平滑的空间结构）。"""
    return (1.0 * X1 + 0.8 * X2 + 0.5 * X3
            - 0.6 * X1 * X2 + 0.4 * X2 * X3 - 0.2)


def gen_landscape(n, rng, cfg=None):
    """多尺度环境场(GSTools) → 阻力面(Zeller 2012 惯例) → 非线性适宜性 → 曲折海洋。
    返回 (X1, X2, X3, R, p_true, ocean)；ocean 为 None 表示无海洋。"""
    cfg = cfg or dict(len_scale=18.0, ridge=3.0)

    def multiscale(len_scale):
        """嵌套双尺度场：长程趋势 + 局地变异（更接近真实气候/环境场）。"""
        f = (0.65 * matern_field(n, rng, len_scale)
             + 0.35 * matern_field(n, rng, max(2.5, len_scale * 0.25)))
        return (f - f.mean()) / f.std()

    X1 = multiscale(cfg['len_scale'])
    X2 = multiscale(cfg['len_scale'] * 0.7)
    X3 = multiscale(cfg['len_scale'] * 0.5)          # 第三协变量（进入训练特征）
    yy, xx = np.mgrid[0:n, 0:n]
    d = (xx + yy - n) / (n * 0.45)                 # 斜向山脊
    ridge = np.exp(-(d ** 2) / 2)
    R = 1.0 + cfg['ridge'] * ridge + 0.3 * rng.normal(size=(n, n))
    R = np.clip(R, 0.5, None)
    ocean = None
    if cfg.get('ocean'):
        # 曲折海岸线：多条正弦叠加的中心线 + 宽带（真实海岸形态）
        center = (n / 2.0
                  + cfg.get('ocean_amp', 0.10) * n * np.sin(2 * np.pi * yy / (n * 0.6))
                  + cfg.get('ocean_amp2', 0.05) * n * np.sin(2 * np.pi * yy / (n * 0.23) + 1.3))
        ocean = np.abs(xx - center) < n * cfg.get('ocean_width', 0.05)
        R[ocean] = 100.0
    ls = cfg.get('logit_scale', 1.0)
    p_true = 1.0 / (1.0 + np.exp(-ls * response_logit(X1, X2, X3)))
    return X1, X2, X3, R, p_true, ocean


def gen_labels(p_true, labels, rng, cfg):
    """观测标签生成。

    mode='bernoulli': 伯努利出现采样 + 分块检测/报告异质性噪声（漏报代理）
    mode='lgcp'     : 非齐次泊松/LGCP —— 像元事件数~Poisson(λ(s))，
                      出现=至少1个事件, λ(s)=ν·e^{0.6Z(s)}·p_true (Z 为随机效应场)
    """
    if cfg.get('mode') == 'lgcp':
        z = matern_field(p_true.shape[0], rng, cfg.get('len_scale', 18.0))
        lam = cfg.get('nu', 0.8) * np.exp(0.6 * z.ravel()) * p_true.ravel()
        p_occ = 1.0 - np.exp(-lam)
        return (rng.uniform(size=len(p_occ)) < p_occ).astype(float)
    sigma = cfg.get('noise_base', 0.08) + cfg.get('noise_diff', 0.10) * (labels.ravel() % 2)
    # 注：labels=-1 为海洋像元（numpy 取模得 1，按奇数块噪声），不进数据集，不影响结果
    p_eff = np.clip(p_true.ravel() + rng.normal(0.0, sigma), 0.001, 0.999)
    return (rng.uniform(size=len(p_eff)) < p_eff).astype(float)


# ==========================================================================
# 2. 分块方式
# ==========================================================================
def default_blocks(n, rng, n_blocks=16, mask=None, iters=6):
    """Lloyd 松弛 Voronoi"国家"块（k-means 迭代）：
    区域紧凑、面积相近、长宽比合理（避免细长三角条），边界不规则。
    mask 给定则只在 mask 内分块（用于按大陆分块），mask 外为 -1。"""
    yy, xx = np.mgrid[0:n, 0:n]
    pts = np.stack([xx.ravel(), yy.ravel()], 1)
    work = pts if mask is None else pts[mask.ravel()]
    pick = rng.choice(len(work), size=min(n_blocks, len(work)), replace=False)
    seeds = work[pick].astype(float)
    for _ in range(iters):                       # Lloyd 松弛迭代
        d = np.array([np.hypot(work[:, 0] - s[0], work[:, 1] - s[1])
                      for s in seeds]).T
        lab = d.argmin(1)
        for j in range(len(seeds)):
            sel = lab == j
            if sel.any():
                seeds[j] = work[sel].mean(axis=0)
    d = np.array([np.hypot(work[:, 0] - s[0], work[:, 1] - s[1])
                  for s in seeds]).T
    lab = d.argmin(1)
    if mask is None:
        return lab.reshape(n, n)
    full = np.full(n * n, -1, dtype=int)
    full[mask.ravel()] = lab
    return full.reshape(n, n)


def grid_blocks(n, k=3, mask=None):
    """规则网格块（k×k），对应 5°×5° 网格分块惯例。"""
    yy, xx = np.mgrid[0:n, 0:n]
    lab = (yy // (n // k)) * k + (xx // (n // k))
    if mask is not None:
        lab = np.where(mask, lab, -1)
    return lab


def resist_blocks(R, k=3, mask=None):
    """生态阻力分区：按阻力面分位数分成 k 块。"""
    q = np.quantile(R, np.linspace(0, 1, k + 1)[1:-1])
    lab = np.digitize(R, q)
    if mask is not None:
        lab = np.where(mask, lab, -1)
    return lab


def make_partition(kind, n, rng, R=None, nblocks=8, mask=None):
    if kind == 'p_voronoi':
        return default_blocks(n, rng, nblocks, mask)
    if kind == 'p_grid':
        return grid_blocks(n, 3, mask)
    if kind == 'p_resist':
        return resist_blocks(R, 3, mask)
    raise ValueError(f'未知分块方式: {kind}')


def split_continents(ocean):
    """把陆地按海洋左右切成两块大陆（按行找海洋段；无海行归左侧）。"""
    n = ocean.shape[0]
    left = np.zeros_like(ocean)
    right = np.zeros_like(ocean)
    for i in range(n):
        o = np.flatnonzero(ocean[i])
        if len(o) == 0:
            left[i, :] = True
        else:
            left[i, :o[0]] = True
            right[i, o[-1] + 1:] = True
    return left, right


def split_blocks_at_barriers(labels, barrier):
    """把跨越屏障（海洋）的块按 4-连通拆开，保证任何块都不跨水。"""
    from scipy.ndimage import label as cc_label
    lab = labels.copy()
    lab[barrier] = -1
    max_id = int(lab.max())
    for b in range(int(lab.max()) + 1):
        mask = lab == b
        comp, n_comp = cc_label(mask)
        if n_comp > 1:
            lab[mask] = max_id + comp[mask]
            max_id += n_comp
    return lab


def merge_small_blocks(labels, min_pixels, barrier=None):
    """小区域合并（对应"小国最近邻合并"）：面积 < min_pixels 的块
    按质心最近邻合并到邻近块；-1 为背景（海洋），不参与合并；
    barrier 给定（海洋）则禁止跨屏障合并（左右大陆各自合并）。"""
    lab = labels.copy()
    land = lab >= 0
    side = None
    if barrier is not None:
        _, right = split_continents(barrier)
        side = np.where(right, 1, 0)
    k = int(lab.max()) + 1
    while True:
        sizes = np.bincount(lab[land], minlength=k)
        small = [b for b in range(k) if 0 < sizes[b] < min_pixels]
        if not small:
            break
        for b in small:
            if side is not None:
                bside = int(np.bincount(side[lab == b].astype(int), minlength=2).argmax())
            centroids = {}
            for c in range(k):
                if c != b:
                    pos = np.argwhere(lab == c)
                    if len(pos):
                        if side is not None:
                            cside = int(np.bincount(side[lab == c].astype(int), minlength=2).argmax())
                            if cside != bside:
                                continue
                        centroids[c] = pos.mean(axis=0)
            if not centroids:
                break
            cb = np.mean(np.argwhere(lab == b), axis=0)
            nn = min(centroids, key=lambda c: float(np.hypot(*(centroids[c] - cb))))
            lab[lab == b] = nn
        uniq = np.unique(lab[land])
        remap = {v: i for i, v in enumerate(uniq)}
        newlab = np.full_like(lab, -1)
        newlab[land] = np.vectorize(remap.get)(lab[land])
        lab = newlab
        k = len(uniq)
    return lab


# ==========================================================================
# 3. 评估指标（纯 numpy）
# ==========================================================================
def auc_score(y, p):
    """Mann-Whitney U 统计量形式的 AUC；单类样本返回 0.5（无判别信息）。"""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(p, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def brier_score(y, p):
    return float(np.mean((p - y) ** 2))


def logloss_score(y, p, eps=1e-7):
    """二分类交叉熵（LogLoss / CE）。"""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def recall_score(y, p, thr=0.5):
    """召回率（阈值 0.5）。"""
    pred = (p >= thr).astype(float)
    tp = float(((pred == 1) & (y == 1)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def spatial_metrics(p, land_mask=None):
    """论文口径的三个空间平滑度指标（与 CHIKV 论文 Spatial Smoothness Metrics 一致）：
    Moran's I（4 邻域自相关）、ContED（4 邻域平均边缘强度）、
    Iso ratio（|p_i − 8 邻域均值| > 0.05 的孤立像元比例）。
    land_mask 可选：只统计陆地像元（邻对两端均须在陆地内）。"""
    p = np.asarray(p, dtype=float)
    n = p.shape[0]
    m = np.ones_like(p, dtype=bool) if land_mask is None else land_mask.astype(bool)
    # 4 邻域有效对（水平右邻 + 垂直下邻，避免重复计数；两端均在陆地）
    h_ok = m[:, :-1] & m[:, 1:]
    v_ok = m[:-1, :] & m[1:, :]
    pi_h, pj_h = p[:, :-1][h_ok], p[:, 1:][h_ok]
    pi_v, pj_v = p[:-1, :][v_ok], p[1:, :][v_ok]
    pi = np.concatenate([pi_h, pi_v])
    pj = np.concatenate([pj_h, pj_v])
    if len(pi) == 0:
        return 0.0, 0.0, 0.0
    # ContED：相邻对绝对差均值
    conted = float(np.mean(np.abs(pi - pj)))
    # Moran's I：z 相对陆地均值
    p_land = p[m]
    z = p - p_land.mean()
    num = float(np.sum(z[:, :-1][h_ok] * z[:, 1:][h_ok])
                + np.sum(z[:-1, :][v_ok] * z[1:, :][v_ok]))
    denom = float(np.sum(z[m] ** 2))
    s0 = float(h_ok.sum() + v_ok.sum())
    moran = (int(m.sum()) / s0) * (num / denom) if (denom > 0 and s0 > 0) else 0.0
    # Iso ratio：|p_i − 8 邻域均值| > 0.05 的陆地像元占比
    # 邻域用切片实现（边界外不算邻域，无 np.roll 卷绕伪影）；只统计陆地邻域
    nb_sum = np.zeros_like(p)
    nb_cnt = np.zeros_like(p, dtype=float)
    pm = np.where(m, p, 0.0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            s_r0, s_r1 = max(0, dy), min(n, n + dy)
            d_r0, d_r1 = max(0, -dy), min(n, n - dy)
            s_c0, s_c1 = max(0, dx), min(n, n + dx)
            d_c0, d_c1 = max(0, -dx), min(n, n - dx)
            nb_sum[d_r0:d_r1, d_c0:d_c1] += pm[s_r0:s_r1, s_c0:s_c1]
            nb_cnt[d_r0:d_r1, d_c0:d_c1] += m[s_r0:s_r1, s_c0:s_c1]
    nb_mean = np.divide(nb_sum, nb_cnt, out=np.zeros_like(p), where=nb_cnt > 0)
    iso = float(np.mean(np.abs(p[m] - nb_mean[m]) > 0.05))
    return moran, conted, iso


# ==========================================================================
# 4. 软标签、冻结门控、ASR 目标
# ==========================================================================
def soft_labels_8nn(p_grid, R, beta, connectivity=True):
    wsum = np.zeros_like(p_grid)
    num = np.zeros_like(p_grid)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            pj = np.roll(p_grid, (dy, dx), axis=(0, 1))
            rj = np.roll(R, (dy, dx), axis=(0, 1))
            w = np.ones_like(p_grid) if not connectivity \
                else np.exp(-beta * (R + rj) / 2.0)
            num += w * pj
            wsum += w
    # 深海像元 8 邻域全为高阻力 → wsum≈0，避免 0/0 NaN
    return np.divide(num, wsum, out=np.zeros_like(num), where=wsum > 0)


def gate_frozen(p_ref):
    """冻结门控 δ(p̂) = 1 − 2|p̂ − 0.5|。"""
    return np.clip(1.0 - 2.0 * np.abs(p_ref - 0.5), 0.0, 1.0)


def make_asr_objective(delta_frozen, soft, lam):
    """ASR 目标（margin 空间梯度/海森）。λ 可为标量或逐样本数组。"""
    lam = np.asarray(lam, dtype=float)

    def obj(preds, dtrain):
        y = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        l = lam if (lam.ndim == 0 or lam.size == 1) else lam
        d = delta_frozen
        s = soft
        pp1 = p * (1.0 - p)
        g = (p - y) + 2.0 * l * d * (p - s) * pp1
        h = pp1 + 2.0 * l * d * (pp1 ** 2) \
            + 2.0 * l * d * (p - s) * pp1 * (1.0 - 2.0 * p)
        return g, np.clip(h, 1e-6, None)
    return obj


def predict_prob(bst, dmatrix):
    """自定义目标模型统一取 margin 再 sigmoid（跨 xgboost 版本一致）。"""
    return 1.0 / (1.0 + np.exp(-bst.predict(dmatrix, output_margin=True)))


def select_lambda_per_block(X, y, tr_idx, lab_tr, soft, delta, lam_grid,
                            params, num_round, nfold=5, rule='1se'):
    """分块自适应 λ 选择：块内 nfold 折 CV，按平均 Brier 网格搜索 λ*。

    rule='min': 取 CV Brier 最小的 λ（平曲线上可能随机落在端点）
    rule='1se' : 取"最优 λ 的一个标准误内"的最小 λ（glmnet 1-SE 惯例）——
                 平曲线上自动选最小充分正则化，避免随机落点与过平滑。
    """
    lam_star = {}
    rows = []
    curves = {}   # 每块的 (λ, CV Brier, CV AUC, CV SE) 曲线，展示选择过程
    rng_cv = np.random.default_rng(0)
    for b in np.unique(lab_tr):
        trb = tr_idx[lab_tr == b]
        if len(trb) < 30:
            continue
        perm = rng_cv.permutation(len(trb))
        folds = np.array_split(perm, nfold)
        curve = []
        for lam in lam_grid:
            briers, aucs = [], []
            for fi, f in enumerate(folds):
                va_f = trb[f]
                tr_f = trb[np.concatenate([x for i, x in enumerate(folds) if i != fi])]
                if len(va_f) == 0:
                    continue
                obj = make_asr_objective(delta[tr_f], soft[tr_f], lam)
                bst = xgb.train(params, xgb.DMatrix(X[tr_f], label=y[tr_f]),
                                num_boost_round=num_round, obj=obj,
                                verbose_eval=False)
                pv = predict_prob(bst, xgb.DMatrix(X[va_f]))
                briers.append(brier_score(y[va_f], pv))
                aucs.append(auc_score(y[va_f], pv))
            curve.append((lam, float(np.mean(briers)), float(np.mean(aucs)),
                          float(np.std(briers) / np.sqrt(nfold))))
        curve.sort(key=lambda t: t[1])          # 按 CV Brier 升序
        if rule == '1se':
            thr = curve[0][1] + curve[0][3]     # 最优 + 1 SE
            chosen = min((t for t in curve if t[1] <= thr), key=lambda t: t[0])
        else:
            chosen = curve[0]
        lam_star[b] = chosen[0]
        rows.append((b, len(trb), chosen[0], chosen[1], chosen[2]))
        curves[b] = sorted(curve, key=lambda t: t[0])   # 打印时按 λ 升序
    return lam_star, rows, curves


# ==========================================================================
# 5. 单次模拟
# ==========================================================================
GENERATIONS = {
    'g_clean':   dict(name='伯努利·低噪声', mode='bernoulli',
                      noise_base=0.02, noise_diff=0.22, ridge=3.0, len_scale=18.0,
                      frag=1.5, ocean=True),
    'g_mid':     dict(name='伯努利·中等噪声', mode='bernoulli',
                      noise_base=0.22, noise_diff=0.02, ridge=3.0, len_scale=18.0,
                      frag=0.0, ocean=True),
    'g_noisy':   dict(name='伯努利·高漏报异质性', mode='bernoulli',
                      noise_base=0.05, noise_diff=0.30, ridge=3.0, len_scale=18.0,
                      frag=1.5, ocean=True),
    'g_noisy2':  dict(name='伯努利·全块高噪声', mode='bernoulli',
                      noise_base=0.35, noise_diff=0.02, ridge=3.0, len_scale=18.0,
                      frag=0.0, ocean=True),
    'g_hard':    dict(name='伯努利·弱信号困难', mode='bernoulli',
                      noise_base=0.35, noise_diff=0.02, ridge=3.0, len_scale=18.0,
                      frag=0.0, ocean=True, logit_scale=0.45),
    'g_barrier': dict(name='伯努利·强地形屏障', mode='bernoulli',
                      noise_base=0.08, noise_diff=0.10, ridge=6.0, len_scale=18.0,
                      frag=0.0, ocean=False),
    'g_lgcp':    dict(name='点过程·LGCP', mode='lgcp',
                      nu=0.8, ridge=3.0, len_scale=18.0, frag=0.0, ocean=False),
}
PARTITIONS = {
    'p_voronoi': 'Lloyd 松弛 Voronoi 国家块（k-means 迭代，紧凑类国家形状）',
    'p_grid':    '规则网格块（3×3）',
    'p_resist':  '生态阻力分区（阻力分位）',
}


def simulate_once(gname, pname, seed, n, nblocks, min_pixels, lam_grid,
                  beta, params, num_round, cv_rounds, cv_folds,
                  custom_blocks=None, verbose=False, split=(0.7, 0.2, 0.1),
                  sample_frac=0.4):
    """一次完整模拟：生成 → 分块 → CE → 软标签/门控 → λ*(块内CV) →
    ASR全局/分块 → 测试集评估。返回 metrics dict（不打印）。"""
    rng = np.random.default_rng(seed)
    gcfg = GENERATIONS[gname]
    X1, X2, X3, R, p_true, ocean = gen_landscape(n, rng, gcfg)

    if custom_blocks is not None:
        labels = custom_blocks.copy()
        if ocean is not None:
            labels[ocean] = -1           # 海洋不隶属任何块
    elif ocean is not None:
        # 左右两块大陆分别分块，海洋不参与；块不跨水
        left, right = split_continents(ocean)
        n_left = max(1, round(nblocks * left.sum() / (left.sum() + right.sum())))
        n_right = max(1, nblocks - n_left)
        labels = np.full((n, n), -1, dtype=int)
        off = 0
        for cont, nb in ((left, n_left), (right, n_right)):
            if cont.sum() < min_pixels:
                continue
            sub = make_partition(pname, n, rng, R, nb, mask=cont)
            sel = cont & (sub >= 0)
            if sel.any():
                labels[sel] = sub[sel] + off
                off += int(sub[sel].max()) + 1
    else:
        labels = make_partition(pname, n, rng, R, nblocks)
    if ocean is not None:
        labels = split_blocks_at_barriers(labels, ocean)
    # min_pixels 语义 = 每块最少【数据集】像元；采样后换算成全栅格阈值
    full_min = int(min_pixels / sample_frac) if sample_frac < 1.0 else min_pixels
    labels = merge_small_blocks(labels, max(1, full_min), barrier=ocean)
    if ocean is not None:
        labels = split_blocks_at_barriers(labels, ocean)   # 合并后再保险一次
    n_blk = int(labels.max()) + 1

    # 块间异质性：偶数块叠加"可学习的锐利边界"（由协变量 X1、X2 阈值驱动的
    # 交叉边界，模型学得到），且这些块标签噪声很低（去噪收益≈0）——
    # 强平滑会抹掉真实边界 → 最优 λ 显著偏小；奇数块平滑渐变+高噪声 → 强 λ 去噪有利。
    if gcfg.get('frag', 0.0) > 0:
        step = (X1 > 0.0).astype(float) + (X2 > 0.0).astype(float) - 1.0
        frag_mask = (labels % 2 == 0).astype(float)
        ls = gcfg.get('logit_scale', 1.0)
        logit = ls * (response_logit(X1, X2, X3) + gcfg['frag'] * step * frag_mask)
        p_true = 1.0 / (1.0 + np.exp(-logit))

    y = gen_labels(p_true, labels, rng, gcfg)

    # 按 sample_frac（默认 40%）采样像元作为数据集（模拟观测稀疏），再在数据集内 7:2:1 划分
    land_idx = np.flatnonzero(np.ones(n * n, bool) if ocean is None
                              else ~ocean.ravel())
    land_mask = np.ones((n, n), dtype=bool) if ocean is None else ~ocean
    n_ds = int(sample_frac * len(land_idx))
    ds = land_idx if sample_frac >= 1.0 else rng.choice(land_idx, size=n_ds, replace=False)
    idx = rng.permutation(ds)
    s = np.asarray(split, dtype=float)
    s = s / s.sum()                       # 比例份数 → 归一化（如 7:2:1）
    n_tr = int(s[0] * len(idx))
    n_va = int(s[1] * len(idx))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    trva = np.concatenate([tr, va])
    Xall = np.column_stack([X1.ravel(), X2.ravel(), X3.ravel()])

    dtrva = xgb.DMatrix(Xall[trva], label=y[trva])
    dgrid = xgb.DMatrix(Xall)

    # CE 基线
    if verbose:
        print(f'  >> 训练 CE 基线（{num_round} 轮）…', flush=True)
    bst_ce = xgb.train(params, dtrva, num_boost_round=num_round, verbose_eval=False)
    p_ce = bst_ce.predict(dgrid)

    # 软标签（连通性加权）与冻结门控
    p_ce_grid = p_ce.reshape(n, n)
    soft = soft_labels_8nn(p_ce_grid, R, beta, connectivity=True).ravel()
    delta = gate_frozen(p_ce)

    # ASR 全局 λ
    obj_g = make_asr_objective(delta[trva], soft[trva], 1.0)
    bst_g = xgb.train(params, dtrva, num_boost_round=num_round,
                      obj=obj_g, verbose_eval=False)
    p_asr_g = predict_prob(bst_g, dgrid)

    # 分块自适应 λ（块内 CV）
    lab_tr = labels.ravel()[tr]
    if verbose:
        print(f'  >> 分块 λ 选择（{n_blk} 块 × {len(lam_grid)} 个 λ × '
              f'{cv_folds} 折 CV）…', flush=True)
    lam_star, cv_rows, cv_curves = select_lambda_per_block(
        Xall, y, tr, lab_tr, soft, delta, lam_grid, params, cv_rounds, cv_folds)
    lam_map = np.full(n * n, 1.0)
    for b, l in lam_star.items():
        lam_map[labels.ravel() == b] = l

    # ASR 分块自适应 λ
    obj_b = make_asr_objective(delta[trva], soft[trva], lam_map[trva])
    bst_b = xgb.train(params, dtrva, num_boost_round=num_round,
                      obj=obj_b, verbose_eval=False)
    p_asr_b = predict_prob(bst_b, dgrid)

    # 测试集评估
    out = dict(gname=gname, pname=pname, seed=seed, n_blocks=n_blk,
               lam_star=lam_star, cv_rows=cv_rows, cv_curves=cv_curves,
               lam_map=lam_map, labels=labels, R=R, p_true=p_true, ocean=ocean,
               y=y, te=te, p_ce=p_ce, p_asr_g=p_asr_g, p_asr_b=p_asr_b,
               n=n, soft=soft, delta=delta, Xall=Xall, trva=trva)
    out['test'] = {}
    for name, p in [('CE', p_ce), ('ASRg', p_asr_g), ('ASRb', p_asr_b)]:
        moran, conted, iso = spatial_metrics(p.reshape(n, n), land_mask)
        out['test'][name] = (auc_score(y[te], p[te]),
                             brier_score(y[te], p[te]),
                             moran, conted, iso,
                             logloss_score(y[te], p[te]),
                             recall_score(y[te], p[te]))
    out['n_land'] = len(land_idx)
    out['n_dataset'] = len(ds)
    out['n_split'] = (len(tr), len(va), len(te))
    return out


def build_spatial_folds(block_ids, y, n_folds=5, min_samples=10, seed=42):
    """final 式空间折：网格块 → 有效块(≥min_samples 样本且含两类) →
    按块大小贪心平衡分到 n_folds 个折（块不跨折，对齐 final.py
    _build_spatial_folds）。无效块(样本不足/单类)的点 fold=-1：
    不参与测试，但可参与训练。返回 point_fold（长度 = len(block_ids)）。"""
    valid = []
    for bid in np.unique(block_ids):
        if bid < 0:
            continue
        m = block_ids == bid
        yb = y[m]
        if len(yb) >= min_samples and len(np.unique(yb)) >= 2:
            valid.append(bid)
    valid = np.array(valid)
    sizes = np.array([(block_ids == bid).sum() for bid in valid])
    order = np.argsort(-sizes)                 # 大块优先
    fold_sizes = np.zeros(n_folds, dtype=int)
    block_fold = {}
    rng = np.random.RandomState(seed)
    for i in order:
        cands = np.where(fold_sizes == fold_sizes.min())[0]
        f = int(rng.choice(cands))
        block_fold[valid[i]] = f
        fold_sizes[f] += sizes[i]
    point_fold = np.full(len(block_ids), -1, dtype=int)
    for bid in valid:
        point_fold[block_ids == bid] = block_fold[bid]
    return point_fold


def spatial_cv_evaluate(X, y, fit_idx, grid_labels, R, beta, params, num_round,
                        lam_map, n_folds=5, min_samples=10, fold_seed=42):
    """final 式空间 CV（3×3 网格块折，块不跨折，无效块点只进训练）。

    每折：折内训练集重训 CE → 全图重算软标签/门控（无泄漏）→
    ASR 用训练侧预选 λ*（lam_map 逐像元，与测试集口径一致，无 oracle）。
    返回 (per_fold, pts)：per_fold 每折 (brier, auc, logloss, recall)；
    pts 收集全部测试折像元的 y/CE/SR/ASR/lam（供子集增益分析）。"""
    block_ids = grid_labels[fit_idx]
    point_fold = build_spatial_folds(block_ids, y[fit_idx], n_folds,
                                     min_samples, fold_seed)
    n = int(round(np.sqrt(len(X))))
    dX = xgb.DMatrix(X)
    per_fold = {m: [] for m in ('CE', 'SR(λ=1.0)', 'ASR')}
    pts = {'y': [], 'CE': [], 'SR': [], 'ASR': [], 'lam': []}
    for f in range(n_folds):
        test_pts = fit_idx[point_fold == f]
        if len(test_pts) == 0:
            continue
        train_pts = fit_idx[point_fold != f]             # 含无效块点
        dtr = xgb.DMatrix(X[train_pts], label=y[train_pts])
        bst_ce = xgb.train(params, dtr, num_boost_round=num_round,
                           verbose_eval=False)
        p_ce_full = bst_ce.predict(dX)                   # 全栅格预测
        soft_f = soft_labels_8nn(p_ce_full.reshape(n, n), R, beta).ravel()
        delta_f = gate_frozen(p_ce_full)
        y_te = y[test_pts]
        p_ce_te = p_ce_full[test_pts]
        # SR(λ=1.0)：固定强度
        obj = make_asr_objective(delta_f[train_pts], soft_f[train_pts], 1.0)
        bst = xgb.train(params, dtr, num_boost_round=num_round, obj=obj,
                        verbose_eval=False)
        p_sr10_te = predict_prob(bst, dX)[test_pts]
        # ASR：训练侧预选 λ*（逐像元，公平版）
        obj = make_asr_objective(delta_f[train_pts], soft_f[train_pts],
                                 lam_map[train_pts])
        bst = xgb.train(params, dtr, num_boost_round=num_round, obj=obj,
                        verbose_eval=False)
        p_asr_te = predict_prob(bst, dX)[test_pts]
        per_fold['CE'].append((brier_score(y_te, p_ce_te),
                               auc_score(y_te, p_ce_te),
                               logloss_score(y_te, p_ce_te),
                               recall_score(y_te, p_ce_te)))
        per_fold['SR(λ=1.0)'].append((brier_score(y_te, p_sr10_te),
                                      auc_score(y_te, p_sr10_te),
                                      logloss_score(y_te, p_sr10_te),
                                      recall_score(y_te, p_sr10_te)))
        per_fold['ASR'].append((brier_score(y_te, p_asr_te),
                                auc_score(y_te, p_asr_te),
                                logloss_score(y_te, p_asr_te),
                                recall_score(y_te, p_asr_te)))
        pts['y'].append(y_te)
        pts['CE'].append(p_ce_te)
        pts['SR'].append(p_sr10_te)
        pts['ASR'].append(p_asr_te)
        pts['lam'].append(lam_map[test_pts])
    return per_fold, pts


# ==========================================================================
# 6. 主流程
# ==========================================================================
CACHE_VERSION = 'v2'   # 缓存结构版本（test 元组含 Moran's I/ContED/Iso ratio 起）；改结构需递增
def main():
    ap = argparse.ArgumentParser(description='ASR-XGBoost minimal demo')
    ap.add_argument('--beta', type=float, default=2.0)
    ap.add_argument('--gen', default='g_clean', choices=list(GENERATIONS.keys()),
                    help='单次演示的数据生成场景（g_hard 弱信号困难场景 ASR 收益更明显）')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--grid', type=int, default=300, help='模拟栅格边长')
    ap.add_argument('--blocks', default=None, help='regions.npy（draw_regions.py 生成）')
    ap.add_argument('--sample', type=float, default=0.4,
                    help='总像元采样比例作为数据集（默认 0.4，模拟观测稀疏但结果稳定）')
    ap.add_argument('--split', default='7,2,1',
                    help='数据集内训练/验证/测试比例（默认 7:2:1，与论文口径一致）')
    ap.add_argument('--nblocks', type=int, default=16)
    ap.add_argument('--min-pixels', type=int, default=800,
                    help='小区域合并阈值（像元数；对应 strategy B 小国合并，保证块内 CV 稳定）')
    ap.add_argument('--lam-grid', default='0,0.5,1,1.5,2,2.5,3,3.5',
                    help='λ 候选：0（无正则化，等价 CE）至 3.5（论文敏感性分析上限）')
    ap.add_argument('--cv-folds', type=int, default=5)
    ap.add_argument('--cv-rounds', type=int, default=100)
    ap.add_argument('--num-round', type=int, default=150)
    # 模拟研究
    ap.add_argument('--study', action='store_true', help='运行多情形模拟研究')
    ap.add_argument('--gens', default='g_clean,g_mid,g_hard',
                    help='模拟研究的数据生成列表（默认三档难度：g_clean 简单 / g_mid 中等 / '
                         'g_hard 困难；可加 g_noisy,g_noisy2,g_barrier,g_lgcp，逗号分隔）')
    ap.add_argument('--parts', default='p_voronoi,p_grid,p_resist')
    ap.add_argument('--reps', type=int, default=2, help='每个组合的重复种子数')
    ap.add_argument('--study-grid', type=int, default=100)
    ap.add_argument('--study-nblocks', type=int, default=8)
    ap.add_argument('--no-cache', action='store_true',
                    help='忽略 study 缓存，强制全部重跑')
    args = ap.parse_args()
    lam_grid = [float(v) for v in args.lam_grid.split(',')]
    split = tuple(float(v) for v in args.split.split(','))
    if args.study:
        run_study(args, lam_grid)
        return

    # ---------- 单次详细演示 ----------
    out = simulate_once(args.gen, 'p_voronoi', args.seed, args.grid,
                        args.nblocks, args.min_pixels, lam_grid, args.beta,
                        dict(objective='binary:logistic', eta=0.1, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, seed=args.seed),
                        args.num_round, args.cv_rounds, args.cv_folds,
                        custom_blocks=(np.load(args.blocks) if args.blocks else None),
                        verbose=True, split=split, sample_frac=args.sample)
    n = out['n']
    n_land = out['n_land']
    n_tr, n_va, n_te = out['n_split']
    print('=' * 66)
    print(f'模拟栅格 {n}×{n}（陆地 {n_land} 像元 | 数据集 {out["n_dataset"]}，'
          f'{args.sample:.0%} 采样）| 训练 {n_tr} | 验证 {n_va} | 测试 {n_te}（{args.split}）')
    print(f'数据生成: {GENERATIONS[args.gen]["name"]} | 分块: {PARTITIONS["p_voronoi"]} | '
          f'块数 {out["n_blocks"]}')
    print(f'超参数: β={args.beta}  λ_global=1.0  λ_grid={lam_grid}  seed={args.seed}')
    print(f'分块自适应 λ*（块内 {args.cv_folds} 折 CV，1-SE 规则）:')
    for b, npx, star, mb, ma in out['cv_rows']:
        print(f'  块 {b}: 像元 {npx:5d}  λ* = {star:>4}  (CV Brier {mb:.4f} / AUC {ma:.4f})')
    print('各块 λ 的 CV Brier 曲线（* = 选中，越低越好）:')
    for b, npx, star, mb, ma in out['cv_rows']:
        pts = ' '.join(f'{l}:{cb:.3f}{"*" if l == star else ""}'
                       for l, cb, _, _ in out['cv_curves'][b])
        print(f'  块 {b:<2}: {pts}')
    print('=' * 66)

    # final 式空间 CV：3×3 网格块折（对应 5°×5° 惯例），块不跨折；
    # 每折重训 CE 并重算软标签/门控（无泄漏）；ASR 用训练侧预选 λ*（公平版）。
    print('>> final 式空间 CV 评估（3×3 网格块折）…', flush=True)
    glab = grid_blocks(n, 3).ravel()
    per_fold, _ = spatial_cv_evaluate(out['Xall'], out['y'], out['trva'], glab,
                                      out['R'], args.beta,
                                      dict(objective='binary:logistic', eta=0.1,
                                           max_depth=4, subsample=0.8,
                                           colsample_bytree=0.8, seed=args.seed),
                                      args.num_round, out['lam_map'],
                                      n_folds=args.cv_folds)
    print(f'空间 CV（3×3 网格块折，{args.cv_folds} 折，mean±std，唯一评估方式）:')
    print(f'{"指标":<10}{"CE":<24}{"SR(λ=1.0)":<26}{"ASR":<14}')
    for mname, idx in [('AUC', 1), ('Brier', 0), ('LogLoss', 2)]:
        row = f'{mname:<10}'
        for m in ('CE', 'SR(λ=1.0)', 'ASR'):
            vals = [x[idx] for x in per_fold[m]]
            row += f'{np.mean(vals):.4f}±{np.std(vals):.4f}  '
        print(row)
    print('=' * 66)

    # 出图（2×3 六联图：真值/分块/λ* + CE/ASR/差值；差值图放大差距，海洋显示为灰色）
    ocean = out['ocean']

    def mask(arr):
        return np.ma.masked_where(ocean, arr) if ocean is not None else arr

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    p_ce_g = mask(out['p_ce'].reshape(n, n))
    p_asr_map = mask(out['p_asr_b'].reshape(n, n))
    diff = p_asr_map - p_ce_g
    vmax = float(np.nanmax(np.abs(diff))) or 1.0   # 差值图对称色标，放大差距
    panels = [
        ('True risk', mask(out['p_true']), 'viridis', None),
        ('Blocks (merged)', mask(out['labels']), 'tab10', None),
        ('Lambda* by block', mask(out['lam_map'].reshape(n, n)), 'viridis', None),
        ('CE prediction', p_ce_g, 'viridis', None),
        ('ASR (block-adaptive)', p_asr_map, 'viridis', None),
        ('ASR − CE', diff, 'RdBu_r', (-vmax, vmax)),
    ]
    for ax, (title, data, cmap_name, vrange) in zip(axes.ravel(), panels):
        cmap = plt.get_cmap(cmap_name)
        cmap.set_bad('0.85')
        kw = dict(origin='lower', vmin=vrange[0], vmax=vrange[1]) if vrange else dict(origin='lower')
        im = ax.imshow(data, cmap=cmap, **kw)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('ASR-XGBoost demo: CE vs ASR (block-adaptive λ)  '
                 f'(beta={args.beta}, seed={args.seed})', fontsize=13)
    fig.tight_layout()
    fig.savefig('demo_result.png', dpi=140)
    print('✅ 全部完成，图片已保存: demo_result.png', flush=True)


def _cache_entry(out):
    """提取聚合所需字段做缓存（不含 Xall/R/soft 等训练用大对象）。"""
    return dict(gname=out['gname'], pname=out['pname'], seed=out['seed'],
                n_blocks=out['n_blocks'], lam_star=out['lam_star'], n=out['n'],
                test=out['test'], cv=out['cv'], cv_pts=out['cv_pts'],
                te=out['te'], y=out['y'], p_ce=out['p_ce'],
                p_asr_b=out['p_asr_b'], lam_map=out['lam_map'])


def run_study(args, lam_grid):
    """多情形模拟研究：生成方式 × 分块方式 × 重复种子 → 平均精度。
    逐模拟缓存（study_cache.pkl，key 含全部相关参数；--no-cache 强制重跑）。"""
    gens = args.gens.split(',')
    parts = args.parts.split(',')
    split = tuple(float(v) for v in args.split.split(','))
    n = args.study_grid
    params = dict(objective='binary:logistic', eta=0.1, max_depth=4,
                  subsample=0.8, colsample_bytree=0.8, seed=args.seed)
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'study_cache.pkl')
    cache = {}
    if not args.no_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            # 仅复用当前版本（v2）的条目；旧版本条目保留在文件中不删除，
            # 避免写回时无谓覆盖丢失（版本不兼容时 key 自然不匹配，不会误用）
            n_ok = sum(1 for k in cache if k.startswith(CACHE_VERSION + '|'))
            print(f'[Cache] 加载 {len(cache)} 条（当前版本可复用 {n_ok} 条）: {cache_path}')
        except Exception:
            cache = {}
            print('[Cache] 缓存读取失败，忽略并重跑')
    results = []
    total = len(gens) * len(parts) * args.reps
    i = 0
    for gi, gname in enumerate(gens):
        for pi, pname in enumerate(parts):
            for rep in range(args.reps):
                seed = args.seed + rep * 100 + gi * 10 + pi
                key = (f'{CACHE_VERSION}|{gname}|{pname}|{seed}|{args.seed}|{n}|{args.study_nblocks}|'
                       f'{args.min_pixels}|{",".join(map(str, lam_grid))}|'
                       f'{args.beta}|{args.cv_folds}|'
                       f'{args.sample}|{",".join(map(str, split))}')
                if key in cache:
                    out = cache[key]
                    cached = True
                else:
                    out = simulate_once(gname, pname, seed, n, args.study_nblocks,
                                        args.min_pixels, lam_grid, args.beta,
                                        params, 60, 40, args.cv_folds,
                                        split=split, sample_frac=args.sample)
                    # final 式空间 CV（3×3 网格块折，每折重训+重算软标签，无泄漏）
                    glab = grid_blocks(out['n'], 3).ravel()
                    pf, pts = spatial_cv_evaluate(out['Xall'], out['y'], out['trva'], glab,
                                                  out['R'], args.beta, params, 60,
                                                  out['lam_map'],
                                                  n_folds=args.cv_folds)
                    out['cv'] = {m: np.asarray(pf[m]).mean(axis=0)
                                 for m in ('CE', 'SR(λ=1.0)', 'ASR')}
                    out['cv_pts'] = {k: np.concatenate(v) for k, v in pts.items()}
                    cache[key] = _cache_entry(out)
                    cached = False
                results.append(out)
                i += 1
                print(f'[{i}/{total}] {gname} × {pname} × seed={seed} '
                      f'({"缓存" if cached else "重跑"}) '
                      f'(λ* 范围 {min(out["lam_star"].values())}~'
                      f'{max(out["lam_star"].values())})', flush=True)
    if not args.no_cache:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f'[Cache] 已写入 {len(cache)} 条缓存: {cache_path}')

    # 汇总：测试集指标 + final 式空间 CV（mean±std）
    disp = {'CE': 'CE', 'ASRg': 'SR(λ=1.0)', 'ASRb': 'ASR'}
    cv_models = ('CE', 'SR(λ=1.0)', 'ASR')
    print('=' * 74)
    print(f'模拟研究汇总：{len(gens)} 种数据生成 × {len(parts)} 种分块 × '
          f'{args.reps} 个种子 = {len(results)} 次模拟（跨次平均，等效重复实验）')
    print('-' * 74)
    print('测试集指标（7:2:1 划分，ASR 用训练侧分块 λ*；mean±std）:')
    print(f'{"指标":<10}{"CE":<24}{"SR(λ=1.0)":<26}{"ASR":<14}')
    agg = {}
    for mname, idx in [('AUC', 0), ('Brier', 1), ('LogLoss', 5),
                       ("Moran's I", 2), ('Cont. ed.', 3), ('Iso ratio', 4)]:
        row = f'{mname:<10}'
        for m in ('CE', 'ASRg', 'ASRb'):
            vals = [r['test'][m][idx] for r in results]
            agg.setdefault(disp[m], {})[mname] = np.asarray(vals)
            row += f'{np.mean(vals):.4f}±{np.std(vals):.4f}  '
        print(row)
    print('-' * 74)
    _asr_note = 'ASR 用训练侧预选 λ*（公平版）'
    print('空间 CV（3×3 网格块折，块不跨折，每折重训+重算软标签，'
          f'{_asr_note}；mean±std）:')
    print(f'{"指标":<10}{"CE":<24}{"SR(λ=1.0)":<26}{"ASR":<14}')
    cv_agg = {}
    for mname, idx in [('AUC', 1), ('Brier', 0), ('LogLoss', 2)]:
        row = f'{mname:<10}'
        for m in cv_models:
            vals = [r['cv'][m][idx] for r in results]
            cv_agg.setdefault(m, {})[mname] = np.asarray(vals)
            row += f'{np.mean(vals):.4f}±{np.std(vals):.4f}  '
        print(row)
    print('-' * 74)
    for mname in ('AUC', 'Brier', 'LogLoss', "Moran's I", 'Cont. ed.', 'Iso ratio'):
        d = np.mean(agg['ASR'][mname] - agg['CE'][mname])
        print(f'平均改善 (ASR − CE, 测试集): Δ{mname} {d:+.4f}')
    for mname in ('AUC', 'Brier', 'LogLoss'):
        d = np.mean(cv_agg['ASR'][mname] - cv_agg['CE'][mname])
        print(f'平均改善 (ASR − CE, 空间CV): Δ{mname} {d:+.4f}')
    print('-' * 74)
    # 子集增益：只在 ASR 实际生效（λ*>0）或门控最强（CE∈[0.4,0.6] 风险带）
    # 的像元池上算 ASR−CE（pooled，跨模拟平均；每模拟池 <50 样本则跳过）
    band = (0.4, 0.6)
    subsets = [
        ('仅λ*>0区域',
         lambda r, te: r['lam_map'][te] > 0.0,
         lambda pts: pts['lam'] > 0.0),
        (f'CE∈[{band[0]:.1f},{band[1]:.1f}]',
         lambda r, te: (r['p_ce'][te] >= band[0]) & (r['p_ce'][te] <= band[1]),
         lambda pts: (pts['CE'] >= band[0]) & (pts['CE'] <= band[1])),
    ]
    print('子集增益（ASR − CE，仅在子集像元池上；跨模拟平均）:')
    for tag, cond_t, cond_c in subsets:
        for src, cond in (('测试集', cond_t), ('空间CV', cond_c)):
            dA, dB, dL, n_used = [], [], [], 0
            for r in results:
                if src == '测试集':
                    te = r['te']
                    m = cond(r, te)
                    if int(m.sum()) < 50:
                        continue
                    y_s = r['y'][te][m]
                    ce_s = r['p_ce'][te][m]
                    asr_s = r['p_asr_b'][te][m]
                else:
                    p = r['cv_pts']              # 已是拼接好的数组
                    m = cond(p)
                    if int(m.sum()) < 50:
                        continue
                    y_s = p['y'][m]
                    ce_s = p['CE'][m]
                    asr_s = p['ASR'][m]
                dA.append(auc_score(y_s, asr_s) - auc_score(y_s, ce_s))
                dB.append(brier_score(y_s, asr_s) - brier_score(y_s, ce_s))
                dL.append(logloss_score(y_s, asr_s) - logloss_score(y_s, ce_s))
                n_used += 1
            if n_used:
                print(f'  {src} {tag:<12} (n={n_used}/{len(results)}): '
                      f'ΔAUC {np.mean(dA):+.4f}  ΔBrier {np.mean(dB):+.4f}  '
                      f'ΔLogLoss {np.mean(dL):+.4f}')
            else:
                print(f'  {src} {tag:<12}: 无足够样本')
    print('-' * 74)
    # 配对显著性：每模拟 Δ = ASR − CE，配对 t + Wilcoxon 符号秩
    from scipy import stats
    print(f'配对显著性（每模拟 Δ = ASR − CE；配对 t 与 Wilcoxon 符号秩，n={len(results)}）:')
    for src, key, mlist, asr_key in (
            ('测试集', 'test', [('AUC', 0), ('Brier', 1), ('LogLoss', 5),
                                ("Moran's I", 2), ('Cont. ed.', 3), ('Iso ratio', 4)], 'ASRb'),
            ('空间CV', 'cv', [('AUC', 1), ('Brier', 0), ('LogLoss', 2)], 'ASR')):
        for mname, idx in mlist:
            d = np.array([r[key][asr_key][idx] - r[key]['CE'][idx] for r in results])
            t, p = stats.ttest_1samp(d, 0.0)
            if np.any(d != 0.0):
                w, wp = stats.wilcoxon(d)
            else:
                w, wp = np.nan, 1.0
            print(f'  {src} {mname:<8} Δ{np.mean(d):+.4f}±{np.std(d):.4f}  '
                  f't={t:+.2f} p={p:.4f}  Wilcoxon p={wp:.4f}')
    print('-' * 74)
    print('按 生成×分块 组合（跨种子平均 AUC / Brier，测试集）:')
    print(f'{"组合":<28}{"CE":<18}{"SR(λ=1.0)":<20}{"ASR":<14}')
    for gname in gens:
        for pname in parts:
            rs = [r for r in results if r['gname'] == gname and r['pname'] == pname]
            row = f'{gname}×{pname:<16}'
            for m in ('CE', 'ASRg', 'ASRb'):
                a = np.mean([r['test'][m][0] for r in rs])
                b = np.mean([r['test'][m][1] for r in rs])
                row += f'{a:.4f}/{b:.4f}  '
            print(row)
    print('-' * 74)
    print('按 生成×分块 组合（跨种子平均 AUC / Brier，空间CV）:')
    print(f'{"组合":<28}{"CE":<18}{"SR(λ=1.0)":<20}{"ASR":<14}')
    for gname in gens:
        for pname in parts:
            rs = [r for r in results if r['gname'] == gname and r['pname'] == pname]
            row = f'{gname}×{pname:<16}'
            for m in cv_models:
                a = np.mean([r['cv'][m][1] for r in rs])
                b = np.mean([r['cv'][m][0] for r in rs])
                row += f'{a:.4f}/{b:.4f}  '
            print(row)
    print('-' * 74)
    print('按 生成 分档精度（跨 3 分块 × 10 种子；每格 CE / SR(λ=1.0) / ASR）:')
    hdr = f'{"生成":<10}'
    for col, key, idx in (('AUC(CV)', 'cv', 1), ('Brier(CV)', 'cv', 0),
                          ('LogLoss(CV)', 'cv', 2), ('Iso(test)', 'test', 4)):
        hdr += f'{col:<28}'
    print(hdr)
    for gname in gens:
        rs = [r for r in results if r['gname'] == gname]
        row = f'{gname:<10}'
        for col, key, idx in (('AUC(CV)', 'cv', 1), ('Brier(CV)', 'cv', 0),
                              ('LogLoss(CV)', 'cv', 2), ('Iso(test)', 'test', 4)):
            ms = ('CE', 'SR(λ=1.0)', 'ASR') if key == 'cv' else ('CE', 'ASRg', 'ASRb')
            row += '/'.join(f'{np.mean([r[key][m][idx] for r in rs]):.4f}' for m in ms).ljust(28)
        print(row)
    print('-' * 74)
    # λ* 选择分布（训练侧块内 CV，对应行政 λ 选择口径）
    from collections import Counter
    cnt = Counter()
    for r in results:
        cnt.update(r['lam_star'].values())
    dist = ', '.join(f'λ={k}: {v} 块' for k, v in sorted(cnt.items()))
    print('训练侧分块自适应 λ* 选择分布（全部模拟的所有块）:')
    print('  ' + dist)
    print('=' * 74)


if __name__ == '__main__':
    main()
