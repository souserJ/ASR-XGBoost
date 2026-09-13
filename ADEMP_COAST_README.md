# 海岸/海陆几何因子（coast factor）——ADEMP 扩展

新增两个文件，**原有文件一律未改**：

| 新文件 | 来源 | 作用 |
|---|---|---|
| `asr_demo_coast.py` | 复制自 `asr_demo_modified.py` | 只新增 `COASTS` + `make_ocean()`，`gen_landscape` 的海陆几何改为按 `cfg['coast']` 生成 |
| `asr_ademp_coast.py` | 复制自 `asr_ademp.py` | 只新增 `--coast` / `--coast-pairing` 因子，其余 ADEMP 流程不变 |

被改动的函数只有：`gen_landscape` 里的 ocean 段落、`generate_data`/`model_regions` 的 seed 前缀、
`summarize` 的分组键、`parse_args` 的参数、`main` 的循环与文件名。

## 因子水平

| coast | 几何 | 海洋占比(grid 60) | 陆地像元 |
|---|---|---|---|
| `vertical` | 纵向海峡（**与原实现逐像元完全一致**） | 0.100 | 3240 |
| `horizontal` | 横向海湾（把海峡转到东西向） | 0.100 | 3240 |
| `archipelago` | 近海群岛（破碎海岸 + 近岸水道，固定模板） | 0.323 | 2437 |
| `none` | 无海洋（全陆地） | 0.000 | 3600 |

`archipelago` 的模板参数（可在 `make_ocean` 里调）：
`arch_ocean_frac=0.25`、`arch_len_scale=7.0`、`arch_channel=0.18`、`arch_channel_w=0.05`、`arch_seed=20260912`（固定模板）。
把 `cfg['coast_jitter']=True` 可让 vertical/horizontal 的振幅/宽度/相位随 rng 抖动（每个重复一套海岸）。

## coast pairing（重要）

- `--coast-pairing shared`（**默认**）：同一 (scenario, rep) 下 4 个 coast 水平**共用同一套**
  landscape/DGP/observation/sample 随机流，减少环境场随机波动对构型比较的影响。由于海陆掩膜会改变
  可用像元、样本量、生成分区以及部分场景的观测过程，这仍然不是“只改变海岸线”的独立因果对照。
  **`coast=vertical` + shared 可以逐种子复现原来的 asr_ademp.py pilot**（已验证 landscape seed 一致）。
- `--coast-pairing independent`：每个 coast 水平各自抽随机流（完全独立的地理位置）。

## 条件与输出

条件 = `scenario × coast × partition × validation × rep`。

- 新列：`summary.csv` / `paired.csv` / `diagnostics.csv` / `lambda_curves.csv` 均含 `coast` 列；
  `seeds.csv` 新增 `coast` 与 `land_pixels` 两列。
- 文件命名：`truth_{scenario}_{coast}_{rep}.npz`、`pred_{scenario}_{coast}_{rep}_{partition}_{protocol}.npz`。
- **默认输出目录是 `results/coast_<YYYYmmdd_HHMMSS>`**，不会覆盖 `ademp_*` 旧结果；
  且 `--out` 目标已存在会直接报错退出（`exist_ok=False`）。

## 运行

快速冒烟（流程检查，约 1 分钟）：

```powershell
python asr_ademp_coast.py --gens g_clean,g_detection --coast vertical,horizontal,archipelago,none `
  --parts p_grid --validation random,spatial --reps 2 --grid 24 --sample 0.7 `
  --inner-folds 2 --outer-folds 2 --num-round 12 --early-stop 4 --lam-grid 0,1 `
  --out results/coast_smoke
```

看海岸效应（3 场景 × 4 海岸 × 20 重复，估计 ~15 分钟）：

```powershell
python asr_ademp_coast.py --gens g_clean,g_mid,g_nonstationary --coast vertical,horizontal,archipelago,none `
  --parts p_grid --reps 20 --out results/coast_pilot
```

完整因子先导（5 场景 × 4 海岸 × 20 重复，估计 ~45 分钟）：

```powershell
python asr_ademp_coast.py --coast vertical,horizontal,archipelago,none --parts p_grid --reps 20 `
  --out results/coast_pilot_full
```

只想看 `vertical` 能否复现旧 pilot（应与 `ademp_20260912_173152` 数值一致）：

```powershell
python asr_ademp_coast.py --coast vertical --parts p_grid --reps 20 --out results/coast_vertical_check
```

## 已知口径差异（看结果时注意）

1. **陆地像元数随 coast 变化**：`--sample` 是陆地像元的比例，所以 `archipelago`（陆地 68%）
   的每重复样本数比 `vertical`（90%）少约 25% → 该水平 MC 精度略低。若要等样本量，
   可对 archipelago 单独提高 `--sample`，或改成固定样本数（需要小改，另行确认）。
2. `none` 水平没有海洋 → `split_blocks_at_barriers` 不生效（代码里本来就有 `if np.any(~land)` 守卫）。
3. 海岸是**固定模板**：同一 coast 水平下不同重复的海岸线完全相同（只有环境场在变）。
   若要让海岸也随重复变化，用 `coast_jitter`，或走 `--coast-pairing independent` 之外再抖动模板。
4. `coast_pilot` 仅覆盖低噪、中噪和非平稳三个场景，并且只使用 `p_grid`。其结果应表述为在这些
   已测试地理构型下的定性稳健性检查，不能推广为对任意海岸或空间分区均不敏感。
5. 标准输出的 `paired.csv` 比较的是同一条件内 ASR 与 CE/SR1/SRcv，并不直接给出海岸水平之间的
   显著性检验。任何 coast-to-coast 对比都需要另存可复现的分析脚本并说明多重比较处理。

## 验证记录（2026-09-12）

- `python -m py_compile asr_demo_coast.py asr_ademp_coast.py` 通过。
- 冒烟运行（grid 24，g_clean+g_detection × 4 coasts × 3 reps × 2 protocols）：
  **0 failures**，输出到 `/tmp`，未触碰 `results/`。
- `coast=vertical` 的 ocean 掩码与原 `gen_landscape` **逐像元 0 差异**；
  `shared` pairing 下 landscape seed 与原 pilot 完全一致。
