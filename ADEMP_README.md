# ASR 模拟研究修正版

运行入口为 `asr_ademp.py`。它复用 `asr_demo_modified.py` 中的环境场和损失函数；旧脚本保留，但其旧验证结果不能与本版本直接拼接。本版本不读取旧 pickle 缓存。

## 快速运行

在本目录执行：

```powershell
python asr_ademp.py --gens g_clean,g_detection --parts p_grid --validation random,spatial --reps 2 --grid 24 --sample 0.7 --inner-folds 2 --outer-folds 2 --num-round 12 --early-stop 4 --lam-grid 0,1 --out results/ademp_smoke
```

这是流程检查，不能用作论文性能结论。每次输出到新目录；已有目录不会被覆盖。

小规模先导研究（完成后按 MCSE 决定正式重复次数）：

```powershell
python asr_ademp.py --gens g_clean,g_mid,g_hard,g_nonstationary,g_detection --parts p_grid --reps 20 --out results/ademp_pilot
```

默认同时运行 random 和 spatial；完整验证会比旧实现昂贵，因为不能复用看过验证标签的参考模型或 λ。先导研究不是对所有空间分区的充分验证。

## ADEMP 对应关系

| 要素 | 本工程中的定义 |
|---|---|
| A：研究目的 | 判断自适应强度在何种条件下改善或损害预测，包括与调参后的全局 SR 比较；不预设 ASR 获胜 |
| D：数据生成 | 保留原场景参数和多尺度环境场。风险突变、噪声强度使用单独生成的 Voronoi 分区；模型分区不控制生成数据 |
| E：评价目标 | 观测出现概率 `p_obs`；潜在风险/适宜性 `p_true` 作为另一个明确区分的诊断目标 |
| M：比较方法 | CE；SR1（λ=1）；SRcv（内层 CV 选择全局 λ）；ASR（同一内层折按区域选择 λ） |
| P：性能指标 | 标签 AUC、Brier、LogLoss、Recall；相对 `p_obs` 和 `p_true` 的 MSE、相对 `p_true` 的平均误差、理论期望 Brier；每个条件内重复间 MCSE 和配对差值 |

参考：Morris、White、Crowther，*Using simulation studies to evaluate statistical methods*，用户提供的 arXiv:1712.03198v3 版本；特别是表 1、表 3、表 6。本实现将其研究设计原则用于预测问题，不照搬参数估计任务的全部指标。

## 验证协议

1. 从陆地像元中采样；所有方法共享数据与外层折。
2. random 为随机像元外层 CV，衡量区域内插值；spatial 为固定 4×4 网格的整组留出，衡量空间留出性能（不是带缓冲区的远距离外推）。内层调参使用相同类型的折。
3. 外层训练数据内生成内层折；每个内层训练折再随机留出 15% 作早停。CE、软标签和门控全部只依赖内层训练/早停标签，内层评分标签只用于选择 λ。
4. 候选 λ 模型在完整内层拟合子集训练；同一模型的内层留出误差分别用于全局、区域评分。因此避免原实现“在小块上训练候选模型、再把所选 λ 用到全局模型”的不一致。
5. 最终各方法使用相同外层训练侧拟合/早停划分，外层留出标签只用于评价。XGBoost 预测明确截取 `best_iteration + 1`。
6. 一个重复内合并所有外层留出预测再计算指标。外层折不是独立 Monte Carlo 重复，不能将折数乘进重复数。

λ 默认按最小内层 Brier 选择；`--rule 1se` 保留“1-SE 范围内选最小 λ”的保守启发式，这不是通常选择更强正则化的 glmnet 规则。折间 SE 仅用于该启发式，不能视为独立重复的 MCSE。区域少于 2 个有效评分折或评分样本不足时，回退到训练侧选择的全局 λ，并记录原因；未见区域也使用全局 λ。

软标签使用有限八邻域，排除海洋和边界环绕。对没有有效邻居的像元保留其 CE 概率。λ=0 直接使用 CE 目标；所有方法统一用 Brier 早停。

## 观测目标及解释限制

- Gaussian 概率扰动场景：原过程是对每个像元独立采样噪声，再由截断概率采样一次标签。修正版解析计算 `E[clip(p_true+epsilon)]`，直接由此采样 Bernoulli 标签；边际标签分布不变。评价目标 `p_obs` 因而不含一次性的不可预测噪声实现。
- detection：`p_obs=clip(p_true*q)`，保存 `q`。直接预测观测标签的模型没有显式校正漏报，不能因为 `mse_true` 偏大就判为实现错误，也不能声称 ASR 可识别未知检测概率。
- LGCP：`p_obs=1-exp(-intensity)`，条件于本次模拟的潜在随机场；`p_true` 仍表示生成适宜性，不等于发生概率。
- 非平稳场保留原来的长/短尺度独立场空间混合；这不是精确指定了某个位置相关长度函数的 Matérn 协方差模型。
- 分区已经解耦，但场景参数沿用原脚本，尚未形成完整因子设计。后续可交叉改变采样比例、噪声与空间尺度。
- 全图已知协变量用于软标签预测；未使用留出标签。这符合栅格制图情境，不应解释为对未知协变量区域的完全归纳式预测。

## 输出

- `manifest.json`：参数、运行库版本、源代码 SHA256 和评价定义。
- `seeds.csv`：独立生成阶段的随机种子；种子按场景/重复/阶段命名，改变场景或分区列表顺序不改变已有数据。
- `truth_*.npz`：真实风险、观测概率、检测概率、标签、采样位置、陆地和生成分区。
- `pred_*.npz`：每种方法的外层留出预测、折号和模型分区；未采样位置是 NaN。
- `replicates.csv`：每条件、每独立重复、每方法的指标。
- `summary.csv`：每个场景×模型分区×验证协议单独报告均值、样本 SD、MCSE 和近似 95% Monte Carlo t 区间。
- `paired.csv`：同一重复上的 ASR−CE、ASR−SR1、ASR−SRcv 差值。损失类指标负值表示 ASR 更好，AUC/Recall 正值表示更好；有符号偏差不能简单按越低越好解释。
- `lambda_curves.csv`：各候选 λ 的内层折平均 Brier、折间 SE、选中值、有效样本数和回退标记。
- `diagnostics.csv`：λ 分布、回退块数、门控、软标签改变幅度、拟合/早停/留出样本数和各模型最佳轮数。
- `failures.csv`：失败条件及错误。不静默补值或把失败当作成功；存在失败时进程返回非零。指标表只含成功条件，需与失败记录共同解读。单类样本 AUC/Recall 不可定义时写 NaN，不伪装成 0.5/0。

每完成一个条件即写表，便于检查中途结果。当前不提供断点续训；完整重新运行需指定新的输出目录。一个重复时 MCSE/区间为 NaN，而不是零。

对于每条件的配对差值 `d_r`，`MCSE=SD(d_r)/sqrt(R)`。根据先导研究的差值 SD 和预先选定的目标精度 ε，可估计正式重复数 `ceil((SD/ε)^2)`；这只是先导估计，不能以“出现显著优势”为停止规则。Monte Carlo 区间仅反映有限模拟重复的不确定性。

## 检查

```powershell
python -m pytest test_asr_ademp.py -q
```

检查留出标签隔离、内层早停隔离、λ=0 一致性、最佳轮数预测、空间组隔离、无环绕邻域、配对 MCSE、AUC ties 和噪声边际化。

## 论文示意图

`make_sim_figure.py` 复现论文里的模拟示意图（5 面板：真值 / CE / ASR / ASR−CE / 校准曲线）。
它只读 ADEMP 的输出，不重跑研究本身。

```bash
# 第一步：生成图所需的一次运行（几秒）
python asr_ademp.py --gens g_mid --parts p_voronoi --validation random --reps 1 --sample 1.0 --out results/figure_pvoronoi

# 第二步：出图（默认参数即为论文图）
python make_sim_figure.py                    # → figures/ademp_sim_demo.png
```

- 默认 `--run results/figure_pvoronoi --scenario g_mid --rep 0 --partition p_voronoi --validation random --fold 0 --grid 60`。
- 重复运行不会互相覆盖：输出目录已存在时 `asr_ademp.py` 自动改名为 `<name>-1`、`<name>-2`；`make_sim_figure.py` 会自动挑选最近一次且陆地覆盖完整的 run。
- 预测画在**外层留出像元**上（ADEMP 协议如此），海洋画灰；论文图用 `--sample 1.0`，陆地全覆盖、无留白（采样比例 <100% 时，未采样的陆地会留白）。
- 若改用 `--partition p_grid`，分块面板会显示规则方格（那张运行数据就是这么分的），属正常但示意性较差。
- 缺输入文件时脚本会直接报错并提示第一步的命令。
