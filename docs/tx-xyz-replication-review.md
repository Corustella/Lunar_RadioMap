# TX 相对坐标三对 seed 复核：结论修正与下一步

日期：2026-10-05。读取 `D:/ssh_download/tx_xyz_v1_seed0`、`seed1`、`seed2`，仅作本地分析；未访问服务器、重新推理或运行新训练。

**决定：tx_xyz 未通过原协议的稳定增益复核，不将它设为后续默认基线。保留实现及全部结果，grad-weight 仍为 0；下一步优先独立检验沿 TX→RX 路径的地形净空表征。** 这修正了 seed 0 的初步判断，也不等于证明坐标永远无用。[上一轮审查](tx-xyz-seed0-review.md) · [下一项具体提案](next-step-visibility-proposal.md)

## 1. 配对主结果

两组都从同一个已训练完成的 λ=0 secondU 全模型 baseline 初始化、冻结 firstU，新列零初始化；150 epochs、相同 masked MSE、优化器、AMP、同步增强与官方 val 选模。比较对象是每个 seed 的同预算 zeros，不能把追加微调的收益都归于坐标。

| seed | zeros RMSE / dB | tx_xyz RMSE / dB | 配对 Δ / dB | 主边界 SSE 变化 | 近缺测 SSE 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 6.570180 | 6.557052 | −0.013128 | −0.2675% | +0.3035% |
| 1 | 6.553584 | 6.558563 | **+0.004979** | **+0.0724%** | −0.0205% |
| 2 | 6.567756 | 6.564873 | −0.002882 | −0.2200% | −1.7750% |
| 三对均值 | 6.563840 | 6.560163 | **−0.003677** | −0.1384% | −0.4973% |

配对 Δ 的运行间样本 SD 为 **0.009080 dB**（ddof=1，n=3），大于平均收益；平均相对 RMSE 变化约 −0.0559%。均值/SD 只是这三次运行的描述，不是置信区间或显著性判断。主指标仍为跨样本有效像素 SSE/数量开方，不是逐图 RMSE 平均；表中跨 seed 均值也不是集成预测的 RMSE。

原 baseline 分数为 6.578612 dB；所有六个微调 checkpoint 均优于它，包括全部 zeros。因此“候选胜过原 baseline”不能替代“坐标胜过同预算对照”。原计划要求 seed 1/2 同时支持整体 masked RMSE 和主边界 SSE 改善，seed 1 两项都退化，复核未通过。[原计划 §6](tx-xyz-seed0-review.md#6-下一步先完成已触发的配对复核)

## 2. 来源与可比性核验

完整核验通过，差异不能简单归咎于协议跑错：

- 三份 manifest 均 complete、非 smoke；无 resume、grad-weight=0。相同 baseline SHA-256 `46c998a3254ee42946a59bbaae9c8596758d02cbdb9150f59b5c0aea8e3eb836`，相同 firstU SHA-256 `941d54c33a6db587df7da5d1712f313a73ee3ff5e242ae1b689c3e868f32f2c2`；本地原 baseline 及六组 best/last 共十二个 checkpoint 均验证。
- seed 0 提交为 `de5a431`，seed 1/2 为 `36adfbf`，后者仅追加上一轮分析材料。十个训练/评估代码文件的指纹一致，均与各自提交的 Git blob 及本地 LF 归一后的代码一致。记录的训练/评估工作区为空，环境、FP16 autocast 和实际数据目录 loader 一致。
- metadata、train/val index 与实际 loader 的本地字节哈希吻合；三份 baseline reference 指标和区域记录相同。完整大数组未逐字节哈希，不声称完成全部数据内容校验。
- 2330 个 val 样本、680 terrains、135,926,253 有效像素，train/val terrain 不相交。九次报告共 1,048,500 条区域记录经流式复核，SSE、区域补集、近/远缺测分解、主 RMSE、comparison.csv 一致。
- 六组完整历史均为 epoch 0–149，best 与各自最小 `val/rmse_db_masked` 相符，评估 checkpoint 指纹一致。训练设置除 seed/out/feature mode 外相同；各对比较仅 out/feature mode 不同。
- 八张既定示例的 input/GT/mask/firstU 跨全部六组一致，缓存评分可复核。没有用不同推理重新生成示例。

[机器可读总审查](analysis/tx_xyz_replication_20261005/replication_audit.json) · [配对 seed 表](analysis/tx_xyz_replication_20261005/paired_seeds.csv) · [各 seed 区域](analysis/tx_xyz_replication_20261005/regions_by_seed.csv)

## 3. 地形分布：seed 2 的小收益明显集中

| seed | 改善 terrain / 680 | 改善样本 / 2330 | 删去单个 terrain 后的整体 Δ 范围 / dB |
| --- | ---: | ---: | ---: |
| 0 | 371 | 1223 | [−0.014218, −0.010524] |
| 1 | 293 | 1059 | [+0.004139, +0.005618] |
| 2 | 309 | 1118 | [−0.006148, +0.009365] |

seed 1 删除任何单个 terrain 后仍退化；seed 2 的方向会因一个 terrain 改变：**02383** 的 SSE 减少 21,292,515 dB²，而该 seed 全集净减少仅 5,144,811 dB²。删除它后 Δ 为 **+0.009365 dB**。该 terrain 在 seed 0/1 反而略退化，不能把它解读为稳定改善的区域。

仅 **45/680** terrains 在三对里都改善，**79/680** 在三对里都退化，按平均 SSE 差改善的为 **316/680**。删除 02383 后，三对各自的整体 Δ 均值变成约 **+0.000161 dB**。这用于说明收益集中度，不用于挑选排除地形；正式主表始终保留全体 val。[跨 seed terrain 表](analysis/tx_xyz_replication_20261005/terrain_seed_consistency.csv)

分组也不支持均匀改善：

| seed | multi 组 ΔRMSE / dB | single 组 ΔRMSE / dB |
| --- | ---: | ---: |
| 0 | −0.014431 | −0.009662 |
| 1 | +0.005277 | +0.004184 |
| 2 | −0.008750 | +0.012784 |

mask-near 代价也不稳定：seed 0 在近缺测退化、远缺测改善；seed 2 则近缺测改善、远缺测退化（SSE +0.6871%）。不能继承 seed 0 的“收益主要来自远缺测”作为跨 seed 机制结论。主边界 t=10、r=3 的像素占比仍为 30.89%，各组贡献约 82% SSE，剩余高变化区域误差仍是主要问题。

各 seed 子目录保留 terrain bootstrap，含义仍只是已拟合、val 选模模型的经验 terrain 重采样敏感性，不覆盖训练不确定性、选模或父地形相关性。本报告不把它当成 seed 置信区间、P 值或独立测试证据。

## 4. 曲线与固定问题：没有得到轮廓修正证据

| seed | zeros best epoch | xyz best epoch | zeros epoch 149 RMSE | xyz epoch 149 RMSE |
| --- | ---: | ---: | ---: | ---: |
| 0 | 26 | 26 | 6.706285 | 6.696773 |
| 1 | 15 | 15 | 6.699943 | 6.686186 |
| 2 | 34 | 26 | 6.706883 | 6.706532 |

epoch 为零起始。xyz 在对应 epoch 上较低的次数分别为 109、101、78；末期有时仍相对较低，但六组末期分数都明显差于各自 best。不能看到这次结果后改用最后 epoch、最后 30 轮平均或加长训练来“挽救”失败的预定比较。学习率、正则化、冻结粗预测可能值得单独检验，当前没有隔离其成因。

固定 `01917_32` 第128行、列80–195、110 dB上升交点：GT为142→143；firstU为109→110；**全部六个 secondU 都为109→110**，仍约提前33 px。该截面沿用之前的定义，没有为本轮重新挑选阈值。

八图误差分解满足 `mean_s((p_s-y)^2) = (mean_s(p_s)-y)^2 + mean_s((p_s-mean_s(p_s))^2)`。403,785个有效像素中，平均预测保留的SSE比例为zeros **99.449%**、xyz **99.435%**；主边界分别为 **99.543%**、**99.541%**。这些固定例子的误差高度共通，增加seed或对这些预测平均不能消除大部分问题；这不等于完整val的集成分数，也不是总体偏差/方差估计。[八图分解](analysis/tx_xyz_replication_20261005/replication_audit.json) · [逐图表](analysis/tx_xyz_replication_20261005/cached_examples.csv)

## 5. 新的只读诊断：沿路径净空值得检验，但仍是假设

在读取GT/mask之前，仅由高度图、TX位置、metadata的TX3m/RX1m，构造直线路径并以双线性地形、间隔≤0.5m的离散点计算最小垂直净空。正净空表示该近似模型没有穿过地形，负值表示被地形阻挡；它不是原模拟器精确LOS，也不预测多径路径损耗。

| 八图区域 | 有效像素 | zeros：三次模型平均MSE开方 / dB |
| --- | ---: | ---: |
| 代理可见 | 354,635 | 4.582700 |
| 代理遮挡 | 49,150 | 8.542123 |
| 代理可见性变化的r=3带 | 34,822 | **13.039976** |

该代理变化带占 **8.624%** 有效像素，却贡献zeros八图平均SSE的 **53.662%**；与GT主边界重叠的部分占GT边界像素 **28.508%**。提示两者相关，但大量GT边界不在代理变化带内，不能将全部真实边界视为遮挡。

既定01917截面中，代理可见性转变的右端列为 **142、175**；GT110dB上升交点右端列143，现有网络为110。第一个转变接近已知错误位置的真实交点，为路径几何假设提供一个项目内线索；单个截面仍不足以证明错误成因或全局泛化。

≤1m和≤0.5m采样比较，在403,785个有效像素中有 **61** 个二值可见性差异（约0.0151%）。这支持这八图离散采样符号的初步稳定性，不能证明连续极值、双线性地形、实际mesh或RX插值完全准确。合成平地、阻挡山脊、常数高度平移及八种旋转/翻转检查通过。官方有效性mask仅参与后验诊断，未作为可见性输入或标签。

CPU参考的粗+细计算为4.46–10.21s/图。随后只对输入构造进行本地CUDA原型计时：RTX4060 Laptop、FP32、batch1、每图预热后3次，八图均值 **0.04639s/图**；包含CPU坐标准备、输入传GPU、ray sampling、插值、min及同步，不含文件IO、结果回CPU、asinh缩放、模型forward和训练。在一张真实图与FP64参考的最大误差为 **1.3045e−5m**，可见性符号无差异。未计时服务器，也未检查其全数据吞吐或比赛总时限。

[几何诊断及配置](analysis/visibility_probe_20261005/visibility_probe.json) · [逐图诊断](analysis/visibility_probe_20261005/examples.csv) · [GPU计时](analysis/visibility_cuda_20261005/cuda_benchmark.json)

## 6. 下一步行动顺序

1. 结束当前xyz验证：记录小且不稳定的结果，保留代码，不追加seed 3/4来寻找更好的结果，不继承xyz best作为新基线。
2. 优先完成净空代理的实现规格、精度/增强/缓存验收以及服务器特征构造计时，见[提案](next-step-visibility-proposal.md)。这是从端点几何转向整条路径的表征检验。
3. 若代理可用且耗时可接受，从原λ=0全模型baseline做 **zeros vs [净空,0,0]** 配对，不同时加xyz、新损失、LR或firstU改动。先seed0，再按预定门槛复核seed1/2；正式训练入口尚未实现。
4. 若没有可复现收益，停止堆叠几何输入。后续单独检验firstU表征或输出定位方式；这些是新假设，当前不改模型结构。

这条路线针对的是“高变化区域误差占比大、已有端点表征未稳定改善、固定宏观位移保留”。文献背景来自Jaensch等城市无线电地图中可见性输入消融；本项目证据来自上述真实产物与近似几何诊断；“净空输入能降低月球masked RMSE”仍是待验证研究假设。详细来源和迁移限制在提案中列出。

## 7. 本地验证及可执行命令

本轮只修改分析脚本/文档并保存新审查结果，不改变模型、训练入口、数据loader、超参数或下载产物。三对完整审查、八图几何诊断、CUDA原型已在本地实际运行。来源指纹保存于各JSON。单对脚本支持manifest中的seed，旧seed0证据目录保留原记录，不覆盖历史结果。

分析复杂度：区域核验O(SRNK)，terrain构造O(SNT)，经验重采样O(SBT)；S=3、R=3组、N=2330、K=25、T=680、B=20000。checkpoint/来源指纹按总文件字节线性读取；几何计算每图O(HWL)，分块空间O(P L+HW)。只用现有NumPy/PyTorch，未安装依赖。

本地提交（尚未commit/push；仅加入本轮材料，不加入用户现有.gitignore改动）：

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git add -f docs/README.md docs/tx-xyz-replication-review.md docs/next-step-visibility-proposal.md docs/analysis/analyze_tx_xyz.py docs/analysis/analyze_tx_xyz_replication.py docs/analysis/probe_terrain_visibility.py docs/analysis/benchmark_visibility_cuda.py docs/analysis/tx_xyz_replication_20261005 docs/analysis/visibility_probe_20261005 docs/analysis/visibility_cuda_20261005
git diff --cached --stat
git commit -m "Review TX geometry replication and propose terrain clearance"
git push origin main
```

服务器在用户已确认的Conda环境中同步并重算审查（全部输出目录须为新目录；通常数分钟）：

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch main
git pull --ff-only
python docs/analysis/analyze_tx_xyz_replication.py --source runs --data-root ~/liuanda/LunarRM --out results/tx_xyz_replication_audit_20261005
python docs/analysis/probe_terrain_visibility.py --source runs --metadata ~/liuanda/LunarRM/metadata.json --out results/visibility_probe_20261005
python docs/analysis/benchmark_visibility_cuda.py --source runs --metadata ~/liuanda/LunarRM/metadata.json --out results/visibility_cuda_20261005
```

以上为已存在、可执行的审查/计时命令；不执行训练。原始产物包含相同服务器目录，路径约定来自下载记录而非实时观察。新净空训练消融需要下一次实现与确认，不能用当前 `ablate_coordinates.py` 冒充净空实验。
