# 三维 TX 相对坐标：seed 0 产物审查与下一步

日期：2026-10-04。来源：用户下载的 `D:/ssh_download/tx_xyz_v1_seed0`。本地只读取 checkpoint、CSV、JSON 和八图缓存，未训练或重新推理，也未访问服务器。

**决定：保留 tx_xyz 为待复核候选，下一步运行原协议的 seed 1/2 配对实验；暂不叠加 LOS，不延长训练，不改变损失、归一化或冻结规则。** seed 0 相对同轮零通道对照小幅改善，符合原定“整体 masked RMSE 与主边界 SSE 均下降”的复核触发条件；但尚不能将其采纳为已确认的稳定方法，也没有解决已知宏观轮廓错位。

## 1. 比较对象及真实性核验

两组都从训练完成的 λ=0 **secondU 全模型 baseline** 初始化，新输入列补零，冻结同一 firstU；使用 masked MSE、grad-weight=0、seed 0、batch 16、Adam、1e-4 cosine 至 1e-6、clip=0、AMP、同步增强，再训练 150 epochs。这是额外微调；零通道对照控制相同微调预算和网络形状。[事先固定的协议](tx-xyz-ablation.md)

本地核验通过：

- 三组评估来自提交 `de5a4314efbc92bbf41a9d1cefdba21aff8b8a9d`，记录的 Git 工作区均为空；评估环境、代码指纹、实际 loader、metadata/index 和精度一致。Windows 部分本地文件为 CRLF，原始字节哈希与服务器不同；转为 LF 后以及直接读取该提交 Git blob 后，十个代码文件均与服务器记录的 SHA-256 相同，未把换行差异当作代码变更。
- 实际 loader 来自数据目录；本地 loader/metadata/train index/val index 的字节哈希与记录相同。完整大数组未逐字节哈希，不能仅据此证明完整数据内容身份。
- 两组 checkpoint args 仅 `out` 和 `second_features` 不同，无 resume；初始化文件 SHA-256 都是 `46c998a3254ee42946a59bbaae9c8596758d02cbdb9150f59b5c0aea8e3eb836`，与本地上一轮 λ=0 baseline 文件匹配。
- best/last 共四个 checkpoint 的 firstU 哈希均为 `941d54c33a6db587df7da5d1712f313a73ee3ff5e242ae1b689c3e868f32f2c2`。zeros 新增权重保持零，tx_xyz 三个输入均有非零权重；权重范数不能当作独立特征重要性或传播机制证据。
- 两份历史记录完整覆盖 epoch 0–149，best 确为按 `val/rmse_db_masked` 选择；best 文件 SHA-256 匹配评估 provenance，loss components 中 grad-weight 始终为 0。
- 三组各有 **2,330 样本、680 terrains、135,926,253 有效像素**，train/val terrain 不相交。流式复核 349,500 条区域记录，验证逐样本 SSE、区域/补集和近/远缺测分解、汇总 RMSE、comparison.csv 及 firstU 区域统计一致性。
- 每组八张缓存的 input、target、mask、firstU 预测及既定截面完全相同；缓存预测重算 masked RMSE 与记录相符。精度为同一评分 forward 的 CUDA autocast float16，未为示例重新推理。

证据与来源哈希见 [audit.json](analysis/tx_xyz_seed0_audit_20261004/audit.json)。

## 2. 主结果与收益拆分

| 组别 | masked RMSE / dB | 相对 zeros / dB | best epoch，零起始 |
| --- | ---: | ---: | ---: |
| 初始化 baseline，未追加微调 | 6.578612 | +0.008432 | 68，上一轮 |
| zeros，相同结构追加微调 | 6.570180 | 0 | 26 |
| tx_xyz，相同预算追加微调 | **6.557052** | **−0.013128** | 26 |

tx_xyz 相对 zeros 的 RMSE 降幅约 **0.1998%**，有效像素总 SSE 降幅 **0.3992%**；相对原 baseline 的 RMSE 降幅为 **0.021560 dB，0.3277%**。原 baseline→zeros 的 0.008432 dB 改善来自同预算零通道微调比较，不能把 baseline→tx_xyz 的全部改善都归于坐标。

这不是逐图 RMSE 的平均。主分数由跨样本有效像素 SSE/有效像素数后开方，区域重建与主评分的浮点差低于 1e−5 dB。该轮初始化与此前从 firstU checkpoint 训练 secondU 的梯度消融不同，不直接用两轮 Δ 数值排名方法优劣。

## 3. 改善在哪里，代价在哪里

下表均为候选相对 zeros；区域来自已固定、mask-safe 的 GT 梯度定义，不代表 LOS/NLOS。

| 区域 | zeros RMSE / dB | tx_xyz RMSE / dB | SSE 变化 |
| --- | ---: | ---: | ---: |
| 主边界，t=10 dB/px、r=3 px | 10.705568 | 10.691240 | **−0.2675%** |
| 主边界补集 | 3.351559 | 3.334759 | **−1.0000%** |
| 距缺测区域 ≤5 px 的有效像素 | 12.095997 | 12.114338 | **+0.3035%** |
| 远离缺测区域的有效像素 | 5.733640 | 5.713211 | **−0.7113%** |
| 主边界且靠近缺测 | 12.902510 | 12.919568 | **+0.2646%** |
| 主边界且远离缺测 | 9.952808 | 9.926117 | **−0.5356%** |

主边界 SSE 减少约 1,287 万 dB²，补集减少约 1,055 万 dB²。远离缺测区域减少约 2,890 万 dB²，靠近缺测区域增加约 548 万 dB²，两者净减少约 2,343 万 dB²；这两种区域划分分别分解同一总误差，不可把不同划分相加。

九种预定边界阈值/半径组合的 SSE 均下降，幅度为约 0.0115%–0.4098%，方向不依赖单个边界定义。但总体仍是小收益，靠近缺测区域出现代价；没有证据支持通过更改官方 mask 或补洞标签来消除这一代价。[全部区域](analysis/tx_xyz_seed0_audit_20261004/regions.csv)

## 4. 按 terrain 审查与统计边界

改善样本：**1223/2330（52.49%）**；改善 terrain：**371/680（54.56%）**。每次删去一个 terrain 并重新累计整体 masked RMSE，候选−对照 Δ 范围为 **[−0.014218, −0.010524] dB**，方向均不反转，当前符号不是由单个 terrain 决定。

multi 组（33 terrains、1683 TX 样本）的 RMSE 降低 0.014431 dB；single 组（647 terrains、647 样本）降低 0.009662 dB，两个组均有小收益。最大五个净收益 terrain 是 03305、03755、00536、05845、05026；最大损害包括 00247、00124 等，完整配对表保留全部 terrain，不排除任何样本。[按 terrain](analysis/tx_xyz_seed0_audit_20261004/paired_terrains.csv) · [按样本](analysis/tx_xyz_seed0_audit_20261004/paired_samples.csv) · [分组](analysis/tx_xyz_seed0_audit_20261004/groups.csv)

补充探索性 terrain cluster bootstrap：680 个 terrain 有放回重采样，每个 terrain 的所有 TX 一起保留；每次按重采样后的 SSE/像素数重新计算两组 RMSE 差。20,000 次，NumPy RNG seed=20261004，2.5/97.5 百分位为 **[−0.021560, −0.005378] dB**。

这只描述**已拟合、已在该 val 选模的两个 checkpoint**对经验 terrain 重采样的敏感性，不覆盖训练 seed、firstU、初始化分布和 checkpoint 选择不确定性，也未建模同一大父地形之间的相关性。不能当作训练收益的 seed 置信区间、独立测试推断、P 值或统计显著性证据。目前模型训练重复数仍是 **一对 seed**，像素、TX 和 150 个 epoch 都不是独立训练重复。

## 5. 曲线与固定示例：收益不等于轮廓修正

两组 best 都在 epoch 26，候选在 109/150 个对应 epoch 的 val RMSE 较低，末 30 epochs 的平均候选−对照 Δ 为 −0.007478 dB。这说明本次收益不只表现为两个不同 best epoch 的比较；它仍只来自同一次训练，不作为额外重复证据。

| 组别 | best RMSE / dB | epoch 149 RMSE / dB | best 时 train MSE | epoch 149 train MSE |
| --- | ---: | ---: | ---: | ---: |
| zeros | 6.570180 | 6.706285 | 0.000469190 | 0.000412811 |
| tx_xyz | 6.557052 | 6.696773 | 0.000470155 | 0.000412433 |

训练 MSE 持续下降，而 val 在 best 后回退，与进一步拟合训练集但未改善该 val 的现象一致；未隔离其学习率、正则化或数据成因。不建议加长至 300 epochs，也不在当前复核中同时改 LR 或预算。仍保留原 best 选择规则。[原始 epoch 差及训练 MSE](analysis/tx_xyz_seed0_audit_20261004/epoch_deltas.csv)

沿用上轮 **01917_32、第 128 行、列 80–195、110 dB 上升交点**，没有为本候选重新选截面或阈值：

- GT：列 142→143；firstU：109→110。
- 原 baseline secondU、zeros、tx_xyz：均为 109→110，约提前 **33 px**。
- 该图总体 masked RMSE：baseline 5.288345、zeros 5.461298、tx_xyz 5.519525 dB；候选在这张已知问题图上还略差。

八个固定示例中 tx_xyz 对 zeros 有六图改善、两图退化。已检查 `01917_32` 的原始零通道和坐标图：过大的高损耗区域及红色环带仍在。单个图/交点不是正式定位指标，也不能代表全部 terrain；结论是“当前证据未显示该已知宏观位移得到修正”，而非“全数据所有轮廓都没有变化”。[缓存示例 RMSE](analysis/tx_xyz_seed0_audit_20261004/fixed_examples.csv) · [固定截面逐点数据](analysis/tx_xyz_seed0_audit_20261004/fixed_profile.csv)

原始图：[zeros](/D:/ssh_download/tx_xyz_v1_seed0/seed0_zeros/boundary/01917_32_58.png) · [tx_xyz](/D:/ssh_download/tx_xyz_v1_seed0/seed0_tx_xyz/boundary/01917_32_58.png)。

## 6. 下一步：先完成已触发的配对复核

**问题→已有依据→原理：**本轮是 0.20% 的总体小收益，需要确认是否依赖训练随机性；项目已有的配对协议要求同一训练起点、同预算零通道控制及 terrain 分组。因此保持模型与数据处理不变，seed 1/2 均从原 λ=0 baseline 重新初始化，与 seed 0 形成三对比较。

| 方案 | 优点 | 代价/局限 | 本轮决定 |
| --- | --- | --- | --- |
| 原设置 seed 1/2 配对复核 | 直接检验当前可重复性，保持可解释的控制变量 | 约 6.9 小时；仍是同一 firstU、同一开发 val | **执行此下一步** |
| 立即叠加 LOS | 更直接提供中间路径几何，契合用户后续方向 | 混入新变量；净空代理定义/模拟对应及耗时尚需独立核验 | 复核后另立受控实验 |
| 同时改 LR、epoch 或 firstU 输入 | 可检验训练策略或粗预测表征瓶颈 | 当前无法分离收益来源；需新协议和授权 | 暂不修改 |

文献支持只用于方向背景：Jaensch 等 [原论文 §3.2、Table 5](https://arxiv.org/html/2402.00878v1)在城市建筑/植被、3.7 GHz 定向 TX 数据上比较相对几何和 LOS 输入；不同于月球连续地形、各向同性 TX 和官方有效性 mask。该论文不能证明本次 33 px 偏移由遮挡缺失造成，也不能证明其 LOS 输入收益会迁移。**研究假设**仍是“端点几何之后，路径遮挡几何可能提供额外帮助”；目前未在本项目验证。

复核保持：同一 baseline SHA-256、相同 firstU、grad-weight=0、150 epochs、相同优化器/精度/增强、zeros vs tx_xyz；仅改变配对 seed，分别使用新输出目录。**不能从本轮 tx_xyz 或 zeros best 再续训来充当 seed 1/2 复核，也不能换成每个 seed 各自的 λ=0 baseline。**这两项会改变训练起点。

两轮完成后，报告三组配对 Δ 的均值、运行间 SD 和全部原始值；按 terrain 复核主 RMSE、主边界 SSE 和近缺测代价。若 seed 1/2 同时支持主 RMSE 和主边界 SSE 改善，可将几何组确认为这个固定 firstU 微调条件下的小幅候选，再单独设计 LOS 对照。若方向明显不一致或不能复现，保留负结果，暂不声称稳定增益。即使复核成功，也不升级为“修正宏观位置机制”或“独立测试泛化已证实”。

这些 seed 改变数据顺序、增强等训练随机性；共享权重从同一完整 baseline 复制、新列均为零，不覆盖新 firstU 训练及全新初始化分布。完整两阶段重新训练属于后续独立问题。

## 7. 交付、提交与服务器命令

本轮新增本报告、只读分析脚本及机器可读核验结果，更新文档索引；没有改模型、损失、超参数、训练入口或原始下载产物。本地审查脚本已运行并通过上述核验。重算（新目录，通常数十秒）：

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
D:\Anaconda\python.exe docs/analysis/analyze_tx_xyz.py --source D:/ssh_download/tx_xyz_v1_seed0 --data-root D:/codes/LunarRadiomap/LunarRM --out docs/analysis/tx_xyz_seed0_audit_repeat
```

复杂度：区域 CSV 按行流式读取，约 O(RNK)；terrain 分组构造 O(NT)，bootstrap O(BT)，checkpoint/来源指纹按总文件字节线性读取。R=3、N=2330、K=25、T=680、B=20000；空间主要为 O(RNK + B + 256T + 单 checkpoint)，加固定八图缓存。不存放 B×N 大矩阵，不增加依赖。

尚未 commit/push。`docs/` 被已有 `.gitignore` 设置忽略，显式只加入本轮材料：

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git add -f docs/README.md docs/tx-xyz-seed0-review.md docs/analysis/analyze_tx_xyz.py docs/analysis/tx_xyz_seed0_audit_20261004
git diff --cached --stat
git commit -m "Audit seed-0 TX geometry and plan paired replication"
git push origin main
```

报告提交可能改变 Git commit，但不改变训练代码；跨轮比较时按代码/数据/初始化指纹判断，仅有文档提交差异不算模型算法变化。

服务器路径依据本轮下载产物记录；若路径未变，在用户已确认的 Conda 环境中执行：

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch main
git pull --ff-only
```

原 runner 已支持 seed 参数，无需修改训练代码。先打印 seed 1 的只读计划，可核对所有参数和 baseline 指纹：

```bash
python ablate_coordinates.py --data-root ~/liuanda/LunarRM --baseline-ckpt runs/boundary_ablation_v1/seed0_grad0/radiownet_58_masked_secondU_best.pt --out-root runs/tx_xyz_v1_seed1 --seed 1
```

正式运行：

```bash
python ablate_coordinates.py --data-root ~/liuanda/LunarRM --baseline-ckpt runs/boundary_ablation_v1/seed0_grad0/radiownet_58_masked_secondU_best.pt --out-root runs/tx_xyz_v1_seed1 --seed 1 --execute
python ablate_coordinates.py --data-root ~/liuanda/LunarRM --baseline-ckpt runs/boundary_ablation_v1/seed0_grad0/radiownet_58_masked_secondU_best.pt --out-root runs/tx_xyz_v1_seed2 --seed 2 --execute
```

每个 out-root 必须不存在；第一轮失败时先定位错误，不继续执行下一行或重用目录强行覆盖。下载记录中 seed 0 两组训练分别用时 6,154/6,169 秒，总流程含三次评估约 **3.43 小时**；seed 1/2 合计预计约 **6.9 小时**，以实际服务器为准。这是历史产物推算，不是实时状态。

完成后下载两个目录，保留 manifest、comparison.csv、best/last checkpoint、history/loss/provenance、val JSON 及 boundary 完整目录。服务器同步与正式训练均由用户执行。
