# 下一步提案：单独检验沿路径地形净空输入

**2026-10-06 执行更新：**用户已要求直接做完整 LOS 主实验；缓存、secondU 输入及完整 RMSE 训练入口已实现，见[当前运行说明](los-main-experiment.md)。下文保留为历史提案，其中待确认、前置消融、配对 zeros、边界诊断和哈希步骤不再适用于本轮执行。

日期：2026-10-05。状态：已完成本地分析原型和八图诊断，**尚未加入训练/推理模型，也未执行正式实验**。当前用户请求是分析产物和设计下一步；下文是供确认的实现规格。[三对seed审查](tx-xyz-replication-review.md)

更新：用户下载的服务器诊断和计时已核验，5090D v2平均8.036ms/图；本地八图GPU/FP64及64项D4检查通过，推荐一次性缓存F进入受控实现阶段。此结果没有提供净空训练成绩，详见[服务器产物审查](visibility-server-review.md)。

进一步更新：服务器八图及64项D4扩展验收也已下载核验，最大GPU/FP64差2.2993e−5m、D4差3.2425e−5m，原型数值验收完成。无需再重复第5节原型检查；实际下一步是确认并落实缓存及模型消融。

## 1. 问题 → 依据 → 原理

**针对问题：**tx_xyz在三对配对seed中只有两次改善，平均−0.003677dB、SD0.009080dB；固定01917约33px提前保留。端点dx/dy/dz描述“RX相对TX在哪里”，但没有显式汇总两点间地形。不能据此证明端点表示是唯一瓶颈，也不能断言CNN不能自行学习遮挡。

**项目内证据：**八张固定缓存的近似可见性变化带占8.624%有效像素，却包含zeros三次平均SSE的53.662%；既定01917截面代理转变列142接近GT110dB交点列143。它只覆盖28.508%GT边界像素，表明只能解释一部分高变化结构。这些是后验诊断、非训练收益，也不构成独立验证。

**文献支持：**Jaensch、Caire、Demir，*Radio Map Estimation — An Open Dataset with Directive Transmitter Antennas and Initial Experiments*，2024预印本，§3.2.3提出可见性及最低可见高度输入，Table5的RadioUNet Basic无LoS归一化RMSE为0.0713，binary为0.0700、连续表示为0.0686。[原文](https://arxiv.org/html/2402.00878v1#S3.SS2.SSS3) · [作者输入实现](https://raw.githubusercontent.com/fabja19/RML/main/lib/torch_datasets.py)

该实验是柏林建筑/植被、3.7GHz、定向TX、WirelessInSite、归一化目标；本项目是连续月球地形、5.775GHz、各向同性TX和有效性mask。论文的城市可见高度、天线锥和遮挡计算不是本提案净空定义，不将其当作逐项复现，也不迁移其误差数值或必然有效的结论。

**待验证研究假设：**提前汇总直线路径的净空，可能帮助secondU利用远处遮挡关系定位部分高变化区域。它提供的是现有输入的另一种计算表示，没有增加观测信息；反射/散射和RX插值仍可能主导某些误差，所以只作为可纠正的输入，不作为硬约束。

## 2. 首选输入：一个连续标量，不先组合多个新变量

对TX像素t、RX像素p，原高度以米为单位，分辨率1m/px：

\[
z_t=h(t)+3,\qquad z_p=h(p)+1,\qquad q(\alpha)=(1-\alpha)t+\alpha p.
\]

\[
C(p)=\min_{\alpha\in\mathcal A_p}\big[(1-\alpha)z_t+\alpha z_p-\widetilde h(q(\alpha))\big].
\]

\(\widetilde h\)为双线性插值地形，\(\mathcal A_p\)包含两端点，采用嵌套采样：令D为水平距离/像素间距，细采样区间数为\(2\max(1,\lceil D\rceil)\)，因此水平步长≤0.5m。TX自身的垂直路径也正常定义，平地净空为1m；不存在除零。

C<0表示这个采样近似中的地形穿过直线，C>0表示未穿过。因为包含RX端点，正净空最多1m，不将C当成Fresnel净空或物理损耗。对模型拟提供：

\[
F(p)=\frac{\operatorname{asinh}(C(p)/(1\,\mathrm m))}{\operatorname{asinh}((496\,\mathrm m)/(1\,\mathrm m))}.
\]

496m来自训练metadata固定高度范围，1m来自网格单位，两者不在val上调参。asinh保留符号、缓和大遮挡值，同时不把米级差异简单除以496压成接近零；这是一项待验证编码选择，不是论文的原公式。仅以训练可用的归一化高度反解米制输入，延续原loader的裁剪；不读取target或mask构造F。

| 表征候选 | 优点 | 局限 | 次序 |
| --- | --- | --- | --- |
| 连续净空F | 一个通道保留正负和遮挡程度，避免额外手工传播模型 | 地形插值/采样误差；正值受RX端点限制；不能描述完整多径 | **首选** |
| 二值可见性V | 简单，直接判断代理遮挡 | 丢失幅度，零附近硬阈值敏感 | 作为解释性对照，首轮不叠加 |
| 最低可见高度/视线角 | 更接近文献连续编码 | 对近TX采样和投影需额外定义/稳定性检查 | F不适合时另立方案 |
| Fresnel/刀刃损耗、FSPL残差 | 可加入更强传播形式 | 与模拟机制、近场/多径等匹配问题更多 | 当前不采用 |

metadata记载原模拟未包含绕射、包含LOS/反射/散射/折射，RX来自地形mesh平移1m后的三角形重心并插值至网格；本次无法取得原mesh和完整生成代码。因此不能称本代理为精确Sionna LOS，也不能对其套用刀刃绕射损耗来制造额外物理目标。依据是本地metadata；本轮官方网页及源页面取回失败，未声称重新核验在线生成实现。

## 3. 实现前检查与成本

已有原型：[CPU参考及诊断](analysis/probe_terrain_visibility.py)、[CUDA原型](analysis/benchmark_visibility_cuda.py)。合成平地/山脊/高度平移/D4检查通过；在八个真实高度图中粗细采样符号差异共61个有效像素。CUDA在一张真实图与FP64参考最大误差1.3045e−5m，低于预先声明的0.001m实现容限。这些检查没有验证mesh或反射路径。

实施检查状态与后续事项：

1. 已完成本地及服务器全部八图GPU数值与D4检查，近零符号差异为0。正式特征固定FP32计算并避开AMP，0.001m容限只用于实现误差，不是物理容限。若用于展示二值V，明确零接触及1e−4m数值容差；正式输入F保持连续，不用V裁剪。
2. 固定或核查双线性插值与原mesh的差异。在训练集按预先固定ID抽样检查高度/TX对齐、统计范围和数值收敛；不根据val交点改TX/RX高度、阈值或采样。需要进一步迭代时先说明变更，不把后验筛选包装成预注册。
3. 服务器原型计时已完成，完整检查轮均值0.008271s/图。后续只在真实缓存构造/读取及模型训练中测量总成本，不重复原型计时；batch1八图结果不是全流程实测。
4. 选择输入构造方式：优先对每个terrain+TX缓存F，训练时与height/TX/GT/mask同步D4增强；标量F在已定义几何下应满足D4等变。缓存记录metadata/输入/代码哈希，不改变优先加载的原数据loader；训练/评估和提交使用同一定义。缓存不能使用GT/mask。

一通道FP32缓存估算：train约6.74GB、val约0.61GB、test约1.30GB（十进制GB）；可用mmap按index逐样本读取，不能按terrain只缓存一份，因为TX不同。是否采用缓存属于实施选择，需结合空间和吞吐。若在线生成，则两组都同样计算代理、只在zeros组把输出置零，以免只计模型forward遗漏候选特征开销。

每图复杂度O(HWL)，其中L是最远射线的采样数，256网格细采样最大约723点；分块空间O(P L+HW)。CPU参考粗+细4.46–10.21s/图，不适合每epochCPU重算。CUDA原型batch1在本地4060Laptop每图约0.041–0.054s，包含坐标准备/输入传GPU/计算/同步，但不含IO、F缩放、模型和训练；不能用该小样本推算正式训练或宣称满足比赛限制。使用项目已有NumPy/PyTorch，无新增依赖。

## 4. 下一项受控训练规格（待确认与实现）

**只向secondU提供F；firstU保持原输入并冻结，grad-weight=0。**不默认继承xyz，也不从这轮任何优胜checkpoint继续训练。

保持现有三列扩展形状，新增模式语义为 **[F,0,0]**，与 **[0,0,0]** 对照。两组共享原λ=0全模型baseline初始化和零填充，参数数目相同，沿用原两阶段输出形式；不同时增加显式残差、定位损失、网络宽度、LR或预算。使用三列而非重建单列网络，目的是复用已验收的迁移和现有zeros结构，控制模型形状；不将两列零输入声称为额外物理信息。

需要的最小实现边界：

- 特征代码独立成模块，保存算法版本、单位、采样/插值/FP32策略及缓存指纹。`radiounet.py`仅扩展mode与注入，保留上游RadioUNet源码。
- train/evaluate共用定义，checkpoint自动恢复mode/config；旧none/zeros/xyz行为保持，init-from允许精确零填充迁移，resume拒绝不匹配模式/config。
- 单独的受控入口从真实baseline继承设置；默认只打印计划，execute只写新目录。固定firstU哈希、来源记录、官方mask指标和既有边界诊断。
- 初始forward与对照在声明精度容限内一致，新增F列有有限梯度，firstU不变，保存加载往返一致。推理只用测试可用输入，输出完整256×256。

第一对使用seed0、各150epochs、与目前同一batch/优化器/调度/AMP/增强、相同原baseline。首先看完整val masked RMSE和t10r3边界SSE；两者都下降才触发seed1/2的配对复核。正式结论仍需报告全部三对Δ/SD、terrain集中度、multi/single和近缺测代价。

预定停止条件：初步整体或主边界变差则不立即叠加更多几何输入；复核任一seed反转、平均收益明显依赖一个terrain，或总体变好但预定关键区域持续明显退化，均不采纳为稳定默认。最后一种情况报告权衡，不临时换指标。发现收益后先完成同一开发条件复核，再进行不同firstU或独立测试的泛化；不把反复开发val上的改善当独立测试证据。

若净空也没有稳定帮助，再分别考虑“让firstU使用路径表征”或“允许输出位移修正”；前者要改变训练起点和阶段，后者要处理幅值/缺失区域及采样，均另立受控协议。当前不存在其优于本路线的本项目验证。

## 5. 可执行的服务器诊断

在提交并同步本轮材料、启用已确认Conda环境后运行，新输出目录必须不存在：

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch main
git pull --ff-only
python docs/analysis/probe_terrain_visibility.py --source runs --metadata ~/liuanda/LunarRM/metadata.json --out results/visibility_probe_20261005
python docs/analysis/benchmark_visibility_cuda.py --source runs --metadata ~/liuanda/LunarRM/metadata.json --out results/visibility_cuda_20261005
```

正式训练模式/入口尚未实现，以上命令只作已有原型诊断和服务器计时。本轮没有随意改动模型或启动新训练。实施得到确认并完成验收后，再交付实际可执行的配对训练命令和时间估计。
