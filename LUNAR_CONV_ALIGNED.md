# 无注意力的 MaxViT 对齐消融

架构名：`lunar_conv_aligned`；代码：[lunar_conv.py](lunar_conv.py)。
本轮先运行 **seed 0**，使用独立输出目录；已有 RadioWNet、MaxViT 及其 checkpoint 保持兼容。
这里把 LOS＋distance 任务配置落实为一个与 MaxViT 对齐的单 U 形对照网络，
不是在原 RadioWNet secondU 中继续添加模块，也不继承冻结的 firstU。

## 本轮改变什么

| 项目 | 原 LOS＋distance RadioWNet | 本轮无注意力对照 |
| --- | --- | --- |
| 阶段 | firstU → secondU，第二阶段冻结 firstU | 单阶段，整个编码器和解码器共同训练 |
| 特征入口 | firstU 输入 H/TX/F；secondU 再接收 rho | H/TX/F/rho 从网络入口一起编码 |
| 编码器 | RadioWNet 的普通卷积结构 | MaxViT 的 MBConv 去 SE，保留两次通道 MLP 残差 |
| 尺度与通道 | RadioWNet 原规格 | stem 64；四组通道 64/128/256/512，块数 2/2/5/2 |
| 解码器 | RadioWNet 原规格 | 复用 MaxViT 的五级卷积解码器、skip 和全分辨率分支 |
| 初始化 | 从历史 LOS firstU 初始化 secondU | 从头训练，共享权重与同 seed MaxViT 初值相同 |
| 优化器 | Adam，无 weight decay | 与 MaxViT 对照相同的 AdamW，weight decay 0.05 |

相对于 **MaxViT＋AdamW 基线**，本轮只移除两个机制：

- 每个块中的 window/grid self-attention 更新，包括其专用 LayerNorm、Q/K/V、输出投影和相对位置偏置。
- MBConv 中的 SE 通道注意力，用 Identity 代替；保留其余卷积、BatchNorm、GELU 和残差投影。

保留原来两次注意力之后的 **LayerNorm → Linear → GELU → Linear → Dropout**
及各自的残差、stochastic depth。这里 Linear 只作用于单个像素的通道维度，
不计算像素间的注意力。去掉注意力后，分窗与网格重排也不再必要。

输入物理定义、已有 LOS 缓存、log-distance、D4 同步增强、masked MSE、归一化、
ReLU 直接预测头、固定验证集和有效像素 RMSE 均与现有 MaxViT 相同。
distance **已经是对数表征**，本轮不更改公式或量纲。

## 数学形式与实现

令 M 表示去掉 SE 的 MBConv 分支，P 表示同一块的残差投影，D 表示 stochastic depth：

\[
u=P(x)+D(M(x)),\qquad
v=u+D(f_1(\operatorname{LN}(u))),\qquad
y=v+D(f_2(\operatorname{LN}(v))).
\]

其中每个 f 是逐位置的 `Linear(C,4C) → GELU → Linear(4C,C) → Dropout(0)`。
不采用 `x + Identity(x)` 来“关闭”注意力，因为那会把输入加倍。
空间信息通过卷积、逐级降采样和 U 形 skip/解码器传播。

编码器保留 11 个 MBConv、22 个通道 MLP；输出尺度为 128/64/32/16/8。
解码器恢复到 16/32/64/128/256，最后与 H/TX/F/rho 拼接后预测完整地图。
参数量为 **26,307,429**，原 MaxViT 为 **35,318,189**，减少约 **25.5%**。
没有为补齐参数量而增加卷积深度或宽度。

构造时先按原 MaxViT 的随机初始化顺序建立模型，再移除注意力和 SE；
因此保留参数及构造后的 CPU RNG 状态与同 seed MaxViT 一致。
这不使用任何预训练权重。被移除参数不会进入优化器、前向或反向图。
两者的 stochastic depth 调用次数不同，训练随机轨迹和预测不保证逐步相同。

固定卷积核及扩张比，编码器时间复杂度为
\(O(B\sum_s L_sN_sC_s^2)\)，训练激活空间为
\(O(B\sum_s L_sN_sC_s)\)，参数及优化器状态空间为
\(O(\sum_s L_sC_s^2)\)，另加两模型共有的解码器开销。
B 为 batch，N 为该尺度像素数，C 为通道数，L 为块数；前向不构造注意力矩阵。
正式耗时仍须测量，且计入 LOS 特征构造。

没有新增依赖，使用项目已有 torch/torchvision。模型配置与架构名写入 checkpoint，
只接受本架构匹配的续训／评估，拒绝直接加载 RadioWNet 或 MaxViT 权重。
训练器默认优化器仍是 Adam；正式命令必须显式传入下方的 AdamW 参数。

## 证据与判读边界

- **实现依据**：以项目已使用的 torchvision MaxViT 为参照，保留其 MBConv、
  MLP、归一化、激活及对应 stochastic depth 概率；此处是有意删除注意力的消融，
  不是完整复现原 MaxViT，也没有论文证明本消融能提升月球传播预测精度。
- **项目内证据**：两条原基线同时存在阶段、冻结、通道规格、解码器、初始化及优化器差异，
  所以历史分差不能直接视为注意力收益。
- **待验证研究假设**：对齐上述非注意力因素后，一个完全可训练的卷积／通道 MLP 网络
  能解释原 MaxViT 优势中的一部分，并成为更清晰的后续模块研究起点。

比较新模型与原 LOS＋distance，衡量的是整套对齐配置的效果，不能分离冻结 firstU、
优化器或模型容量的单独贡献。比较新模型与 MaxViT＋AdamW，衡量的是移除
**self-attention＋SE 这一组合**的影响，且伴随参数量变化，不能声称证明了纯自注意力的净贡献。

训练后对齐 seed 0 的完整 history、best/last、验证逐样本 SSE/count 与推理成本。
主分数使用有效像素 SSE 总和除以有效像素总数后开方，不能平均逐图 RMSE。
若新模型接近 MaxViT，可优先在无注意力基线上研究功能明确的传播模块；
若差距仍大，后续才分别研究 SE 与空间注意力。这里只是决策分支，不自动追加训练。
单 seed 仅用于本轮筛选；跨 seed 或泛化结论需要后续配对实验。

## 本地验证记录

辅助脚本、日志和报告保存在仓库外的
`D:/codes/LunarRadiomap/local_experiments/conv_alignment_20261019/`，不提交 Git。
已通过逐位置 MLP 数学／梯度核对、移除分区与保留分区实现的一致性、模型图检查、
共享初始化、batch 16 AMP 梯度，以及下载的旧 RadioWNet／MaxViT checkpoint 严格加载。
完整网络在 FP32 下与保留分区、将注意力更新置零且去 SE 的参考实现输出相同。
AdamW 衰减组有 26,216,948 个参数，不衰减组有 90,481 个参数。

梯度诊断发现，两种架构在随机初始化时都有极小的深层梯度，FP16 下可下溢为零，
FP32 下链路连通。保持正式 AMP／初始化配置的本地 batch 16 重复样本检查中，
第 31、32 次更新前，各尺度监测的卷积、MLP、解码器权重均获得非零有限梯度。
没有为此改变模型初始化、损失缩放或训练精度。该观察只说明这次短程检查未持续阻断
深层更新，不能证明正式训练始终没有下溢，亦不能将早期零梯度误称为参数冻结。

已通过真实 CLI 的两轮小样本训练／续训、checkpoint 自动识别评估和现场 LOS 推理计时。
检查保持 cosine 总长度 150，验证优化器和 scaler 恢复、全部参数可训练以及有效像素
汇总指标；正式命令的训练参数已逐项对照下载的 MaxViT＋AdamW provenance。
具体结果见 `verification.json`、`diagnose_gradients.json` 和 `integration.json`。
小样本检查不代表正式训练效果；上游 torchvision 的 meshgrid Future-interface Warning
不影响上述检查，不为消除警告而修改上游实现。

## 本地提交与推送

在本地 PowerShell 检查本次五个文件后提交；不包含忽略的 docs 或辅助实验：

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git status
git diff --stat
git add lunar_conv.py model_factory.py train.py README.md LUNAR_CONV_ALIGNED.md
git diff --cached --stat
git commit -m "Add attention-free MaxViT alignment control"
git push -u origin research/conv-alignment
```

## SSH 服务器同步与正式训练

先由用户确认并激活现有训练环境。以下命令供用户执行，Codex 未在服务器运行。
LOS 缓存复用 `results/los_cache_v1`。与 MaxViT＋AdamW 对齐：batch 16、150 epochs、
lr 1e-4、cosine 至 1e-6、AMP、seed 0；不使用先前 secondU 的 30 轮截断。
总耗时尚未测量，可先按历史 MaxViT 约 3–4 小时的量级安排，并根据实际每轮耗时更新。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git fetch origin
git switch research/conv-alignment
git pull --ff-only

python train.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --arch lunar_conv_aligned --phase single --band 58 --loss masked \
  --optimizer adamw --weight-decay 0.05 \
  --epochs 150 --batch-size 16 --lr 1e-4 \
  --lr-sched cosine --lr-min 1e-6 \
  --num-workers 8 --device cuda --amp --seed 0 \
  --init-from none --resume auto \
  --panels 6 --panels-every 5 --plots-every 5 \
  --out runs/lunar_conv_aligned_v1/seed0 \
  --run-name lunar_conv_aligned_58_masked_single_los_distance_adamw

python evaluate.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --ckpt runs/lunar_conv_aligned_v1/seed0/lunar_conv_aligned_58_masked_single_los_distance_adamw_best.pt \
  --split val --batch-size 16 --num-workers 8 --device cuda --amp \
  --out results/lunar_conv_aligned_v1/seed0/val.json

python benchmark_inference.py \
  --data-root ~/liuanda/LunarRM \
  --ckpt runs/lunar_conv_aligned_v1/seed0/lunar_conv_aligned_58_masked_single_los_distance_adamw_best.pt \
  --split val --device cuda --samples 16 --warmup 3 --amp \
  --out results/lunar_conv_aligned_v1/seed0/inference.json
```

重复同一训练命令仅从本次输出目录的 last 续训；best/last 都保留。
现有训练器恢复优化器、学习率和 AMP scaler，但不保证中断前后的随机轨迹完全一致。
推理计时包含现场 LOS 特征生成；benchmark 拒绝覆盖已有输出，需要复测时使用新文件名。
