# C 基线上的 L/M 空间聚合实验

本轮只训练 **L、M 各 seed 0/1**，复用 C 的已有结果。
复用现有 `train.py`、`evaluate.py`、`benchmark_inference.py`，没有新训练入口或依赖。

| 组别 | 架构名 | 32×32 stage 第二个 MBConv 的空间卷积 | 参数量 |
| --- | --- | --- | ---: |
| C | `lunar_conv_aligned` | 原单支 DW3×3，d=1 | 26,307,429 |
| L | `lunar_conv_local3` | 三支 DW3×3，d=[1,1,1] | 26,316,645 |
| M | `lunar_conv_multiscale` | 三支 DW3×3，d=[1,2,3] | 26,316,645 |

## 修改范围和原理

只替换 `encoder.blocks[1].layers[1].layers.MBconv.layers.conv_b[0]`。
该位置为 stride 1，扩展特征为 512×32×32。对于扩展后的 U：

\[
Z=\sum_{j=1}^{3}\operatorname{DWConv}_{3\times3,d_j}(U;W_j).
\]

各分支零 padding 等于 dilation，保持空间尺寸；求和后执行原来的 BN/GELU。
分支内部没有新增归一化、激活、门控或 stochastic-depth 调用。
四组 `[2,2,5,2]` 深度、通道、两次逐位置 MLP、skip、decoder、直接预测头均保持 C。
输入仍为 H/TX/F，模型在同步增强后计算原定义的三维 log-distance。

先完整构造 C，保留原空间核 W1，新增 W2/W3 零初始化。因此同 seed 的共同权重、
缓冲区和初始函数与 C 一致；新增卷积构造使用 CPU RNG fork，不改变后续 CPU 随机状态。
这是从头训练，不加载 C 的已训练 best。所有参数参与训练。

两组均新增 `2*9*512=9,216` 个权重，DW 部分新增约 9.44M MAC/图。
对 N=HW、扩展通道 Ce，新增计算为 O(18BNCe)，局部激活空间仍为 O(BNCe)，
但多分支增加常数。端到端耗时、显存须实测，MAC 不等于实际加速比。

L 的线性三支可合并成普通 3×3；M 可合并成有 25 个独立空间偏移的稀疏 7×7。
两者同参数数量，但不同有效表达容量。L 的重复参数化还可能改变有效优化步长。
M−L 检验采样范围及其表达空间的额外价值，不是固定有效容量下的纯感受野效应。

## 固定训练设置与判读

与 C 的下载 provenance 对齐：单阶段、58、masked MSE、150 epochs、batch16、AdamW、
weight decay 0.05（仅 Conv/Linear 权重）、lr=1e-4、cosine 至 1e-6、clip=0、
D4 同步增强、AMP、8 workers。固定官方 val，依据有效像素累计 RMSE 选择 best；保留 last。
不修改 loader、数据归一化、LOS 缓存、输出定义或损失。

- **项目证据**：C 两 seed 的 best 均值为 5.111476 dB，优于 B 的 5.167185 dB；因此保留 C 深度。
  C 的深层已有图幅尺度名义支持，不能把此实验描述成解决“没有全局感受野”。
- **文献启发**：[REM-Net+ 作者实现](https://github.com/YNUniversityCQ/REM-NET/blob/master/models/modules.py)
  在城市 RadioMap 使用多膨胀聚合；[EMCAD Table 5](https://arxiv.org/html/2405.06880v1#S5.SS2)
  在医学分割比较不同核组合。任务、预算、输入与本项目不同；本实现不是完整复现。
- **待验证假设**：在一个中间层同时聚合近邻与较远信息能改善 C；32×32 和 d=[1,2,3]
  是本项目工程选择，尚未证明最优。首轮不增加几何条件 gate 或差分模块。

M 同时优于 C/L 且两个 seed 方向一致，才进入条件尺度选择研究。
L/M 同幅改善优先解释为参数化/优化变化；M 无收益则停止此具体版本，不立即扩展到各层。
微小或方向不一致的变化用 best 的逐 terrain 配对评估判断，不能以单图 RMSE 平均替代主指标。

## A/B 清理与兼容范围

按用户确认，仅移除 `lunar_firstu`、`lunar_conv_shallow` 的实现/注册及专用文档。
原始 RadioWNet firstU/secondU、有效 LOS 路线、MaxViT、C 和更早其他实验不变。
没有删除下载的训练产物或 checkpoint。当前版本不再加载 A/B checkpoint；需要回顾时使用
提交 `9a40414aa8d805e2f911fc777d7a42ef97e0ba77` 的独立 checkout。
清理前完整 tracked 代码已归档到仓库外：
`D:/codes/LunarRadiomap/local_experiments/conv_spatial_lm_20261021/before_cleanup_9a40414.zip`。

## 本地提交和推送

如果本次修改尚未提交推送，在本地 PowerShell 执行以下命令。先检查 diff，仅提交列出的文件：

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git status
git diff --check
git add lunar_conv.py model_factory.py README.md LUNAR_CONV_SPATIAL.md lunar_firstu.py ARCHITECTURE_ABLATION.md
git diff --cached --stat
git commit -m "Add controlled local and multiscale spatial ablations"
git push origin research/conv-alignment
```

## 服务器同步和四次训练

以下命令由用户在 SSH GPU 服务器执行。先激活之前训练 C 使用的环境，确认仓库/数据路径有效；
不更换 torch/torchvision 或其他依赖。Codex 未在服务器执行。
复用 `results/los_cache_v1`；不需要重建 LOS。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch research/conv-alignment
git pull --ff-only
```

下面串行运行 L0、L1、M0、M1，不重训 C。任一任务失败即停止循环。
历史 C0 日志累计轮次时间约 3.95 小时，可先为四次训练预留约 16 小时，再依据实际每轮时间调整；
这不是 L/M 实测时长，评估、加载和其他 GPU 负载另计。

```bash
for arch in lunar_conv_local3 lunar_conv_multiscale; do
  for seed in 0 1; do
    python train.py \
      --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
      --arch "$arch" --phase single --band 58 --loss masked \
      --optimizer adamw --weight-decay 0.05 \
      --epochs 150 --batch-size 16 --lr 1e-4 \
      --lr-sched cosine --lr-min 1e-6 --clip 0 \
      --num-workers 8 --device cuda --amp --seed "$seed" \
      --init-from none --resume auto \
      --panels 6 --panels-every 5 --plots-every 5 \
      --out "runs/${arch}_v1/seed${seed}" \
      --run-name "${arch}_58_masked_single_los_distance_adamw" || break 2
  done
done
```

每组、每个 seed 使用独立目录。首次从头训练；同一命令重跑从本组 last 续训，已完成的组不重训。
C/L/M 的架构名和 model_config 严格匹配，禁止相互混用 checkpoint。
现有训练器恢复模型、optimizer、scheduler、scaler，但不保证中断后随机轨迹完全一致。

## 完整验证与推理成本

正式训练完成后，生成逐样本 SSE/count，供按 terrain 配对分析。C 两个 best 只评估、不重训；
下列 C 路径沿用已有服务器目录，运行前确认两个 checkpoint 都在那里。

```bash
for arch in lunar_conv_aligned lunar_conv_local3 lunar_conv_multiscale; do
  for seed in 0 1; do
    python evaluate.py \
      --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
      --ckpt "runs/${arch}_v1/seed${seed}/${arch}_58_masked_single_los_distance_adamw_best.pt" \
      --split val --batch-size 16 --num-workers 8 --device cuda --amp \
      --out "results/conv_spatial_v1/${arch}/seed${seed}/val.json" || break 2
  done
done
```

如需比较推理成本，在同一空闲 GPU 条件下执行以下命令；计时包含现场 LOS、内部 distance 和全图输出。
benchmark 拒绝覆盖已有输出，重复测量时使用新的文件名。

```bash
for arch in lunar_conv_aligned lunar_conv_local3 lunar_conv_multiscale; do
  for seed in 0 1; do
    python benchmark_inference.py \
      --data-root ~/liuanda/LunarRM \
      --ckpt "runs/${arch}_v1/seed${seed}/${arch}_58_masked_single_los_distance_adamw_best.pt" \
      --split val --device cuda --samples 16 --warmup 3 --amp \
      --out "results/conv_spatial_v1/${arch}/seed${seed}/inference.json" || break 2
  done
done
```

下载 L/M 两个 run 根目录中的 history、summary、provenance、best/last、曲线及示例图，
以及 `results/conv_spatial_v1/`。C 的已有训练产物继续复用。

## 本地验证记录

辅助脚本与结果保存在实际仓库外的
`D:/codes/LunarRadiomap/local_experiments/conv_spatial_lm_20261021/`。

- C 与修改前的配置、参数、缓冲区、CPU RNG 和前向逐位一致；下载的 C0 best 严格加载通过。
- L/M 两 seed 的共同初值、CPU/CUDA RNG、初始 eval/train 输出与 C 逐位一致。
  新分支的 FP32 梯度非零；非零随机核与独立构造的等效稀疏核前向/输入梯度一致。
- 新核全部进入原 AdamW 权重衰减组。严格配置检查拒绝 L/M 混用、错误 dilation 和已移除 A/B 架构。
- batch16 AMP 首步新核梯度下溢为零。保持设置的重复样本短程检查中，两组 seed0 均在第18次、
  seed1 均在第14次更新开始得到非零梯度，后续连续更新通过，新核实际离开零值。
  本机 RTX 4060 Laptop 峰值 allocated 约3.31 GiB；不代表服务器显存或完整训练保证。
- 两个架构的实际 CLI 训练→续训→评估→现场 LOS 推理均通过。夹具仅2 train/2 val、每轮一个batch，
  cosine horizon 仍为150；核对 optimizer/scheduler/scaler 恢复、best/last、有效像素汇总及产物。
- 文档 Python 命令经现有 argparse 解析；训练设置逐项与下载的 C provenance 对齐。

详细结果：`verification.json`、`amp_updates.json`、`integration.json`。
以上检查不构成正式验证集成绩。没有修改上游 meshgrid 未来接口警告涉及的程序逻辑。
