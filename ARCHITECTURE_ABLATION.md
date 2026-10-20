# A/B/C：firstU 与无注意力网络的深度消融

本轮只做用户指定的三组结构比较，不增加 SE、空间注意力或新传播算子。
A 已确认采用旧 firstU **架构从头训练**，不加载旧 firstU 权重。

| 组别 | `--arch` | 结构 | 参数量 |
| --- | --- | --- | --- |
| A | `lunar_firstu` | 旧 firstU 独立运行，H/TX/F/rho 共同输入；没有 secondU | 10,922,563 |
| B | `lunar_conv_shallow` | C 每组只保留第一个块，块数 `[1,1,1,1]` | 13,113,765 |
| C | `lunar_conv_aligned` | 当前网络，块数 `[2,2,5,2]` | 26,307,429 |

三组均为 `--phase single`：这是训练入口的统一命名，A 的内部结构仍是旧 firstU。
所有参数共同训练，完整输出 256×256 地图。

## 固定条件与比较边界

- 三组输入均为 H、TX、连续净空 F、既有 log-distance rho；外部 loader 提供
  H/TX/F，模型在同步增强后按相同公式计算 rho。
- 固定 5.8 GHz、官方 train/val、masked MSE、D4 增强、batch 16、150 epochs、
  AdamW（Conv/Linear 权重衰减 0.05，bias/norm 不衰减）、lr 1e-4、cosine 至
  1e-6、AMP、无梯度裁剪，均从头训练。
- A 使用 C 的优化器和训练协议，以比较架构；不是复现历史 Adam/继承权重实验。
- A/B 使用 seed 0/1；C 复用已完成 seed 0，再训练 seed 1，共新增五次正式训练。
  不将两个 seed 的预测集成；分别按对应 seed 比较后汇总。
- 复用 C0：`runs/lunar_conv_aligned_v1/seed0/`，run name 为
  `lunar_conv_aligned_58_masked_single_los_distance_adamw`；
  原记录 commit `916e96f`、best `5.110348788065096 dB`，epoch 63。
  本轮 C 的配置、同 seed 初值和前向须保持兼容，不能从 A/B 加载权重。
- 所有组保留 best/last，主指标为有效像素总 SSE / 总有效像素数后开方。
  逐样本 JSON 用于按 terrain 配对分析，不平均逐图 RMSE 作为主成绩。

**A 与 B/C** 比较的是同输入、同训练协议下两类单 U 网络，仍同时涉及卷积类型、
归一化、残差、尺度和宽度。它不能独立证明 distance、取消冻结或某一个块的作用。
若以后要严格测 distance 增益，需要另设完全同训练协议的 A 无 distance 对照，
本轮不擅自增加第四组。

**B 与 C** 检验删去每组重复块后的表现和成本。深度、容量、有效感受野及残差
随机路径数量随之变化，不能将结果表述为等参数条件下的“深度纯效应”。

## A：保留旧 firstU 的具体实现

代码：[lunar_firstu.py](lunar_firstu.py)。复用 `RadioUNet/modules.py` 的 firstU
层定义及其 MIT 许可，保留普通卷积、ReLU、MaxPool、转置卷积和全部 skip 连接。
瓶颈仍为 4×4，编码层通道为 `6/40/50/60/100/100/150/300/500`。

H/TX/F/rho 同时进入三个原始输入连接：`layer00`、`conv_up00`、`conv_up000`，
它们的输入通道分别为 4、30、24。没有直接调用上游 `inputs=4`，因为那会连带将
第一隐藏层从 6 加宽到 10，超出本轮“旧 firstU 加入 distance”的结构范围。
这三个卷积按原 Conv2d 的默认方式重新随机初始化所有输入列，其余层使用上游
从头初始化。保留原 ReLU 直接预测头，没有套用 C 的 0.5 输出偏置初始化。

模型没有注册或执行 secondU，也没有冻结、detach 或第一阶段预测作为输入。
原双 U 的文件和 checkpoint 加载路径保持不变。

固定卷积核时，时间复杂度为
`O(B * sum(H_l W_l C_in_l C_out_l k_l^2))`，训练激活空间为
`O(B * sum(H_l W_l C_l))`，另有 `O(P)` 参数与优化器状态。

## B：只删除重复块

代码：[lunar_conv.py](lunar_conv.py)。先按 C 的原顺序初始化，再保留每组首块，
因此同 seed 下所有保留参数（含 stem、decoder、输出头）与 C 初值相同，
构造后的 CPU RNG 状态也相同。未加载任何训练 checkpoint。

四个首块负责原有降采样，输出尺度仍是 64/32/16/8，通道仍是 64/128/256/512；
stem、全分辨率分支、五级 decoder、输入/输出定义不变。
MBConv 数量从 11 降到 4，逐位置通道 MLP 从 22 降到 8，仍无 SE/self-attention。

保留块的 stochastic-depth 概率沿用 C 中同位置的 **0/0.04/0.08/0.18**；
不重新按四个块拉伸到 0.2。因此 B 实际最大概率是 0.18，配置中明确记录这一点。
删除的块不再产生随机调用，训练随机轨迹不保证与 C 相同。

时间复杂度仍为 `O(B * sum(L_s N_s C_s^2))`，训练激活为
`O(B * sum(L_s N_s C_s))`，本轮只降低 `L_s`，decoder 的成本不变。
参数减少约 50.2% 不等于实际推理时间减半，须在同环境测量。

## 依据与结果判读

- **实现依据**：A 复用项目中的上游 RadioWNet firstU；B 是已有 C 的层删除消融，
  不宣称复现新论文，也不宣称原创算法。
- **项目内证据**：C0 为 5.110349 dB，已接近完整 MaxViT+AdamW 的 5.097016 dB；
  旧 LOS+distance secondU 为 5.416208 dB，但历史比较混有训练、容量等差异。
- **研究假设**：单阶段输入与训练对齐后，旧 firstU 能接近现代卷积网络；
  或当前 `[2,2,5,2]` 的重复块有冗余。两者尚待这轮配对结果验证。

优先比较每个 seed 的 A−C、B−C、A−B，汇总两 seed 的均值和差值范围，
同时保留 best/last、训练曲线和总训练时间。固定验证集上的多次结构筛选不算独立测试。
如 B 与 C 接近且推理成本更低，可转向 B；若 B 明显退步则保留 C 的深度。
A 接近 B/C 才支持继续以原 firstU 为简洁结构基线，不能预先下结论。

## SSH 同步与训练

以下命令供用户在服务器执行；Codex 只在本地验证。先确认并激活现有训练环境，
确认仓库与数据路径仍有效。复用 `results/los_cache_v1`。
历史 C0 的日志累计训练轮次时间约 3.95 小时；A/B 正式耗时尚未测量，
五次新训练可先预留约 10–20 小时，再按实际每轮耗时修正，另计评估和计时。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git fetch origin
git switch research/conv-alignment
git pull --ff-only
```

A 的两个 seed：

```bash
for seed in 0 1; do
  python train.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --arch lunar_firstu --phase single --band 58 --loss masked \
    --optimizer adamw --weight-decay 0.05 \
    --epochs 150 --batch-size 16 --lr 1e-4 \
    --lr-sched cosine --lr-min 1e-6 \
    --num-workers 8 --device cuda --amp --seed "$seed" \
    --init-from none --resume auto \
    --panels 6 --panels-every 5 --plots-every 5 \
    --out "runs/firstu_distance_v1/seed${seed}" \
    --run-name lunar_firstu_58_masked_single_los_distance_adamw
done
```

B 的两个 seed：

```bash
for seed in 0 1; do
  python train.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --arch lunar_conv_shallow --phase single --band 58 --loss masked \
    --optimizer adamw --weight-decay 0.05 \
    --epochs 150 --batch-size 16 --lr 1e-4 \
    --lr-sched cosine --lr-min 1e-6 \
    --num-workers 8 --device cuda --amp --seed "$seed" \
    --init-from none --resume auto \
    --panels 6 --panels-every 5 --plots-every 5 \
    --out "runs/lunar_conv_shallow_v1/seed${seed}" \
    --run-name lunar_conv_shallow_58_masked_single_los_distance_adamw
done
```

C 只补 seed 1，seed 0 不重训：

```bash
python train.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --arch lunar_conv_aligned --phase single --band 58 --loss masked \
  --optimizer adamw --weight-decay 0.05 \
  --epochs 150 --batch-size 16 --lr 1e-4 \
  --lr-sched cosine --lr-min 1e-6 \
  --num-workers 8 --device cuda --amp --seed 1 \
  --init-from none --resume auto \
  --panels 6 --panels-every 5 --plots-every 5 \
  --out runs/lunar_conv_aligned_v1/seed1 \
  --run-name lunar_conv_aligned_58_masked_single_los_distance_adamw
```

同一命令再次执行会从该组、该 seed 的 last 续训；三组 checkpoint 不互相迁移。
现有训练器恢复模型、optimizer、scheduler、scaler，但不保证中断前后的随机轨迹一致。

## 完整验证与在线推理成本

在同一硬件与空闲条件下测量，计时包含现场 LOS 构造、模型内 distance 和完整地图输出。
以下也为复用的 C0 补充逐样本验证；不会重训 C0。

```bash
for group in A B C; do
  case "$group" in
    A) run_root=runs/firstu_distance_v1; run_name=lunar_firstu_58_masked_single_los_distance_adamw ;;
    B) run_root=runs/lunar_conv_shallow_v1; run_name=lunar_conv_shallow_58_masked_single_los_distance_adamw ;;
    C) run_root=runs/lunar_conv_aligned_v1; run_name=lunar_conv_aligned_58_masked_single_los_distance_adamw ;;
  esac
  for seed in 0 1; do
    python evaluate.py \
      --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
      --ckpt "${run_root}/seed${seed}/${run_name}_best.pt" \
      --split val --batch-size 16 --num-workers 8 --device cuda --amp \
      --out "results/architecture_ablation_v1/${group}/seed${seed}/val.json"
    python benchmark_inference.py \
      --data-root ~/liuanda/LunarRM \
      --ckpt "${run_root}/seed${seed}/${run_name}_best.pt" \
      --split val --device cuda --samples 16 --warmup 3 --amp \
      --out "results/architecture_ablation_v1/${group}/seed${seed}/inference.json"
  done
done
```

benchmark 拒绝覆盖已有文件，重复计时时请使用新的输出文件名。
下载 A/B/C 的 history、summary、provenance、best/last、曲线与样本图，
以及 `results/architecture_ablation_v1/` 下的 val.json、inference.json；
C0 已下载的训练产物可以复用。

## 本地验证

独立验证脚本与产物位于实际 Git 仓库之外：
`D:/codes/LunarRadiomap/local_experiments/architecture_ablation_20261020/`。
已通过 A 与上游 firstU 的输出/梯度对照、distance 三处输入权重梯度、B 与直接
删层参考的一致性、C 的配置/初值/前向不变、下载 C0 best 的严格加载。
A/B 的 batch 16 AMP 检查通过，本机峰值 allocated 分别约 1.44/2.22 GiB。

实际 CLI 的 train → resume → evaluate → 在线推理检查均通过；
使用 2 train / 2 val 夹具、每轮一个 batch，cosine horizon 保持 150。
验证了架构识别、optimizer/scaler 恢复、全部参数可训练、有效像素累计口径、
曲线/样本图产物和文档命令参数。小样本检查不是正式训练或验证集成绩。

B 的首次 CLI 子进程异常退出，没有留下 Python 错误栈，原退出码未保存；
保留原日志后，使用相同模型/训练参数及未缓冲输出重跑，训练、续训、评估、
计时均通过。首次退出原因尚未确定，没有为此改变模型、损失、精度或超参数。
详细记录见 `verification.json`、`integration.json`、`compatibility.json`。
