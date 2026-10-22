# C 上的两段几何地形交互：T / TG

本轮从现有 C（`lunar_conv_aligned`）出发，从头训练 **T、TG 各 seed 0/1**。
复用 C 的训练结果，不加载其 best 初始化，不叠加 L/M、路径 GRU、FADE 或新损失。
训练、评估与计时继续使用 `train.py`、`evaluate.py`、`benchmark_inference.py`。

| 组别 | `--arch` | 改动 | 参数量 |
|---|---|---|---:|
| C | `lunar_conv_aligned` | 原卷积/MLP 编码器 | 26,307,429 |
| T | `lunar_conv_tx_content` | 单次 TX 条件内容交互 | 26,365,093 |
| TG | `lunar_conv_tx_geometry` | T + 两段几何软偏置 | 26,365,269 |

T/TG 的 Q/K/V、head 数、候选位置和输出投影相同；TG 多176个几何 MLP 参数。
这是内容交互与几何条件交互对照，不声称严格等有效容量。
旧 `lunar_maxvit_c` 是不同骨干上的历史实验，不代替当前 T。

## 功能与证据边界

问题：现有 F/log-distance 显式描述 TX→RX 净空/距离，候选地形相对 TX、RX 的关系
由网络隐式组合。C 已有全图尺度名义感受野和完整地形输入，不能说它完全看不到远处地形。

文献/实现依据：[GRAPHWAVE](https://arxiv.org/html/2605.30525v1) 在城市传播任务研究
TX→表面→RX 图表示；本项目 metadata 声明包含镜面/漫反射，固定 Sionna RT 1.1.0
[材料实现](https://raw.githubusercontent.com/NVlabs/sionna-rt/v1.1.0/src/sionna/rt/radio_materials/radio_material.py)
包含表面方向相关的反射/散射。两跳思想已有先例，本实现不宣称首创或复现完整射线追踪。

项目证据：L/M 两 seed 的微小收益不一致；C 的 terrain 难度与有效邻接标签变化相关，
但不能据此证明像素误差由某种反射引起。研究假设是：两段几何条件帮助选择相关的地形特征。
尚无本方案正式训练成绩，不预先承诺提高 RMSE。

softmax 权重是隐藏特征选择，**不是**真实路径概率、功率、可见性或反射系数。
粗 anchor 不保证对应有效反射面元。F 的终点在地表上方1m，不能作为地表 anchor 的精确受照判断。
不使用标签、mask、F 硬遮挡，不加绕射、相位、功率守恒或严格 FSPL 下界。

## 固定算子与坐标

插入位置是 `encoder.blocks[1]` 输出：128×32×32，先交互，再进入下一编码块和对应 skip。
保持原 C 的所有 MBConv、逐位置 MLP、stochastic depth、decoder 和输出头。

T 直接复用已有 `_TXSceneAttention` 的内容聚合实现：

- 1024个 query；2×2 average pooling 得到256个 anchor。
- TX 特征按 stride8、origin0 的实际网格双线性采样，越界使用 border extension。
- Q/K/V 各由当前位置/anchor 与 TX 特征拼接后线性投影，总宽64，4 heads，每head16维。
- softmax 聚合后投影回128维，残差相加。无额外 FFN、归一化或 dropout。
- K 中共同的 TX 项在 softmax 中抵消；TX 条件主要通过 Q 与 V 作用，沿用历史控制定义。

TG 在 attention logits 上加 `6→16→4`、GELU 的几何 MLP 偏置。
最后 Linear 不加 bias，因为每 head 的全局常量会在 softmax 中抵消。

令 t 为 TX 坐标，s 为地表 anchor，r 为接收位置：

\[
t=(x_t,y_t,H_t+3),\quad s=(x_s,y_s,H_s),\quad r=(x_r,y_r,H_r+1).
\]

高度偏移和米/像素尺度实际读取 `feature_config`，上述3/1是当前数据值。
H 从归一化值恢复为米；x=column，y=row，与增强后数组保持一致。
query 坐标为 `0,8,...,248`，anchor 为池化中心 `4,20,...,244`。
anchor 的几何高度是该地表位置的高度，不是 pooled feature 的高度平均，也不加 RX 高度。
法向由原网格中心差分计算：

\[
n=\operatorname{normalize}(-\partial_xH,-\partial_yH,1).
\]

anchor 均在图内，±1像素的差分模板无需边界延拓。几何在 D4 同步增强后重算。
卷积采样网格本身并非严格 D4 等变，不能以此声称整网旋转等变。

令 \(d_1=\|s-t\|,d_2=\|r-s\|,d_0=\|r-t\|\)，
\(u=(s-t)/d_1,v=(r-s)/d_2,\bar u=u-2(u\cdot n)n\)。六项描述为：

\[
\gamma=\left[
\frac{\log(1+d_1)}{\log(1+D)},
\frac{\log(1+d_2)}{\log(1+D)},
\frac{\log(1+\max(d_1+d_2-d_0,0)/\max(d_0,\epsilon))}{\log(1+2D/d_{\min})},
-n\cdot u,\ n\cdot v,\ \frac{\|v-\bar u\|^2}{4}
\right].
\]

距离以1m为单位；D 为两段距离的保守解析上界，由图幅、训练高度跨度及天线高度计算；
\(d_{\min}=\max(\epsilon,\min(\text{pixel spacing},|h_t-h_r|))\)，\(\epsilon=10^{-6}\rm m\)。
这是固定尺度，不使用 val、test 或 batch 的拟合统计。额外路程保留相对直达距离的定义，
再做对数压缩，防止接近 TX 的位置使关系 MLP 输入失衡。
有符号入/出射余弦保持符号，不据此硬删除连接。

\[
\alpha_{pj}=\operatorname{softmax}_j(q_p^\top k_j/\sqrt{16}+b_\phi(\gamma_{pj})),\qquad
\widetilde U_p=U_p+W_o\sum_j\alpha_{pj}v_j.
\]

所有几何和该交互分支的计算在 AMP 下保持 FP32，更新后转回主干 dtype。
输入几何无需梯度；几何 MLP 保留 autograd。W_o 的 weight/bias 零初始化，Q/K/V 与几何 MLP
采用原 PyTorch 默认初始化。共享主干先构造，新模块在 CPU RNG fork 中构造，保持同 seed
的 C 初值和后续随机状态；T/TG 的 Q/K/V 初值也一致。
第一步 W_o 获得梯度，内部参数待 W_o 离开零值后才获得任务梯度，这是残差初始化的预期行为。

配置保存在 `model_config.tx_scene_attention`，物理尺度仍保存在原 `feature_config`。
三个新旧架构不能互相 init/resume/evaluate；现有严格检查会拒绝混用。C、L/M、旧 MaxViT
配置/权重键不变，旧内容交互默认不使用新增的几何接口。

## 成本与固定训练协议

交互时间约 O(BNKd)，Q/K/V 投影 O(B(N+K)Cd)；TG 另加 O(BNKr(6+h))，
N=1024、K=256、d=64、r=16、h=4。朴素 attention 空间 O(BhNK)，TG 输入和 MLP 保存激活
另为 O(BNK(6+r+h))，不能把176个参数误当几乎没有计算成本。
QK+AV 约33.55M MAC/图；几何 MLP 约41.94M MAC/图，不含投影、几何构造和激活。
单份FP32 attention矩阵约4MiB/图、batch16约64MiB，实际训练显存还有梯度和中间激活。
最终以同机完整推理计时为准，现场 LOS/内部 distance/新几何均计入。

与 C 一致：58、single、masked MSE、150 epochs、batch16、AdamW（Conv/Linear weights-only
decay0.05）、lr1e-4、cosine到1e-6、clip0、D4、AMP、8workers；原固定val选best，保留last。
不修改 loader、缓存或数据归一化。原 F 缓存可复用。

先比较 TG−C 和 TG−T，并按 terrain、seed、single/multi 检查收益。
仅TG有效后再研究几何与anchor错配对照，不在首轮同时扫描位置、anchor数和多跳深度。
两个seed方向一致且TG优于T才支持几何增量；微小变化、子集之间转移或训练loss降低不构成结论。

## 本地提交与推送

如果尚未提交，在本地 PowerShell 检查并提交本次文件：

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git status
git diff --check
git add lunar_conv.py lunar_maxvit.py model_factory.py terrain_interaction.py README.md LUNAR_TERRAIN_INTERACTION.md
git diff --cached --stat
git commit -m "Add geometry-conditioned terrain interaction controls"
git push origin research/conv-alignment
```

## 服务器同步与运行

以下命令由用户在 SSH GPU 服务器执行，先激活训练 C 时的既有环境并确认目录有效。
不更换依赖；Codex 未在服务器执行。保留 `results/los_cache_v1`。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch research/conv-alignment
git pull --ff-only
```

串行训练 T0/T1/TG0/TG1。历史 C 约4小时/次，四次训练应先预留至少约16小时；TG额外计算、
加载/评估和服务器负载另计。以实际首轮时间调整预期，这不是TG实测耗时。

```bash
for arch in lunar_conv_tx_content lunar_conv_tx_geometry; do
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

第一次从头训练，重跑同命令从本组last续训；现有训练器不保证中断后的随机轨迹逐位一致。
不要指向旧 C/TX/MaxViT 的输出目录，也不要以 `--init-from` 加载它们。

训练完成后全量验证，获得逐样本 SSE/count：

```bash
for arch in lunar_conv_tx_content lunar_conv_tx_geometry; do
  for seed in 0 1; do
    python evaluate.py \
      --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
      --ckpt "runs/${arch}_v1/seed${seed}/${arch}_58_masked_single_los_distance_adamw_best.pt" \
      --split val --batch-size 16 --num-workers 8 --device cuda --amp \
      --out "results/terrain_interaction_v1/${arch}/seed${seed}/val.json" || break 2
  done
done
```

C已有 `results/conv_spatial_v1/lunar_conv_aligned/seed*/val.json` 可复用，输入/split一致时不重复评估。
同一空闲GPU下分别计时，C和新模型使用相同样本与条件：

```bash
for arch in lunar_conv_aligned lunar_conv_tx_content lunar_conv_tx_geometry; do
  for seed in 0 1; do
    python benchmark_inference.py \
      --data-root ~/liuanda/LunarRM \
      --ckpt "runs/${arch}_v1/seed${seed}/${arch}_58_masked_single_los_distance_adamw_best.pt" \
      --split val --device cuda --samples 16 --warmup 3 --amp \
      --out "results/terrain_interaction_v1/${arch}/seed${seed}/inference.json" || break 2
  done
done
```

下载两个新架构的run目录：history、summary、provenance、best/last、曲线/示例图，
以及 `results/terrain_interaction_v1/`。沿用已有C产物进行配对分析。

## 本机验证

辅助脚本及结果保存在实际仓库外
`D:/codes/LunarRadiomap/local_experiments/terrain_interaction_20261022/`。

- 平地解析值、斜平面全部8种D4输入、TX角点、整体高度平移及实际stride/池化中心检查通过；
  相对独立 NumPy 几何计算的最大绝对差为 `4.35e-6` 以内。此检查不声称整网D4等变。
- seed0/1 的 C/T/TG 共享权重、缓冲区、构造后CPU/CUDA RNG及初始AMP eval/train输出逐位一致；
  T/TG 的 Q/K/V 初值一致，所有参数可训练。
- C/L/M 配置保持修改前定义；旧内容注意力在非零输出投影下与旧实现逐位一致；
  下载的C0 best严格加载通过，错误架构及几何坐标配置被拒绝。
- batch16 AMP 重复样本检查中，T/TG 两seed在第11次更新开始全部被检查分支出现非零梯度。
  更严格的任务学习检查将更新与仅AdamW衰减的更新作比较：TG seed0/1 分别于第28/26次更新
  开始全部被检查的Q/K/V/输出/几何权重出现任务更新，随后连续两次通过。
  这是有限短程可训练性检查，不说明实际收益或所有样本的启动时刻。
- 本机 RTX4060 Laptop、torch2.5.1+cu121、torchvision0.20.1+cu121：batch16 AMP
  T峰值allocated约3.35GiB，TG约4.00GiB。不是服务器显存或完整训练的保证。
- 两个新架构的实际训练→续训→评估→现场LOS推理入口通过。夹具仅2train/2val，每轮1batch，
  验证了optimizer/scheduler/scaler恢复、150轮日程、best/last及有效像素累计评价。
- 文档12条展开后的Python命令经现有argparse解析，正式训练设置逐项与C下载provenance对齐。

复现脚本/记录：`verify_models.py` / `verification.json`、`check_learning.py` /
`learning_updates.json`、`check_cli.py` / `integration.json`。
原torchvision引出的meshgrid未来接口警告未修改；无新增依赖。
这些检查不替代正式验证集性能或服务器运行，未执行完整C逐像素误差诊断或正式训练。
