# 固定 T 的细化对照：仅 seed0

本轮使用 `lunar_conv_tx_refine_frozen`，从已训练的 T seed0 best 初始化。
仅固定第一阶段预测器，现有256/128/64浅层细化器、跨阶段连接和零修正头不变。
这是上一轮联合训练的机制对照；按用户要求只训练 **seed0**。

## 研究问题与固定条件

项目证据：上一轮T再训练与联合细化的两个seed均未超过原T起点；后期训练loss下降，
验证误差回升。联合细化seed0训练轮次内best为5.097454，原T为5.059348 dB。
旧RadioWNet的secondU在固定firstU后曾获得小幅收益。

文献动机：[RadioUNet §4.3.1](https://arxiv.org/html/1911.09002)的回顾式训练固定第一U。
其数据为城市RadioMapSeer，不是月面；它没有证明冻结总是优于联合训练。
本轮研究假设：固定初稿与特征，让细化器拟合稳定的误差，比同步重优化成熟T更有效。
不声称已复现完整RadioWNet或已经证明该假设。

\[
(y_0,D_0)=T_{\theta_0}(X),\quad \theta_0\text{固定},\qquad
\hat y=\operatorname{ReLU}\big(y_0+R_\phi(X,y_0,D_0)\big),
\]
\[
\min_\phi\;\frac{\sum_i m_i(\hat y_i-y_i)^2}{\sum_i m_i}.
\]

- `X=[H,TX,F,log-distance]`，同步D4增强、官方mask、归一化和数据loader不变。
- 冻结T的所有参数，T前向使用`no_grad`；每次外层`model.train()`后，T的全部子模块仍处于eval，
  固定BatchNorm的running mean/variance/count并关闭随机深度。只冻结参数不足以实现此对照。
- 优化器只包含`refiner.*`，可训练参数248,225；总参数仍26,613,318。
- 细化器仍用AdamW、Conv/Linear权重衰减0.05、lr1e-4、cosine至1e-6、150轮、batch16、
  clip0、AMP；只监督最终masked MSE，不增加阶段或边界损失。
- 零头仍保证初始输出等于T；第一步内部细化参数的任务梯度为零，输出头可学习。
  本轮主要改变T是否固定，不同时改变该初始化。
- T以eval处理增强后的训练输入。相比联合对照，梯度、BN和随机深度行为共同改变；
  这是“固定预测器”的整体对照，不可把收益单独归因于其中一项。

前向结构没有增加计算；新增细化卷积仍约2.640G MAC/图。细化时间复杂度
`O(BΣHW k² cin cout)`，保留激活`O(BΣHW c)`。冻结移除了T的反向图与优化器状态，
仍需T前向以及传给细化器的特征；实际时间与显存以实测为准。

## 初始化与恢复

`--init-from`必须指向同seed的原T checkpoint，不能使用上一轮细化器last。
`--eval-initial`保留epoch−1起点；best只在有效像素RMSE严格改善时更新。
同一命令的`--resume auto`恢复本轮last，已完成150轮则不再训练。

固定版、联合版有独立架构名和model_config。它们不可互相resume或直接加载替代；
旧联合版checkpoint的配置保持兼容。新checkpoint包含T与细化器全部权重，
评估/推理不需要另外加载原T文件。`--phase single`表示一个最终预测接口，
在固定版中不表示全部参数一起更新。

## 本地提交与服务器同步

在本地PowerShell检查并提交、推送本轮修改；不添加`docs/`或`local_experiments/`：

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git diff --check
git add lunar_refinement.py model_factory.py train.py README.md LUNAR_T_REFINEMENT_FROZEN.md
git diff --cached --stat
git commit -m "Add fixed-T refinement control"
git push origin research/conv-alignment
```

由用户在SSH服务器激活运行T时的现有环境，确认项目/数据路径后执行：

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch research/conv-alignment
git pull --ff-only
```

## 单 seed 正式训练

复用现有LOS缓存和T seed0 best，输出使用新目录。上一轮联合细化的150轮日志约
2.9–4.7小时；本次冻结后时间需要实测，不能将不同服务器负载下的时间差当作加速结论。

```bash
python train.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --arch lunar_conv_tx_refine_frozen --phase single --band 58 --loss masked \
  --optimizer adamw --weight-decay 0.05 \
  --epochs 150 --batch-size 16 --lr 1e-4 \
  --lr-sched cosine --lr-min 1e-6 --clip 0 \
  --num-workers 8 --device cuda --amp --seed 0 \
  --init-from runs/lunar_conv_tx_content_v1/seed0/lunar_conv_tx_content_58_masked_single_los_distance_adamw_best.pt \
  --eval-initial --resume auto \
  --panels 6 --panels-every 5 --plots-every 5 \
  --out runs/t_refinement_frozen_v1/seed0 \
  --run-name lunar_conv_tx_refine_frozen_58_masked_single_los_distance_adamw
```

启动信息应为`trainable params 248,225 / 26,613,318`。若不匹配，先检查同步版本与架构参数。

## 验证与完整在线推理计时

```bash
python evaluate.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --ckpt runs/t_refinement_frozen_v1/seed0/lunar_conv_tx_refine_frozen_58_masked_single_los_distance_adamw_best.pt \
  --split val --batch-size 16 --num-workers 8 --device cuda --amp \
  --out results/t_refinement_frozen_v1/seed0/val.json

python benchmark_inference.py \
  --data-root ~/liuanda/LunarRM \
  --ckpt runs/t_refinement_frozen_v1/seed0/lunar_conv_tx_refine_frozen_58_masked_single_los_distance_adamw_best.pt \
  --split val --device cuda --samples 16 --warmup 3 --amp \
  --out results/t_refinement_frozen_v1/seed0/inference.json
```

下载本轮`initial/history/summary/provenance`、评估JSON和样例图；保留服务器best/last，
不必首先下载全部权重。与原T seed0=5.059348及联合细化训练内best=5.097454比较。
只超过退化的联合版本、却没超过原T，不算净收益。单seed结果仅作筛选，不能宣称跨seed稳定。

本地验证记录位于仓库外`local_experiments/t_refinement_frozen_20261024/`。
正式训练由用户在服务器执行，本地小样本验收不构成准确率结论。

本地seed0已通过train/eval和FP32/AMP初始一致性、12步AMP的T权重/BN/特征逐位不变、
细化器任务更新、实际epoch−1恢复→训练→续训→评估，以及16输入在线推理检查。
本机重复样本batch16峰值allocated约0.845GiB；没有正式精度结果。临时测试权重已清理。
