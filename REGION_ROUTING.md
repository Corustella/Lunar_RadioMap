# T上的区域软路由与边界修正

新增架构：`lunar_conv_tx_region_uniform`、`lunar_conv_tx_region_fixed`、
`lunar_conv_tx_region_learned`。主候选为`learned`；以原T
`lunar_conv_tx_content`为结构和训练配方基础，不叠加已停止扩展的嵌套连接。
旧T入口、权重配置、loader、LOS缓存、训练损失和评估口径保持原样。

## 预测结构

所有分支共享T编码器、32×32 TX场景交互、16/32/64三级解码和原生skip。
T原有最后128/256两级解码及输出作为可见区完整损耗估计`Lv`。
一个独立的128/256解码分支学习有符号修正`delta_s`，遮挡区完整估计为
`Ls=Lv+delta_s`。边界分支从共享D64、原生128/256特征及输入几何读取上下文，
经两个16通道卷积块生成有符号边界修正`Rb`。

```text
H/TX/F -> 原T共享编码器、TX交互、解码至64
                         |-> 原T的128/256解码 -> Lv
                         |-> 独立128/256解码 -> delta_s -> Ls=Lv+delta_s
                         |-> 轻量128/256边界解码 -> Rb、门控特征
H/TX/F -> 同步增强后的距离/法向/入射代理 -> 门控与边界解码
Lv + (1-p)*delta_s + 4*p*(1-p)*Rb -> 原有ReLU -> 完整256x256预测
```

这是`p*Lv+(1-p)*Ls+b*Rb`的等价重参数化。Lv、Ls是完整的归一化路径损耗，
不解释为真实LOS/non-LOS功率，也不在dB域相加物理路径功率。
重参数化使三个模式均从原T函数起步，同时避免“两个完全同值的复制预测头在等权
融合下永远对称”的问题。所有参数端到端训练，不冻结现有主干，没有第二个编码器。

## 门控与输入几何

外部输入仍为`[H,TX,F]`，内部rho仍使用原公式。新增几何仅由这些测试可用输入构造：

- 可见性近似为`F>=0`：原F是最低竖直净空的奇函数编码，零点对应零净空。
  它不是官方LOS标签，也不是有效性mask。
- `d`为到另一类像素中心的**截断Chebyshev距离**，减半像素形成界面距离，
  正号表示几何可见侧。16轮3×3侵蚀，范围为±15.5像素，再按分辨率转成米。
  这不是欧氏最短距离。它在所定义的L∞网格度量和带宽内精确，D4对称。
- 侵蚀忽略图像外部，不把地图边缘当作相反类别；全可见/全遮挡时返回恒定截断距离。
- 地形法向使用米制高度的中心差分、复制边界；入射代理是该法向与RX→TX方向点积。
  它不是未知反射点/散射点法向。
- 五通道为归一化d、nx/ny/nz和点积。它们在同步D4后现场构造，纯PyTorch、FP32、
  无autograd、无CPU往返、无新缓存、无RT、无新依赖。已有F缓存可直接复用。

三种模式：

| 模式 | 路由 | 边界权重 | 用途 |
| --- | --- | --- | --- |
| uniform | p=0.5 | b=1 | 相同几何/预测分支的无区域路由对照 |
| fixed | p=sigmoid(d/(2r)) | b=4p(1-p) | 固定几何先验 |
| learned | p=sigmoid((d+delta_d)/tau) | b=4p(1-p) | 可纠偏路由，主候选 |

r是米/像素。learned的1×1门控读取边界特征和五几何通道，输出两个实数：
`delta_d=r*raw_delta`，`tau=r*(0.5+softplus(raw_tau))`。
初始delta_d=0、tau=2r；位置修正允许正负值，宽度始终为正。
门控和融合在AMP下保持FP32，门控保留梯度。
几何只是可纠偏先验；p不是经过校准的真实LOS概率。

uniform/fixed共用完全相同的分支参数量；learned仅额外增加44个门控参数。
各模式都执行全部分支，软门控不意味着自动跳过计算。

## 初始化与checkpoint

共享T先按原顺序构造。新参数在CPU RNG fork中构造，shadow解码复制原T，
shadow修正输出与边界输出置零。因此同seed初始共享权重、构造后RNG状态和完整
预测均保留T；它是从头训练的稳定初始化，不是加载已训T再冻结它。
修正输出第一步有梯度，内部参数及门控待修正/两完整估计分化后得到任务梯度。

推荐正式比较使用`--init-from none`，沿用原T从头端到端训练配方。
也实现显式`--init-from <T_best.pt>`的权重初始化：严格核对T键/尺度，复制shadow解码，
重置修正与路由，初始预测与该T相同、全部参数可训练。它是另一个训练流程，
不能将其成绩与从头训练控制直接解释为纯架构增量。
T权重只允许显式初始化，不能作为区域模型的resume或evaluate。
三个区域模式也不能互相续训；完整model_config写入checkpoint并严格验证。

`forward(x)`保持`[prediction]`接口，train/evaluate/benchmark入口自动支持新架构。
`forward_regions(x)`额外返回p、b、d、delta_d、tau、两完整估计及修正图，
供后续检查门控塌缩、位置/宽度和区域误差。无额外辅助loss、分组权重或采样变化。

## 复杂度、证据与限制

新增卷积工作量为O(BHW c²)，c为固定小通道数；几何距离O(16BHW)，
法向和方向O(BHW)，新增工作激活O(BHW c)。参数增量小不等于没有高分辨率计算，
端到端延迟必须实测。距离/法向在模型内计算，因此现有benchmark的`network_ms`
包含它们；`total_ms`仍包含现场F、传输、rho、全部分支和CPU dB输出，不可只看网络。

- 项目内证据：4B分组中高依赖近边界区误差较大；但低RT误差筛选同时选出了较易样本，
  且局部RT边界仍失配。旧高梯度开关/冻结细化的微增益不证明本方案有效。
- 文献支持：[MEPSNet](https://arxiv.org/abs/2009.14563)在空间异质图像退化任务中
  研究共享表示与专家特化；不是月球传播上的直接证据，也不是本代码逐行复现。
- 研究假设：共享全局传播上下文、内部区域特化和可纠偏门控能改善官方损耗回归。
  本实现未把该假设当作已证实成因；没有读取RT分量、标签或mask来构造门控。

正式评价使用完整固定val、seed0/1；比较原T、uniform、fixed、learned。
主指标是跨有效像素累计RMSE。用几何可见性/距离分组检查区域转移，同时检查非边界区，
不能只以最优8例、边界RMSE或图像更锐利来认定收益。门控和头分工未被官方机制标签监督，
不能据权重图宣称恢复了真实机制份额。

## 服务器同步、计时和训练

以下由用户在SSH GPU服务器执行；Codex只在本地修改/验证。先激活原训练环境，
不假定环境名称。数据/旧缓存路径沿用已有约定。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git fetch origin
git switch research/visibility-routing
git pull --ff-only
```

先做完整链路计时，随机权重仅检查成本，不代表精度或官方500ms认证。
输出用新路径，已有JSON不会被覆盖。

```bash
for arch in lunar_conv_tx_content lunar_conv_tx_region_learned; do
  python benchmark_inference.py \
    --data-root ~/liuanda/LunarRM --arch "$arch" --band 58 --split val \
    --device cuda --amp --samples 64 --warmup 3 \
    --out "results/visibility_routing_v1/latency_random/${arch}.json" || break
done
```

三模式各两个seed，串行训练；150轮、batch16、AdamW0.05、lr1e-4、cosine至1e-6、
clip0、D4、AMP和8workers与T的下载provenance一致。没有更改学习率或训练预算。
历史T约4小时/次，六次先按至少约24小时量级预留，再根据实际首轮耗时更新；
这不是新结构的服务器实测。若只启动主候选，将arch列表保留learned即可，仍一起运行两个seed。

```bash
for arch in lunar_conv_tx_region_uniform lunar_conv_tx_region_fixed lunar_conv_tx_region_learned; do
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
      --out "runs/visibility_routing_v1/${arch}/seed${seed}" \
      --run-name "${arch}_58_masked_single_los_distance_adamw" || break 2
  done
done
```

全量官方val评估；新输出目录不覆盖T或之前实验。

```bash
for arch in lunar_conv_tx_region_uniform lunar_conv_tx_region_fixed lunar_conv_tx_region_learned; do
  for seed in 0 1; do
    python evaluate.py \
      --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
      --ckpt "runs/visibility_routing_v1/${arch}/seed${seed}/${arch}_58_masked_single_los_distance_adamw_best.pt" \
      --split val --batch-size 16 --num-workers 8 --device cuda --amp \
      --out "results/visibility_routing_v1/${arch}/seed${seed}/val.json" || break 2
  done
done
```

主候选训练后再用实际权重计时；固定样本、batch=1、全部现场特征成本。

```bash
for seed in 0 1; do
  python benchmark_inference.py \
    --data-root ~/liuanda/LunarRM \
    --ckpt "runs/visibility_routing_v1/lunar_conv_tx_region_learned/seed${seed}/lunar_conv_tx_region_learned_58_masked_single_los_distance_adamw_best.pt" \
    --split val --device cuda --amp --samples 64 --warmup 3 \
    --out "results/visibility_routing_v1/latency_trained/learned_seed${seed}.json" || break
done
```

本机辅助验证/日志放在仓库外`local_experiments/visibility_routing_20261101`；
不提交docs、测试临时权重或本地数据。

2026-11-01本地验证：三模式初始FP32/AMP预测与同seed T逐位一致，距离/几何D4、
严格checkpoint读写、显式已训T初始化、门控梯度与uniform非永久对称检查通过。
真实CLI的两轮微型训练/续训及评估通过；并验证batch16 AMP一次完整优化更新。
learned为26,473,931参数，较T增加108,838（约0.413%）；uniform/fixed为26,473,887。

RTX4060 Laptop、随机权重、batch1、16个不同val样本、3个独立warmup，现有benchmark：

| 模型 | 全链路均值 ms | P95 ms | 最大 ms |
| --- | ---: | ---: | ---: |
| 原T | 107.31 | 116.32 | 119.25 |
| learned | 132.66 | 145.22 | 146.14 |

计入现场F、输入传输、模型内rho/新几何/全部分支、CPU dB输出；磁盘读取单列。
16/16观察样本≤500ms，不是官方认证、服务器实测或模型精度证明。
batch16更新的PyTorch峰值已分配显存约4.0GiB；训练开销须以服务器实际情况为准。
