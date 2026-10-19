# LOS＋distance 路径剖面编码实验

本轮仅运行 G 方案的 **seed 0**。固定 LOS＋distance RadioWNet，在 secondU
的 Wlayer10.0（64×64，40→50 通道）联合编码局部空间特征和 TX–RX 地形剖面。
架构名为 radiownet_path_profile；它不包含上一轮 TX 局部方向采样分支。

## 功能与证据范围

现有 LOS 沿完整 TX–RX 路径计算净空，但仅保留最小值。新算子保留 64 段
净空的最小值、均值、相对位置和三维段长，由共享的单层 GRU（hidden=32）
从 TX 到 RX 编码。原 5×5 卷积及 bias 保留；50 维局部输出与 32 维末状态
经 82→50 线性融合，再进入原 ReLU、编码器及 skip。

Agile/REM-U-Net 的遮挡比例、IPP-Net 的路径条件输入提供邻近任务启发：
[REM-U-Net](https://arxiv.org/abs/2309.11898)、
[IPP-Net](https://arxiv.org/abs/2501.06414)。
这些城市/室内研究没有验证本项目的有序月面剖面、GRU 或插入位置。
本方案是待验证研究假设，学习状态不是电磁场、功率、相位或累计 dB 损耗；
月球数据没有 diffraction，原二维主干仍负责侧向环境上下文。

## 固定几何与计算定义

高度仅从输入和训练 metadata 恢复。源点 t 来自同步增强后的 one-hot TX；
RX 随地形，使用 metadata 的 TX/RX 离地高度（当前为3/1米）。

\[
q_p(u)=t+u(p-t),\quad
c_p(u)=(1-u)[H(t)+h_T]+u[H(p)+h_R]-H(q_p(u)).
\]

64×64 接收点对应原图中心 (4j+1.5,4i+1.5)，即两次 2×2 pooling 的中心。
每条路径分 K=64 段；每段取 m+1 个等间距点：

\[
m=2\max(1,\lceil\|p-t\|_{\rm px}/64\rceil),\quad
u_{kj}=(k+j/m)/64,\quad k=0,\ldots,63;\ j=0,\ldots,m.
\]

在原 256×256 高度上双线性取样，align_corners=True、border padding。
间距不大于0.5像素，每段至少包含两端和中点；分界点在相邻段各计一次，
mean 为段内采样点的等权平均。零水平长度也有稳定的竖直净空序列。

段 min/mean 采用 asinh(c/1m)/asinh((Hmax−Hmin)/1m) 编码；
相对位置为 (k+0.5)/64；三维段长为
sqrt(horizontal_distance_m²+(zRX−zTX)²)/64，编码为
log1p(length/1m)/log1p(Dmax/1m)，Dmax 复用原 distance 的训练边界。
不使用 target、有效性 mask、路径损耗观测或验证集统计。

几何、GRU 和融合在 FP32 下执行，融合结果转回原卷积 dtype 供 AMP 主干使用。
几何按256个接收点分块，GRU 按4096条序列分块；分块不改变路径顺序。
没有新增依赖，只有现有 PyTorch 算子。

新增7,798参数：GRU 3,648，融合4,150。总参数13,284,523，secondU可训练
2,362,539。融合初始化为 [I,0]、bias=0；保留旧卷积参数，新增初始化不消耗
共享主干之后的 CPU RNG 状态。训练起点与相同 LOS 权重加载后的基线一致。

N=64²、K=64、hidden=C、batch=B 时，几何约 O(BNL)，GRU
O(BNKC(C+4))，训练激活 O(BNKC)。GRU 约0.906 G MAC/图，原卷积
约0.205 G MAC/图，另有融合和几何成本；不是整网增加5.4倍。
分块限制临时张量，但不消除反向所需状态。

## 初始化、训练和结果解释

复用 seed0 的历史 LOS firstU checkpoint，它同时包含继承的 secondU 权重。
新算子与旧权重严格迁移；禁止将 TX-direction 权重或已经训练好的
distance secondU best 作为本轮初始化。resume/evaluate 要求架构、几何、
全部算子配置和权重严格匹配。

训练仍为 secondU 150轮、batch16、Adam、weight_decay=0、lr=1e-4、
cosine到1e-6、AMP、原同步增强及 masked MSE；firstU 完全冻结。
checkpoint 仍按 val/rmse_db_masked 选择。secondU 为直接预测，
没有改成 out1＋residual。

本轮仅做单 seed 工程筛选，与历史 U0 的5.416208 dB对照；不据此声称跨seed
稳定收益或 GRU 的因果贡献。0.10 dB仅沿用为实际推进门槛，不是预计降幅。
若值得继续，再按新的授权补同输入集合聚合和容量对照。

## SSH 服务器同步与运行

在用户已确认的训练环境中执行；沿用既有仓库、数据、LOS缓存和初始化路径。
本地助手未在服务器执行这些命令。新结构耗时应以服务器前几轮日志估算；
150轮属于数小时级以上的正式任务，不能用本机小样本检查推断完整耗时。

先同步并检查完整推理开销。计时不读取标签/mask/LOS缓存，在线计算所有特征；
随机权重计时不是精度证据。计时输出路径已存在时脚本会拒绝覆盖。

~~~bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git fetch origin
git switch research/path-profile-unet
git pull --ff-only

python benchmark_inference.py \
  --data-root ~/liuanda/LunarRM --arch radiownet_path_profile \
  --band 58 --device cuda --samples 16 --warmup 3 \
  --out results/path_profile_v1/inference_random.json
~~~

仅训练 seed 0，不调用默认安排两个 seed 的 run_los.py：

~~~bash
python train.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --arch radiownet_path_profile --phase secondU --band 58 --loss masked \
  --first-features los --second-features los_distance \
  --optimizer adam --weight-decay 0 \
  --epochs 150 --batch-size 16 --lr 1e-4 \
  --lr-sched cosine --lr-min 1e-6 \
  --num-workers 8 --device cuda --amp --seed 0 \
  --init-from runs/los_both_v1/seed0/firstU/radiownet_58_masked_firstU_los_both_best.pt \
  --resume auto --panels 6 --panels-every 5 --plots-every 5 \
  --out runs/path_profile_v1/seed0/secondU \
  --run-name radiownet_58_masked_secondU_path_profile
~~~

完成后用 best 进行完整验证：

~~~bash
python evaluate.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --ckpt runs/path_profile_v1/seed0/secondU/radiownet_58_masked_secondU_path_profile_best.pt \
  --split val --batch-size 16 --num-workers 8 --device cuda --amp \
  --out results/path_profile_v1/seed0/val.json
~~~

下载整个 runs/path_profile_v1/seed0/secondU 和 results/path_profile_v1，
包含 best/last、history、summary、provenance、图像、逐样本验证结果及计时。
沿用有效像素累计 SSE/count 的指标与既定高变化区定义共同分析。
