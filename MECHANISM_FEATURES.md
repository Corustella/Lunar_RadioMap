# 直达面积与单次镜面反射输入实验

本实现针对 **band 58（实际 5.775 GHz）**，在现有 T 的 H/TX/F/rho 之外加入 D 和 S。
这仍是待验证的输入表征假设，不代表已经改善验证集 RMSE。
当前 D 可进入服务器成本复测；S/DS 保留实验接口，尚未通过完整成本及面积近似验收，暂不启动正式训练。

## 定义与依据

| 输入 | 解决的问题 | 当前实现 |
| --- | --- | --- |
| D_A | 原 F 是点到点最小净空，不能直接表示接收三角形的部分可见面积 | 每个 RX 三角形先做三点求积，部分可见的面再细分为 64 个等面积子三角形，以其质心估计可见比例 |
| S_exist | 局部坡度、距离或方向接近不能证明镜面路径成立 | 对每个 RX 三角形质心判断是否存在合法单次反射；再投影至像素网格 |
| S_gain | 描述有效反射路径的相对强度 | `log1p(project(sum(g))/project(G0))`，G0 是相同 RX 点上的自由空间功率 |
| S_extra | 描述有效反射的绕行程度 | `log1p(project(sum(g*(L/d-1)))/project(sum(g)))`，无反射时为 0 |

**官方实现依据：** slab 的 TE/TM 系数、垂直极化投影与镜面功率缩放依据
Sionna RT 1.1.0 的 [electromagnetics.py](https://github.com/NVlabs/sionna-rt/blob/38ff0af9ffcade89ab06391b2ebb39293ae484b2/src/sionna/rt/utils/electromagnetics.py)
和 [radio_material.py](https://github.com/NVlabs/sionna-rt/blob/38ff0af9ffcade89ab06391b2ebb39293ae484b2/src/sionna/rt/radio_materials/radio_material.py)。
它们支持反射系数的计算方法，不证明这些特征对本任务的学习收益。
实现按 metadata 读取介电常数、电导率、厚度和散射系数，使用 `(1-s²)`，不加入 Monte Carlo 选路概率补偿。

**项目内证据：** 重建场景消融支持镜面机制优先；旧 TG 的粗几何偏置没有带来稳定收益。
新 S 增加的区别是镜像源求解、落在实际三角形内的交点、TX/RX 同侧及双段可见性约束。

**研究假设：** 显式可见面积及成立的单次反射路径能提供额外可学习信息。
S 只覆盖一阶镜面路径，不等同于 `full-no_specular`，后者还包含多次反射与混合传播的变化。
本实现不需要原始 Sionna 生成脚本，不调用 Sionna 求解器；仅使用地形、TX 和 metadata。

网格按存储行列坐标构建 NW–SE 对角线的原生三角形，使用平面法向；TX/RX 高度来自 metadata。
FP64 可见性逐一检查直线经过的 x、y、x-y 网格边界，容差 `1e-7 m`，不使用固定步长漏采样。
反射候选用保守的镜面锥体与 RX 四叉树包围盒相交检索，不使用最近 K 面或角度截断。
仅对共面邻面共享边上的重复路径去重；源与面共面的零立体角退化情形跳过并计数。
内部整数像素的线性投影使用相邻两个质心的中点；外围一圈单独作 RX 点求值。
边界规则是完整输入代理，不复现官方 NaN/补洞流程。特征构造不读取标签或有效性 mask。
原始生成网格及完整后处理尚未获得，这些几何约定不能被描述为官方场景的逐像素精确复现。

特征定义版本为 **2**：初始三点 D 在训练地形的部分可见面上有约 11% 平均面积误差，
因此加入局部 64 点加密。四个固定训练地形的检查中，该误差降至约 0.8–0.9%；
这是局部数值筛选，不是全域误差保证。S 的单质心存在性比例误差仍明显，不能声称准确面积估计。

## 模型、缓存与兼容性

| arch 后缀（完整前缀 `lunar_conv_tx_mechanism_`） | D_A | 三个 S 通道 |
| --- | --- | --- |
| `a` | 0 | 0 |
| `d` | 开启 | 0 |
| `s` | 0 | 开启 |
| `ds` | 开启 | 开启 |

四组外部输入均为 `[H,TX,F,D_A,S_exist,S_gain,S_extra]`，模型内部插入 rho。
三个物理输入入口同时扩展：encoder stem、全分辨率分支和输出头物理 shortcut。
新增列初始化为零，共享 T 权重及 CPU RNG 状态完全保留；四组参数量均为 26,368,009。
初始输出接近 T 不代表完整训练轨迹逐位相同，因此正式比较仍保留配对 seed 的 A。
骨干、损失、优化器配置和 D4 规则由原训练入口控制；旧架构与旧 checkpoint 路径保留。

F v1 缓存与 D/S v2 缓存使用不同目录。新缓存为 `[N,4,256,256]` FP32，
train+val 约 **27.39 GiB**；D-only 也保留四个槽位，未计算的 S 槽为零。
manifest 记录几何/材料/求积规则、实际 loader、行索引、计算模式、完成状态和环境。
仅 D 的缓存不能被 S/DS 模型误用；不完整或定义不匹配的缓存拒绝加载，不自动覆盖。
所有标量图在原场景生成后与 H/TX/标签/mask 同步增强；旋转后不重新固定 NW–SE 重建特征。

## 依赖与成本

现有 NumPy/PyTorch 用于数据和网络；新增可选 Numba/CUDA Python 用于几何循环及 CUDA kernel。
只从缓存训练/评估不需要这两个额外包。实时计算和建缓存需安装：

```bash
python -m pip install -r requirements-mechanisms.txt
```

已本地验证 Python 3.12、NumPy 1.26、Numba 0.61.2、CUDA Python 12.6.0、CUDA Toolkit 12.4。
CUDA 计算需要 NVVM、libdevice 和 CUDA runtime；只有 PyTorch CUDA wheel 不一定足够。
Numba 优先查找 Conda toolkit，再查找 `CUDA_HOME`，最后查找 `/usr/local/cuda`。
若缺少 toolkit，可在已确认的实验环境中按 [Numba 官方安装说明](https://numba.readthedocs.io/en/0.61.2/cuda/overview.html#software) 安装，例如：

```bash
conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version=12.4"
```

不在已有可用 toolkit 的环境重复安装。程序使用官方 NVIDIA driver binding；不修改系统驱动。
CPU 后端可作数值参照，不预期满足 500 ms。首次 JIT 可能需要数十秒，计时报告单列 warmup。

令 M 为反射面数、P 为接收求积点数、B 为树测试数、K 为候选对数、L 为可见性遍历长度。
D 约为 `O(P*L)`；S 约为 `O(M+P+B+K*L)`，最坏仍可退化至 `O(M*P*L)`。
CUDA 分两遍精确计数/填充候选，再并行检查路径，工作数组为 `O(M+P+K)`；
CPU 参考版另需 `O(threads*P)` 累积存储。候选缓冲超过 2 GiB 时明确报错，不截断路径。
FP64 原子累加的顺序可能产生舍入级差异，不承诺 CPU/GPU 逐位相等。

完整计时包括新地形的网格/树构建、原 F、新 D/S、rho、搬运、网络及 CPU dB 输出。
磁盘读取另报，训练缓存不代替推理成本。PyTorch 显存统计不覆盖 Numba；报告另列反射工作数组字节数，
它也不等于进程总峰值显存。小样本计时不是比赛官方认证或最坏情况证明。

2026-10-27–28 本机 RTX 4060 Laptop、PyTorch 2.5.1、AMP、batch=1，16 个不同训练样本，
排除 3 个独立 warmup，随机权重下的完整计时如下。包括现场特征构造，不使用训练缓存。

| 版本 2 | 均值 ms | P95 ms | 最大 ms | 不超过 500 ms |
| --- | ---: | ---: | ---: | ---: |
| D | 232.54 | 272.49 | 290.88 | 16/16 |
| D+S | 619.79 | 1304.89 | 1463.29 | 8/16 |

两次独立运行的 GPU 时钟/系统负载未锁定，不用差值推断严格的模块加速比。
S 的主要成本在无截断候选树遍历；服务器需重新计时。此处没有正式训练或完整 val 精度结论。

## 服务器同步与先行复测

在用户已确认的训练环境中运行；不预设 Conda 环境名称。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch research/conv-alignment
git pull --ff-only

python benchmark_inference.py \
  --data-root ~/liuanda/LunarRM \
  --arch lunar_conv_tx_mechanism_d --band 58 --split train \
  --samples 16 --warmup 3 \
  --out results/mechanism_v2_d_latency.json

python benchmark_inference.py \
  --data-root ~/liuanda/LunarRM \
  --arch lunar_conv_tx_mechanism_ds --band 58 --split train \
  --samples 16 --warmup 3 \
  --out results/mechanism_v2_ds_latency.json
```

输出文件已存在时换一个名字，不覆盖结果。两条命令使用随机权重，只检查成本。
即使服务器 DS 计时通过，S 的面积近似仍需解决，不能直接据此启动 S/DS 正式训练。

## D/A 成本通过后的训练与评估

如果已有匹配的 F v1 缓存，可复用并替换以下 `--los-cache` 路径。
建缓存可能需要几十分钟至数小时，依硬件与磁盘而定；先确认至少约 35 GiB 空间供两套缓存使用。

```bash
set -e
python build_los_cache.py \
  --data-root ~/liuanda/LunarRM \
  --cache-root ~/liuanda/LunarRM/los_cache_v1 \
  --splits train val --device cuda

python build_los_cache.py \
  --data-root ~/liuanda/LunarRM \
  --cache-root ~/liuanda/LunarRM/mechanism_v2_d \
  --feature-set mechanisms --mechanism-mode d \
  --splits train val --device cuda --log-every 500

for seed in 0 1; do
  for mode in a d; do
    python train.py \
      --data-root ~/liuanda/LunarRM \
      --arch lunar_conv_tx_mechanism_${mode} --band 58 --phase single --loss masked \
      --los-cache ~/liuanda/LunarRM/los_cache_v1 \
      --mechanism-cache ~/liuanda/LunarRM/mechanism_v2_d \
      --epochs 150 --batch-size 16 \
      --optimizer adamw --weight-decay 0.05 \
      --lr 1e-4 --lr-sched cosine --lr-min 1e-6 \
      --seed "$seed" --amp --out runs/mechanism_v2 \
      --run-name mechanism_${mode}_seed${seed}

    python evaluate.py \
      --data-root ~/liuanda/LunarRM \
      --ckpt runs/mechanism_v2/mechanism_${mode}_seed${seed}_best.pt \
      --los-cache ~/liuanda/LunarRM/los_cache_v1 \
      --mechanism-cache ~/liuanda/LunarRM/mechanism_v2_d \
      --split val --band 58 \
      --out results/mechanism_v2_${mode}_seed${seed}.json
  done
done
```

四次训练按历史 T 的耗时估计约 18 小时，另计建缓存和评估；没有在服务器执行过这些命令。
比较同 seed 的全有效像素 RMSE，并按 terrain 汇总差异；两 seed 一起分析。
不把合成数据冒烟、初始函数等价或面积近似误差当作正式精度收益。
