# Transport point：仅替换TX相对坐标编码

模型名：`lunar_transport_point_polar`。对照：`lunar_transport_point`。
这是用户确认的坐标编码实验，保留256²输入/输出网格，不把整张图重采样成极坐标图。

## 1. 核心定义

x=列、y=行，长度单位为米，TX为原点：

```text
dx=(col-colTX)*resolution, dy=(row-rowTX)*resolution
r=sqrt(dx²+dy²)
cos(theta)=dx/r, sin(theta)=dy/r  (r>0)
er=(cos(theta),sin(theta),0)
etheta=(-sin(theta),cos(theta),0)
normal_r=nx*cos(theta)+ny*sin(theta)
normal_theta=-nx*sin(theta)+ny*cos(theta)
```

r=0时固定sin=0、cos=1，即er=+x、etheta=+y，避免除零。该点方位角本来未定义；
旋转/翻转检查在该点使用固定基约定，其余像素按正常矢量变换。
直接编码sin/cos而非原始atan2标量，避免角度接缝。radius使用原网格对角线归一化，
dz继续使用训练metadata的高度跨度，原rho的三维log-distance编码不变。

| 直角坐标几何9通道 | 极坐标几何10通道 |
| --- | --- |
| dx/diag,dy/diag,dz/range,nx,ny,nz,n·toTX,F,rho | r/diag,sin(theta),cos(theta),dz/range,normal_r,normal_theta,nz,n·toTX,F,rho |

原四个方向槽仍对应world +x,+y,-x,-y；仅将其二维方向分量投影到各位置的er/etheta。
没有增加方向槽、改变节点位置或修改邻居连接。

外部输入仍是H/TX/F三通道，网络内部physical仍是H/TX/F/rho。
极坐标只作用于源状态、初始几何、交互Q/K、接收查询及局部读出的几何条件入口。
坐标/法向在联合D4增强后由输入现场计算，FP32、无autograd，不读取标签/mask/RT。
既有F缓存可直接复用，不需要重建数据集或缓存。

## 2. 控制变量与初始化

编码器、五级解码、1024状态/4轮交互、查询邻居、普通点值读出、masked MSE、
训练增强和配方与旧point一致。没有增加高分辨率通路、GNN、双曲模块、场输出或边界辅损。

sin/cos展开比dx/dy多一维，因此8个几何入口权重各增加一个输入列；
cos(theta)新列初始化为0且参与训练，其余列按对应几何量复制同seed旧point的随机初始化。
所有其他权重与构造后CPU随机流逐位一致。复制的是同seed新构造的初始权重，不是已训练旧模型。

| 模型 | 参数量 |
| --- | ---: |
| Cartesian point | 26,679,121 |
| Polar point | 26,680,113 |

增加992参数，约0.0037%。这是周期角度编码必需的输入维数变化，不声称参数量完全相同；
旧nx/ny入口转换为径向/切向语义后，初始函数也不会与旧point完全相同。
正式训练从头开始，不将旧point权重误当新坐标版本的resume或跨架构init-from。

坐标构建时间/额外工作内存均为O(BHW)；仅多一张几何通道及O(BHW)的旋转投影，
原交互/查询的复杂度阶数不变。初始化扩列只在构造时执行。

**共同限制：** 为保持坐标编码单变量，本版本和旧point共用原有笛卡尔采样/插值路径。
前轮确认的64→256查询特征半像素对齐问题，以及8→16上下文的名义格点差异，没有混入本轮修复。
本实验只回答在相同网格约定下更换几何编码的效果；不能据此声称已测出校正网格后的polar收益。
若随后修正插值，必须给新旧坐标同时建立清楚的对照，不能静默改变旧checkpoint语义。

## 3. Checkpoint与本地验证

使用独立arch和version2 model_config，记录通道顺序、原点基、方向投影和初始化规则。
旧point/field默认配置和权重语义保留，严格加载拒绝跨坐标模型或篡改坐标约定。
仅支持band58、phase single、边界辅助权重0；旧训练脚本的损失和模型选择规则不变。

本地验证包括：

- r/sin/cos还原dx/dy；径向/切向法向还原nx/ny；四个物理方向槽还原。
- TX原点、四象限/轴线、角落与8种D4几何变换；物理输入、dz、nz、入射标量、F/rho不变。
- 真实旧point/field best的AMP输出与12ce334原代码逐位相同。
- 新旧随机权重除新增零列外逐位相同，构造后CPU RNG相同。
- polar seed0/1真实batch16各两次AMP更新；全部可训练参数参与反向，梯度有限，新增列有非零梯度。
- FP32/AMP有限输出、batch隔离、严格配置及生产CLI的训练→恢复→评估。

工程检查不代表正式训练收益。本地记录在实际仓库外`local_experiments/transport_polar_20261105/`，
不加入Git，不把tiny检查分数作为新RMSE。

本机RTX4060 Laptop、PyTorch2.5.1+cu121、batch1、AMP、4 CPU threads，随机权重，
64个不同val输入和3个额外warmup，现场F且无label/mask/F缓存：
总耗时mean/P95/max为118.44/136.66/147.02ms，64/64观测≤500ms。
两个seed的batch16工程更新峰值allocated均约6.08GiB；不是服务器训练显存承诺。
上述计时包含几何/极坐标投影和全部网络，磁盘输入另报，不是官方认证或精度结果。

## 4. 服务器同步及整链路先行计时

由用户在SSH服务器运行，先激活现有训练环境；不猜测Conda环境名。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git fetch origin
git switch research/transport-polar
git pull --ff-only
```

现场F、输入传输、全部坐标和法向、网络、CPU dB输出一起计时；无标签/mask/F缓存。
脚本不覆盖已有JSON，重测应换一个输出名。先检查结果，再启动正式训练。

```bash
python benchmark_inference.py \
  --data-root ~/liuanda/LunarRM \
  --arch lunar_transport_point_polar --band 58 --split val \
  --device cuda --amp --samples 64 --warmup 3 --geometry-threads 8 \
  --out results/transport_polar_v1/latency_random.json
```

随机权重计时不证明精度或最坏情况时延，也不是官方认证。

## 5. 正式训练：仅polar point，seed0/1

复用既有Cartesian point的两seed完整训练作为配对性能对照，
不自动重跑已结束配置。新输出目录独立；两个polar运行均从头训练。

```bash
for seed in 0 1; do
  python train.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --arch lunar_transport_point_polar --phase single --band 58 --loss masked \
    --field-boundary-weight 0 \
    --optimizer adamw --weight-decay 0.05 \
    --epochs 150 --batch-size 16 --lr 1e-4 \
    --lr-sched cosine --lr-min 1e-6 --clip 0 \
    --num-workers 8 --device cuda --amp --seed "$seed" \
    --init-from none --resume auto \
    --panels 6 --panels-every 5 --plots-every 5 \
    --out "runs/transport_polar_v1/seed${seed}" \
    --run-name lunar_transport_point_polar_58_masked_single_los_distance_adamw || break
done
```

旧point的history累计每seed约14.8小时，只是既有服务器运行的参考，非本次速度承诺；
新运行以首轮实测epoch_seconds估计，不能用本地tiny运行时间外推。

## 6. 完整官方val评估与判断

```bash
for seed in 0 1; do
  python evaluate.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --ckpt "runs/transport_polar_v1/seed${seed}/lunar_transport_point_polar_58_masked_single_los_distance_adamw_best.pt" \
    --split val --batch-size 16 --num-workers 8 --device cuda --amp \
    --out "results/transport_polar_v1/seed${seed}/val.json" || break
done
```

主要对照为Cartesian point seed0/1的5.041559/5.076248dB，均值5.058904。
各seed均用pooled官方有效像素RMSE，不用逐图RMSE均值，不把两个seed分数均值称为预测集成。
同时查看突变区/普通区、几何边界附近与远处是否发生抵消，以及训练稳定性和完整推理成本。
收益是待验证研究假设，不能将极坐标表征本身说成已证实能降低月球任务RMSE。
