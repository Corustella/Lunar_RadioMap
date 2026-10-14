# 单次漫反射物理特征：轻量输入与参考实现

日期：2026-10-14。新增轻量构造器；原精确双线性求交版作为物理参照保留。
本次不修改 LOS/distance、模型、checkpoint、损失、训练流程或原始数据。

## 轻量输入与 500 ms 计时口径

官网[任务说明](https://github.com/lunarradiomapchallenge/lunarradiomapchallenge.github.io/blob/main/index.html)
写明方法整体运行时间最多 500 ms；[提交说明](https://github.com/lunarradiomapchallenge/lunarradiomapchallenge.github.io/blob/main/dataset.html)
说明由组织方运行 notebook 核验每图时间。公开材料未细分读盘、模型初始化、batch、
硬件以及平均/尾延迟规则。因此本地采用保守工程预算：

`原始 H/TX 读取 → LoS → 漫反射 → 编码/传输 → 两 U-Net（含 distance）→ dB 数组回传`。

**特征单独低于 500 ms 不等于整体达标。** 训练时可缓存；新图推理的地形相关计算
仍须计时。固定方向和材质查表可随模型一次初始化，不能将新图几何伪装成这类静态准备。
初始化和首次热身在报告中单列；这不代表已经确认组织方也排除它们。

新增 `diffuse_light.py` / `diffuse_light_geometry.py`：

- 采用 TX/RX 双中心分区，两个权重分别为 `rr²/(rt²+rr²)` 和 `rt²/(rt²+rr²)`，
  相加为 1；各自做完整球面积分，缓和单中心求积遗漏另一中心附近贡献的问题。
- 两个积分均按真实 TX→地形的入射方向计算 V 偏振反射率，不假定传播互易性。
- TX 的首次交点、法向和反射率每图仅构造一次，供所有 RX 共享。
- 使用原始 1 m DEM 的半像素并行采样、6 次二分、secant 和地表投影，替代逐 cell DDA。
  这是数值近似，可能漏掉相切或单步内进出地形，不能称为精确求交。
- 几何为 FP32；有限板层的 TE/TM 反射功率先用复 FP64 公式生成 4097 点查表，随后插值。
  S、介电常数、电导率、厚度、AGL 高度、频率均不因提速而改变。
- 默认 `16×16 RX 锚点、每中心 4×16=64 个方向`；RX 高度在原始 DEM 上插值后加 1 m。
  先插值线性功率至 256×256，再编码；不会将 DEM 本身降采样用于遮挡判断。
- 默认是保留运行余量的工程选择，并非精度最优结论。该图是粗尺度间接照明上下文，
  **不能声称保留像素级遮挡边界或已经改善 RMSE**。全分辨率 LOS/distance 仍保留。

双中心表达式为：

\[
G=C\frac{S^2}{\pi}\left[
\int_{\Omega_p}\frac{R_{\rm eff}c_iV_t}{r_t^2+r_r^2}\,d\Omega_p+
\int_{\Omega_t}\frac{R_{\rm eff}c_oV_p}{r_t^2+r_r^2}\,d\Omega_t\right],
\quad C=(\lambda/4\pi)^2.
\]

每项 first hit 处理本侧可见性，另一段仍独立检查；两项都限制反射半球。
材料、偏振和功率权重依据官方固定版本代码；分区等式由变量替换得到；
作为模型输入能否提高泛化和 RMSE，仍是研究假设。

独立有限平面检查：每中心 64 方向的双中心增益误差约 −0.408 至 +0.024 dB，
并非每例都优于单中心；这不是预测标签 RMSE。小平滑地形上 300 条射线对 DDA
无 hit-mask 差异、最大距离误差 1.26e-6 m，但不构成复杂地形的误差上界。
查表反射功率绝对误差最大 6.17e-7；小场景 D4 相对误差 1.89e-7。

本机 RTX 4060 Laptop，58 单频、batch=1、真实 train 前三行各重复两次，预热后：

| 构造设置 | 漫反射均值 | 完整预算均值 | 完整预算最大值 | 超过 500 ms |
| --- | ---: | ---: | ---: | ---: |
| 32×32 原型 / 每中心 64 方向 | 466.19 ms | 541.47 ms | 586.59 ms | 6/6 |
| 24×24 生产版 / 每中心 64 方向 | 317.84 ms | 405.42 ms | 423.46 ms | 0/6 |
| 16×16 生产版 / 每中心 64 方向 | 174.47 ms | 258.24 ms | 294.74 ms | 0/6 |

16×16 的描述性小样本 P95 为 291.92 ms。只检查三张图，不是正式尾延迟保证；
初始化约 12.45 s、首次热身约 1.43 s，均另列。旧 checkpoint **未消费新特征**，
这里测的是构造新特征加现有网络的计算预算，不是新模型的正式推理或竞赛达标证明。
若后续训练新输入、换硬件、加两频联合处理/TTA/集成，必须重新按实际链路验收。
曾测到仅特征 138 ms，但完整链环境下明显更慢；没有用较快值替代失败记录。

`LightDiffuse(config).feature(raw_height, tx)` 返回设备上的 `[1,1,H,W]` FP32 张量，
适合后续输入拼接；`.feature_numpy` 仅供离线缓存。训练脚本尚未增加新特征模式，
不能把轻量缓存当作 LOS 缓存、不能用旧 checkpoint 直接恢复新输入训练。

在现有 PyTorch 环境中同步代码后，可先预检（不写缓存）：

```bash
python build_diffuse_cache.py \
  --data-root ~/liuanda/LunarRM \
  --cache-root results/diffuse_light_58_rx16_mu4_phi16 \
  --method light --band 58 --rx-grid 16 --n-mu 4 --n-phi 16 \
  --splits train val --device cuda --dry-run
```

实际构造仍须计入运行成本。限量检查使用新目录、去掉 `--dry-run` 并加
`--limit-rows 2`；正式全量构造去掉限制。缓存保存完整近似配置，和 reference 不混用。

服务器现有 checkpoint 的完整预算检查入口（不训练、不读 target/mask）：

```bash
python benchmark_physical_inputs.py \
  --data-root ~/liuanda/LunarRM \
  --ckpt runs/los_distance_v1/seed0/secondU/radiownet_58_masked_secondU_los_distance_both_best.pt \
  --split val --indices 0 1 2 --repeats 2 \
  --rx-grid 16 --n-mu 4 --n-phi 16 --ray-chunk 8192 \
  --out results/diffuse_light_58_rx16_budget.json
```

checkpoint 路径需替换为服务器实际存在的 LOS-distance 文件；这里没有声称已访问服务器。
输出 JSON 分别保存静态初始化、首次热身、特征耗时和含/不含读盘的完整预算，拒绝覆盖已有报告。

以下章节保留原参考实现的推导和验收记录。

## 计算对象与依据

输出为仅由原始米制高度图、TX 位置和 `metadata.json` 构造的一次 Lambertian
漫反射增益代理。无标签、无 mask、无 firstU 预测参与。缓存存储经过固定 soft-log
压缩的单通道 float32 图，按 index 行排列；线性增益可通过 `diffuse_gain` 单独获取。

官方机制依据为 **Sionna RT v1.1.0**，不是当前最新版行为：

- [材质及散射 Jones 矩阵](https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/radio_materials/radio_material.py)
- [有限厚度板层系数](https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/utils/electromagnetics.py)
- [Lambertian 方向分布](https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/radio_materials/scattering_pattern.py)
- [iso/V 天线](https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/antenna_pattern.py)
- [各路径功率累计](https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/radio_map_solvers/mesh_radio_map.py)

5.8 GHz 分支实际读取 5.775 GHz、TX/RX AGL 3/1 m、介电常数 3.411、
电导率 0.009324 S/m、板层厚度 10 m、散射系数 S=0.632、XPD=0.1。
这些均来自元数据，不作为可任意调节的超参数。415 MHz 使用其自己的频率和电导率，
不能跨频复用同一物理缓存。

原表面积分为：

\[
G(p)=\left(\frac{\lambda}{4\pi}\right)^2
\int_A \frac{S^2 R_{\mathrm{eff}}(q)c_i c_o}
{\pi r_{tq}^2r_{qp}^2}V(t,q)V(q,p)\,dA.
\]

RX 仅离地 1 m，少量粗面片中心可能漏掉近区贡献。实现改用同一积分的变量替换
\(d\Omega_p=c_o dA/r_{qp}^2\)：

\[
G(p)=\left(\frac{\lambda}{4\pi}\right)^2
\int_{\mathbb S^2}\frac{S^2 R_{\mathrm{eff}}(q)c_i}
{\pi r_{tq}^2}V(t,q)\,d\Omega_p.
\]

其中 q 是从 RX 向某方向首次命中的地形点；无命中、TX 段遮挡、TX/RX 位于
散射面不同侧时，该方向贡献为零。采样整个球面，包括地平线上方的山坡。
首次命中已处理 RX 段可见性，并吸收出射余弦和 RX 段距离因子，不能再次乘入。
接收端为各向同性探针，不另乘接收天线方向增益。

`R_eff` 是按入射 V 偏振投影到 TE/TM 后的板层反射功率系数，不做非偏振平均。
漫反射功率乘 S²，不再乘镜面项的 (1-S²)。v1.1.0 的 XPD 矩阵保持范数，
一次散射到各向同性 RX 时，不把 Kx 或 (1-Kx) 当作功率损耗。
不同路径在功率域累加，不引入相干双径干涉或衍射项。

## 明确的近似与未核实项

1. 公开输入只有高度栅格，不能精确恢复原始三角 mesh。使用双线性地形、其解析导数
   法向；网格线上的导数采用对称平均。逐 cell DDA 解析求交，仅对这一重建表面精确。
2. RX 位于栅格高度+1 m；未复现原生成器的三角形面积平均、中心采样及 griddata 插值。
3. 只计算一次漫反射，不包含多次反射/散射/透射组合，不是官方总路径损耗或分量真值。
4. 元数据声明 iso/V，但未记录 TX 姿态。本实现明确采用 identity 姿态、世界 z 垂直偏振
   约定；原始生成代码中的历史姿态尚未独立核实。配置会保存这一限制，不伪称已核实。
5. 仅在输入栅格范围内追踪，不做地图边界外地形外推。
6. 方向求积使用 cos(theta)、phi 均匀中点，phi 个数须是 4 的倍数，方向集保持 D4。
   它仍是有限角求积，**没有经过验证的默认方向数**。
7. 几何与功率累计为 float64，起点自交排除距离为 1e-4 m；不会平移一个像素来躲开遮挡。

固定编码为 `log1p(G/g_floor)/log1p(g_ref/g_floor)`，其中
`g_floor=10**(-PL_train_max/10)`、`g_ref=10**(-PL_train_min/10)`。
使用训练元数据的全频 PL 边界，不重估 val/test 范围；G=0 保持为 0，不是 PL=0 dB。

## 已完成的数值检查及实用性限制

以下是本机数值检查，不是 Sionna 运行或模型训练结果：

- 复板层系数对独立 NumPy 公式最大误差 2.22e-16；无损 R+T 守恒误差 3.33e-16。
- S² 缩放、单次各向同性接收的 XPD 不变性通过；高度整体增加 400 m 不变。
- 小 DEM 的全部 D4 变换最大相对误差 2.45e-16。
- 几何求交通过解析平面、鞍面、相切、边界及 180 条随机射线的独立全格求根对照。
- 仅含输入的合成 test 缓存、partial 标识、复用、不匹配拒绝与失败缓存保护通过。

独立高密度面积积分对照显示：

| 方向求积 n_mu × n_phi | 平面相对误差 | 倾斜平面相对误差 |
| --- | ---: | ---: |
| 16 × 32 | 29.87% | 15.40% |
| 64 × 128 | 3.28% | 1.81% |
| 128 × 256 | 0.762% | 0.0934% |
| 256 × 512 | 0.0162% | 0.00332% |

角求积误差可能非单调；这些仅为两个简单场景，不能保证真实地形精度。
本地 RTX 4060 Laptop 8 GB 上，真实 train 第 0 行完整 256×256 图、4×8 方向、
ray chunk=262144、float64，`diffuse_gain` 计算耗时 **87.66 s**，未计输入读盘和缓存写盘；
峰值 PyTorch CUDA 分配约 190 MB。
该低方向数只用于计时与真实输入通路检查，没有通过角收敛验收。
本实现尚不适合直接作为在线推理输入，也不推荐目前启动全量缓存或模型训练。
提高速度和减少角求积误差仍是必须解决的数值问题，不能通过修改物理参数掩盖。

项目内证据：现有 LOS/distance 未显式表达间接路径。本实现是否改善 RMSE，
仍为待验证研究假设；没有新增训练结果。

## 命令与缓存

使用已有 PyTorch/NumPy，不需安装 Sionna、SciPy 或新依赖。
脚本优先采用数据目录内的实际 loader，仅读取 `_heightmap`、`_tx` 原始输入。
下面仅为配置预检，**不创建缓存、不计算地图**；方向数是示例，不是推荐正式参数。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch main
git pull --ff-only
python build_diffuse_cache.py \
  --data-root ~/liuanda/LunarRM \
  --cache-root results/diffuse_reference_58_mu16_phi32 \
  --band 58 --splits train val \
  --n-mu 16 --n-phi 32 --device cuda --dry-run
```

正式生成会执行耗时物理求积。确需限量数值检查时，在新 cache-root 使用
`--limit-rows N` 并去掉 `--dry-run`；会明确记录 partial，不能当作完整训练缓存。
`--chunk` 是射线批量；实际每批至少保留一个 RX 的全部方向。
`--splits test` 支持无标签输入，但本地实际数据没有 test split，未声称完成真实 test 构造。

每 split 输出 `<split>.npy/.json`；JSON 保存频段、完整物理和数值配置、实际 loader、
index 行、partial 范围、代码/环境版本、完成状态及逐图耗时。
完整且相符的缓存只读复用，失败或不匹配缓存拒绝覆盖。
此处尚未添加 Dataset/model 适配器；不能直接把缓存塞进现有 checkpoint 并声称兼容。

## 复杂度

N=H×W，K=n_mu×n_phi，L 为逐射线跨越 cell 数（上界 H+W）。
时间 O(NKL)，额外工作内存 O(N+max(chunk,K))；落盘每图 256 KiB。
全部 train/val 约 7.35 GB/频段，仅是数组大小，不含 JSON 和物理构造耗时。
