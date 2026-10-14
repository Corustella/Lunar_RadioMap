# 单次漫反射物理特征：参考实现

日期：2026-10-14。**本版本用于核对物理表达与数值计算，不是已经满足比赛推理耗时的输入模块。**
它不修改 LOS/distance、模型、checkpoint、损失、训练流程或原始数据。

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
