# TX 相对三维几何输入消融

日期：2026-10-03。用户确认：保留当前代码，`grad-weight=0`，从上一轮 λ=0 baseline 初始化；仅向 secondU 加入三个通道，固定 firstU，并提供零通道对照。尚未完成正式训练。

## 依据、问题与假设

- **项目内证据**：现有训练代码仅输入归一化高度图与稀疏 TX one-hot；此前三个 seed 的梯度损失收益平均约 0.00708 dB，固定示例的大轮廓偏移仍在。数值和适用范围见 [配对审查](boundary-replication-review.md)，不能据此认定所有偏移均由几何表征不足造成。
- **文献支持**：Jaensch、Caire、Demir 的 [Radio Map Estimation — An Open Dataset with Directive Transmitter Antennas and Initial Experiments](https://arxiv.org/html/2402.00878v1) §3.2 与 Table 5，在柏林建筑/植被、3.7 GHz 定向 TX 的射线追踪数据上比较相对平面坐标和相对高度编码；RadioUNet 无 LOS 的归一化 RMSE 从 Basic 0.0713 变为欧氏坐标 0.0697。该实验使用其城市几何/天线输入包，并非对月球 dx/dy/dz 的独立验证；也不是当前两阶段模型的固定 firstU 微调实验。
- **待验证研究假设**：针对从稀疏 TX 隐式恢复远距离位置关系的负担，直接给每个像素提供相对 TX 的三维端点几何，可能改善有效像素 RMSE 或区域范围。月球连续地形、各向同性 TX、Sionna 模拟机制和缺测 mask 均不同，不能保证城市论文收益迁移。端点几何不描述中间地形遮挡。

本轮只检验这一假设；不加入 LOS、距离、方位角、物理损失或额外训练数据。本协议按用户最新确认替代之前提案中“两通道、λ=0.03、从 firstU checkpoint 重训 secondU”的未实施方案，历史报告保持原样。

## 特征定义与最小实现

对增强后的 TX `(r_t,c_t)` 与像素 `(r,c)`：

```text
dx_m = (c-c_t) * resolution_m_per_px
dy_m = (r-r_t) * resolution_m_per_px
dz_m = H(r,c) + rx_height_m_agl - H(r_t,c_t) - tx_height_m_agl

dx = dx_m / ((W-1)*resolution_m_per_px)
dy = dy_m / ((H-1)*resolution_m_per_px)
dz = dz_m / (metadata.heightmap_range_m[1]-metadata.heightmap_range_m[0])
```

当前 metadata：256×256、1 m/像素、TX 离地 3 m、RX 离地 1 m、高度范围 `[0,496]` m，所以 `dz=(H-H_tx-2)/496`。x 正方向为图像列增加，y 正方向为图像行增加，不赋予其未核实的地理朝向。TX 像素的 dx/dy 为 0，dz 为 −2/496；有符号特征不截断到 `[0,1]`。

`radiounet.py` 在进入上游 forward 前，从已增强的高度与 TX 重算三通道，再追加到 base input。firstU 仍由上游切片只接收原始 2 通道；`band=both` 时仍只接收原始 3 通道，包括频段标记。secondU 的输入是 `[out1,height,TX,(band_flag),dx,dy,dz]`。zeros 模式构造同样特征后替换为零，以保持相同的特征检查和网络结构。

现有 loader 在归一化高度时会 clip。当前本地全部 train/val 高度范围分别是 `[0,495.091675]` 与 `[0,443.500763]` m，均在 metadata 范围内；本地无 test 高度数组，未检查其范围。若新数据越界，由归一化输入重建的 dz 会继承裁剪，属于已有输入上的几何代理；不能声称重新恢复了被裁掉的原始高度。接收点本身也来自模拟三角网格及插值，规则网格端点高差不是完整传播路径或精确 LOS 标签。

仅扩展三处卷积，保留原层宽度、bias、上游 forward 和 MIT 源码：

| 层 | 5.8 GHz 通道变化 | 新增权重 |
| --- | --- | ---: |
| `Wlayer00[0]` | 3→6 | 540 |
| `Wconv_up00[0]` | 43→46 | 1,500 |
| `Wconv_up000[0]` | 23→26 | 75 |
| 合计 | | 2,115 |

旧 checkpoint 初始化时，原输入列和其他参数逐元素复制，新增末三列补零；不使用 `strict=False` 跳过不匹配层。新模式/尺度保存在 checkpoint，resume 必须一致，evaluate 自动恢复并核对数据 metadata；旧 checkpoint 缺字段按 `none` 加载。特征只依赖测试时可用输入，不使用 target/mask。loader 保持数据目录优先，本地核验实际文件为 `D:/codes/LunarRadiomap/LunarRM/lunar_dataset.py`；每次训练/评估记录其指纹。

特征构造时间、额外空间为 O(BHW)；固定卷积核下新增卷积成本也是 O(BHW)，约 138,608,640 MAC/图。参数增量固定。模型内部 TX/高度检查有 GPU 同步，真实吞吐和全流程推理时间须在服务器测量；这里不把复杂度估算称为实测。没有新增依赖。

## 对照与训练来源

| 设置 | 对照 | 候选 |
| --- | --- | --- |
| 模式 | `zeros` | `tx_xyz` |
| 新增通道 | 0、0、0 | dx、dy、dz |
| 初始化 | 同一训练完成的 λ=0 secondU baseline 全部权重，新列补零 | 同左 |
| firstU | 同一权重，冻结 | 同左 |
| 训练 | masked MSE，grad-weight=0，seed 0，150 epochs | 同左 |
| optimizer/scheduler | 重新启动，继承 baseline 记录的配置 | 同左 |
| 选模 | 全 val `val/rmse_db_masked` 最优 | 同左 |

这是在已训练 secondU 上额外微调，不是从 firstU checkpoint 中未训练的 secondU 开始，也不是 resume。两组都重启 Adam 和 cosine，从同一 baseline 开始额外训练 150 epochs。零通道组控制卷积形状、名义参数量、初始化和训练预算；新增权重在零输入下不能学习有效贡献，不能声称消除了所有功能容量差异。

本地读取的 baseline 是 `D:/ssh_download/boundary_ablation_v1/seed0_grad0/radiownet_58_masked_secondU_best.pt`：best 6.5786123225 dB、epoch 68（零起始）。SHA-256：`46c998a3254ee42946a59bbaae9c8596758d02cbdb9150f59b5c0aea8e3eb836`。其中 firstU 哈希：`941d54c33a6db587df7da5d1712f313a73ee3ff5e242ae1b689c3e868f32f2c2`。这些是已下载产物的来源记录，不是服务器实时状态。

继承设置来自 checkpoint 实际 args：batch 16、Adam、1e-4 cosine 至 1e-6、clip=0、AMP、同步增强、150 epochs、workers=8。runner 检查 phase/band/loss/grad_weight/完整配置，拒绝以 λ=0.03 或已有几何模型充当本轮 baseline。每组训练后检查 firstU 哈希，记录完整命令、checkpoint、数据 metadata 和耗时；训练及评估各自记录环境、Git、代码、实际 loader/index 指纹。大数组没有逐字节哈希，不声称仅靠 index/metadata 证明完整数据身份。

主比较是 tx_xyz 对同轮 zeros；原 baseline 另评估作为上下文。采用候选还应要求优于原 baseline，避免只是比退化的微调对照更好。检查整体 masked RMSE 和预设主边界 SSE，区分定位、幅值和过渡宽度；固定图只作示例。若两项均有改善，冻结协议再做 seed 1/2 配对复核及按 terrain 的配对分析，不能把像素当独立样本或称调参 val 为独立测试。

## 本地验证与局限

- 23 项单元/集成检查通过（原有 17 项 + 新增 6 项，约 24 秒）：边角/中心及批量坐标、天线偏移、八种增强、非法 TX、形状及新增权重补零、梯度可达、firstU 冻结、checkpoint 保存/恢复/评估、模式/尺度不匹配拒绝、消融命令共享设置。
- 本地真实 baseline、三张 val 输入的初始输出与原始高度差核验，数值见 [验证记录](analysis/tx_xyz_validation_20261003.json)。CPU 双精度最大差约 2.22e−16 归一化单位；本地 AMP 三图初始输出均相同；GPU FP32 最大逐像素差约 0.004674 dB，初始 1e−6 归一化容限未通过。结合严格权重补零核验和双精度结果，这与卷积形状改变引起的内核/累积舍入一致；随后以 5e−5 归一化容限检查 FP32、1e−3 检查 AMP，并保留初始失败记录，不声称跨形状 bitwise 相等。
- 本地 CUDA 可用只用于小样本初始化检查，不代表已在服务器训练或测得全 val 新成绩。官方固定 val 正式训练、吞吐及收益均待用户运行。
- 测试中的 `torch.load` FutureWarning 是当前 PyTorch 对未来默认行为的提示；测试通过，不为消警告修改训练逻辑。

本地重跑测试：

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
D:\Anaconda\python.exe -c "import torch,unittest; torch.set_num_threads(2); result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')); raise SystemExit(not result.wasSuccessful())"
```

## Git 与服务器运行

本轮没有 commit/push；`.gitignore` 已有的用户改动继续保留。`docs/` 被忽略，若连同协议归档须显式 force-add，仅添加本轮相关文件：

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git add radiounet.py train.py evaluate.py experiment_provenance.py ablate_coordinates.py tests/test_coordinates.py README.md
git add -f docs/tx-xyz-ablation.md docs/analysis/tx_xyz_validation_20261003.json
git commit -m "Add paired secondU TX-relative 3D geometry ablation"
git push origin main
```

以下命令依据下载 checkpoint 的服务器路径记录生成，仅供用户执行。若路径已经变更，替换仓库/data/baseline 路径；先激活用户已确认的 Conda 环境，不推测环境名称。Codex 没有访问或执行服务器。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch main
git pull --ff-only
test -f ~/liuanda/LunarRM/metadata.json
test -f runs/boundary_ablation_v1/seed0_grad0/radiownet_58_masked_secondU_best.pt
```

先打印只读计划，核对 baseline、所有设置和新输出目录：

```bash
python ablate_coordinates.py --data-root ~/liuanda/LunarRM --baseline-ckpt runs/boundary_ablation_v1/seed0_grad0/radiownet_58_masked_secondU_best.pt --out-root runs/tx_xyz_v1_seed0 --seed 0
```

短程检查使用另一个新目录（每组 1 epoch、3 batches，评估限制到约 48 张，含示例图，预计数分钟，实际依服务器而定）：

```bash
python ablate_coordinates.py --data-root ~/liuanda/LunarRM --baseline-ckpt runs/boundary_ablation_v1/seed0_grad0/radiownet_58_masked_secondU_best.pt --out-root runs/tx_xyz_v1_smoke --seed 0 --smoke --execute
```

正式配对训练加全 val 评估（由历史每组约 6,100 秒估算，两组训练约 3.4 小时，另加评估；新增输入结构和微调来源可能改变耗时，以 smoke 为准）：

```bash
python ablate_coordinates.py --data-root ~/liuanda/LunarRM --baseline-ckpt runs/boundary_ablation_v1/seed0_grad0/radiownet_58_masked_secondU_best.pt --out-root runs/tx_xyz_v1_seed0 --seed 0 --execute
```

输出：`manifest.json`、`comparison.csv`，以及 baseline_reference / seed0_zeros / seed0_tx_xyz 的 val JSON、boundary 区域与逐样本 CSV、固定八图和 provenance；新组另有 best/last checkpoint、训练历史和 loss components。运行目录必须不存在。中途失败先查 manifest 记录的失败命令，不用相同 out-root 强行重新执行整轮。

特征 checkpoint 推理构造示例（输入 loader 应使用默认归一化、不增强；输出仍是完整地图）：

```python
ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
mode, config = checkpoint_features(ck)
model = RadioWNet(inputs=2, phase="secondU", second_features=mode,
                 feature_config=config).to(device)
load_model_state(model, ck)
model.eval()
with torch.no_grad():
    prediction_db = model(normalized_height_and_tx)[1] * 208.0 + 20.0
```

这里的 208/20 仅对应本轮 metadata；通用推理应读取并核对训练保存的 pathloss 范围，不能用 test 数据重估。
