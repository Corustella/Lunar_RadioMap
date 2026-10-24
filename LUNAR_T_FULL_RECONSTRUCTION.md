# 固定 T 的完整第二 U 与分层重建：U/P，各 seed0

本轮从同一原T seed0 best初始化，固定T，只训练第二阶段：

| 配置 | `--arch` | 第二阶段 |
| --- | --- | --- |
| U，对照 | `lunar_conv_tx_refine_unet` | 完整第二U＋256尺度单头修正 |
| P，主候选 | `lunar_conv_tx_refine_pyramid` | 同骨干＋32尺度均值及64/128/256细节修正 |

按用户批准只做seed0，两组各150轮，不追加T续训或P05重训。

## 依据与功能

项目证据：P05固定T后从5.059348降至5.051296 dB，幅度小；T既有全有效8×8块
误差分解中约58.2%为块均值项、41.8%为块内项。这不是全图频谱分解，也不能
将其解释为LOS/反射等物理机制。已有浅层细化器也已接收T的上下文特征。

文献支持：[RadioUNet §4.3.1](https://arxiv.org/html/1911.09002)采用冻结第一U、
第二U重新解释输入和初稿，数据为城市RadioMapSeer；[LapSRN](https://arxiv.org/abs/1704.03915)
在自然图像超分中逐级重建残差，还采用深监督和Charbonnier损失。
本次借鉴重建组织方式，不是上述算法复现，其月球收益与零均值约束均是待验证假设。

U检验完整第二次区域编码；P检验区域均值与局部形状分工是否比同骨干的单头更有效。
不新增注意力、物理特征、mask输入或损失。

## 结构、初始化与复杂度

第二U编码分辨率为256/128/64/32/16/8，通道16/32/64/96/128/192，随后完整解码。
沿用原double-conv/ReLU、stride2卷积和转置卷积；D256/D128/D64仍接入同尺度编码。
U/P公共模块按相同顺序构造，新模块处于独立RNG作用域，不改变后续数据随机流。
两组第二U从随机权重开始，不加载P05细化权重；所有输出头为零，起点均等于原T。

实现检查发现必须调整新第二U的初始化：沿用浅层默认初始化时，12步小样本AMP
更新后，U的8尺度卷积任务梯度为0，同权重FP32约1.42e−16；Kaiming诊断下
两者均恢复至约1e−8量级。故U/P共同采用Kaiming normal（fan_in、ReLU），
卷积bias为零，再将输出头清零。初始化只影响本轮两个新架构，旧P04/P05不变。
这针对深层梯度衰减，不是验证集调参；也不证明初始化会带来正式精度提升。
因此U对P05同时涉及深层结构与初始化，不能称纯深度消融；U/P公共初始化一致。
初始化使用现有PyTorch的[`kaiming_normal_`](https://docs.pytorch.org/docs/2.14/nn.init.html#torch.nn.init.kaiming_normal_)，
其ReLU增益提供初始化尺度依据；具体可训练性由本地任务梯度检查确认。

令A为不重叠2×2平均，Up为最近邻2倍复制，P定义：

\[
c_{32}=h_{32}(F_{32}),\quad q_s=h_s(F_s),\quad d_s=q_s-\mathrm{Up}(Aq_s),
\]
\[
r_{32}=c_{32},\quad r_s=\mathrm{Up}(r_{s/2})+d_s\quad(s=64,128,256),
\qquad \hat y=\mathrm{ReLU}(y_T+r_{256}).
\]

因此激活前`A d_s=0`、`A r_s=r_(s/2)`；细节头不能改变父级块的平均修正。
池化、中心化、合成和与T相加使用FP32，保留梯度。细节头不设bias，因为它会
被零均值投影抵消；仅粗尺度头保留bias。相较设计估算去掉3个无作用标量。
该表示可表达任意残差图；共享特征、masked loss及最终ReLU意味着训练并未正交解耦。
固定网格仍可能出现块状伪影，不能仅凭视觉锐利认定有效。

| 配置 | 可训练参数 | 全部参数 | 第二阶段卷积GMAC/图 |
| --- | ---: | ---: | ---: |
| U | 3,251,681 | 29,616,774 | 4.024435 |
| P | 3,253,409 | 29,618,502 | 4.032397 |

MAC不含T、激活、池化、加法和bias，不等于完整推理时间。卷积时间复杂度为
`O(BΣHW k² cin cout)`，保存激活约`O(BΣHW c)`；分层合成额外`O(BΣHW)`。
仍需T的前向，固定T只移除其反向图与优化器状态。显存和服务器速度以实测为准。

## 训练与checkpoint约束

- T参数、BN运行状态和随机深度都固定在eval行为，仅`refiner.*`可训练。
- H/TX/F/log-distance、同步增强、masked MSE、归一化和实际loader选择规则不变。
- AdamW，Conv/Linear衰减0.05，lr1e-4，150轮cosine至1e-6，batch16，AMP，clip0。
- 必须`--eval-initial`，保留epoch−1起点；best只有在有效像素RMSE严格改善时更新。
- `--init-from`只接受相同seed的原T；`--resume auto`恢复该配置的last。U/P/P05不得相互resume。
- 每个checkpoint包含T和第二U全部权重，评估只需要该文件。`--phase single`表示一个最终输出。
- 本地检查不能替代正式val。建议工程继续门槛为较原T降低0.03 dB，非预测或统计显著性阈值。
  若P与U接近，不能归因分层输出；若只有区域项改善，不能称边缘恢复。

## 本地提交与服务器同步

本地PowerShell：

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git diff --check
git add lunar_refinement.py model_factory.py train.py README.md LUNAR_T_FULL_RECONSTRUCTION.md
git diff --cached --stat
git commit -m "Add full second-U and mean-detail reconstruction controls"
git push origin research/conv-alignment
```

服务器先激活运行原T时的现有环境；以下沿用项目和数据路径：

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch research/conv-alignment
git pull --ff-only
```

P05日志轮次耗时约3.12小时。本轮增加深层计算，建议先按每组3–6小时预留预算，
两组顺序运行；这是预算估计，尚未在服务器测量。不要同时运行两组造成资源竞争。

## U：普通完整第二U，seed0

```bash
python train.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --arch lunar_conv_tx_refine_unet --phase single --band 58 --loss masked \
  --optimizer adamw --weight-decay 0.05 \
  --epochs 150 --batch-size 16 --lr 1e-4 \
  --lr-sched cosine --lr-min 1e-6 --clip 0 \
  --num-workers 8 --device cuda --amp --seed 0 \
  --init-from runs/lunar_conv_tx_content_v1/seed0/lunar_conv_tx_content_58_masked_single_los_distance_adamw_best.pt \
  --eval-initial --resume auto \
  --panels 6 --panels-every 5 --plots-every 5 \
  --out runs/t_full_reconstruction_v1/lunar_conv_tx_refine_unet/seed0 \
  --run-name lunar_conv_tx_refine_unet_58_masked_single_los_distance_adamw
```

## P：分层重建，seed0

```bash
python train.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --arch lunar_conv_tx_refine_pyramid --phase single --band 58 --loss masked \
  --optimizer adamw --weight-decay 0.05 \
  --epochs 150 --batch-size 16 --lr 1e-4 \
  --lr-sched cosine --lr-min 1e-6 --clip 0 \
  --num-workers 8 --device cuda --amp --seed 0 \
  --init-from runs/lunar_conv_tx_content_v1/seed0/lunar_conv_tx_content_58_masked_single_los_distance_adamw_best.pt \
  --eval-initial --resume auto \
  --panels 6 --panels-every 5 --plots-every 5 \
  --out runs/t_full_reconstruction_v1/lunar_conv_tx_refine_pyramid/seed0 \
  --run-name lunar_conv_tx_refine_pyramid_58_masked_single_los_distance_adamw
```

## 验证与完整在线推理计时

```bash
python evaluate.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --ckpt runs/t_full_reconstruction_v1/lunar_conv_tx_refine_unet/seed0/lunar_conv_tx_refine_unet_58_masked_single_los_distance_adamw_best.pt \
  --split val --batch-size 16 --num-workers 8 --device cuda --amp \
  --out results/t_full_reconstruction_v1/lunar_conv_tx_refine_unet/seed0/val.json

python evaluate.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --ckpt runs/t_full_reconstruction_v1/lunar_conv_tx_refine_pyramid/seed0/lunar_conv_tx_refine_pyramid_58_masked_single_los_distance_adamw_best.pt \
  --split val --batch-size 16 --num-workers 8 --device cuda --amp \
  --out results/t_full_reconstruction_v1/lunar_conv_tx_refine_pyramid/seed0/val.json

python benchmark_inference.py \
  --data-root ~/liuanda/LunarRM \
  --ckpt runs/t_full_reconstruction_v1/lunar_conv_tx_refine_unet/seed0/lunar_conv_tx_refine_unet_58_masked_single_los_distance_adamw_best.pt \
  --split val --device cuda --samples 16 --warmup 3 --amp \
  --out results/t_full_reconstruction_v1/lunar_conv_tx_refine_unet/seed0/inference.json

python benchmark_inference.py \
  --data-root ~/liuanda/LunarRM \
  --ckpt runs/t_full_reconstruction_v1/lunar_conv_tx_refine_pyramid/seed0/lunar_conv_tx_refine_pyramid_58_masked_single_los_distance_adamw_best.pt \
  --split val --device cuda --samples 16 --warmup 3 --amp \
  --out results/t_full_reconstruction_v1/lunar_conv_tx_refine_pyramid/seed0/inference.json
```

下载两组initial/history/summary/provenance、完整results及best权重，保留服务器best/last。
best权重用于沿用既定高变化区及块误差口径的复核，不需要下载last。
统一比较原T、P05、U、P；分清训练前起点与真正训练后best，单seed结论只作筛选。
