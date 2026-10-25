# T 内部嵌套跳连：全尺度主候选与局部范围消融

本轮按用户决定只做 **seed0**：`lunar_conv_tx_nested_full` 为主候选，
`lunar_conv_tx_nested_local` 为范围消融。对照复用原 T0（`lunar_conv_tx_content`），
两组均从头端到端训练，不加载 T best。原 T 仍是强基线，正式收益待实验。

## 问题、依据与可检验假设

- **项目证据**：T 已有原分辨率地形分支与多尺度 skip，高变化区仍是主要误差来源。
  以成熟 T 为起点的后置细化收益有限，尚未检验首次预测内部的嵌套特征融合。
- **文献支持**：[UNet++](https://arxiv.org/abs/1807.10165) 及
  [扩展论文](https://arxiv.org/abs/1912.05074)在医学分割中研究嵌套密集跳连，
  通过逐级处理缩小浅层特征与解码特征的语义差异；
  [作者实现](https://github.com/MrGiovanni/UNetPlusPlus/blob/master/keras/segmentation_models/xnet/builder.py)
  使用同尺度累计拼接与相邻尺度上采样。
- **研究假设**：让地形细节在最终解码前逐级结合上下文，可能改善月面路径损耗重建。
  医学分割与月面连续电磁回归的数据、标签、损失不同，文献不直接证明本项目收益。
  直接 skip 本身不丢失其张量数据；本轮检验的是融合方式是否限制了特征利用。

这是 UNet++ 启发的任务适配：保留 T 的通道、双卷积、转置上采样和单头回归，
不称原论文完整复现，也不将普通嵌套连接本身包装为原创算法。
“全尺度”表示嵌套覆盖全部尺度，不是 UNet3+ 的所有尺度同时直接拼接。

## 数学拓扑

原生特征 E0…E5 对应 256、128、64、32、16、8 分辨率，通道为
`[16,64,64,128,256,512]`。E3 是经过原 T 的 TX 内容交互后的特征。
各行新节点的输出通道为 `[16,32,64,128,256]`，其中 E1 为64通道而新节点为32通道。

全尺度令 d=5、X[i,0]=Ei。所有 i+j≤d、j≥1 的节点为：

\[
X_{i,j}=H_{i,j}\left([\operatorname{Up}_{i,j}(X_{i+1,j-1}),
X_{i,0},X_{i,1},\ldots,X_{i,j-1}]\right).
\]

Up 是原 4×4、stride2、padding1 转置卷积＋ReLU；H 为两个3×3卷积＋ReLU，
节点之间不共享参数。拼接顺序为上采样、原生特征、同尺度前序节点。
对角线 i+j=d 对应原 T 的最终解码路径；其余是新增的旁路节点。
最终沿用原头 `ReLU(Conv3×3([X[0,d],H,TX,F,rho]))`，直接预测归一化路径损耗。

局部版本 d=3：X[0,0]=E0、X[1,0]=E1、X[2,0]=E2，
X[3,0] 为原 T 经8→16→32解码得到的 D32；再使用同一递推。
它新增 X[0,1]、X[1,1]、X[0,2]，拓宽原 D128/D256 的融合输入。

| 结构 | 嵌套输出尺度 | 整个 decoder 的节点数 | 总参数 | 相对 T 增量 |
| --- | --- | ---: | ---: | ---: |
| T | 无 | 5 | 26,365,093 | — |
| 全尺度 | 16/32/64/128/256 | 15 | 28,430,213 | 2,065,120 |
| 局部 | 64/128/256 | 8 | 26,489,445 | 124,352 |

没有新增深监督、边缘损失、输出残差、门控、第二个 U 或物理特征。
全部参数可训练；推理只读取 H/TX/F，内部计算原 log-distance。
GT 梯度及官方有效性 mask 不参与模型输入或选区。

## 初始化与 checkpoint

先按 seed0 构造随机 T，再在 CPU RNG fork 内构造嵌套节点。
两候选共有的3个旁路节点先构造，确保它们的初值一致；公共编码器、TX交互、
上采样、输出头及原融合权重保持 T 的同 seed 初值，构造后的随机状态也保持。

拓宽对角线融合卷积时，复制原上采样＋原生 skip 的随机权重，新增输入权重置零；
其他新增节点正常随机初始化。数学上的初始函数与随机 T 相同，浮点卷积算法可能产生舍入差。
首步新增连接权重可获得梯度；旁路内部要等连接权重离开零值后才获得任务梯度。
这不是冻结、加载成熟 T 或增加训练阶段。有限 AMP 验证须确认内部节点确实更新，
不能用 AdamW 衰减导致的参数变化代替任务学习证据。

`model_config` 记录递推、节点表、宽度、范围和初始化；物理尺度仍为原 `feature_config`。
full/local/T 不能交叉初始化、恢复或评估；`--resume auto` 仅恢复本架构的 last。
旧 T、C、MaxViT 和 RadioWNet 的定义与权重键未修改。

## 计算成本与判定

深度 d 的三角形有 d(d+1)/2 个节点。每个节点的卷积时间约为
O(B HW (4bc + 9c(e+jc) + 9c²))，b 为粗尺度输入宽度、e 为原生 skip 宽度、
c 为输出宽度；转置卷积按低分辨率输入格点计数。参数不含 HW 因子，
激活还要保存密集拼接，约 O(B Σ节点 HW(e+jc+b))。随嵌套深度增加，
节点数与拼接宽度均增长；256尺度的计算开销不可由参数比例代替。

卷积解析估计（1 MAC 为一次乘加，不计激活、归一化、拼接、访存）：
原 decoder 为3.087008 GMAC/图；全尺度12.566135，增量9.479127；
局部5.737808，增量2.650800。编码器、TX交互、物理特征与输出头不变。
服务器实际吞吐/显存以日志为准；同机 `benchmark_inference.py` 包含现场 LOS 构造。

沿用 T：58、single、masked MSE、150 epochs、batch16、AdamW（Conv/Linear
weights-only decay0.05）、lr1e-4、cosine到1e-6、clip0、D4、AMP、8workers。
不更换 loader、split、归一化或 LOS 缓存；固定 val 的 pooled valid-pixel RMSE 选 best。

先比较 full−T0、local−T0、full−local。约0.03dB仅作工程推进参考，
不是统计显著性阈值。单 seed 只用于本轮筛选，不声明跨 seed 稳定提升。
正式分析同时检查历史固定的高变化区、远离缺测邻域的高变化区、非高变化区和
全有效8×8块均值误差，避免把子集转移当作普遍改善。
本轮范围变化伴随容量/计算变化；若有效，后续再考虑匹配容量的普通卷积对照，
不能仅凭本轮将全部收益归因于嵌套拓扑。

## 本地提交、服务器同步

本地尚未提交/推送时，检查并提交这四个文件；不要提交忽略的 docs 或本机 QA 产物。

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git status
git diff --check
git add lunar_nested.py model_factory.py README.md LUNAR_T_NESTED_SKIPS.md
git diff --cached --stat
git commit -m "Add full and local nested skip controls on T"
git push origin research/conv-alignment
```

用户在 SSH GPU 服务器激活训练 T 的既有环境，并确认以下目录仍有效。
不安装新依赖；Codex 未在服务器执行。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch research/conv-alignment
git pull --ff-only
```

## 两候选各 seed0，串行运行

以下共两次150轮训练。全尺度增加较多高分辨率计算，请根据首次 epoch 的实际耗时
预估余时，不将本机小样本耗时当作服务器训练时间。两组使用独立新目录。
训练失败会退出循环；首次从头训练，重跑同命令从各自 last 恢复。
不要加载旧 T best，也不要添加 `--eval-initial`。

```bash
for arch in lunar_conv_tx_nested_full lunar_conv_tx_nested_local; do
  python train.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --arch "$arch" --phase single --band 58 --loss masked \
    --optimizer adamw --weight-decay 0.05 \
    --epochs 150 --batch-size 16 --lr 1e-4 \
    --lr-sched cosine --lr-min 1e-6 --clip 0 \
    --num-workers 8 --device cuda --amp --seed 0 \
    --init-from none --resume auto \
    --panels 6 --panels-every 5 --plots-every 5 \
    --out "runs/${arch}_v1/seed0" \
    --run-name "${arch}_58_masked_single_los_distance_adamw" || break
done
```

两组完成后，完整验证并导出逐样本 SSE/count：

```bash
for arch in lunar_conv_tx_nested_full lunar_conv_tx_nested_local; do
  python evaluate.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --ckpt "runs/${arch}_v1/seed0/${arch}_58_masked_single_los_distance_adamw_best.pt" \
    --split val --batch-size 16 --num-workers 8 --device cuda --amp \
    --out "results/t_nested_skips_v1/${arch}/seed0/val.json" || break
done
```

同一空闲 GPU 测 T/full/local 的完整在线推理成本。原 T 路径沿用既有 runs 约定，
若已归档，仅将它改为实际 T0 best 的位置。输出文件已存在时脚本拒绝覆盖，请保留原测量
或选择新的结果路径，不为计时重训 T。

```bash
for arch in lunar_conv_tx_content lunar_conv_tx_nested_full lunar_conv_tx_nested_local; do
  python benchmark_inference.py \
    --data-root ~/liuanda/LunarRM \
    --ckpt "runs/${arch}_v1/seed0/${arch}_58_masked_single_los_distance_adamw_best.pt" \
    --split val --device cuda --samples 16 --warmup 3 --amp \
    --out "results/t_nested_skips_v1/${arch}/seed0/inference.json" || break
done
```

下载两组 `runs/lunar_conv_tx_nested_*_v1/seed0/` 下的 history、summary、provenance、
best/last、曲线/示例图，以及 `results/t_nested_skips_v1/`。
训练器记录实际 Git 版本、loader、环境、数据路径、配置和 checkpoint；
需先提交同步再正式运行，便于与原 T0 配对解释。

## 本机验证记录

辅助脚本和报告位于实际仓库外
`D:/codes/LunarRadiomap/local_experiments/t_nested_skips_20261025/`。

- 与独立手写的全尺度15节点/局部8节点计算图逐项对照，通过；测试使用非零新增连接
  和非零 TX 输出投影，避免零初始化掩盖错误接线。
- T 公共权重、两候选共有旁路权重、构造后的 CPU/CUDA RNG 一致；旧 T best 严格加载通过。
  初始 FP32 最大差为全尺度2.98e-8、局部0；AMP train/eval 最大差均为0.00024414
  （归一化输出），属于本机卷积舍入差，不声称逐位相同。
- batch16 AMP、两个真实样本重复的24步检查通过；第23、24步全部被检查的旁路及主解码
  卷积权重均有非零梯度，更新超出仅权重衰减。所有参数进入优化器，未冻结编码器。
- RTX4060 Laptop、torch2.5.1+cu121：全尺度/局部峰值 allocated 约5.07/3.99GiB。
  16个不同输入、现场 LOS 的 batch1 总推理均值约107.91/104.28ms；包含输入读取则为
  109.50/104.58ms。这是本机短程成本观察，不是服务器吞吐或最坏情况保证。
- 两组实际 tiny train→resume→evaluate→online benchmark 均通过，验证150轮日程、
  optimizer/scaler恢复、完整参数训练、best选择和物理/结构配置。夹具只有2train/2val，
  每组仅两轮各1batch；临时权重已清理，日志保留。
- 文档7条展开后的 Python 命令均经原 argparse 解析，正式训练设置逐项匹配原 T0 provenance。

结果见 `verification.json`、`integration.json` 和 `README.md`。沿用 torchvision 的
meshgrid 未来接口警告，未因此修改模型逻辑；无新增依赖。
本机检查不替代正式完整验证集精度，未进行服务器训练。
