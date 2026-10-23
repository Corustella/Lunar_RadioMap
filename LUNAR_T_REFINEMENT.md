# T 的跨阶段浅层细化

本轮以已训练 T（`lunar_conv_tx_content`）为起点。对照 T-cont 与候选
`lunar_conv_tx_refine` 各运行 seed0/1；后者有两个结构阶段，但使用 `--phase single`
进行联合优化，只监督最终输出。它不是冻结 firstU 的旧 RadioWNet secondU。

## 功能与依据

项目证据：T 的高变化区两 seed 平均 RMSE 从 C 的8.3931降至8.3109 dB，但高跳变
相邻像素对的梯度误差几乎不变。细化阶段同时针对区域幅值和局部形状，不限定为锐化器。

[RadioUNet §4.3.1](https://arxiv.org/html/1911.09002)支持对初稿进行回顾细化；
[MPRNet](https://arxiv.org/abs/2102.02808)及
[作者实现](https://github.com/swz30/MPRNet/blob/main/Deblurring/MPRNet.py)支持跨阶段特征交换。
它们分别研究城市无线电地图和自然图像恢复；本实现是月球任务的待验证适配，
不复现整套 MPRNet，也不声称联合训练已被证明优于冻结。

## 固定结构

第一阶段保留 T 的全部卷积/MLP、32×32 TX 内容交互、decoder 与输出头。
第二级为 `256→128→64→128→256`，宽度 `16/32/64/32/16`：

- 输入为原四通道 `H/TX/F/rho` 与第一阶段预测，共5通道。
- 第一阶段解码后的 D256/D128/D64 分别经同宽1×1投影，拼入第二级对应的编码块。
- 每个编码/融合块为两次3×3卷积和ReLU，无新增归一化或dropout。
- 两次下采样为3×3、stride2、padding1卷积；上采样复用原4×4转置卷积和concat skip融合。
- 最后3×3线性头产生有符号修正；weight/bias均初始化为零。

\[
(\hat y_1,D)=T_\theta(X),\qquad
\hat y_2=\mathrm{ReLU}\big(\hat y_1+R_\phi(X,\hat y_1,P(D))\big).
\]

修正相加保持FP32，避免AMP下的小修正被低精度加法舍去；卷积仍使用原AMP策略。
最终预测量、归一化和非负约束不变。两阶段之间没有detach，所有参数参与原masked MSE。
不增加阶段损失、边界损失、注意力、物理通道或GT/mask模型输入。

`lunar_maxvit.forward_features`仅暴露原前向已有的物理输入及解码状态，保持旧模型预测路径和权重键。
新参数统一为 `refiner.*`，在CPU RNG fork内构造，保留同seed T的初始化及后续随机状态。
第一步零输出头可学习；内部细化参数的任务梯度在该头离开零值后出现。

时间复杂度为新增各尺度卷积的 `O(BΣHW k² cin cout)`，激活空间为
`O(BΣHW c)`（层数影响常数），没有第二套全局注意力。新增 **248,225** 参数，
总参数从 **26,365,093** 到 **26,613,318**；新增卷积约 **2.640G MAC/图**，
转置卷积按输入位置计算，不含激活/拼接/逐元素操作。参数少不代表计算量可忽略。

## 初始化、选择与恢复

正式设置沿用 T：5.8GHz、masked MSE、AdamW、Conv/Linear权重衰减0.05、
150轮、batch16、lr1e-4、cosine到1e-6、clip0、D4增强、AMP、8workers。
保持数据loader、LOS缓存、log-distance定义不变，无新增依赖。

两组均加载各自seed的T best，只初始化模型权重；优化器与150轮日程重启。
这是相同**新增训练步数**的对照，不是相同总算力：新模型每步额外计算另计。

`--eval-initial`在epoch0前用当前验证设置重新评估，保存`*_initial.json`，
并将初始状态保存为epoch **−1** 的best/last。后续只有有效像素RMSE更低时才替换best。
初始验证不写入训练history；比较时须包含initial.json，best若停留在−1表示本轮没有超过起点。
初始验证会保留torch随机状态，避免其DataLoader迭代改变随后训练的随机流。

- `lunar_conv_tx_refine`新训练只允许从**同seed T**显式`--init-from`，且必须使用`--eval-initial`。
- T→新结构的权重迁移仅在初始化时允许，新增参数保留构造值；其他共享权重严格匹配。
- `--resume`要求架构、物理配置、优化器分组和初始验证策略一致；恢复本轮last时忽略init-from。
- 新checkpoint包含完整T与细化器，评估/推理不再需要原始T文件。
- epoch−1状态也可恢复：在初始验证之后、第一轮之前中断，重跑命令即可。
- 原框架仍不保证中断后的随机轨迹逐位重放；本轮不改变其恢复语义。

## 本地提交与服务器同步

在本地 PowerShell 检查本次差异后提交、推送：

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git diff --check
git add lunar_maxvit.py lunar_refinement.py model_factory.py train.py README.md LUNAR_T_REFINEMENT.md
git diff --cached --stat
git commit -m "Add joint cross-stage refinement from T checkpoints"
git push origin research/conv-alignment
```

`docs/`和仓库外的`local_experiments/`仅作本地记录，不加入提交。
以下服务器命令由用户执行，先激活运行T时的现有环境，并确认项目/数据路径仍有效：

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch research/conv-alignment
git pull --ff-only
```

## 四组正式训练

沿用已存在的 `results/los_cache_v1`。两个原始T best应位于下述init-from路径，
本轮使用新的`runs/t_refinement_v1/`，不覆盖原T产物。
原T日志约4.54小时/150轮；四次至少约18小时，细化的额外计算及评估另计，
具体时间以服务器首轮实测为准，不按本机计时直接推算服务器。

```bash
for arch in lunar_conv_tx_content lunar_conv_tx_refine; do
  for seed in 0 1; do
    python train.py \
      --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
      --arch "$arch" --phase single --band 58 --loss masked \
      --optimizer adamw --weight-decay 0.05 \
      --epochs 150 --batch-size 16 --lr 1e-4 \
      --lr-sched cosine --lr-min 1e-6 --clip 0 \
      --num-workers 8 --device cuda --amp --seed "$seed" \
      --init-from "runs/lunar_conv_tx_content_v1/seed${seed}/lunar_conv_tx_content_58_masked_single_los_distance_adamw_best.pt" \
      --eval-initial --resume auto \
      --panels 6 --panels-every 5 --plots-every 5 \
      --out "runs/t_refinement_v1/${arch}/seed${seed}" \
      --run-name "${arch}_58_masked_single_los_distance_adamw" || break 2
  done
done
```

重跑同命令从各组last继续；已完成150轮的组会跳过。不要使用原T的last作为本轮resume，
也不要一组沿用旧优化器、另一组重启。loss与训练预算已固定，不在本轮同时调整学习率。

## 全量验证与计时

```bash
for arch in lunar_conv_tx_content lunar_conv_tx_refine; do
  for seed in 0 1; do
    python evaluate.py \
      --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
      --ckpt "runs/t_refinement_v1/${arch}/seed${seed}/${arch}_58_masked_single_los_distance_adamw_best.pt" \
      --split val --batch-size 16 --num-workers 8 --device cuda --amp \
      --out "results/t_refinement_v1/${arch}/seed${seed}/val.json" || break 2
  done
done
```

同一空闲GPU下计时完整在线特征构造与预测：

```bash
for arch in lunar_conv_tx_content lunar_conv_tx_refine; do
  for seed in 0 1; do
    python benchmark_inference.py \
      --data-root ~/liuanda/LunarRM \
      --ckpt "runs/t_refinement_v1/${arch}/seed${seed}/${arch}_58_masked_single_los_distance_adamw_best.pt" \
      --split val --device cuda --samples 16 --warmup 3 --amp \
      --out "results/t_refinement_v1/${arch}/seed${seed}/inference.json" || break 2
  done
done
```

下载 `runs/t_refinement_v1/` 中的initial/history/summary/provenance、best/last、曲线/样例，
以及 `results/t_refinement_v1/`。以新方案减T-cont及新方案减原T的配对seed差值判断，
按terrain比较，并复用固定的高变化区定义。初始best属于保留起点，不能当作新训练收益。
若有效，下一轮才检验去掉跨阶段特征连接的机制对照；冻结对照不在本轮同时展开。

## 本地验证范围

本机检查与日志放在仓库外 `local_experiments/t_refinement_20261024/`。
seed0/1已经检查同初始化RNG、旧前向兼容、预训练T→细化后的train/eval和FP32/AMP初始输出一致，
以及batch16短程任务梯度、超出AdamW纯衰减的参数更新、特征连接干预。
这些检查不代表正式性能改善。模型本机torch2.5.1+cu121、RTX4060 Laptop的峰值allocated
约3.84–3.85 GiB；服务器版本和实际显存需以训练记录为准。

两组真实入口的初始保存→epoch−1中断→恢复→两轮小样本训练→独立评估均通过，
优化器/scaler/150轮调度状态正确恢复。该小样本检查中两组best均保留在−1，
说明初始成绩没有被后续退化覆盖。文档展开的12条Python命令经argparse解析，
除权重初始化和初始验证外，训练设置逐项与原T provenance对齐。

完整在线特征推理也通过16张不同val输入，计时包含现场LOS与内部distance；
本机不同进程的T/新结构总时间均值为50.18/33.84ms，存在明显时段波动，
不能由此宣称新结构加速或得到可靠开销比。正式成本按上面的服务器同机计时比较。
所有临时验证权重已清理，原T/C/TG及其他正式权重未变；日志和统计保留。
