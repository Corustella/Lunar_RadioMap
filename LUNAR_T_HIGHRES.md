# T-H64

`lunar_conv_tx_highres` 从 T（`lunar_conv_tx_content`）的 64×64 编码特征出发，在编码期间与低分辨率主干交换两次信息。第一轮使用 TX 注意力之后的 32×32 特征，第二轮使用由更新后的 32×32 主干生成的 16×16 特征。更新后的 64×64 特征接入 T 的 64×64 解码跳连。

每轮将粗特征投影到 32 通道并上采样，与 64 通道的细特征拼接。两个 1×1 卷积及中间的 3×3 深度卷积生成 64 通道更新量。细分支直接加上更新量，粗分支接收经中心平均池化和 1×1 投影的更新量。两轮独立训练。

上采样按原图像素中心对齐。32→63 和 16→61 使用 `align_corners=True`，再向右侧和底部复制到 64。回写时 64→32 使用 `kernel=3, stride=2, padding=1`，64→16 使用 `kernel=7, stride=4, padding=3`。边缘池化只平均图内像素。

从头训练采用同 seed 的 T 共享权重和 CPU 随机流。两轮细分支末层投影初始化为零，其余新增投影使用常规随机初值。第一步由两个末层投影收到任务梯度，后续梯度进入更新块和回写投影。训练、恢复和评估通过 `model_factory.py` 校验独立架构、`model_config` 与物理输入尺度。

AMP 下，两轮交换使用 FP32 参数精度计算，再把结果转回各支路的输入 dtype。同一组权重、输入和上游梯度的局部重放确认了 FP16 对启动期小梯度的下溢。在本机固定的 16 个真实训练样本上，两个粗尺度回写投影分别在第 11 和第 23 步收到非零梯度。

模型读取原 `H/TX/F`，在模型内计算对数距离 `rho`，输出一张原始 256×256 网格上的总路径损耗图。使用 T 的单阶段 masked MSE 和 LOS 缓存。新增 58,496 个参数，总计 26,423,589。卷积计算增量为 0.10748 GMAC/图，插值、池化、激活和访存另计。固定两轮的中间激活随批量和 64×64 网格面积线性增长。

本地 RTX 4060 Laptop、batch 1、AMP、现场 F、随机权重，同一组 64 个验证输入下，T 的平均、P95 和最大值为 91.66、105.73、109.72 ms，T-H64 为 101.95、116.65、129.11 ms。正式训练的 RMSE 和服务器时延待测。

## 服务器同步与计时

激活已有训练环境后同步分支。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git fetch origin
git switch research/t-highres64
git pull --ff-only
```

先在同一 GPU 上按相同的验证行比较整链路时延。脚本计算现场 F，计入输入传输、模型和 CPU dB 输出，另报磁盘读取。随机权重用于测量计算成本。重测时选择新 JSON 路径。

```bash
for arch in lunar_conv_tx_content lunar_conv_tx_highres; do
  python benchmark_inference.py \
    --data-root ~/liuanda/LunarRM --arch "$arch" --band 58 --split val \
    --device cuda --amp --samples 64 --warmup 3 --geometry-threads 8 \
    --out "results/t_highres_v1/${arch}_latency_random.json" || break
done
```

## 配对训练与完整验证

5.8 GHz 按 T 的既有配方训练 seed0/1，各使用独立输出目录。同一目录再次执行时由 `--resume auto` 恢复相同模型的 last。

```bash
for seed in 0 1; do
  python train.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --arch lunar_conv_tx_highres --phase single --band 58 --loss masked \
    --optimizer adamw --weight-decay 0.05 \
    --epochs 150 --batch-size 16 --lr 1e-4 \
    --lr-sched cosine --lr-min 1e-6 --clip 0 \
    --num-workers 8 --device cuda --amp --seed "$seed" \
    --init-from none --resume auto \
    --panels 6 --panels-every 5 --plots-every 5 \
    --out "runs/t_highres_v1/seed${seed}" \
    --run-name lunar_conv_tx_highres_58_masked_single_los_distance_adamw || break
done
```

```bash
for seed in 0 1; do
  python evaluate.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --ckpt "runs/t_highres_v1/seed${seed}/lunar_conv_tx_highres_58_masked_single_los_distance_adamw_best.pt" \
    --split val --batch-size 16 --num-workers 8 --device cuda --amp \
    --out "results/t_highres_v1/seed${seed}/val.json" || break
done
```

按官方有效像素累计 SSE/count 后开方，配对 T seed0/1 的 5.059348 / 5.058840 dB。下载两 seed 的 history、summary、provenance、best 和 val.json，结合逐图 P99、历史高变化区域及全有效 8×8 块内误差复核。
