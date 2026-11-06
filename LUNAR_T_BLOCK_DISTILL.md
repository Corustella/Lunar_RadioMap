# T 的 8×8 块均值蒸馏

用 5.8 GHz 全尺度嵌套模型的 seed0/1 best 作为固定教师。两个模型在官方 train 上按原图顺序前向，对归一化路径损耗图等权平均，再做步长为 8 的 8×8 平均池化。只保留 64 个官方 mask 像素全部有效的块，其他位置在缓存中填零。仿真空洞由官方 mask 识别。

学生从头训练 `lunar_conv_tx_content`。学生损失为官方有效像素 MSE 加 `0.25 × MSE(学生块均值, 教师块均值)`。蒸馏项只在全有效块上取均值。缓存目标与高度、TX、F、官方标签和 mask 同步做 D4 增强。保留 T 原有的模型、输入和推理入口。验证只读取官方数据及原 LOS 缓存，并按 `val/rmse_db_masked` 选择 best。

每张训练图缓存一张 32×32 float32 块图，25,720 张图约 100.47 MiB。缓存构建对每张图运行两次教师前向，分批写入 mmap。训练增加一次学生块池化、一次 mask 块池化以及对应的损失计算，时间随批量和像素数线性增长。推理计算量和参数量与 T 相同。

缓存目录含 `train.npy` 和最后写入的 `train.json`。manifest 记录 train 行顺序、物理特征尺度、路径损耗归一化范围、教师 checkpoint、全有效块数和构建环境。训练校验索引与尺度。续训需要相同缓存身份和损失权重，可以移动完整缓存目录。若构建中断，选择新输出目录重新生成。

## 服务器同步与教师缓存

激活已有训练环境后执行。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git fetch origin
git switch research/t-block-distill
git pull --ff-only
```

用既有 LOS train 缓存和两个嵌套 best 生成一次。两个学生 seed 共用此缓存。

```bash
teacher=lunar_conv_tx_nested_full_58_masked_single_los_distance_adamw_best.pt
python build_block_teacher_cache.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --teacher-checkpoints "runs/lunar_conv_tx_nested_full_v1/seed0/$teacher" \
                        "runs/lunar_conv_tx_nested_full_v1/seed1/$teacher" \
  --batch-size 16 --num-workers 8 --device cuda --amp \
  --out results/t_block_distill_v1/teacher
```

## 配对训练和完整验证

```bash
for seed in 0 1; do
  python train.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --block-teacher-cache results/t_block_distill_v1/teacher \
    --arch lunar_conv_tx_content --phase single --band 58 --loss masked \
    --optimizer adamw --weight-decay 0.05 \
    --epochs 150 --batch-size 16 --lr 1e-4 \
    --lr-sched cosine --lr-min 1e-6 --clip 0 \
    --num-workers 8 --device cuda --amp --seed "$seed" \
    --init-from none --resume auto \
    --panels 6 --panels-every 5 --plots-every 5 \
    --out "runs/t_block_distill_v1/seed${seed}" \
    --run-name lunar_t_block_distill_58 || break
done
```

```bash
for seed in 0 1; do
  python evaluate.py \
    --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
    --ckpt "runs/t_block_distill_v1/seed${seed}/lunar_t_block_distill_58_best.pt" \
    --split val --batch-size 16 --num-workers 8 --device cuda --amp \
    --out "results/t_block_distill_v1/seed${seed}/val.json" || break
done
```

配对比较原 T 的 seed0/1 成绩 5.059348 / 5.058840 dB。除完整 val 的 masked RMSE，复核全有效 8×8 块均值和块内残差，并检查高变化区及逐图 P99。history 中的 `train/mse_block_teacher_online` 为训练期全有效块累计 SSE 除以 `train/block_teacher_valid_blocks`，单位为归一化路径损耗的平方。
