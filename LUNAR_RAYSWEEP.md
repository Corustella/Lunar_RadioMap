# LunarRaySweepNet

5.8 GHz 的三个独立配置从 T 的共享主干初始化分布出发，从头联合训练。

| 架构 | 64×64 径向扫描 | 32×32 地表交互 |
| --- | --- | --- |
| lunar_raysweep_sweep | 开启 | 原 T 场景注意力 |
| lunar_raysweep_scatter | 关闭 | 新坡面源、坡面间交互及 RX 查询 |
| lunar_raysweep_full | 开启 | 新坡面交互并读取扫描上下文 |

三者均包含一个只读取 H 派生形态的轻量地形编码器。它输入归一化 H、原生米制 DEM 的 nx/ny 法向和有符号 9×9 局部起伏 asinh(H−mean9(H))，经两次 stride-2 卷积得到 64×64、32 通道形态特征。像素中心为 0,4,…,252。形态特征在同地形不同 TX 下共享定义，TX 在扫描及交互中调制这些特征。已有 T 主干继续读取 H/TX/F/rho，原生网格跳连和解码器负责输出。

## 径向扫描

以实际 TX 平移一个 512 方位、2 m 径向步长的网格，范围覆盖整张地形的对角线。1 m H 与 F、4 m 形态特征分别按自己的原点及步幅采样。保留相对地表高度、RX 仰角、相对上游地平线的余量、半径、主要上游遮挡脊位置、角度阴影积分、F 和几何有效性。

地平线采用地表 H 的排他前缀最大值，RX 仰角采用 H+1 m。角度阴影积分是 viewshed 描述量。它与当前每条 TX—RX 连线的遮挡积分含义不同，也不作为衰减定律。法向和形态特征包含局部邻域，因果性描述采样序列上的计算顺序。

16 通道、两轮输入条件径向递推采用

```text
tau_r = 16 + 240 * sigmoid(decay(input_r))       # metres
a_r = exp(-dr / tau_r)
s_r = a_r * s_previous + (1-a_r) * value(input_r)
```

利用有界的 FP32 前缀乘积与加权累计和并行计算整段序列。最大半径约362 m，指数范围受控。每轮之后进行方位周期深度卷积和逐点混合。图外样本冻结递推并从地平线排除。回采样使用周期角向 halo 和有效支撑归一化，将上下文送入主干64×64特征。四个轴向方向固定为精确的0/±1，覆盖贴边TX。

该结构没有 Mamba 扩展依赖。固定宽度和层数时，扫描计算随极坐标采样点数 AR 线性增长，点卷积另有 O(ARC²) 项。往返采样提供近似上下文，原网格通路提供细节。

## 地表坡面交互

地表锚点为 16×16，物理中心位于 4+16k 像素，高度取 H。TX 高度为 H(TX)+3 m，RX 查询点位于 0+8k 像素、高度 H+1 m。法向先在1 m米制地形上计算，再于锚点采样。

TX 到地表锚点采用不超过半像素间隔的内部采样，形成地表最小净空、阻挡比例和主要阻挡位置。锚点形态、TX处形态、二者乘积以及12项源几何经非线性MLP构造源状态。full 配置还使用扫描上下文。

源状态先做一轮坡面间交互，再由32×32主干特征查询。内部坡面交互和几何偏置MLP采用框架常规非零初始化，主干融合输出采用std=0.001的小幅非零初始化。几何偏置末层不含会在softmax中相消的逐头常数bias。每条关系具有距离、法向出射余弦、相对TX入射方向的镜面偏差、额外路径长度、出射z分量及8点第二段净空代理。几何进入软注意力偏置。阴影地表锚点继续参加交互。所有状态均为可学习上下文，未标定为真实路径功率。

查询分成128点一块计算。内容注意力量级为 O((M²+NM)d)，M=256、N=1024、d=64，另有第二段几何采样与偏置MLP的开销。新增分支使用FP32并保留autograd，返回主干时转换为其dtype。输出投影采用小幅非零初始化，任务梯度和优化器一阶矩需实际核查。checkpoint记录完整物理输入及模型配置。

## 早停与恢复

训练新增以下参数。旧命令默认 patience=0。

```text
--early-stop-patience 25
--early-stop-min-epochs 60
--early-stop-min-delta 0
```

每轮在完整官方val上按 val/rmse_db_masked 更新早停。新实验最大150轮，至少60轮后，连续25次验证未取得严格改善即停止。调度的 cosine T_max 始终为150。best继续按真实最小masked RMSE选择，min_delta仅控制耐心计数。

last保存早停阈值、计数、参考best与触发状态。resume校验相同策略并接续计数。已早停的last再次resume会补全汇总和已有best的面板，训练循环结束。checkpoint使用临时文件写完后替换。summary中的completed_epochs表示实际完成轮数。

在已有T的两个150轮history上回放本策略，分别在完成91/74轮时触发，保留原best 5.059348/5.058840 dB。该回放用于选择工程起点，新模型的收敛时间仍需实测。

## 本地核查

RTX4060 Laptop、AMP、随机权重、batch1，同一组32个val输入和3个独立预热样本。计时包括现场F、全部模型内几何、网络、输入传输与CPU dB输出，磁盘读取和模型加载另计。

| 配置 | 参数量 | 平均 ms | P95 ms | 最大 ms |
| --- | ---: | ---: | ---: | ---: |
| T | 26,365,093 | 84.09 | 97.30 | 102.91 |
| sweep | 26,375,733 | 103.10 | 133.61 | 145.98 |
| scatter | 26,417,445 | 115.44 | 133.84 | 138.10 |
| full | 26,423,525 | 159.69 | 179.36 | 188.20 |

联合版在16个真实train输入上的batch16 AMP短训练，39步内51个新增参数张量均出现有限非零任务梯度、非零Adam一阶矩和实际参数更新。峰值allocated约6.95 GiB。共享T初始化和构造后的CPU随机流逐位一致。启动期32×32分支存在FP16上游梯度下溢，FP32对照确认路径接通；内部投影采用正常尺度，避免多级小投影继续削弱学习。观察这些启动步数不用于估计正式精度。

坐标斜坡、完整前缀递推与逐步参考、未来序列扰动、地表/RX受照区别、四角TX回采样支撑、真实CLI训练/恢复/评估及早停状态恢复均已核查。早停触发测试采用受控验证序列。正式masked RMSE、服务器显存和服务器整链路时延待完整实验确认。

## 服务器同步和延迟检查

激活已有训练环境。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git fetch origin
git switch research/lunar-raysweep
git pull --ff-only
```

先在目标GPU测整链路。现场构造F和模型内全部地形/扫描/坡面几何均计时，磁盘读取另报。选择新JSON路径保存。随机权重仅用于测成本。

```bash
for arch in lunar_conv_tx_content lunar_raysweep_sweep lunar_raysweep_scatter lunar_raysweep_full; do
  python benchmark_inference.py \
    --data-root ~/liuanda/LunarRM --arch "$arch" --band 58 --split val \
    --device cuda --amp --samples 64 --warmup 3 --geometry-threads 8 \
    --out "results/raysweep_v1/${arch}_latency_random.json" || break
done
```

## 配对训练及完整评估

以下按full、sweep、scatter顺序，每种配置seed0/1。使用既有LOS缓存，现场派生的新几何不依赖RT分量、教师缓存或标签。最大150轮，采用T原有AdamW配方及同步D4。历史T约4.5小时/seed，新配置耗时依据首轮日志估计。

```bash
for arch in lunar_raysweep_full lunar_raysweep_sweep lunar_raysweep_scatter; do
  for seed in 0 1; do
    python train.py \
      --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
      --arch "$arch" --phase single --band 58 --loss masked \
      --optimizer adamw --weight-decay 0.05 \
      --epochs 150 --batch-size 16 --lr 1e-4 \
      --lr-sched cosine --lr-min 1e-6 --clip 0 \
      --early-stop-patience 25 --early-stop-min-epochs 60 --early-stop-min-delta 0 \
      --num-workers 8 --device cuda --amp --seed "$seed" \
      --init-from none --resume auto \
      --out "runs/raysweep_v1/${arch}/seed${seed}" --run-name "$arch" || break 2

    python evaluate.py \
      --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
      --ckpt "runs/raysweep_v1/${arch}/seed${seed}/${arch}_best.pt" \
      --split val --batch-size 16 --num-workers 8 --device cuda --amp \
      --out "results/raysweep_v1/${arch}/seed${seed}/val.json" || break 2
  done
done
```

与T的seed0/1 5.059348/5.058840 dB配对，复核总masked RMSE、突变区、远离几何LOS边界的突变区、块均值/块内误差和逐图P99。每种配置评估整套改动的收益。sweep与scatter都包含形态编码器，不能仅由它们相对T的差异将收益归因到单个子模块。已有嵌套双seed等权4.97306 dB作为部署得分参照。
