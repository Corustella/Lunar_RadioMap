# TX条件场景传播与局部几何化损耗场

新增独立分支上的两个架构：`lunar_transport_point`、`lunar_transport_field`。
它们共享同一局部卷积主干、TX条件交互与接收查询，只改变最后读出。
训练直接使用官方总损耗图，不需要补齐全量RT分量。原T、区域路由模型、
loader、F缓存和原有训练命令保持原样。

本模型是待验证研究假设。工程测试/随机权重计时不等于精度收益；
所有正式成绩仍须以完整官方val、配对seed0/1的有效像素RMSE判断。

## 1. 问题、依据与迁移边界

项目内完整val诊断发现：局部法线移动的理想剖面SSE下降约44%，同自由度切向移动约17%；
预测跨边界对比度约为标签的60%，理想对齐后仍约70%。远离几何可见性边界的损耗突变
邻域仍占总SSE约59.5%。这些是描述性证据，不证明LOS份额导致模型失败，
也不证明本网络一定有效。

本方案尝试同时学习**场景传播上下文、总损耗转变的位置/宽度和两侧数值**：

- [Perceiver IO](https://arxiv.org/abs/2107.14795)支持有限latent与输出查询处理大输入/输出；
  其光流等任务不是月球传播，latent不自动具有物理路径意义。
- [Boundary Attention](https://arxiv.org/abs/2401.00935)支持连续局部边界原语和重叠区域结构；
  它从图像本身提取边界，而这里必须由地形/TX预测损耗边界，这是尚未验证的迁移。
- 本代码为任务定制的原创组合，不是上述论文的逐行复现。没有RT路径枚举、测试期优化、
  复电场相位、Helmholtz约束或官方未建模的绕射。

## 2. 输入、局部编码与坐标

外部输入仍是`Bx3x256x256 [H,TX,F]`，已有`results/los_cache_v1`直接复用。
F仍为输入地形/TX构造的最低净空奇函数编码，不是官方LOS标签。rho按原函数在模型内计算。

复用强基线C/T的局部卷积编码器及五级解码器，从头联合训练；移除原单次T注意力和旧输出头，
以新的多轮场景交互替代。保留256原分辨率支路、128 stem和全部局部skip。
编码分辨率/通道为256/16、128/64、64/64、32/128、16/256、8/512。

在同步D4增强之后，由H/TX重新计算九个标量：归一化dx/dy/dz、向上单位法向nx/ny/nz、
法向与RX→TX方向点积、F、rho。全部FP32、无autograd、不读标签或mask。
距离用米，高度按metadata反归一化，TX=H(TX)+3m、RX=H+1m；法向采用中心差分和复制边界。
这些是输入几何代理，不是实际散射点/反射点属性。

## 3. 场景传播算子

- 16×16地表位置，中心与实际stride16编码器一致：0,16,...,240。
- 每位置4个方向条件槽位，共1024个状态，宽度64；方向代码为±x/±y。
  槽位不是四条真实射线、极化分量或反射阶数。
- 状态从16尺度局部内容、8尺度深层上下文及几何/方向条件初始化。
- 连续4轮pre-LN单头SDPA＋4倍宽FFN；每轮重新注入TX源状态。
  源状态来自TX处stride16内容、原分辨率内容及输入几何。
- 几何与方向通过学习投影进入Q/K；所有状态可交互，无硬LOS图裁剪。
- TX在stride16格的采样使用0..240的真实格点及边界延拓，不能按256端点任意缩放。

传播状态、源注入、接收查询保持FP32，即便卷积使用AMP。初版AMP实测会吞掉早期小梯度；
现实现已通过FP32对照定位并修正。新增层使用0.1的非零初始残差尺度。
局部卷积主干/解码、读出前卷积仍按既有AMP模式运行。

## 4. 接收查询与直接特征通路

在64×64、中心0,4,...,252的位置查询传播状态。每查询固定32个候选key：
4个相邻地表格点×4方向，以及16个TX条件全局摘要。最后条带中因边界夹紧而重复的局部key
被显式屏蔽，避免重复计权。始终至少有有效全局key，不产生全屏蔽softmax。

查询残差进入D64，继续解码至128/256。同时，未缩小的64通道查询上下文直接作为特征到达
最终读出，避免远程状态只通过多层小幅卷积接收梯度。这里插值的是上下文特征，不是最终标量图。
完整本地特征32通道加查询上下文64通道，最终读出输入96通道。

接收查询按512个位置分块，既限制临时内存，也避免batch16时将4096查询展平为65536个
SDPA batch触发CUDA launch限制。计算包含全部分块，没有省略或预缓存传播状态。

## 5. 两个可对照的读出

两版均为`Conv96->32 k3,pad1,ReLU -> FP32 1x1 output`。
点值版stride1、输出1通道；场版stride4、输出9通道。共享主干权重、共同层的初始化和
构造后CPU随机流一致；新读出在RNG fork内构造。读出初始函数不强制完全相同，
这种输出参数化/初始化差异属于实验处理，不能声称只改变一个无关开关。
两组新读出均使用输出权重std=0.01，以减弱新长链路初始梯度的衰减；旧T的初始化不变。
FP32、直接特征通路和该初始化是整体工程选择，没有将梯度修正单独归因于某一项。

| 架构 | 输出 | 参数量 |
| --- | --- | ---: |
| lunar_transport_point | 完整256²的普通总PL点值 | 26,679,121 |
| lunar_transport_field | 64²局部参数，渲染完整256²总PL | 26,679,385 |

场版仅多264参数。两版读出步幅和计算量不同，必须单独计时，不能仅凭参数近似相等声称成本相等。

### 场渲染公式

局部中心为c=(4i,4j)，patch为9×9，覆盖偏移u∈[-4,4]²。
每patch预测9个参数：a、两侧各2个斜率、jump、角度、边界位移、过渡宽度。

```text
Pminus(u) = a + 0.1*tanh(bminus) dot (u/4)
Pplus(u)  = a + jump + 0.1*tanh(bplus) dot (u/4)
n         = (cos(pi*tanh(angle)), sin(pi*tanh(angle)))
d         = 4*tanh(offset) pixels
tau       = 0.15 + 1.85*sigmoid(width) pixels; initial 1px
w(u)      = sigmoid((n dot u-d)/tau)
Pk(u)     = Pminus(u) + w(u)*(Pplus(u)-Pminus(u))
```

以上是归一化**总路径损耗dB标量场**的分段函数。两侧不是LOS/nonLOS功率，不能当作物理分量解释。
边界由模型预测，不绑定几何F=0；jump有正负，既可表示两侧差异也可退化为连续场。
宽度有固定数值范围，表示有限过渡而非强制锐边；这些范围是初始模型定义，未通过val扫参。

使用正的可分离cos²窗口加权，再以同样的fold覆盖权重归一化。patch中心0..252、fold padding4；
图外贡献裁去，四边和角落覆盖严格为正。最后沿用非负归一化输出ReLU，不做最大值裁剪。
渲染、三角函数、归一化及输出始终FP32。

局部单直线有表达限制：角点、多重细条纹可能欠拟合，重叠也可能重新平滑。
归一化重叠产生单张完整输出，但不保证各局部函数一致；首轮没有额外重叠一致性loss。
这需要正式结果检验；不宣称可以无损表示任意RadioMap。

## 6. 可选的官方边界监督

主对照默认`--field-boundary-weight 0`，仍是原来的官方masked MSE，选择checkpoint仍只用
`val/rmse_db_masked`。实现也支持独立的辅助监督实验：

1. 在官方train的有效相邻像素间找≥10dB跳变，标记两端点。
2. 只在完整3×3官方有效模板内评分。改变无效位置填充值不改变目标或辅助loss。
3. 几何场产生软边界强度：`4*w*(1-w)`乘以由局部两侧差值构造的软强度门。
   门的阈值10dB、温度2dB；该幅度门detach，辅助loss直接指导几何，不靠改变损耗幅度规避监督。
4. 对存在的正、负类别分别取BCE均值再平均；空mask返回有定义的0。
5. 总loss=`masked_MSE + field_boundary_weight * boundary_BCE`。
   两者量纲不同，权重没有精度调参依据；首轮结构对照保持0，不混入监督改变。

非零权重只允许场模型和masked loss。history额外记录main/aux loss；resume严格检查权重一致。
这与历史对最终PL加梯度L1不同：此处直接约束内部几何边界强度。它是否有效也需要同结构对照。
正常forward/评估/计时不构造边界目标、不读取标签或mask，不增加推理分支。

## 7. 复杂度与验证范围

N=1024，d=64，L=4，Q=4096，K=32，patch边长b=9：

- 几何O(BHW)；继承卷积成本与C相同，另有共享本地读出特征与新上下文。
- 场景交互O(BL(N²d+Nd²))，注意力后端可能占O(BN²)工作内存。
- 全局摘要O(B*16*N*d)；接收查询O(BQKd)，分块限制临时内存。
- 场渲染O(BQb²)，另有96通道读出卷积。反向还需保存相应激活。

本地RTX4060 Laptop、PyTorch2.5.1+cu121、batch1、AMP、4 CPU threads，
随机权重，64个不同val样本、3个另取warmup，现场F、无F缓存/标签/mask：

| 架构 | 总耗时均值 ms | P95 ms | 最大 ms | 观测≤500ms |
| --- | ---: | ---: | ---: | ---: |
| lunar_transport_point | 140.17 | 165.71 | 202.03 | 64/64 |
| lunar_transport_field | 125.00 | 135.97 | 147.28 | 64/64 |

两次运行的F构造时间不同，不能把总耗时差全部归因于场读出。它们均不是服务器计时、
官方认证、最坏情况保证或训练后的精度结果。

本地真实batch16 AMP更新通过，峰值allocated显存约5.84/5.77GiB，所有可训练参数参与反向、
全部梯度有限；场读出9个参数通道均有梯度。实际CLI的点值、场、场＋辅助监督均通过
训练1轮→恢复第2轮→评估，150轮cosine总长度和AdamW状态正常恢复；改变辅助权重的resume被拒绝。
常量/全局仿射场、正覆盖、九参数梯度、无效填充值不变性、空mask、几何D4标量、
latent重排、batch隔离和严格checkpoint检查通过。小样本更新不计为正式RMSE。

本地验证和计时记录保存在实际仓库外的`local_experiments/transport_field_20261102/`。
随机权重计时包含现场F、传输、rho/法向、所有交互/查询/读出及CPU dB输出，磁盘读取另列。
没有将核心算子时间充作500ms整系统认证，也没有在服务器执行或训练。

## 8. 服务器同步与先行计时

先激活原训练环境。以下由用户在SSH服务器运行，不假定Conda环境名称。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git fetch origin
git switch research/transport-field
git pull --ff-only
```

先实测服务器整链路；计时脚本不覆盖已有JSON，重测请使用新的输出名。

```bash
for arch in lunar_transport_point lunar_transport_field; do
  python benchmark_inference.py \
    --data-root ~/liuanda/LunarRM --arch "$arch" --band 58 \
    --split val --device cuda --amp --samples 64 --warmup 3 \
    --geometry-threads 8 \
    --out "results/transport_field_v1/${arch}_latency_random.json" || break
done
```

## 9. 正式结构对照：各两个seed

沿用T配方：AdamW、Conv/Linear decay0.05、150轮、batch16、lr1e-4、cosine至1e-6、clip0，
原联合增强、完整官方val。两个新模型均从头训练，不加载T或区域路由权重。运行4次，
不能在未知新主干速度时把历史约4小时/次当成实测预计；可从首轮epoch_seconds估算剩余时间。

```bash
for arch in lunar_transport_point lunar_transport_field; do
  for seed in 0 1; do
    python train.py \
      --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
      --arch "$arch" --phase single --band 58 --loss masked \
      --field-boundary-weight 0 \
      --optimizer adamw --weight-decay 0.05 \
      --epochs 150 --batch-size 16 --lr 1e-4 \
      --lr-sched cosine --lr-min 1e-6 --clip 0 \
      --num-workers 8 --device cuda --amp --seed "$seed" \
      --init-from none --resume auto \
      --panels 6 --panels-every 5 --plots-every 5 \
      --out "runs/transport_field_v1/${arch}/seed${seed}" \
      --run-name "${arch}_58_masked_single_los_distance_adamw" || break 2
  done
done
```

```bash
for arch in lunar_transport_point lunar_transport_field; do
  for seed in 0 1; do
    python evaluate.py \
      --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
      --ckpt "runs/transport_field_v1/${arch}/seed${seed}/${arch}_58_masked_single_los_distance_adamw_best.pt" \
      --split val --batch-size 16 --num-workers 8 --device cuda --amp \
      --out "results/transport_field_v1/${arch}/seed${seed}/val.json" || break 2
  done
done
```

如之后检验边界辅助监督，使用独立输出目录、同场模型/配对seed及完全相同预算，
仅改变`--field-boundary-weight`；不要覆盖上面loss0的运行，也不能从loss0使用resume改变目标。
本实现不预设一个已有效权重。

## 10. Checkpoint与诊断接口

`forward(x)`仍返回`[prediction]`，适配现有train/evaluate/benchmark。
`forward_field(x, pl_scale)`返回预测及64²的两侧中心值、法线、位移、宽度和256²边界强度。
PLscale应与metadata归一化跨度一致，本数据为208；它只影响边界强度的dB量纲。

model_config完整记录坐标、方向、交互、查询、FP32策略、patch窗口、宽度和辅助目标定义。
点/场不能相互resume/evaluate，旧T也不能直接灌入这些模型。架构、尺度或模型配置变化都会被拒绝。
从相同新架构checkpoint使用init-from仅加载权重，resume才恢复优化器/调度器。

正式决策同时查看完整val、突变/普通区、几何边界内/外和远离缺失区，以及局部位置、
对比度、过渡是否确有改善。不把latent/几何参数图解释成已恢复真实传播机制。
