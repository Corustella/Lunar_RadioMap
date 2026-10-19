# LOS＋distance 双尺度反馈重建

本轮仅安排 seed 0。架构 radiownet_biscale 从原 LOS＋distance RadioWNet 出发，
firstU 冻结，只在 secondU 末端引入 64↔128 的细节与上下文交互。
不包含路径剖面或 TX 方向采样模块；原上采样层、全分辨率融合和直接预测头保留。

## 研究问题与证据

原 secondU 已有逐级 skip；待验证的问题是：128尺度细节反向修正64尺度上下文，
再将更新后的上下文返回细节，是否优于原逐级拼接。

- 文献支持：[REM-Net 作者代码](https://github.com/YNUniversityCQ/REM-NET/blob/master/models/modules.py)
  在城市无线电地图中保留并行尺度并重复融合；
  [HRNet 原文](https://arxiv.org/html/1908.07919v2)在姿态估计与分割中检验重复融合。
  两者都没有直接验证本项目末端两尺度、共享权重和冻结 firstU 的设计。
- 项目证据：路径剖面 G 的29轮 best 为5.419521 dB，原U0为5.416208；
  历史高变化区贡献大部分剩余SSE。它们不能证明解码器是根因，也不能证明GRU普遍无效。
- 研究假设：下述有限次特征反馈可改善月球有效像素RMSE。这是项目适配，
  不是REM-Net复现，也不是物理场、可逆变换或有收敛保证的求解器。

## 数学与固定规格

C0 = concat(Wconv_up10输出, Wlayer1输出)，形状为 B×80×64×64；
H0 = Wlayer0输出，形状为 B×30×128×128。两个输入投影 AC、AH 均为带bias
的1×1卷积，输出32通道。每轮为：

\[
a_k=A_C(C_k),\quad b_k=A_H(H_k),
\]
\[
C_{k+1}=C_k+P_C\,\mathrm{ReLU}\!\left(K_C[a_k,D(b_k)]\right),
\]
\[
H_{k+1}=H_k+P_H\,\mathrm{ReLU}\!\left(K_H[b_k,U(A_C(C_{k+1}))]\right).
\]

固定两轮，全部参数跨轮共享。KC、KH 为64→32的3×3卷积，padding=1、bias开启；
PC、PH 为32→80和32→30的1×1卷积。D为2×2/stride2无padding平均池化；
U为到128×128的双线性插值，align_corners=False，匹配两级pool网格中心。
细节分支自身一直保留，不被下采样结果替代。没有新增BN、注意力、膨胀卷积或损失项。
所有计算跟随主干AMP。粗尺度更新先完成，细尺度必须使用更新后的C，不能误用旧C。

重建接回原网络：
\[
Z_{256}=\mathrm{Wconv\_up0}[
\mathrm{Wconv\_up1}(C_2),H_2].
\]
随后仍为原全分辨率skip、Winput、Wconv_up00和Wconv_up000，输出仍为直接预测。
上游RadioUNet/modules.py不修改；新子类以三处局部hook衔接已有forward，
每次forward结束或异常时清空临时特征。新增参数名以W开头，沿用原secondU训练规则。

PC/PH权重与bias全零初始化，其余卷积用PyTorch默认初始化；保留原权重，
新增初始化隔离CPU RNG。验收要求初始两阶段输出与原U一致，更新后两方向均有有效梯度。
只允许从历史LOS firstU checkpoint初始化，该文件也含继承的secondU权重；
禁止用已训练的distance secondU、G或方向采样best混入预算。续训/评估严格核对架构配置与权重。

新增44,142参数。B为batch、K=2、d=32、两尺度位置数为N64/N128时，
时间O(BK(N64+N128)d(d+通道数))，训练激活O(BK(N64+N128)(d+通道数))。
共享权重不共享激活。按上述未缓存投影计算，新增卷积约0.8808 G MAC/图，
不包括池化、插值、激活和数据移动；真实延迟、显存须实测。

## 预算与判断

训练设定保持secondU150轮、batch16、Adam/weight_decay=0、lr1e-4、cosine到1e-6、
AMP、原同步增强与masked MSE；选best按有效像素累计SSE/count后的RMSE。

新增可选 --stop-after-epochs 30 表示训练到累计30轮（epoch0–29）后正常退出，
每轮仍保存last及改进时的best，并在停止点输出完整曲线、best示例和summary。
--epochs 150 始终定义cosine周期；禁止用 --epochs 30 代替筛选停止。
summary区分计划epochs与实际completed_epochs。重复同一停止命令不会再追加训练；
若获准继续，去掉 --stop-after-epochs 30，保持其余参数及 --resume auto 不变，
从last恢复optimizer/scaler/scheduler，无需改变学习率或重新初始化。
沿用现有resume机制，不声称跨进程随机增强轨迹逐位复现。

30轮筛选资源门槛：相对U0的5.416208至少降低0.05 dB（约≤5.366208）才继续考虑；
最终推进门槛沿用0.10 dB（约≤5.316208），同时查看既定高变化区和其余区域SSE。
它们不是统计显著性或预期收益；提前结束可能错过后期改善。单seed只作工程筛选。
有明确收益后再研究单向反馈、相近参数/MAC的同尺度细化及跨seed复核，本轮不自动安排。

## SSH服务器同步和seed0运行

在用户已确认的训练环境中执行，沿用既有仓库、数据、LOS缓存和初始化路径。
本地助手没有在服务器执行。正式30轮耗时以服务器前几轮日志估算，
不能把上一版G的146秒/轮直接当作新结构耗时。预留数小时实验窗口。

~~~bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git fetch origin
git switch research/biscale-reconstruction
git pull --ff-only

python benchmark_inference.py \
  --data-root ~/liuanda/LunarRM --arch radiownet_biscale \
  --band 58 --device cuda --samples 16 --warmup 3 \
  --out results/biscale_v1/inference_random.json
~~~

随机权重计时包含在线LOS/distance和完整两阶段推理；不是精度证据。
计时文件已存在时脚本拒绝覆盖。

~~~bash
python train.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --arch radiownet_biscale --phase secondU --band 58 --loss masked \
  --first-features los --second-features los_distance \
  --optimizer adam --weight-decay 0 \
  --epochs 150 --stop-after-epochs 30 --batch-size 16 --lr 1e-4 \
  --lr-sched cosine --lr-min 1e-6 \
  --num-workers 8 --device cuda --amp --seed 0 \
  --init-from runs/los_both_v1/seed0/firstU/radiownet_58_masked_firstU_los_both_best.pt \
  --resume auto --panels 6 --panels-every 5 --plots-every 5 \
  --out runs/biscale_v1/seed0/secondU \
  --run-name radiownet_58_masked_secondU_biscale
~~~

停止后用best做完整验证：

~~~bash
python evaluate.py \
  --data-root ~/liuanda/LunarRM --los-cache results/los_cache_v1 \
  --ckpt runs/biscale_v1/seed0/secondU/radiownet_58_masked_secondU_biscale_best.pt \
  --split val --batch-size 16 --num-workers 8 --device cuda --amp \
  --out results/biscale_v1/seed0/val.json
~~~

下载完整 runs/biscale_v1/seed0/secondU 和 results/biscale_v1，
包含best/last、history、summary、provenance、示例图、逐样本验证与计时。
G的best/last仍需单独保留；本机当前仅有G的CSV，未完成其路径贡献检查。
