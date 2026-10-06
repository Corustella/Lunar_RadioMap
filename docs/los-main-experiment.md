# LOS 完整主实验

日期：2026-10-06。按用户要求，直接运行一次完整 LOS 主实验，以完整官方 val 的有效像素 RMSE 判断效果。此前提案中的前置消融、zeros 配对训练、边界诊断、重复几何验收和哈希核验均不作为本轮步骤。代码已接入训练和评估；尚未取得服务器正式训练成绩。

## 问题、依据与假设

- **项目内证据：**三维 TX 相对坐标的三个 seed 收益方向不一致，详见[已有结果](tx-xyz-replication-review.md)。端点坐标未显式汇总 TX 到 RX 的中间地形；已经完成的净空原型检查支持复用该计算实现，但不证明预测收益。
- **文献支持：**Jaensch、Caire、Demir 的[原始论文](https://arxiv.org/html/2402.00878v1#S3.SS2.SSS3)研究城市无线电地图的可见性及可见高度输入。其柏林建筑/植被、3.7 GHz 定向 TX、WirelessInSite 数据与本项目的月球连续地形、5.775 GHz 各向同性 TX 和有效性 mask 不同。它支持尝试可见性表征的方向，不直接支持本净空公式或保证月球任务收益；原文分析见[前期提案](next-step-visibility-proposal.md)。
- **待验证研究假设：**将沿直线路径的最小地形净空显式交给 secondU，可能降低有效像素 RMSE。它是由已有输入计算的表示，反射、散射等仍由网络学习。

## 输入定义与实现

对 TX 像素 t、RX 像素 p，天线高度分别为地形高度加 metadata 中的 TX/RX AGL（当前 3 m / 1 m）。在连线的嵌套等距采样点上计算直线高度与双线性地形高度的差，取最小值 C(p)。采样包含端点，区间数为 `2*max(1,ceil(horizontal_distance_px))`，当前网格下间隔不超过 0.5 m。

输入标量固定为：

`F(p) = asinh(C(p) / 1 m) / asinh((height_max - height_min) / 1 m)`

高度范围只取训练 metadata（当前 496 m）；不按验证集重估。正负值保留近似可见/遮挡关系。这是连续净空代理，不是原 Sionna mesh 的精确 LOS 标签，也不是绕射损耗。特征计算不读取 target 或有效性 mask。

- `los_features.py`：复用已验收的 FP32 双线性净空算法，缓存适配器同步增强 height、TX、F、target 和 mask，调用实际数据 loader 的原增强函数。
- `build_los_cache.py`：一次生成 train/val 的 FP32 mmap 缓存，按官方 index 行存储；缺失、未完成或配置/索引不匹配直接报错。完整且匹配的缓存可复用，无哈希计算。
- `radiounet.py`：新增 `second_features=los`，仅向 secondU 注入 `[F,0,0]`。复用原三列扩展与零填充初始化，firstU 使用原输入。
- `train.py` / `evaluate.py`：读取缓存、保存特征定义，完整 val 的 `val/rmse_db_masked` 选模；最终从 best checkpoint 重新评分。
- `run_los.py`：直接执行缓存 → 一组训练 → 一次完整评估，汇总 RMSE、历史 baseline 差值和全流程时间。训练路径关闭可选梯度诊断和 SHA256；不运行边界报告。

算法计算时间为 O(HW L)，工作内存 O(P L + HW)，P 为分块大小，L 为最长采样点数。单样本缓存读取时间/内存均 O(HW)。当前 train/val 一通道缓存约需 **7.35 GB** 磁盘；缓存构造计时包含输入读取和写入，完整流水线另记录训练及评估时间。无新增依赖。

## 训练设置与结果口径

只做 seed 0 的 LOS 主实验，从原 `boundary_ablation_v1/seed0_grad0` 完整 secondU baseline 初始化，不从 xyz best 继续。firstU 冻结，grad-weight=0；150 epochs、batch size、Adam 学习率/调度、AMP、增强与 clip 均从该 checkpoint 继承；新的优化器和调度从头开始。

跨全部样本累计有效像素 SSE 和数量后开方，换算为 dB，作为主 RMSE。保存完整 val 的逐样本报告，不用逐图 RMSE 平均代替主指标。

`comparison.json` 的差值为 `LOS RMSE - 原 baseline RMSE`，负值表示改善。单次主实验可以回答完整方案是否改善当前参考成绩；由于包含 secondU 继续训练，不能单独把全部差值归因于 F。后续依实际 RMSE 决定下一步，不自动添加消融或前置核验。

## 本地验证

`tests/test_los.py` 两项检查通过：生成并复用缓存；一次本地小样本训练；确认 firstU 保持、F 对应权重获得更新；保存/加载 LOS checkpoint；经 DataLoader worker 完整读取测试夹具并重新评分，与训练 val 分数一致。检查确认新主实验路径未调用哈希或诊断梯度函数，入口只生成一组完整训练和一次完整评估。

这是代码集成验证，不是挑战数据完整训练结果。未在服务器运行本轮实验，也未重跑此前的几何或边界诊断。

## 本地提交

PowerShell：

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git add radiounet.py train.py evaluate.py experiment_provenance.py los_features.py build_los_cache.py run_los.py tests/test_los.py
git add -f docs/README.md docs/los-main-experiment.md docs/next-step-visibility-proposal.md docs/visibility-server-review.md
git commit -m "Add full LOS main experiment with cached terrain clearance"
git push origin main
```

这里只列本轮代码和说明文件，不包含工作区已有的 `.gitignore` 修改。

## 服务器同步与运行

在已确认的 Conda 环境中执行。按历史 150 epochs 记录估计约 **2 小时**，实际耗时由新流水线记录；该估计不是服务器实时状态。`runs/los_v1_seed0` 必须是新目录，缓存会自动构造或复用完整版本，原数据不改写。

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch main
git pull --ff-only
python run_los.py \
  --data-root ~/liuanda/LunarRM \
  --baseline-ckpt runs/boundary_ablation_v1/seed0_grad0/radiownet_58_masked_secondU_best.pt \
  --cache-root results/los_cache_v1 \
  --out-root runs/los_v1_seed0 \
  --seed 0 --device cuda --num-workers 8
```

命令直接开始完整实验，无需另跑 smoke、原型验收或 zeros 对照。缓存目录有失败/未完成产物时选择新的缓存路径，不覆盖旧缓存。

完成后下载 `runs/los_v1_seed0`。优先查看：

- `comparison.json`：LOS、原 baseline RMSE 及差值。
- `seed0_los/val.json`：best checkpoint 的完整 val 指标和逐样本分数。
- `seed0_los/radiownet_58_masked_secondU_los_history.csv`：逐 epoch 验证 RMSE。
- `manifest.json` / `seed0_los/radiownet_58_masked_secondU_los_provenance.json`：命令、seed、代码版本、数据/loader 路径、环境和时间记录。

分析 RMSE 无需下载约 7.35 GB 的缓存。
