# 净空诊断服务器产物审查与实现决策

**2026-10-06 执行更新：**用户已要求直接做完整 LOS 主实验，相关训练实现和命令见[当前运行说明](los-main-experiment.md)。本报告保留历史结果；其中待确认、前置验收、zeros 配对和重复核验不再是进入训练的前提。

日期：2026-10-05。读取用户下载的三个目录：`D:/ssh_download/tx_xyz_replication_audit_20261005`、`visibility_probe_20261005`、`visibility_cuda_20261005`。本轮未连接服务器；服务器信息及时间均来自下载记录。

**判断：服务器重复审查通过，净空构造的数值和成本支持进入受控训练的实现阶段。推荐一次性缓存连续净空F，训练只向secondU提供[F,0,0]，与同形状zeros比较；正式模型改动待用户确认。** 此次产物不是净空训练成绩，不能判断它已经改善RMSE。xyz三对seed的不稳定结论保持。

**服务器扩展验收更新：**用户随后下载`D:/ssh_download/visibility_cuda_fullcheck_20261005`，八图GPU/FP64与64项D4检查均通过，数值与本地一致，最大差分别为2.2993e−5m、3.2425e−5m，可见性符号差异为0。该轮八图平均计时8.271ms；原型的两环境数值验收已结束，下一步应落实缓存与模型输入消融，不继续重复同类诊断。[扩展验收核验](analysis/visibility_server_review_20261005/server_fullcheck.json)

## 1. 下载结果核验

- 服务器与本地的三对seed分数、配对差值、terrain诊断、固定截面和八图误差分解一致，数值以rtol=1e−12、atol=1e−8比较。平均Δ仍为−0.003677dB，运行间SD0.009080dB；xyz未通过原定复核。
- 几何summary/config/checks与本地一致：403,785有效像素，粗细采样61个可见性符号差异；代理变化带占8.624%像素、覆盖28.508%GT主边界；01917固定截面代理转变右端列142/175。没有新增预测或完整val几何统计。
- metadata及八张输入缓存的来源哈希匹配实际本地文件；三个seed审查中原始来源指纹集合与先前本地审查一致。脚本指纹匹配已提交的`97660e8` Git blob。benchmark本身未记录执行时Git HEAD，因此只证明脚本内容相符，不声称取得了服务器实时版本。
- 重新计算24个原始计时的均值及每图均值，匹配保存值。文件记录GPU为**NVIDIA GeForce RTX5090 D v2**，PyTorch为**2.14.0+cu132**。

[机器可读核验](analysis/visibility_server_review_20261005/server_review.json) · [核验脚本](analysis/review_visibility_server.py)

## 2. 服务器成本与记录错误

| 项目 | 记录结果 |
| --- | ---: |
| 图数×每图计时重复 | 8×3 |
| 每图平均，八图等权 | **8.036ms** |
| 各图均值范围 | 5.003–12.177ms |
| 单次重复范围 | 3.964–12.206ms |
| 原记录首张图GPU与FP64最大差 | 1.3045e−5m |
| 首张图可见性符号差异 | 0 |

计时包含CPU坐标准备、输入传GPU、FP32采样/插值/min和同步，batch1、预热后计时；不含文件IO、结果传回CPU、F的asinh缩放、模型forward和训练。仅八张图，不能当作全数据吞吐或完整推理时间。较大的重复变化也说明不宜只引用最短一次计时。

原脚本在limits中写死了“Local laptop benchmark”，与这次GPU记录不符。这是本轮发现的记录文字错误；已修正为按调用主机的gpu/torch解释。下载文件原样保留，修正不改变既有计时数值。

如果简单按batch1均值逐样本累加：25,720训练样本×150epochs的在线净空计算为约**8.61小时**，train+val共28,050样本一次计算约**3.76分钟**。这仅是窄计时范围的算术投影，不考虑batch并行、缓存、IO或模型，**不是新训练耗时预测**。它支持“避免每轮重复计算同一静态输入”的实施选择。

一通道FP32缓存train+val约**7.353GB**（十进制）；包括test则约8.651GB。按terrain+TX逐样本缓存、使用mmap，不按terrain只存一份，也不重复为两种频段生成相同几何。缓存保存输入/metadata/索引/算法指纹，不能使用GT/mask构造。

## 3. 本轮补做的本地数值验收

原产物只对一张真实图比较GPU/CPU。为补齐此前提案的局部验收，本轮新增benchmark可选`--verify-all`，保留默认计时范围和采样定义。已在本地4060Laptop、PyTorch2.5.1+cu121实际运行：

- 八张真实图全部与FP64双线性/离散采样参考比较，最大绝对差**2.2993e−5m**；按原1e−4m展示阈值，可见性符号差异共0。
- 每图八种D4变换共64项GPU等变检查，最大差**3.2425e−5m**；均低于预先声明的0.001m实现容限。
- 原平地、山脊、高度平移和D4合成检查已通过；实际输入/目标/有效性mask的来源一致。

这些是**近似算法的数值实现检查**，不验证原Sionna地形mesh、重心RX和网格插值是否完全相同，也不构成净空表征的预测效果验证。随后下载的5090D v2/PyTorch2.14服务器完整检查已通过，八图与64项D4数值均和本地参考吻合；来源metadata/缓存哈希吻合，两个脚本内容匹配提交62062d0的Git blob。计时由24条原始重复重新核算，为8.271ms/图，仍不包含完整IO或模型训练。以上均来自记录，没有实时访问服务器。

[八图/64变换验收](analysis/visibility_cuda_fullcheck_20261005/cuda_benchmark.json)

## 4. 下一版的具体规格

沿用[已说明的单通道净空提案](next-step-visibility-proposal.md)，只落实一个可证伪假设：把沿整条路径的最小净空提前汇总，能否稳定降低现有高变化区域误差。文献/项目证据/研究假设的区分沿用该提案，不新增物理收益结论。

推荐下一版实现：

1. 从测试可用height/TX计算F，固定TX3m/RX1m、双线性插值、≤0.5m嵌套采样和FP32计算。F定义不变，保持`asinh(C/1m)/asinh(496m/1m)`；不根据这次val图调参。
2. 在GPU上一次性构造train/val缓存并写入新的缓存目录，原始数据和优先loader保持。按官方index寻址，增强时同步旋转/翻转scalar F；缓存缺失、版本或索引不匹配时报错。
3. 两组均从原`seed0_grad0`完整baseline初始化，固定同一firstU，grad-weight=0，150epochs和原优化设置。secondU输入为zeros或[F,0,0]，保持已有三列扩展形状、零填充初始化。不从xyz best续训。
4. 先做迁移/初始输出/梯度/冻结/缓存增强/checkpoint往返和短程smoke验收，再输出seed0正式配对运行命令。记录一次性缓存成本、IO、全部训练和评估时间。
5. 整体masked RMSE与主边界SSE都下降才触发seed1/2复核；报告terrain集中度、multi/single及近缺测代价，不把单图轮廓改善替代主指标。

当前尚未修改模型、训练入口或loader。采用缓存增加约7.35GB磁盘需求，正式训练前须确认输出位置/剩余空间；不删除已有产物。缓存构造时间/空间仍是O(HWL)/O(P L+HW)，读取单通道O(HW)，训练输入只增加现有三列中的一项有效特征。无新增依赖。

## 5. 本轮提交与服务器检查命令

上一版新增benchmark完整检查；本次仅增加完整检查记录的只读核验脚本与结果，并更新报告/提案状态，模型与训练代码未变。尚未commit/push；用户已有.gitignore修改不加入本次提交。

```powershell
cd D:\codes\LunarRadiomap\lunar-radiomap-challenge
git add -f docs/README.md docs/next-step-visibility-proposal.md docs/visibility-server-review.md docs/analysis/verify_visibility_fullcheck.py docs/analysis/visibility_server_review_20261005/server_fullcheck.json
git diff --cached --stat
git commit -m "Confirm server visibility fullcheck acceptance"
git push origin main
```

服务器在已确认Conda环境中同步。下面的归档核验命令仅读取已有JSON/缓存，不重新运行GPU，通常数秒；输出文件必须不存在：

```bash
cd ~/liuanda/lunar-radiomap-challenge
git status
git switch main
git pull --ff-only
python docs/analysis/verify_visibility_fullcheck.py --artifact results/visibility_cuda_fullcheck_20261005/cuda_benchmark.json --cache-root runs --data-root ~/liuanda/LunarRM --out results/visibility_fullcheck_verified_20261005.json
```

归档核验是可选的，无需再下载其输出作为进入下一版的前提。原型数值检查已通过，下一步是按第4节实施净空训练消融，等待对模型输入/缓存读取变更的明确确认；不是再重复xyz训练或净空原型计时。`review_visibility_server.py`用于本地下载目录重算，要求审查目录及原始runs同处一个source根目录；不直接将服务器results目录当作满足该结构的source。
