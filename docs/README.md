# 5.8 GHz 预测研究记录

本目录保存实验审查、文献依据和可检验的研究方案；长期协作规则见工作区根目录 `AGENTS.md`。

建议阅读顺序：

1. [baseline 结果审查](baseline-audit.md)：哪些不足已经有数据支持，哪些仍是假设。
2. [文献检索与适用性分析](literature-review.md)：可借鉴的方法、原始来源及迁移限制。
3. [研究逻辑与实验方案](research-plan.md)：假设、机制、消融设计、评价与停止条件。
4. [环状误差诊断与梯度损失消融](boundary-ablation.md)：当前实现、预注册协议、诊断口径、本地验证与服务器运行命令。
5. [seed 0 消融审查](boundary-ablation-v1-review.md)：初步结果和原始配对复核计划。
6. [seed 1/2 配对复核与下一步计划](boundary-replication-review.md)：三个 seed 的实际结果、结论修正及 TX 相对坐标消融提案（2026-10-03）。
7. [下一步坐标输入的文献依据与实施规格](next-step-coordinate-proposal.md)：补充无线电地图输入消融原文、八图误差一致性检查、代码改动位置和验收条件；尚未实施模型修改。
8. [三维坐标消融实施协议](tx-xyz-ablation.md)：用户确认后的 dx/dy/dz、grad-weight=0、固定 firstU 与同形状零通道对照。
9. [三维坐标 seed 0 产物审查](tx-xyz-seed0-review.md)：完整来源核验、terrain 配对、训练回退与轮廓复查，以及 seed 1/2 复核命令（2026-10-04）。

当前已审查梯度消融的三个配对 seed，以及随后三维 TX 相对坐标的 seed 0：tx_xyz 相对同轮 zeros 的 masked RMSE 降低 0.013128 dB、主边界 SSE 降低 0.2675%，靠近缺测区域略有退化，已知宏观轮廓偏移仍在。下一步保持设置复核 seed 1/2，尚未执行该复核；具体证据及结论范围以上述最新报告为准。

原始 baseline 审查日期：2026-09-27；实现更新：2026-09-28。机器可读证据：

- [审查汇总及来源哈希](analysis/baseline_2_20260927/baseline_audit.json)
- [逐样本配对诊断](analysis/baseline_2_20260927/sample_diagnostics.csv)
- [分组诊断](analysis/baseline_2_20260927/group_diagnostics.csv)
- [可复现分析脚本](analysis/analyze_baseline.py)

从仓库根目录重算（输出目录必须是新目录）：

```powershell
python docs/analysis/analyze_baseline.py --source D:\ssh_download\baseline_2 --out <new-audit-directory>
```

脚本只读取已保存的 JSON/CSV，依赖 Python 标准库及项目已有的 NumPy；不需要模型、GPU 或原始地图。对自己生成的三个审查文件重新计算时可显式传入 `--overwrite`。时间复杂度约为 `O(N log N + B G)`，空间复杂度 `O(N + G + B)`，其中 N 为样本数、G 为分组数、B 为 bootstrap 次数；重采样逐次执行，不存储 `B × N` 大矩阵。

服务器仓库、数据与环境已由用户确认：`~/liuanda/lunar-radiomap-challenge`、`~/liuanda/LunarRM`、`conda activate gc1`。Codex 仅在本地分析下载产物及修改、验证文件，服务器同步与正式实验由用户执行；最新报告已核查本地下载的 checkpoint 和实验来源记录，不代表实时访问服务器。
