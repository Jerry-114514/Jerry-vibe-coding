# 独立路线 B：过程记录与复核索引

本页说明必要过程证据的存放方式。完成状态以原始记录、独立校验和 GitHub 服务端资产摘要为准；尚在运行的最后一次评测及最终统计不计作已完成。实验是独立流程验证，不代表原 ARP 阶段 C 已通过。

## 代码、计划与小报告

代码和配置在本仓库版本历史中保留。入口为[执行计划](route_b_execution_plan.md)、[最小循环报告](route-b-minimal-report.md)、[独立验收](route-b-minimal-independent-verification.json)、[训练汇总](route-b-formal-training-report.md)及[逐模型评测进度](route-b-formal-evaluation-progress.md)。

## 原始过程数据

大文件由 AutoDL 直接上传至 [7B GitHub Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-protocol-v2-20261016)，不以只上传代码代替实验归档。每批压缩包配套 `*-inventory.json`，记录原路径、字节数和 SHA256；必要时按清单定位旧批次，不能仅下载最后一个压缩包。

以下路径均相对于远程项目根目录。公共前缀 `OUT` 指 `runs/route-b-staged-7b-20261016`。

| 过程 | 原始证据及用途 |
|---|---|
| 预算、轴定义与冻结输入 | `OUT/protocol.json`、`OUT/phase2-frozen.json`、仓库 `configs/route_b_formal_inputs.frozen.json`；核对臂、seed、更新数、输入哈希及停止规则 |
| 教师缓存与 EOS | `runs/route-b-7b-20261016/teacher-cache-bf16-eos/` 及 `OUT/full-eos-cache-independent-verification.json`；3,545 条缓存、逐文件摘要、张量对齐和 EOS 证据 |
| 最小循环五项验收 | `OUT/minimal-acceptance.json`、`OUT/minimal-independent-verification.json`、`OUT/minimal-report.json` 及 `OUT/minimal/`；包含 A1 第 50 步中断后新进程恢复至 100 步的证据 |
| 每次训练 | `OUT/formal/{臂}-seed{seed}/loss.jsonl`、`train.meta.json`、`complete.json`；逐步双 CE、梯度、耗时、显存及真实更新数 |
| 适配器与恢复状态 | 同目录 `adapter/`、`resume.pt`、`independent-training-verification.json`；核对优化器步数、非零 LoRA 更新、导出张量与恢复状态一致性 |
| 逐题评测 | 同目录及 `OUT/formal/N-seed42/` 的 `eval.jsonl`、`scores.json`；原始生成、题目 ID 和四轴分数 |
| 行数、重评分与上传检查 | `OUT/evaluation-aggregate-independent-verification.json`、`OUT/evaluation-delivery-checks/`；逐模型题数/ID/重评分以及服务端摘要证据 |
| 显存、耗时、损失图 | `OUT/completion-reports/training/` 以及训练、评测日志；汇总不替代逐步原始记录 |
| 最终统计和图表 | 完成后归档 `OUT/completion-reports/multiplicity/`、`OUT/completion-reports/final-figures/`；五 seed、12 项比较、校正区间、统计来源哈希及 PNG/PDF |
| 磁盘清理与异常处理 | `OUT/disk-cleanup-20261017/`、`OUT/delivery-execution-override.json` 和运行日志；保留被清理旧缓存的恢复指针，说明分批归档方式 |
| 最终完整性交付 | 完成后生成 `OUT/objective-content-verification.json`、`OUT/finalization/`；同时核对内容和新获取的 GitHub 资产字节数/SHA256，不以退出码或上传收据单独证明完成 |

增量快照保留原路径于 `runs/route-b-staged-delivery-20261016/<批次>/files/` 下；`binary-asset-references.json` 指向此前已核验的二进制资产。重复元数据快照不表示重复训练，二进制引用也不表示漏传恢复状态。

## 9B 接力的独立证据

9B 使用单独的 [GitHub Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-9b-122b-20261017)。教师请求/响应、HTTP 400 诊断、按来源隔离记录、缓存哈希、事前登记和后续验收均分别保存。API 密钥、SSH 私钥及其他认证信息不属于实验过程数据，不纳入归档。9B 的训练完成状态不能由基础设施就绪或教师缓存完成推定。
