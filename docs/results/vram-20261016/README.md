# B / C / D 显存占用报告

采集时间（UTC）：2026-10-16T00:39:12.450534+00:00

历史峰值来自完成文件中的 `torch.cuda.max_memory_allocated()`；GiB = 1024³ 字节。
该数值不包含 CUDA 上下文和 PyTorch 缓存池保留显存，不能直接等同于 nvidia-smi 总占用。

| 阶段 | 任务 | PyTorch 峰值（GiB） |
|---|---|---:|
| B | qwen3b | 8.776 |
| B | qwen7b | 17.2936 |
| B | intern1b | 2.7438 |
| B | intern2b | 4.8815 |
| B | intern8b | 15.79 |
| B | llava05b | 3.457 |
| B | llava7b | 16.817 |
| C | CPU analysis | 历史峰值未采集（详见 JSON） |
| D1 | Training-set construction | 历史峰值未采集（详见 JSON） |
| D2 | Qwen7B teacher cache | 历史峰值未采集（详见 JSON） |
| D3 | A1 | 9.0102 |
| D4 | A1 | 8.8035 |
| D3 | A2 | 9.1405 |
| D4 | A2 | 8.8035 |
| D3 | U | 9.1339 |
| D4 | U | 8.8035 |
| D3 | N | 零更新对照，无独立训练峰值 |
| D5 | Homogeneous-floor control | 历史峰值未采集（详见 JSON） |

报告采集时 E：nvidia-smi 总占用 1 MiB（0.00 GiB），总容量 24564 MiB。

C 为 CPU 分析，不调用 CUDA；没有历史板卡采样，不能据此填写“总显存占用为 0”。
D2 已完成，但当时未记录峰值，因此不会用后来采样冒充历史峰值。
后续 D 阶段每 15 秒采样一次；采样最大值只是监测开始后的观测值。
D 已全部完成；报告包含 A1、A2、U 的训练与完整评测峰值。

文件：`vram-summary.json`、`peak-allocated.csv`、`nvidia-smi-samples.jsonl`、`evidence/` 原始完成记录。

其中采样和 evidence 原始文件位于 [最终显存 Release 附件](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/download/arp-formal-20261014/arp-formal-20261014-20261016T004031Z.tar.gz)。上表“历史峰值未采集”表示监测记录缺失，不表示该阶段尚未完成。
