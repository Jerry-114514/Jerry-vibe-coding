# B / C / D 显存占用报告

采集时间（UTC）：2026-10-15T15:00:18.176478+00:00

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
| C | CPU analysis | 未采集 / 尚未完成（详见 JSON） |
| D1 | Training-set construction | 未采集 / 尚未完成（详见 JSON） |
| D2 | Qwen7B teacher cache | 未采集 / 尚未完成（详见 JSON） |
| D3 | A1 | 9.0102 |
| D4 | A1 | 未采集 / 尚未完成（详见 JSON） |
| D3 | A2 | 未采集 / 尚未完成（详见 JSON） |
| D4 | A2 | 未采集 / 尚未完成（详见 JSON） |
| D3 | U | 未采集 / 尚未完成（详见 JSON） |
| D4 | U | 未采集 / 尚未完成（详见 JSON） |
| D3 | N | 未采集 / 尚未完成（详见 JSON） |
| D5 | Homogeneous-floor control | 未采集 / 尚未完成（详见 JSON） |

当前 D4-A1：nvidia-smi 总占用 12359 MiB（12.07 GiB），总容量 24564 MiB。

C 为 CPU 分析，不调用 CUDA；没有历史板卡采样，不能据此填写“总显存占用为 0”。
D2 已完成，但当时未记录峰值，因此不会用后来采样冒充历史峰值。
后续 D 阶段每 15 秒采样一次；采样最大值只是监测开始后的观测值。
D 尚未全部完成，本报告会在后续阶段补充。

文件：`vram-summary.json`、`peak-allocated.csv`、`nvidia-smi-samples.jsonl`、`evidence/` 原始完成记录。
