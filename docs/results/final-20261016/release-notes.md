# B、C、D 正式实验结果

2026-10-16：原协议正式运行、独立重评分、训练权重及恢复状态核验已完成。这里是已完成的实验归档；早期阶段性附件继续保留，优先下载下列最终文件。

- [在线查看结果、表格与图表](https://github.com/Jerry-114514/Jerry-vibe-coding/blob/main/docs/results/final-20261016/README.md)
- [B/C/D 显存报告](https://github.com/Jerry-114514/Jerry-vibe-coding/blob/main/docs/results/vram-20261016/README.md)
- [必要文件备份核验](https://github.com/Jerry-114514/Jerry-vibe-coding/blob/main/docs/results/final-20261016/github-backup-verification.json)

## 下载实验数据

请展开下方 **Assets**。GitHub 自动生成的 **Source code (zip/tar.gz)** 仅含 Git 文件，不含这些实验数据。

| 文件 | 内容 | 字节数 |
|---|---|---:|
| [arp-formal-20261014-20261016T004018Z.tar.gz](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/download/arp-formal-20261014/arp-formal-20261014-20261016T004018Z.tar.gz) | 最终完整实验包，包含逐题账本、分析、教师缓存、LoRA、优化器恢复状态、冻结输入、训练和评测图片 | 1,064,822,655 |
| [arp-formal-20261014-20261016T004018Z-inventory.json](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/download/arp-formal-20261014/arp-formal-20261014-20261016T004018Z-inventory.json) | 完整包逐文件路径、大小、SHA256 | 2,232,331 |
| [arp-formal-20261014-20261016T004031Z.tar.gz](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/download/arp-formal-20261014/arp-formal-20261014-20261016T004031Z.tar.gz) | 最终显存汇总、后期采样与证据 | 22,606 |
| [arp-formal-20261014-20261016T004031Z-inventory.json](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/download/arp-formal-20261014/arp-formal-20261014-20261016T004031Z-inventory.json) | 显存包逐文件清单 | 4,484 |

完整实验包 SHA256：`fcf9780dab9d9533072851e9ee2aa567d6926431e611d75d62d4700d9ab4ec68`。

显存包 SHA256：`1b3b45ff87e2abf470491bcc1799d4a5a4af3aace5ce964543f50de5f9a42d86`。

2026-10-16 再次核对 GitHub 实时附件大小和服务端 SHA256：10,561 个必要文件被已上传清单完整覆盖，缺失 0 个。大文件由 AutoDL 直接上传，无需经本地电脑中转。

## 已完成规模与结论边界

- B：7 个模型各 16,900 条推理，共 118,300 条。
- C：3 个家族各扫描 120 设置，1,000 次配对重采样；原协议总体判定 go。
- D：3,545 条教师缓存；A1/A2/U 各 1,000 次实际优化器更新、各 16,900 条训练后评测；N 为零更新对照。
- A1/A2/U 的四轴原始增益之和分别为 +3.031、+2.714、+1.417 个百分点；这些是单 seed 点估计，未证明 A1 显著优于 A2。
- 定位测量、原量尺公式、平局与小分母等限制见在线结果及补实验方案。计算完成不等于全部研究主张得到验证。新补实验尚未运行。
- C 和早期 D2 的历史整卡显存没有采集，报告保留缺测，不用后续采样冒充历史峰值。
