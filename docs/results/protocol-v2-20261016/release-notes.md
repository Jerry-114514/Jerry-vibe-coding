# 补实验阶段性归档：P0 与显存预检

这是 `protocol-v2` 的阶段性产物，**不表示全部补实验或正式多 seed 训练已完成**。

- [在线结果与当前实施边界](https://github.com/Jerry-114514/Jerry-vibe-coding/blob/main/docs/results/protocol-v2-20261016/README.md)
- [API 教师与内存方案核查](https://github.com/Jerry-114514/Jerry-vibe-coding/blob/main/docs/大模型方案_核查意见_20261016.md)
- [原 v1 完整实验归档](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-formal-20261014)

每个 `tar.gz` 是一个已完成阶段快照；同名 `inventory.json` 给出路径、大小及 SHA256。后续快照包含当时已完成且核验的阶段，优先查看最新时间戳。未完成模型的部分账本不计入已完成归档。

已完成 P0 探索性重算、Qwen 3B/7B 全 BF16 的各 800 条开发预检、3B 的 100 次实际训练更新。后续 InternVL/LLaVA 和 7B 训练预检完成后会自动补充。

归档保存逐题生成、分析、显存采样、短程训练 LoRA/恢复状态与验收文件。现有 7B 教师缓存来自原 v1 Release，保持固定，不冒称 API 大教师缓存。

新 C gate 尚未开放；独立确认、人工定位盲审和正式多 seed 结果尚待执行。CPU 结果仍来自已查看的旧数据，不消除选择后偏差。

文件由 AutoDL 直接上传，服务端大小与 SHA256 已校验；大数据与权重不经本地电脑中转。GitHub 自动生成的 Source code 附件不包含这些实验数据快照。
