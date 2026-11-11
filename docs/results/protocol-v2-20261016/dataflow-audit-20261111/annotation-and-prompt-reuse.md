# 数据流审计：原始标注回溯与跨图题面复用

更新：2026-11-12（北京时间）。本轮使用 CPU，没有租 GPU、训练新模型或替换既有正式结果。

## 从正式提示回到原始标注

[标注溯源凭证](annotation-source-replay.json)逐一比对了历史正式清单的 **16,900 条**评测提示与原始资料。读取的 POPE、REFER、MMStar 和 MMBench 原始文件均与退役恢复映射 SHA256 匹配。

| 分支 | 已对齐 | 具体核对 |
|---|---:|---|
| POPE 历史评测 | 9,000 | 官方样本 ID、三类负采样分支、图像文件名、问题文本与 yes/no 标签 |
| RefCOCO 系列历史评测 | 2,400 | 三库各 800；原始 val 划分、ref/ann/sent/image ID、语句、图像尺寸、原始 xywh 转 xyxy |
| MMBench 派生一致性 | 4,000 | 800 原题，各五种模板；选项顺序、正确选项、题面及图像字节 SHA |
| MMStar 历史评测 | 1,500 | 原题、标签、类别和图像字节 SHA |
| 训练代理 | 3,545 | POPE 600、定位 600、一致性 1,745、ScienceQA 600 |

训练 POPE 的 600 条标签可按源 COCO 目标类别重新生成，正负各 300；其中 “no” 指源标注未列出该类别，不能证明图中物理不存在。训练定位的三库各 200 条可回溯到各自 train 标注；MMBench 训练的 1,745 提示（349 原题）能回溯到原始题、模板、选项和图像 SHA。

ScienceQA 做了两级核对：对照[作者训练划分及题目 JSON](https://github.com/lupantech/ScienceQA)，600 条代理的题面与答案均能找到来源，但仅靠文字匹配有 136 条不唯一；再用 [冻结 Parquet 文件](https://huggingface.co/datasets/derek-thomas/ScienceQA/blob/f18b0a70359ebfb41f658fd564208d0355b013f4/data/train-00000-of-00001-1028f23e353fbe3e.parquet)的行号消歧。该文件的 LFS SHA 与归档相同，只读取 262,893 字节的页脚和标签列，**600/600 精确行号的题面、选项、标签全部一致**。见[Parquet 复核凭证](scienceqa-parquet-replay.json)。没有下载或复核 ScienceQA 图像列，也没有将其混称 MMStar。

## 新发现：跨图题面复用影响一致性结论的稳健性

对题面作 Unicode NFKC、大小写和空白归一化后，训练与评测 MMBench 一致性有 **105 个共享题面键**，涉及 28 个训练原题、81 个评测原题。笛卡尔配对后为 665 对，图像文件 SHA 全都不同，其中 **435 对正确选项不同**。典型例子是同样的选项和问题被用于不同图片；没有证据将这 665 对全归为泄漏或错误标注。

基于已有 9B v4.1 逐题账本，所有模型统一排除这 **81/800** 个评测原题及其五条改写，保持训练 seed 配对、图像簇重采样及原 12 项 Bonferroni 口径。A1−N 一致性从全量 **+5.05 pp，校正区间 [0.34, 9.28]** 变为 **+3.84 pp，校正区间 [−0.81, 7.91]**。这说明该结论在这一事后敏感性分析中不再排除零。由于 A1 本身训练的是 ScienceQA，而不是 MMBench，不能从这项比较推断 A1 直接记住了这些 MMBench 题面。原因也可能是子集构成、图像难度或格式适配。历史正式结果保留，论文应并列两种结果，不把一致性提升写成无条件稳健。

完整配对索引和区间见[题面复用敏感性证据](prompt-reuse-replay.json)。上一轮的图像 dHash 排除与这次题面排除是两个不同诊断，不能把前者的稳健性外推到后者。

另一项设计偏移：ScienceQA 训练的 A/B 标签占 434/600（72.33%），历史 MMStar 评测为 876/1,500（58.40%）。这是标签先验差异，不是证明模型利用了先验。借鉴 [MMBench 的 CircularEval](https://github.com/open-compass/MMBench#mmbench)，可把选项轮换和答案重映射做成独立敏感性条件；参考 [MMStar 的图像/无图对照](https://github.com/MMStar-Benchmark/MMStar/blob/main/README.md#leaderboard)，同一模型需分别报告图像条件与无图条件。新确认集冻结前同时审查图像、归一化题面、选项集合和标签分布，不在看到新分数后筛题。五改写全对仍只是本项目的操作定义，不是官方 MMBench CircularEval 分数。

复算入口：[原始标注](../../../../tools/audit_archived_annotation_sources.py)、[ScienceQA 行号](../../../../tools/audit_scienceqa_pinned_rows.py)、[题面复用](../../../../tools/audit_archived_prompt_reuse.py)。

## 9B 补实验训练来源已扩展至六个新端点

流式读取四个归档分卷，仅保留完整性标记、训练元数据和逐步 JSONL。正式补训练 **6/6** 个端点（N-D、N-G、P-G 各两个 seed）均对齐：共 **6,000 步**，每次四轴各 250 步，逐步样本 ID 和轴与注册训练输入一致；实际优化器步标记、梯度、各项 CE 及总损失匹配约定。六个端点各自的适配器 SHA 同时与恢复映射以及开发集、保留集的评测元数据一致；恢复文件 SHA 与映射一致。历史状态校验记录也随证据保留，本次未重新加载六份优化器张量。见[六端点凭证](factorial-training-replay-all.json)和[复算脚本](../../../../tools/audit_archived_factorial_training.py)。
