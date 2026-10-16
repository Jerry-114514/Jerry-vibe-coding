# 路线B必要数据的增量交付

最小循环原始检查点/优化器/适配器、逐步损失和四臂评测已完整上传GitHub Release：前缀 `20261016T134829Z`，约1.7GiB压缩归档，分为1GiB与约694MiB两片；上传器校验服务器大小及SHA256后保存回执。

在新增9B权重后，全实验末尾若重新打包所有既有检查点，临时归档加分片会额外占用约双倍压缩包空间，超过当前磁盘余量。因此新增 `tools/run_route_b_storage_safe.py`，导入原冻结数值执行器而不修改它，仅改变备份方式并复用已验收的阶段一产物。

- 首次生成的每臂适配器/恢复状态完整上传；已有二进制只有在其当前SHA与此前已验收Release inventory一致时才复用。
- 后续评测更新只发布新增文本/JSON/JSONL与既有二进制资产引用，避免反复上传同一优化器状态。
- 全量教师缓存独立完整上传。总报告保存所有逐题数据及必要回执，结合Release inventory可定位原二进制内容；不是删除训练恢复数据。
- 数值源文件和协议SHA保持不变，新增交付覆盖脚本有独立SHA，记录在运行目录的delivery-execution-override.json。
- 新版缓存已原子保存的条目按哈希复用；此次交付调度续跑不重复执行已验收训练或覆盖原评测显存记录。

Release：[arp-protocol-v2-20261016](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-protocol-v2-20261016)。本机仅获取小型JSON/Markdown报告，权重与大检查点直接从AutoDL上传。
