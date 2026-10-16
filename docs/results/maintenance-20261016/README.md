# AutoDL 磁盘清理验收

2026-10-16，按用户要求清理可恢复缓存与已验证的重复快照。

- 实际释放约 17.09 GiB；数据盘可用空间从 `df` 的 7.3G 增至 25G，占用率从 86% 降到 52%。
- 移除可按锁定 revision/SHA256 重新下载的 LLaVA 7B、InternVL 1B 权重；保留小型配置与版本信息。
- 删除已经上传、且逐文件 SHA256 与已验证 GitHub inventory 一致的历史快照载荷；保留上传回执与 inventory。
- 保留 Qwen 3B/7B、正式 B/C/D 原始运行、全部教师缓存、LoRA、优化器恢复状态、图片、运行环境和精度异常历史。
- 清理后再次核验 GitHub：10,561 个必要文件被上传清单完整覆盖，缺失 0 个。

[机器可读清理汇总与清理后备份核验](cleanup.json)。详细删除列表保留于 AutoDL `runs/maintenance/disk-cleanup-20261016.json`。
