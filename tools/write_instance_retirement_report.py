"""Render a user-facing release decision from the completed remote audit only."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/instance-retirement-20261020'
load=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
ready=load('release-ready.json');coverage=load('coverage.json');before=load('coverage-before-backup.json');gap=load('gap-backup-complete.json');restored=load('restore-roundtrip.json')
assert ready['status']=='passed' and ready['missing_files']==0 and restored['status']=='passed'
stamp=datetime.fromisoformat(ready['utc']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
release='https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-instance-retirement-20261020'
text=f'''# AutoDL实例释放前验收报告

北京时间 {stamp} 完成验收。**本次选定研究资产已备份并通过校验，可以释放此实例。** 本任务没有代为释放实例，也没有删除原始实验数据。

## 最终检查

| 检查项 | 结果 |
|---|---|
| 现存研究文件内容核验 | {ready['selected_files']:,}个文件，缺失0 |
| 加入恢复映射的历史已移走文件 | {ready['previously_offloaded_files_added']:,}个 |
| 恢复映射总文件数 | {ready['restore_map_files']:,}个 |
| 关键端点哈希及恢复状态 | {ready['verified_endpoint_digests']}/{ready['required_endpoint_digests']}通过 |
| 补传前未覆盖文件 | {before['missing_files']:,}个，逻辑体积{before['missing_bytes']/2**30:.3f}GiB |
| 本次去重补传 | {gap['unique_files']:,}个内容，{gap['unique_bytes']/2**30:.3f}GiB，{gap['batches']}批 |
| 服务端归档核验 | 核验时{coverage['server_assets']}个附件、{coverage['inventories_checked']}份清单；按附件大小与SHA256核对 |
| 实际恢复演练 | 从GitHub重新下载备份，在AutoDL校验{restored['verified_members']}个归档成员 |
| 适配器读取 | 7B、9B原轮、9B补充实验共3类适配器，张量可读且全部有限 |
| 最终文件稳定性 | 与最终哈希快照一致 |
| 训练／评测进程 | 无活动实验任务；遗留实验监视器已停止 |

“逻辑体积”包含相同内容的多路径副本；补传按内容去重。恢复映射还补入早先归档后从实例删除的文件，所以映射文件数高于现存文件数，映射内总逻辑量为{ready['selected_logical_bytes']/2**30:.3f}GiB。成功的服务端校验与恢复演练都已有凭证，不以“上传命令退出”为唯一依据。

## 已保存的有价值内容

1. **原v1 B/C/D**：七检查点逐题账本、120量尺分析、底线与对照、3B←7B缓存及训练端点、最终核验和显存记录。
2. **ARP补分析与候选数据**：公式／平局／有效域复核、配对门槛、角色隔离与图像去重清单、后续审计候选图片、原始parquet与标注、API失败与工程排查记录。
3. **7B独立实验**：3,545条教师缓存和EOS证据，最小循环与恢复测试，15次训练、全部原评测和38,400条定位复测；已移走恢复状态保留精确归档位置。
4. **9B原轮**：3,540条可用教师文本及隔离记录，15次训练适配器／状态，全部270,400条正式输出、坐标与EOS各版校准失败、五seed统计与图表。
5. **9B补充对照**：6次新训练、2个历史端点，110,040条新增输出、独立保留集、24项校正结果、中断恢复和全部日志。
6. **运行代码与环境**：实例实际运行的源码、测试、配置，两个虚拟环境的精确依赖，Python／驱动／系统信息，公共基础模型配置、tokenizer与精确聊天模板，以及项目外API测试脚本的安全副本。
7. **论文与交接**：中英文初稿、参考文献、完整报告、数据来源与哈希索引仍在Git主分支；本次另附离线恢复工具与指南。

本次真正新补的内容主要为后续审计候选数据、未打包的原始数据、早期精度失败／预检及零散日志和脚本。全部最终训练端点在补传之前已经有归档，本次再次核验了它们。

## 归档位置

- **[释放前新增备份与复现资料]({release})**：本次补传、复现资料包、实际恢复凭证、完整恢复映射及终止验收。
- [原v1](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-formal-20261014)。
- [7B与原ARP补分析](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-protocol-v2-20261016)。
- [9B及补充对照](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-9b-122b-20261017)。

四个Release共同组成恢复来源；不要只下载GitHub自动生成的源码ZIP。新Release内的`INSTANCE_RELEASE_READY.json`和`INSTANCE_RESTORE_MAP.compact.json.gz`提供直接验收与恢复入口。

最终审计包：{ready['report_receipt']['assets'][-1]['name']}。

复现资料包：{ready['capsule_receipt']['assets'][-1]['name']}。

[恢复指南](RESTORE.md) · [机器可读验收](release-ready.json) · [实际恢复验证](restore-roundtrip.json) · [离线恢复工具](../../tools/restore_instance_retirement.py)。

## 没有重复保存的内容

没有将公开可重新下载的预训练基础权重大文件、可重装虚拟环境二进制、重复导出副本或与项目无关的MNIST示例重复上传。已保存精确模型身份、已有权重核验信息、配置、tokenizer与依赖版本；这是一份研究资产备份，不是整机镜像。

API密钥、SSH私钥、`.bashrc`和实际环境变量值没有上传GitHub，未来需要时在新实例重新配置。基础数据与图像的后续使用仍遵循其原许可。

本次验收只说明实例中的选定研究资产已保全，不改变论文结论，也不解除原ARP阶段C的科学门槛。
'''
(OUT/'README.md').write_text(text,encoding='utf-8')
print(OUT/'README.md')
