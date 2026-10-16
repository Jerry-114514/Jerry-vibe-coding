# API 开发实验结果

日期：2026-10-16。两批实际调用均已完成，并通过独立 CPU 验收：覆盖和去重、账本/清单 SHA256、严格重评分、原图 SHA256、返回模型身份、token 用量和概率有限性。

## 四轴严格旧协议比较

每轴 8 个原题单元，一致性保留五种改写，共 32 单元、64 条提示。API 模型为 `qwen3-vl-plus`，精确不可变版本未确认；不能认作 Qwen2.5-VL 32B/72B。

| 轴 | 本地 3B BF16 | 本地 7B BF16 | API 教师 |
|---|---:|---:|---:|
| POPE | 100% | 100% | 100% |
| 定位，原图像素契约 | 100% | 83.33% | 0% |
| 五改写全部正确的一致性 | 87.5% | 87.5% | 100% |
| MMStar | 50% | 75% | 75% |

分数按原协议先在各 dataset/subset 内平均，再对分层等权；例如定位不能简单把 83.33% 换算为 8 题中的正确题数。本地比较使用同一组历史开发题和同一语义提示，但服务端图像处理与本地后端不同。样本很小，不能据此判断教师或模型家族总体优劣。

API 定位 8 题中 7 题超出原图范围，严格解析失败；余下一题可解析但 IoU 不足。MMStar 有 2 条输出在 128 token 上限处截断，保留为失败，不擅自提取推测的选项。`enable_thinking=false` 并不保证回答一定简短。该批消耗 17,903 token。

## 显式原生坐标校准

依据 [Qwen3-VL 官方技术报告 3.2.4](https://arxiv.org/html/2511.21631v1#S3.SS2.SSS4)，另冻结归一化到 `[0,1000]` 的坐标契约。重跑全部 8 条定位开发题，显式要求该坐标格式，再固定使用 `x*width/1000`、`y*height/1000` 映射到原图。禁止按输出大小猜单位、裁剪越界框或根据答案选择坐标换算。

结果：8/8 均可解析，8/8 的原图 IoU >= 0.5，分层等权得分 100%；消耗 3,390 token。三个解析测试验证了已知几何映射、小数值不猜单位和越界拒绝。

这是在已观察开发样本上进行的格式校准，提示和坐标契约都发生了变化；0%→100% 只能支持这 8 条上的测量约定影响，不能称为独立验证或模型能力提升，也不能事后覆盖原严格协议结果。下一步应在未用于本次校准的开发样本验证契约，然后冻结用于后续审计；人工盲审仍未完成。

已启动后续验证：从原 BF16 开发清单的 100 条定位题中严格排除上述 8 条，其余 92 条全部使用相同的已冻结原生提示与映射，过程中不改提示或按答案选择换算。运行目录为 AutoDL 的 `runs/protocol-v2/calibration/api-native-grounding-validation/`；最多 92 次调用、每次 128 输出 token，累计 token 停止阈值 100,000。它仍是历史开发验证，不是独立最终确认；状态以逐题账本、独立验收及 Release 上传回执为准。

## 真实产物和限制

- [64 条比较完成摘要](calibration/api-teacher-development/complete.json)、[独立验收](calibration/api-teacher-development/verification.json)、[冻结协议](calibration/api-teacher-development/protocol.json)。
- [8 条坐标校准完成摘要](calibration/api-native-grounding/complete.json)、[独立验收](calibration/api-native-grounding/verification.json)、[冻结协议](calibration/api-native-grounding/protocol.json)。
- 两批实验合计 72 次调用、21,293 token；另有接入测试 1 次、91 token。金额以服务商账单为准。
- [GitHub Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-protocol-v2-20261016) 保存远程原始输出、每 token 概率、评分、数据清单和来源证据；本地仅保存小型摘要，不下载原始图像或模型。
- API 实测最多 top-5 logprobs，不能直接复现原 top-50 加尾部桶 KL。序列蒸馏须另登记为不同训练损失。
- 所有输入均为公开基准图片/题目；不向 API 发送答案，不含用户私有文件，不保存密钥。
- 这些都是历史数据上的开发实验，未建立新审计和独立确认数据，阶段 C 的 `training_allowed=false` 仍有效；未启动正式多种子训练。
