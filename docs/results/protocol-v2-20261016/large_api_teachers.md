# 大参数视觉 API 教师筛选（2026-10-16）

当前默认教师为 `qwen3-vl-plus`。探测和开发实验从 `configs/api_teacher.json` 读取模型名称；恢复旧实验时若配置与冻结模型不同即停止，不混写模型结果。精确 32B 访问失败记录保留为历史事实，不再作为当前 API 教师的阻塞条件。商业 API 实验单独登记，不能充当固定参数检查点的尺度实验。

## 本次实际调用

四个候选均通过当前工作空间合成图请求返回 HTTP 200，各调用一次、输出上限 128 token、无自动重试。启用 JSON object 请求格式，并要求 color 和 bbox 字段。以下是严格契约结果，而非模型能力排名。

| 模型 | 总参数 / 激活参数 | 实际输出 | 当前严格契约 |
|---|---|---|---|
| qwen3.5-397b-a17b | 397B / 17B | bbox_2d + label 列表，框 [250,250,748,748] | 未通过字段结构 |
| qwen3.5-122b-a10b | 122B / 10B | bbox_2d + label 列表 | 未通过字段结构 |
| qwen3.5-35b-a3b | 35B / 3B | bbox_2d + label 列表 | 未通过字段结构 |
| qwen3.6-35b-a3b | 35B / 3B | 空列表 | 未通过 |

建议优先校准 397B 作为大教师，122B 作为成本和能力对照，35B 作为规模对照。MoE 总参数与激活参数分别报告；单个合成题不能证明实际任务优劣。若更改为模型原生 bbox_2d 协议，应先冻结新的独立校准协议，不事后修改本次失败判定。

`qwen3-vl-plus` 的两请求复验通过：颜色识别正确，显式 JSON object 格式的定位回答符合 color/bbox 结构，归一化坐标 IoU 为 0.8969575200918485；实测 top-5 概率仍可用。首次未指定 response_format 的回答带 Markdown 围栏，严格 JSON 解析失败，原始失败报告一并保留。

当前可先做答案/序列蒸馏设计；这些新候选尚未测试 logprobs，不可声称满足原 top-50 KL。大批生成、正式训练仍受实验 C 判定约束。

原始记录见 [大模型探测](data-preparation/api-validation/large-teachers-20261016T101000Z.json)、[当前教师复验](data-preparation/api-validation/probe-20261016T101003Z.json)、[首次严格解析失败](data-preparation/api-validation/probe-20261016T100916Z.json)。没有密钥或用户私有图片。

官方依据：[397B 模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)、[122B 模型卡](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)、[35B 模型卡](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)、[Qwen3.6-35B 百炼说明](https://help.aliyun.com/zh/model-studio/qwen3-6-35b-a3b)。
