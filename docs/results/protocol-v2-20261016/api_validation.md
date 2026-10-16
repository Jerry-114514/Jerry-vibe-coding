# API 接入验证

验证时间：2026-10-16 13:31（北京时间）。在 AutoDL 新交互式 Bash 会话读取用户配置的环境变量；不保存或输出密钥。

- 模型列表接口：HTTP 200，共 176 个模型 ID。列表可见不代表逐一调用权限已验证。
- 视觉模型 `qwen3-vl-plus`：实际调用 HTTP 200，对 128×128 合成红色图片回答 `Red`。
- 用量：输入 89 token（图片 66、文本 23），输出 2 token，共 91 token。未据此推算价格。
- 请求 `logprobs=true, top_logprobs=5` 成功，实际返回最多 5 个候选 token 的概率。
- 这不足以复现原实验 top-50 token 加尾部概率的 KL 蒸馏；API 生成文本可用于另行登记的序列蒸馏实验。
- `qwen2.5-vl-32b-instruct`、`qwen2.5-vl-72b-instruct` 不在该端点当前模型列表中，尚未验证其调用能力。
- 响应模型名称仍为 `qwen3-vl-plus`，尚未确认不可变版本。不能把它当作确定的 32B/72B 检查点或原模型规模梯度的一部分。
- 当前仅完成接入验证，不构成阶段 C 训练许可；未启动正式训练或批量付费调用。

完整脱敏记录位于 AutoDL：`/root/autodl-tmp/arp/runs/protocol-v2/api-validation/probe-20261016T053138Z.json`。
