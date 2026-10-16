# protocol-v2：补分析与显存预检

日期：2026-10-16。本目录为阶段性产物，**不是全部补实验已经完成**。旧 v1 结果仍保留在 [原结果目录](../final-20261016/README.md)。

## 已完成

| 内容 | 真实产物与边界 |
|---|---|
| P0 公式与估计量审查 | [审查说明](p0/formula_audit.md)、[旧新分析对照](p0/analysis_v1_vs_v2.csv)；旧数据探索性重算，不是独立确认 |
| 严格翻转、并列与 guard | [敏感性表](p0/tie_and_guard_report.csv)、[逐类向量](p0/class-vectors.json)、[headroom 分母诊断](p0/headroom_denominator_diagnostics.csv) |
| 全量尺与共同题集 | [120 设置×4 臂](p0/arm_gains_all_120.csv)、[共同覆盖](p0/arm_common_support.csv) |
| 训练臂配对分析 | [原始/floor 增益](p0/signed-gains-v2.json)、[配对差值区间](p0/arm_paired_contrasts.csv)；1,000 次共享图像簇重采样，仍只有一个训练 seed |
| Qwen 3B BF16 开发预检 | [800 条完成记录](calibration/precision-bridge/qwen3b/complete.json)；每轴 100 原题，一致性保持五改写 |
| Qwen 7B BF16 开发预检 | [800 条完成记录](calibration/precision-bridge/qwen7b/complete.json) |
| InternVL 2B / 8B BF16 开发预检 | [2B 的 800 条完成记录](calibration/precision-bridge/intern2b/complete.json)、[8B 的 800 条完成记录](calibration/precision-bridge/intern8b/complete.json) |
| LLaVA 0.5B BF16 开发预检 | 800 条已全部完成；真实账本和完成标记见本阶段 Release，五模型合计 4,000 条 |
| 阿里云视觉 API 接入 | [实际调用验证](api_validation.md)；`qwen3-vl-plus` 视觉调用及 top-5 logprobs 成功，不等同于指定的 32B/72B 检查点 |
| 3B 训练显存与蒸馏可行性 | [100 次实际更新](calibration/training-memory-3b/complete.json)、[独立验收](calibration/training-memory-3b/verification.json)、[冻结配置](calibration/training-memory-3b/protocol.json)；属于工程预检，不是正式训练臂 |
| 7B 教师文本 CE 训练预检 | [100 次实际更新](calibration/training-memory-7b/complete.json)、[独立验收](calibration/training-memory-7b/verification.json)、[冻结配置](calibration/training-memory-7b/protocol.json)；没有调用 API 教师 |

| 实测项目 | PyTorch allocated 峰值 | reserved 峰值 |
|---|---:|---:|
| Qwen 3B 全 BF16 推理 | 7.573 GiB | 7.754 GiB |
| Qwen 7B 全 BF16 推理 | 16.006 GiB | 17.400 GiB |
| InternVL 2B 全 BF16 推理 | 4.880 GiB | 5.463 GiB |
| InternVL 8B 全 BF16 推理 | 15.787 GiB | 15.953 GiB |
| LLaVA 0.5B 全 BF16 推理 | 2.664 GiB | 以 Release 原始完成记录为准 |
| Qwen 3B 全 BF16、100 步 LoRA+KL | 7.755 GiB | 8.457 GiB |
| Qwen 7B 全 BF16、100 步 LoRA+教师文本 CE | 16.275 GiB | 18.852 GiB |

上述峰值不等于整卡占用；nvidia-smi 采样和原始输出在下方 Release。3B 训练独立验证了逐步实际更新、有限损失/梯度、非零 LoRA B、adapter 与第 100 步恢复状态完全一致。

## 已知发现

- Qwen 原九个经验类中的一个 headroom 全平局类，解释了旧 `F=0.08333`；严格唯一最大轴反转为零。采用结构商集后 F 数值改变，不能当成新的能力证据。
- A1−A2 的原始增益差为 `0.0031667`，共享图像簇的探索性 95% 区间约为 `[-0.01849, 0.02587]`，跨零；floor 增益差区间同样跨零。没有证明 A1 优于 A2。
- InternVL/LLaVA 的共同四轴分析受到大量无效量尺域限制；不能只报告有效少量类中的大翻转比例。
- [P0 gate](p0/gate.json) 明确 `formal=false`、`training_allowed=false`。新的正式审计与独立确认尚未完成；工程更新不代表开放预算训练门槛。

图像簇区间使用全局来源图像共同权重，逐子集计算均值后等权聚合，保留同图跨轴关联；不是各轴独立抽样。它只反映旧数据的评测抽样不确定性，不能替代多 seed 或去除旧选择偏差。

## 显存受限任务完成情况及 API 开发实验

五模型推理预检和 3B/7B 两个训练预检均已完成。本批于北京时间 2026-10-16 12:29 完成。LLaVA 0.5B 曾在 GPU 进程切换时触发空闲检查退出；增加最多 45 秒的显存释放等待后，已恢复并完成 800 条。所有 GPU 进程共用互斥锁，单进程 PyTorch 显存上限 23GiB；3B 训练预检上限为 12GiB。

这批任务只使用现有已缓存模型，不重新下载 32B/72B。7B 的 CE 预检采用已缓存 7B 教师文本，不调用付费 API，也不使用 logits；它检验序列训练的工程可行性，不声称检验了大教师效应。状态以各阶段完成标记、逐题记录和独立验收为准。

另于北京时间 13:35 启动有上限的 API 开发比较：每轴 8 个原题单元，共 32 单元、64 条提示；一致性保留五改写，使用历史开发数据，不进入训练集或独立确认。输入全部与公开基准原始清单匹配，不含用户私有文件，不向 API 发送答案。每次最多输出 128 token，累计用量停止阈值 200,000 token；失败及无法确认结果的调用不自动重试。原图像素、原提示和严格解析器保持不变，API 与本地 BF16 3B/7B 在同题上比较；不同服务端图像处理仍是协议差异。本实验不能解锁正式训练，产物完成后由 AutoDL 直接备份 GitHub Release。

13:36 已完成全部 64 条；发现定位坐标约定不匹配后，另冻结显式 `[0,1000]` 坐标契约，于 13:39 完成 8 条原生坐标校准。两批均通过独立验收，完整原始结果已上传 Release。详见 [API 开发实验结果与边界](api_development_results.md)。

13:45 又完成排除上述 8 条后的 92 条开发验证，并通过独立验收和上传核验。16:45 完成六组 [配对量尺与噪声分析](paired_gate_development.md)，仍未满足正式训练放行条件。

## 独立数据与持续监控

新增 AutoDL 后台 `arp-v2-monitor`，每 5 分钟记录真实进程、状态、日志活跃度、GPU 占用、验收与备份回执是否存在；进程缺失、阶段失败和 30 分钟无进度单独记录。不重试付费调用，不覆盖科学 gate。心跳位于 `runs/protocol-v2-monitor/heartbeat.json`，变化事件位于同目录 `events.jsonl`。这是远程日志监控，不是 Codex 定时唤醒或独立通知服务。

已启动 AI2D 与 MMMU test 的独立数据源准备，共 34 个 Parquet、3,397,133,636 字节。官方小型元数据从控制机获取并锁定 revision/文件 SHA256；数据本体通过可用镜像直接下载至 AutoDL，并与官方 LFS SHA256 验证。状态位于 `runs/protocol-v2-data/source-download/status.json`。原始源文件下载完成不等于四轴独立清单已经完成，仍须去重、图像簇划分和清单冻结。

## GitHub 原始产物与自动备份

[补实验阶段性 Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-protocol-v2-20261016) 保存完成阶段的原始输出、逐题日志、显存采样、短程训练 adapter 和恢复状态。首次完整上传约 109MB，后续每个模型完成后由 AutoDL 直接上传新快照，并验证服务端大小和 SHA256；不把未完成账本冒充完成结果。

## 仍待完成

独立训练/校准/审计/确认清单、近重复复核、人工定位盲审、新 C 配对噪声 gate、多种子正式训练、精确大模型/API 接入、第二家族训练与论文修订均未宣布完成。最新硬件路径参考 [大模型方案核查意见](../../大模型方案_核查意见_20261016.md)，原研究路线见 [整体改进方案](../../ARP_整体改进与补实验方案_20261016.md)。
