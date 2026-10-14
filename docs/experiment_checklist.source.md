# ARP 实验清单：距完整论文数据还需要什么

> 生成时间：2026-10-11
> 硬件约束：单卡 RTX 4090 (24GB)
> 论文方法共 13 步（S1–S13），demo (`arp-demo.html`) 用合成数据覆盖了 S4–S7 的简化版

---

## 一、已有资产

| 资产 | 位置 | 说明 |
|---|---|---|
| idea card (JSON) | `ideaspark_run/mllm-audit-distill/phase4/phase4_expansion.json` | 含 method_flow 13 步、5 个公式、scoop2 六条建议已落实 |
| 中英文标准卡 | `phase4/idea.std.zh.pdf`, `idea.std.en.pdf` | 渲染完毕 |
| 审稿人版详细卡 | `phase4/idea.detail.en.md` | 40KB，含 13 条文献区分 + 5 条攻防 |
| 量尺交互 demo | `arp-demo.html` | 6 把尺 + 等价类折叠 + pi_f + F_scale/F_noise，合成数据，UI 专业级 |
| scoop-check 记录 | `scoop2/step1.md`–`step7.md` | Level 3，ReAD + Spec Curve 各 2 轴匹配 |
| 干跑脚本 | `tools/argmax_dryrun.py` | 验证了 floor correction 的 1/(1-c_a) 代数 + 等价类折叠逻辑 |

## 二、待完成清单

### 阶段 A：数据准备（无需 GPU，约 1 天）

- [ ] **A1. 下载 benchmark 数据集**
  - POPE (random/popular/adversarial)：`arxiv:2305.10355v3`，GitHub 有现成 JSON
  - RefCOCO / RefCOCO+ / RefCOCOg 验证集：需要 COCO 2014 图片 + refer API
  - MMStar：HuggingFace 直接下载
  - 确认每条轴 >= 800 题（n_min = 800）

- [ ] **A2. 编写 5-paraphrase 一致性轴脚本**
  - 对 MMBench 式题干套用 p_para = 5 条固定模板改写
  - 产出：题目清单 + 答案键
  - 确认题量 >= 800

- [ ] **A3. 生成冻结的题目清单（item manifest）**
  - 统一格式：每行 = (axis, item_id, image_path, question, answer_key)
  - 产出文件：`data/manifest.jsonl`

- [ ] **A4. 编写 4 个退化答题器**
  - POPE: 一律 "no"
  - 定位: 图像中心框，面积 = 训练集框面积中位数
  - 一致性: 固定回答
  - MMStar: 固定种子均匀随机选项
  - 产出：每个答题器的逐题对错记录，用于计算 c_a

### 阶段 B：推理账本（GPU 密集，约 24–36h）

- [ ] **B1. 编写统一推理脚本**
  - 支持 7 个检查点：Qwen2.5-VL 3B/7B, InternVL3 1B/2B/8B, LLaVA-OneVision 0.5B/7B
  - fp16 + greedy decoding
  - 输出：per-item (checkpoint, axis, item_id, raw_generation, parsed_answer, correct_bit, parse_fail_flag)
  - 分片 + 断点续跑（可中断）

- [ ] **B2. 跑退化答题器，计算实测 c_a 和 u_a**
  - c_a = 退化答题器在每条轴上的分数
  - u_a = 同家族最大成员在每条轴上的分数（从 B1 结果中取）
  - GPU 时间：~1–2h

- [ ] **B3. 跑全部 7 个检查点的推理**
  - 产出：缓存账本 L (`data/ledger.parquet` 或 `.jsonl`)
  - GPU 时间：~24–36h（瓶颈步骤）
  - 注意：LLaVA-OneVision 7B 在 24GB 下 fp16 可能需要 offload 或 4-bit 量化推理

### 阶段 C：CPU 分析（无需 GPU，约 0.5 天）

- [ ] **C1. 量尺算子 + 三道护栏**（对应 S4–S5）
  - 实现 5 坐标全组合：drop × floor × headroom × OWA(ρ) × parse_fail
  - 三道护栏：sub-floor guard / zero-denom guard / headroom-sign guard
  - 用 `argmax_dryrun.py` 的逻辑扩展，接真实账本

- [ ] **C2. 等价类折叠**（对应 S6）
  - 标称 2×2×3×5×2 = 120 → 可辨识商集
  - 产出：折叠对照表（论文 Table 1 候选）

- [ ] **C3. Bootstrap + argmax 保持率**（对应 S7）
  - n_boot = 1000 次题目级重采样
  - 产出：per-ladder 的 pi_f、F_f、stable-axis set、F_scale、F_noise + CI
  - **关键判定点：** 如果 stable-axis set 在每条阶梯上都非空，且 F_scale 落在 F_noise 的 CI 内 → 按论文自己的 falsification prediction，没有 paper。在这一步之后做 go/no-go 决策

- [ ] **C4. 坐标归因表**（对应 S8）
  - 逐坐标冻结，报告翻转份额
  - 验证 floor correction 是否排第一（预测：是）

- [ ] **C5. 植入 argmax 预言实例**（对应 S9）
  - 合成对错记录，planted worst axis + planted floors
  - 跑 S4–S8，验证：同质 floor → 恢复 planted axis；异质 floor → 复现 (0.5102, 1.0) 边界

### 阶段 D：LoRA 训练臂（GPU，约 20h）

> 只有阶段 C3 确认 go 之后才进入此阶段

- [ ] **D1. 准备 LoRA 训练数据**
  - 按 pi_f 排名选 top 轴和 runner-up 轴
  - 构造 4 份训练集：A1(top轴), A2(runner-up轴), U(混合), N(no-op)

- [ ] **D2. 缓存教师 logit**（对应 S10）
  - Qwen2.5-VL 7B 在训练混合数据上，top-50 logits/position 缓存到磁盘
  - GPU 时间：~4–6h

- [ ] **D3. 训练 4 个 LoRA 臂**（对应 S11）
  - 基座：Qwen2.5-VL 3B
  - 超参：S_steps=1000, r=16, alpha=32, lr=1e-4
  - GPU 时间：~4×1.5h = 6h

- [ ] **D4. 臂后重新推理 + 净跨轴增益**（对应 S12）
  - 4 个适配学生 × 4 轴 × >=800 题
  - 产出：G_net per arm per ruler class，附 per-axis signed components
  - GPU 时间：~8–12h

- [ ] **D5. 同质 floor 负对照**（对应 S13）
  - 把 c_a 全换成 c_bar，重跑 S4–S8 + 选臂
  - CPU 几分钟

### 阶段 E：论文写作素材

- [ ] **E1. 生成论文中的所有表格/图**
  - Table 1: 等价类折叠对照表（C2）
  - Table 2: per-ladder pi_f / F_f / stable-axis set（C3）
  - Table 3: 坐标归因表（C4）
  - Table 4: 4 臂净跨轴增益 + per-axis 分量（D4）
  - Figure 1: 量尺扫描可视化（可从 demo 改造）
  - Figure 2: F_scale vs F_noise with bootstrap CI（C3）

- [ ] **E2. 补充材料**
  - `arp-demo.html` 作为交互附件（修复 F_noise 的尺子说明）
  - 完整折叠表 120 → K 的映射
  - 植入预言实例的恢复率图

---

## 三、时间线估算

| 阶段 | 耗时 | 依赖 | 备注 |
|---|---|---|---|
| A 数据准备 | 1 天 | 无 | 可以现在开始 |
| B 推理账本 | 2–3 天挂机 | A 完成 | **关键路径** |
| C CPU 分析 | 0.5 天 | B 完成 | 出 go/no-go 结论 |
| D LoRA 训练 | 1–1.5 天挂机 | C 确认 go | |
| E 写作素材 | 与 C/D 并行 | 各步产出 | 表格代码可以提前写 |
| **总计** | **~5–7 天实验 + 写作并行** | | 在 3 周内可控 |

## 四、风险点

1. **falsification prediction 成立 → 没有 paper**
   如果 stable-axis set 在三条阶梯上全部非空且 F_scale ≤ F_noise，按论文自己的声明这个想法被杀死。阶段 C3 是 go/no-go 门槛，在它之前不投入训练算力。

2. **LLaVA-OneVision 7B 显存溢出**
   fp16 下约 14GB 权重 + 推理 overhead 可能吃紧。备选：4-bit 量化推理（但需要在论文中说明），或者只用 0.5B 端点跳过 7B。

3. **RefCOCO IoU 计算的解析率**
   0.5B 模型输出 bounding box 格式极不稳定，parse_fail_flag 可能很高。这已经被设计成量尺的一个坐标（parse-failure handling），但如果失败率 > 50% 需要在论文中讨论。

4. **F_scale 和 F_noise 分离度太小**
   n_min = 800 是设计的功效下限，但真实数据上分离度如果 < 0.1 可能不够 convincing。备选：提高到 n_min = 1200 但一致性轴的题量可能不够。

5. **demo 的 F_noise 不对称**
   `simulateNoise` 只在绝对掉分下模拟。论文定义的 F_noise 就是"固定常规设置下重采样"，所以逻辑一致，但建议在 demo 里加一行说明固定的是哪把尺。
