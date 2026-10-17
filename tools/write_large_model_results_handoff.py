"""Write evidence-backed Chinese result and handoff documents; no experiments run."""
import hashlib
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'docs/results/protocol-v2-20261016'
EVIDENCE = DEST / 'handoff-evidence-20261017'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    snap = read(EVIDENCE / 'handoff-live-snapshot.json')
    nine = read(DEST / '9b-training-final/training-results.json')
    health = read(DEST / '9b-training-final/training-health.json')
    seven_cal = read(EVIDENCE / '7b-calibration-proof.json')
    failure = read(EVIDENCE / '9b-failed-processor-calibration.json')
    budget = read(EVIDENCE / '7b-active-gpu-budget.json')
    assert len(nine['runs']) == 15 and nine['updates'] == 15000
    assert all(x['updates'] == 1000 for x in nine['runs'])
    assert health['status'] == 'passed'
    assert seven_cal['accuracy'] == .8 and seven_cal['parse_failures'] == 2
    assert failure['scores']['registered_processor_pixels']['accuracy'] == .01
    hours = sum(x['step_seconds'] for x in nine['runs']) / 3600
    peak = max(x['reserved_peak_bytes'] for x in nine['runs'])
    maxgrad = max(x['max_grad_norm_before_clip'] for x in health['runs'])
    count = sum(snap['seven_models'].values())
    live = snap.get('live_pids', {})
    completed = snap.get('seven_completed_scores', {})
    completed_table = '\n'.join(
        f"| {name} | {row['rows']:,} | {100*row['accuracy']:.3f}% | {row['parse_failures']} |"
        for name, row in completed.items())
    stamp = datetime.fromisoformat(snap['utc']).astimezone(
        __import__('datetime').timezone(__import__('datetime').timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    source = ROOT / 'docs/references/大模型方案_用户修订_20261016.md'
    refs = '\n'.join(f'| `{p.name}` | `{sha(p)}` |' for p in [
        source, EVIDENCE / '7b-active-processor-protocol.json',
        EVIDENCE / '7b-csv-schema-amendment.json', DEST / '9b-processor-pixel-protocol.json',
        DEST / '9b-training-final/training-results.json'])
    training_table = '\n'.join(
        f"| {r['arm']} | {r['seed']} | {r['seconds_per_step']:.3f} | {r['step_seconds']/60:.2f} | "
        f"{r['reserved_peak_bytes']/1e9:.3f} | {r['last_loss']:.6g} |" for r in nine['runs'])
    report = f'''# 基于《大模型方案》的实验结果报告

更新日期：{stamp[:10]}；状态快照：北京时间 **{stamp}**。文件名保留首次交付日期。这是阶段性结果报告，原ARP全部补实验尚未完成。配套[实验交接文档](大模型方案_实验交接文档_20261017.md)。

## 1. 依据、研究问题与执行变更

依据仓库归档的[用户版《大模型方案》](../../references/大模型方案_用户修订_20261016.md)，结合用户后续明确授权的执行变更：4090释放后统一使用RTX 5090；阶段C未放行时先运行独立路线B；完成7B自蒸馏后接力9B学生与122B API教师；最新批准不重训、先统一定位测量。

方案原定P1-4为7B学生加122B教师。**实际完成的7B是7B教师/7B学生自蒸馏；9B是9B学生/122B教师的独立扩展。二者均不能冒充原定P1-4确认性实验。** P1-2应为3B学生/7B教师、原KL协议，也不能用7B自蒸馏替代。

原方案中的估价、商业模型参数量映射和文献条目属于参考材料，本报告不将其直接当成已验证事实；费用采用用户指定的5090单价2.70元/小时，API费用未取得账单时不编造金额。

## 2. 方案逐项落实情况

| 方案项目 | 当前实证 | 完成边界 |
|---|---|---|
| P0-1 公式核对 | [审计](p0/formula_audit.md)、[新旧表](p0/analysis_v1_vs_v2.csv)已输出 | 旧数据探索性重算，不是新独立确认 |
| P0-2 平局/反转/分母 | [敏感性及guard表](p0/tie_and_guard_report.csv)已输出 | 不将定义性平局当作严格翻转 |
| P0-3 全120设置 | [120设置×4臂](p0/arm_gains_all_120.csv)、[共同题集](p0/arm_common_support.csv)已输出 | 旧训练结果的补分析 |
| P0-4 定位测量审计 | 7B全16模型旧输出CPU审计完成；明确提示的GPU复测进行中 | 双人独立盲审未完成；9B共同契约校准失败 |
| P1-1 独立确认集 | 尚无全部冻结并验收完成的证据 | 当前16,900提示是既有开发基准，不能称独立确认集 |
| P1-2 3B←7B KL桥接 | 尚未完成15次原协议桥接实验 | 最小工程预检不能替代正式桥接 |
| P1-3 扩展阶梯及§4.5精度桥接 | 精确3B/7B API上次403；商业审计暂停于1,365条 | 四轴严格差异<0.005未获证明，不能合并阶梯 |
| P1-4 原定7B←122B | 原ARP阶段C未放行，原主线未完成 | 独立7B与9B结果分别列出，不改名充当主线 |
| P2-1 第二家族、P2-2机制消融 | 尚无完整验收结果 | 不声称序列CE优于KL或跨家族泛化 |
| §12 存储与可复核性 | 冻结配置、缓存、日志、适配器、原始输出及Release校验链已建设 | 新复测逐模型归档，未完成部分不标记交付完成 |

原C的[当前判定](p0/gate.json)为`decision=indeterminate, formal=false, training_allowed=false`。工程成功不解除该科学门槛。

## 3. 独立7B自蒸馏：已完成的原轮结果

Qwen2.5-VL-7B-Instruct固定同一教师/学生检查点；BF16语言注意力和MLP LoRA，r=16、alpha=32、lr=1e-4、micro-batch=1、梯度累积1、梯度裁剪1.0。双CE为0.5真实答案CE+0.5教师回答CE；虽然保留top-50缓存，本实验KL权重为0。

A1使用mmstar轴的ScienceQA训练代理，A2使用consistency，U四轴等比例，N不更新。训练池3,545提示；全量EOS证据已核验。seed固定42/137/256/512/1024，每臂每seed固定1,000次真实更新。五项工程验收通过，A1最小循环第50步退出后从51步恢复至100步。完整方法和逐seed结果见[7B详报](7B独立自蒸馏实验结果报告.md)。

15次训练、15,000更新及原轮16×16,900=270,400条评测已完成。下表是五seed均值，N只有一个未训练基线；均为百分比，**定位列保留为旧测量结果，不作纯能力解释**。

| 臂 | POPE | 定位：旧计分 | 一致性 | MMStar |
|---|---:|---:|---:|---:|
| N | 87.567 | 67.792 | 84.125 | 62.933 |
| A1 | 87.807 | 66.133 | 85.325 | 61.840 |
| A2 | 87.476 | 67.458 | 85.225 | 61.613 |
| U | 88.791 | 77.658 | 83.775 | 61.640 |

原轮五seed配对seed/图像簇bootstrap，10,000次重采样，3组×4轴共12项Bonferroni校正；A1−A2、A1−N的四轴区间均包含0。唯一排除0的是**旧定位口径**A1−U：−11.525个百分点，校正区间[−16.157,−5.648]。由于定位单位混杂，不能据此主张U定位能力提高；含该轴的宏均值或G_floor也须重新审视。区间跨0表示证据不足，不表示已证明相等。

纯训练更新2.306小时，折算6.23元；原轮纯生成15.803小时，折算42.67元；合计约48.89元。最高训练reserved为24.547GB/22.861GiB，原轮评测最高20.122GB/18.740GiB。费用不含缓存构建、模型装载、保存、上传、排队及其他开机时间，不能当作账户总账单。

## 4. 测量审计：目前最关键的发现

原始输出按预先固定的不同坐标契约作CPU诊断时，7B N从原图像素口径67.792%变为处理器像素映射口径85.500%，U均值则从77.658%变为73.558%。这是单位/格式混杂的证据，**不是可以选择较高分替换主分的理由**；旧提示要求、模型原生格式、训练目标的交互尚未分离。见[全16模型CPU审计](7b-grounding-cpu-report.md)。

最新注册共同映射为：`x_orig=x_proc×W_orig/W_proc`、`y_orig=y_proc×H_orig/H_proc`，与原图标注计算IoU≥0.5；每条断言处理器实际网格。禁止按GT、训练臂或数值大小自动猜单位。

7B新提示要求“四个数”，第一版JSON-only解析器错误拒绝了严格CSV输出。为此在重新计分前发布纯语法修订：所有16模型统一接受既有JSON格式及严格四数CSV，不改变坐标值、单位、样本或IoU。保留原100条及旧协议，不新增生成。相同N前100条在修订后为**80%准确率、2次解析失败**，恰好达到冻结门槛≥80%、≤5次失败；属于事后测量修订，须披露，不能表述为首次协议直接通过。

快照时7B新定位已生成 **{count:,}/38,400** 条，完整16模型尚未结束，新五seed终局区间尚未产生；这不是最终定位能力分数。固定GPU预算截止为UTC `{budget['end_utc']}`（北京时间10月18日11:42:49），未因语法修订重置。完整复测后报告A1−A2、A1−U、A1−N及U−N，独立4项Bonferroni族，不与原12项混成事前检验。

新契约下已完成的逐模型结果如下；训练臂暂为单seed，不能据此作五seed显著性结论。

| 模型 | 已完成定位题 | 新契约定位分 | 解析失败 |
|---|---:|---:|---:|
{completed_table}

## 5. 独立9B←122B：15次训练完成，正式评测未放行

学生为官方Qwen3.5-9B，教师为Qwen3.5-122B-A10B API；从3,545训练池得到3,540条可用教师回答，另5条因供应方图片检查按来源组隔离，不能称完整3,545条全部成功。已完成前缀未重复请求。API文本目标不是top-50 logits，也不以`finish_reason=stop`冒充原始EOS token证据。

9B最小循环五项工程检查通过；之后15次固定1,000步训练全部完成，共15,000更新，完成时间北京时间10月17日22:07:22。端点是固定第1,000步，没有根据结果加训或选最好检查点。

| 臂 | seed | 秒/步 | 纯更新分钟 | reserved GB | 第1,000步损失 |
|---|---:|---:|---:|---:|---:|
{training_table}

![9B十五次训练损失曲线](9b-training-final/loss-curves.png)

纯更新合计 **{hours:.3f}小时，约{hours*2.70:.2f}元**；reserved最高 **{peak/1e9:.3f}GB/{peak/2**30:.3f}GiB**。训练健康检查确认两路CE与梯度有限、更新数及导出匹配；但最大裁剪前梯度范数为 **{maxgrad:.2f}**，使用1.0裁剪，因此不能声称全程没有梯度尖峰。末步损失是单样本端点，不是验证集均值，也不证明泛化能力。详见[训练数值](9b-training-final/training-results.json)、[健康核验](9b-training-final/training-health.json)。

9B共同处理器像素契约的N校准100条仅**1%准确率、95次解析失败**，未达到≥80%及≤5次失败，已停止；**正式16模型评测为0/16，没有9B正式四轴结论或多seed区间**。严格CSV语法也不能修复该单位问题。同一批输出按固定0–1000解释的CPU事后诊断为86%、0次失败，只支持单位不一致诊断，不能替换注册分数或据此放行。见[失败原证据](handoff-evidence-20261017/9b-failed-processor-calibration.json)。

9B的15分钟原校准绝对预算已截止，不重置，不反复修改提示追求80%。可讨论另立“家族原生前端→统一处理器像素→原图”的协议，但必须在新推理前固定四臂校准与独立预算；目前该方案未执行。

## 6. 对研究假设的结论与限制

1. 已验证单卡完整多臂、多seed双CE蒸馏、保存/加载、恢复及逐题归档流程可运行；不等于原ARP量尺选轴因果命题成立。
2. 7B现有证据未证明A1优于A2或N；旧U定位优势存在测量混杂，必须等待新统一契约复测，不能据旧优势写论文主结论。
3. 9B训练完成但测量未通过，暂不能判断122B教师带来的能力收益。
4. 两轮同时改变学生代际/容量、教师、tokenizer/视觉处理器及可用训练池，不能将差异单独归因于7B→9B容量变化或教师规模。
5. 当前结果主要来自既有开发基准，未完成独立确认、同检查点API桥接、双人盲审及有效阶段C；不能声称P0–P2全部完成、商业阶梯等于精确参数阶梯或ARP主线已验证。

## 7. 数据交付、下一步与证据身份

7B原轮3,738/3,738必要文件的交付覆盖已核验，缓存、适配器、恢复链、逐步日志及原始评测见[7B Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-protocol-v2-20261016)。9B缓存、训练适配器、恢复链及必要过程见[9B Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-9b-122b-20261017)。新7B定位复测逐模型核对2,400唯一ID、SHA和服务器归档后续跑；尚未生成的结果不能视作已上传。

快照时数据盘空闲{snap['data_free_bytes']/2**30:.2f}GiB，系统盘{snap['root_free_bytes']/2**30:.2f}GiB，DiskGuard运行。优先完成7B复测和独立4项区间；9B保留失败记录，另行固定可审阅的新接口校准；原ARP八项前置工作详见[执行计划](route_b_execution_plan.md)。执行入口、监控与恢复注意事项均在[交接文档](大模型方案_实验交接文档_20261017.md)。

| 证据文件 | SHA256 |
|---|---|
{refs}

本报告只反映上述时间快照；活动任务以远程完成标记、实际行数与独立验收为准。
'''
    handoff = f'''# 大模型方案实验交接文档

快照：北京时间 **{stamp}**。配套[实验结果报告](大模型方案_实验结果报告_20261017.md)。本文件是交接说明；表中的状态不是未来完成承诺。所有时间未注明时均为北京时间。

## 1. 接手时必须知道的状态

| 工作流 | 状态 | 接手动作 |
|---|---|---|
| 原ARP主线 | C未放行，`training_allowed=false` | 不将独立实验改名为原主线，不擅自放行 |
| 独立7B原轮 | 15次训练及16模型四轴评测、旧12项统计和归档完成 | 保留全部历史结果；定位能力解释待审计 |
| 7B定位复测 | 正在运行，快照{count:,}/38,400条，{len(completed)}/16模型完成 | 优先保持队列；每模型2,400条，最终4项区间 |
| 独立9B训练 | 15/15、15,000更新完成 | 不重训、不追加预算，不恢复无用优化器做评测 |
| 9B正式评测 | N共同契约校准失败，0/16正式评测 | 保留停止；不能用86%事后诊断替换1%注册成绩 |
| 磁盘 | 数据盘{snap['data_free_bytes']/2**30:.2f}GiB、根分区{snap['root_free_bytes']/2**30:.2f}GiB空闲 | DiskGuard继续运行；删除前先验证云端成员与SHA |

## 2. 环境与连接

- 本地工作区：`D:\\gpt data`，PowerShell；仓库[main](https://github.com/Jerry-114514/Jerry-vibe-coding)，工作分支`codex/arp-autodl`。
- 当前远程：`ssh -p 20479 root@connect.bjb2.seetacloud.com`。旧4090已释放，不连接34903旧端口。
- 项目：`/root/autodl-tmp/arp`；RTX 5090，整卡32,607MiB。
- 7B/控制环境：`/root/autodl-tmp/arp-venv/bin/python`；9B环境：`/root/autodl-tmp/arp-qwen35-venv/bin/python`。
- 使用`tools/autodl_new_instance.py`适配器，不直接用指向旧实例的`autodl_transport.py`入口。SSH私钥及known_hosts位于本地`tmp/autodl/`，不写入GitHub或报告。
- API key及GitHub凭证均不在本交接中复制。运行中服务从环境读取；发布使用本地凭证助手，经加密SSH标准输入传递。不要打印环境变量或开启shell trace。

只读状态检查示例：

```powershell
python tools/autodl_new_instance.py run "df -h / /root/autodl-tmp"
python tools/autodl_new_instance.py run "pgrep -af 'continue_7b_processor|run_7b_processor|protect_9b_disk'"
```

看到screen和Python多个匹配并不等于重复计算；必须区分控制器、实际eval子进程和检查命令本身。不要因历史PID消失就重跑已完成训练。

## 3. 四臂、预算与冻结身份

A1=mmstar轴ScienceQA训练代理；A2=consistency；U四轴等比例；N不更新。五seed为42/137/256/512/1024。两个独立实验均为3臂×5seed×1,000真实更新，固定最后一步，禁止按分数延长。N一个基座，不复制成五个伪重复。

7B损失为双CE，KL权重0；不冒充3B←7B原KL桥接。原始7B数值引擎`route_b_staged.py` SHA为`bb8fb654d254912abaf291b59aeec7f301aaa5437930cfc9e9080cce79de2c30`。预检配置SHA为`781d0f024aa74a0ec02f4628380c5a5d4176d6bac123f84de7a60e78171d060a`。

当前7B复测协议SHA：`{sha(EVIDENCE/'7b-active-processor-protocol.json')}`；[副本](handoff-evidence-20261017/7b-active-processor-protocol.json)。9B失败协议SHA：`{sha(DEST/'9b-processor-pixel-protocol.json')}`。冻结数值源码与哈希必须一致；不要用改文件的方式掩盖历史失败。

## 4. 7B正在运行的任务

工作目录：`runs/7b-processor-pixel-audit-20261017`。

| 内容 | 路径/入口 |
|---|---|
| 独立控制器 | `tools/continue_7b_processor_grounding.py` |
| 数值评测器 | `tools/run_7b_processor_grounding.py` |
| 原共同单位函数 | `tools/processor_pixel_contract.py`，保持9B历史版本不变 |
| 新纯语法包装 | `tools/processor_pixel_contract_csv.py` |
| 控制器screen / 快照PID | `7b-independent-processor-audit` / {live.get('7b_controller', '未发现')} |
| 当前评测快照PID | {live.get('7b_worker', '未发现')}；模型见`queue-status.json`，PID会随模型切换变化 |
| 控制台 | `runs/7b-processor-pixel-independent-console.log` |
| 逐模型输出 | 工作目录下`models/<arm>-seed<seed>/eval.jsonl`及`scores.json` |
| 校准原证据 | `calibration-proof/eval-first100.jsonl`与`calibration.json` |
| 时间预算 | `gpu-wall-budget.json`，原绝对截止UTC `{budget['end_utc']}` |
| 最终产物 | `results.json`、`report.md`、`complete.json`及`terminal-proof/` |

原控制器把7B错误串在9B校准后；现已登记只改变队列依赖的独立控制器。历史protocol内旧queue文字保留，实际执行应结合`independent-queue-amendment.json`解释，不静默改写历史注册记录。

7B首100条的JSON-only解析错误通过严格CSV联合语法修复：所有模型同一语法，不改变坐标值或单位。修订在重评分前登记，旧100条原样保留于`schema-v1-history/`。修订后80%、失败2/100，无新增生成；CSV修复不是9B单位问题的解决方案。

固定总量38,400包含这100条；不能再额外跑一批100条，不重置12小时预算。新映射只用真实处理器宽高及原图尺寸，禁止按输出数值、臂、GT择单位。逐模型验收后立即压缩并上传7B Release，核对服务端大小和SHA。

异常恢复前先检查当前数值子进程、`queue-status.json`、`stopped.json`、预算截止和已完成行数。**运行中不启动第二个控制器。** 仅瞬时网络错误可按已锁定重试策略恢复；OOM、非有限值、哈希不符、门槛失败或预算耗尽不能当网络错误重试。不要盲删完成标记或失败记录。

全部结束后核对16×2,400唯一题目、原始输出重计分及Release回执，生成A1−A2、A1−U、A1−N、U−N的五seed配对区间，10,000 bootstrap、独立4项Bonferroni族。完成状态应为`all16_processor_pixel_grounding_verified_and_delivered`；不能只看进程退出码就宣布完成。

独立CPU图表收尾服务`tools/finish_7b_processor_report.py`已排队，screen为`7b-processor-figures`。它等待全部16模型完成和交付标记后，逐条复核38,400行的映射/IoU、生成PNG/PDF并上传`final-report/`；不增加预测。N全2,400行的独立映射与IoU核验已通过。详见[零点后续跑记录](measurement-progress-20261018.md)。

## 5. 9B停止原因与下一步边界

根目录：`runs/route-b-9b-122b-20261017/student-experiment`；当前失败测量目录：`processor-pixel-evaluation-v3/`。

15次训练结束UTC2026-10-17T14:07:22.180141，`formal-training-complete.json`的历史evaluation字段不是评测完成证明。当前有15份恢复可用链：11份本地、4份已验证Release；评测只需基座加适配器，不加载优化器，不为评测下载这4份恢复状态。

| 当前证据 | 结论 |
|---|---|
| N新定位100条，处理器像素注册口径 | 1%，95解析失败：未通过 |
| 同100条加入严格CSV语法 | 仍1%，95失败：语法修复无效 |
| 同输出固定0–1000的CPU事后解释 | 86%，0失败：只作单位诊断 |
| 正式16模型四轴预测 | 尚未启动，不存在正式多seed结论 |

原校准预算UTC14:08:39.291756至14:23:39.291756已截止。不要自动重启`run_9b_native_evaluation.py`，不要沿用原budget反复调提示。旧`native-evaluation-v2`、旧像素最小循环与当前v3均保留为历史，不覆盖。

可审阅但**未执行**的后续接口：7B固定原生处理器像素；9B固定原生0–1000，先变换到处理器像素，再用共同映射变回原图。若执行，须另立版本并在新增推理前登记四臂相同前端、固定校准样本、独立预算及停止规则；特别验证U学习像素训练目标后是否遵循原生提示。建议400条/15分钟上限仅为待定方案，不是已经启动的新预算。详见[失败与接口提案](handoff-evidence-20261017/9b-failed-processor-calibration.md)。

## 6. 磁盘保护与发布

DiskGuard：`tools/protect_9b_disk.py`，快照PID{live.get('disk_guard', '未发现')}，约45秒采样两个分区；数据盘15GiB触发保护/清理，5GiB硬停止规则保留。当前数据{snap['data_free_bytes']/2**30:.2f}GiB、根分区{snap['root_free_bytes']/2**30:.2f}GiB；这不是“只剩100MB”。

18份旧7B恢复副本已核对归档成员、当前SHA、Release服务端SHA并先上传恢复指针后删除，释放8.13GiB。不要把已云端归档的本地缺失误报为产物丢失；需要恢复时按指针校验后下载。后续清理仍优先旧7B已完成且确认备份的大文件，保留适配器、损失曲线、数据源、模型权重及9B必要恢复链。不做无证据的批量递归删除。

大产物从AutoDL直接上传，不拉回Windows：

- [7B Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-protocol-v2-20261016)：原轮及新7B定位审计。
- [9B Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-9b-122b-20261017)：122B教师缓存、9B适配器/恢复/日志、校准证据。
- 代码和小型表格/证据进入Git；私钥、API key和大权重不进入Git。

必要过程包括协议/修订、清单与SHA、教师EOS或API终止原因、逐步双CE/梯度/显存/耗时、恢复点、逐题原始输出、异常及重试、归档成员清单、服务端校验回执。上传请求成功不能替代内容/行数/成员/SHA核验。

## 7. 托管与时间估算

早报服务`tools/report_9b_native_progress.py`（快照PID{live.get('morning_report', '未发现')}）预定10月18日08:00，内容以实际产物为准；已修正为明确报告9B失败及附带修订证据，不再误写等待中。传输观察器不应绕过校准失败。控制台与完成标记优先于旧进度文案。

7B复测原外推纯生成约6.47小时，含装载/校验/上传暂估6–8小时；新增提示及实测可能改变，绝对上限仍是上述12小时。快照只记录已生成条数，不用短时间N速度冒充全部适配器实际耗时。新7B总费用上限约32.40元（12h×2.70）；9B新校准尚未获有效测量，因此正式评测完成日期**不能可靠承诺**。新协议若通过，才依据新实际延迟更新全量评测ETA。

## 8. 原ARP八项问题与工作量

以下为计划估算而非本轮实耗，详细来源见[锁定执行计划](route_b_execution_plan.md#原arp八项阻塞的工作量估算)。API费按实际输入/输出usage和控制台价计算；供应方等待无保证日期。

| 项 | 未完成内容/工程复用 | 工作量估算 | 费用与依赖 | 可并行性 |
|---|---|---|---|---|
| 1 | 本地7B新审计余2,200及正式配对C | GPU20–35分钟，分析1–2工程小时；若跨卡回放失败则另1–2GPU小时 | 约0.90–1.58元，重跑另2.70–5.40元；无API；完整有效测量是硬依赖 | CPU可并行，GPU在7B复测后排队 |
| 2 | 精确3B/7B 403、33,800同检查点桥接及商业余8,235 | 诊断0.5–2工程小时；桥接6–12小时、补审1.8–3小时；供应方等待未知 | API实价待核；混合阶梯硬依赖；必须每轴差异严格<0.005 | 权限与CPU准备可并行，成功后才桥接 |
| 3 | 商业qwen3-vl-plus不等于精确2.5参数节点 | 论域修订0.5–1工程小时，精确新节点部署待确认 | 改文案0新增推理费；精确节点是纯同系列阶梯硬依赖 | 论域修订可并行 |
| 4 | 双人独立定位盲审 | 导出1–2小时；两模型共6.7–13.3标注人时、裁决1–2小时 | 已有输出无新增GPU/API费；需真实独立标注者 | 可与GPU并行；本次自动审计不替代双人审查 |
| 5 | 原C选轴及共同暴露预算 | 有效C后1–2工程小时 | CPU为主；有效C和预定不同选轴是硬依赖 | 清单工具可先准备，科学判定不能提前 |
| 6 | 原7B←122B目标契约及5条隔离数据 | 复核/重新分词0.5–2工程小时，验收10–30分钟GPU | 约0.45–1.35元；3,540回答可复用不重付；5条处理依赖供应方或另注册可用池 | CPU并行，GPU排队；不得复用9B token IDs当7B输入 |
| 7 | 原ARP执行器接入与新目标端到端验收 | 1–3工程小时后小循环 | 多臂/多seed/恢复工程可复用；新配置GPU费待预检 | CPU可并行；不代表原主线完成 |
| 8 | 资源/评测 | 5090替代4090，旧7B原轮16/16已完成；新定位正在复测 | 旧纯生成15.803h/42.67元；新原ARP耗时需重新估计 | 资源阻塞解除，科学前置未解除 |

## 9. 接手后的验收顺序

1. 查看真实行数、进程、两个分区空间、停止记录及绝对预算；保护正在运行的7B。
2. 每完成一个7B模型立即核对2,400唯一ID、提示/网格/单位契约、原始输出重评分和Release SHA。
3. 全16模型完成后生成新4项区间与图表，明确U−N优势是否仍有证据；保留旧12项结果和全部测量修订。
4. 核实7B最终完成和交付标记后更新结果报告；9B仍未放行则保持该事实，不能为了接力空跑而绕过门槛。
5. 为9B另立可审阅的原生前端协议；在固定新预算前不生成新校准预测。通过后再启动其16模型正式评测，不加载优化器/不保留梯度。
6. 最终9B四轴五seed及12项校正区间齐备后报告；原ARP另外补齐八项依赖、有效C和独立确认，不能顺势宣布ARP完成。

## 10. 交接证据与复现

[快照JSON](handoff-evidence-20261017/handoff-live-snapshot.json)、[7B首100校准](handoff-evidence-20261017/7b-calibration-proof.json)、[CSV修订](handoff-evidence-20261017/7b-csv-schema-amendment.json)、[绝对预算](handoff-evidence-20261017/7b-active-gpu-budget.json)、[9B失败证据](handoff-evidence-20261017/9b-failed-processor-calibration.json)、[9B训练总表](9b-training-final/training-results.json)、[恢复可用性](9b-training-final/recovery-availability.json)。

文档生成器`tools/write_large_model_results_handoff.py`只读取已保存的小型证据，不运行GPU/API、不改变实验状态。后续更新须先刷新证据快照，再运行生成器，避免只修改叙述而不更新证据。
'''
    for name, content in [('大模型方案_实验结果报告_20261017.md', report),
                          ('大模型方案_实验交接文档_20261017.md', handoff)]:
        (DEST / name).write_text(content, encoding='utf-8', newline='\n')
        print(name)


if __name__ == '__main__':
    run()
