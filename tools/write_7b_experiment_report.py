"""Generate the user-requested7B report from verified final small artifacts."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/results/protocol-v2-20261016'
AXES = ('pope', 'grounding', 'consistency', 'mmstar')
SEEDS = (42, 137, 256, 512, 1024)


def read(relative):
    return json.loads((DOC / relative).read_text(encoding='utf-8'))


def generate():
    data = read('route-b-final/multiplicity-results.json')
    evaluation = read('route-b-final/evaluation-results.json')
    training = read('route-b-formal-training-results.json')
    minimal = read('route-b-minimal-report.json')
    delivery = read('route-b-final/delivery-verification.json')
    sampling = read('route-b-final/sampling-identity-verification.json')
    expected = {'N-seed42'} | {f'{a}-seed{s}' for s in SEEDS for a in ('A1', 'A2', 'U')}
    assert set(data['scores']) == expected and data['matched_seeds'] == list(SEEDS)
    assert data['bootstrap_replicates'] == 10000 and data['family_size'] == 12
    assert evaluation['completed_models'] == 16 and evaluation['processed_prompts'] == 270400
    assert len(training['runs']) == 15 and training['updates'] == 15000
    assert delivery['status'] == 'passed' and delivery['verified_files'] == delivery['required_files'] == 3738
    assert all(r['sampling_rows'] == r['unique_prompt_ids'] == 600 for r in sampling['runs'])
    for label, values in data['paired_contrasts'].items():
        left, right = label.split('-')
        for axis, value in values.items():
            delta = statistics.mean(data['scores'][f'{left}-seed{s}'][axis] -
                                    data['scores'][f'{right}-seed{42 if right == "N" else s}'][axis] for s in SEEDS)
            assert abs(delta - value['difference']) < 1e-12
    seconds_training = training['step_seconds_total']
    seconds_eval = sum(r['total_generation_seconds'] for r in evaluation['scores'].values())
    peak_training = max(r['reserved_peak_bytes'] for r in training['runs'])
    peak_eval = max(r['peak_reserved_bytes'] for r in evaluation['scores'].values())
    lines = [
        '# 独立路线B：Qwen2.5-VL-7B自蒸馏实验结果报告', '',
        '日期：2026年10月17日。状态：最小循环、正式训练、完整评测、终局统计及必要数据交付均已完成。', '',
        '> 测量补充：全16模型CPU审计发现原图/处理器像素尺度混杂；旧分数与12项区间在下文作为历史结果保留。2026-10-18 05:34，明确处理器像素提示的38,400条独立复测及图表已完成并归档。新定位五seed均值N/A1/A2/U为84.708%/85.592%/84.975%/82.275%；独立4项校正中仅A1高于U的区间排除0，A1相对N/A2未显著。旧U优势未保留，不能解释为纯能力提升。见[复测终局](7b-processor-final/report.md)、[完整更新分析](大模型方案_实验结果报告_20261017.md)、[原CPU诊断](7b-grounding-cpu-report.md)。', '',
        '## 摘要', '',
        '本实验在单张RTX 5090上，以相同Qwen2.5-VL-7B-Instruct检查点作为教师和学生，比较推理代理训练A1、一致性训练A2、四轴等比例混合训练U与不更新基线N。三个训练臂分别运行五个seed、每次固定1,000次真实更新；N与15个适配器各评测16,900提示，共270,400条输出。所有必要产物经独立内容核验，GitHub服务端确认3,738/3,738项文件覆盖。', '',
        '完整五seed、三组对比乘四轴共12项Bonferroni校正bootstrap区间中，只有定位轴A1−U区间排除0：A1平均低11.525个百分点，校正区间为[-16.157,-5.648]个百分点。A1−A2和A1−N四轴区间全部包含0，未给出推理代理单轴训练的可靠优势证据。实验完成了多臂训练与恢复流程验证，结果属于已有开发基准上的探索性分析，不能替代原ARP阶段C判定。', '',
        '## 1. 研究目的与实验边界', '',
        '研究问题是在学生检查点、优化器及更新预算固定时，不同训练轴覆盖如何影响四轴评测表现。A1沿用v1的mmstar轴定义，但使用ScienceQA训练代理，不能把训练集称为MMStar评测题。各臂由用户预先指定，不由本轮量尺审计或阶段C动态选择。', '',
        '本实验是独立路线B的序列级自蒸馏，不是原v1 top50-tail KL协议的数值复现，也不是122B教师蒸馏实验。它不验证量尺反转、由量尺导致的选轴因果效应、跨代API/本地阶梯可拼接性，或大教师相对小教师的收益。原ARP阶段C维持未放行。', '',
        '## 2. 数据与训练设置', '',
        '### 2.1 数据规模与计分单位', '',
        '| 轴 | 训练池提示 | 训练来源 | 每模型评测提示 | 评测原题 | 计分方式 |',
        '|---|---:|---|---:|---:|---|',
        '| pope | 600 | COCO训练图像正负物体问答 | 9,000 | 9,000 | 严格yes/no准确率 |',
        '| grounding | 600 | RefCOCO train | 2,400 | 2,400 | 原图像素框，IoU≥0.5 |',
        '| consistency | 1,745 | 未进入评测的MMBench原题改写 | 4,000 | 800 | 每原题五改写全部正确才计1 |',
        '| mmstar | 600 | ScienceQA train推理代理 | 1,500 | 1,500 | MMStar严格选项准确率 |',
        '| 合计 | 3,545 | 四轴训练池 | 16,900 | 13,700 | 四轴分别报告 |', '',
        '训练池引用2,132幅图像，完整评测引用4,705幅图像；数据清单及图像摘要见[数据交付核验](route-b-data-delivery-audit.md)。训练来源与评测来源分开构建，不能把训练行数、独立原题数和图像簇数混为一谈。评测使用既有开发基准，不声称是未使用过的独立确认集。', '',
        '### 2.2 四臂和固定暴露预算', '',
        '| 臂 | 更新数据 | 每次训练的提示清单 | 每次真实更新 |',
        '|---|---|---|---:|',
        '| A1 | mmstar轴的ScienceQA代理 | 600提示 | 1,000 |',
        '| A2 | consistency | 600提示 | 1,000 |',
        '| U | 四轴等比例 | 每轴150提示，共600 | 1,000，每轴250 |',
        '| N | 不更新 | 无训练清单/优化器/适配器 | 0 |', '',
        '五个seed固定为42、137、256、512、1024。实际采样先按seed打乱、截取提示清单，再循环暴露至1,000步；已额外核对[全部15次实际采样清单](route-b-final/sampling-identity-verification.json)，每次均为600个不同提示ID。一致性轴含相关改写，600提示不等于600个独立原题。旧协议说明中的replacement措辞不够准确，本报告以冻结数值源码和真实采样清单为准，不据该措辞虚构有放回采样。', '',
        '相同更新数也不等于相同token量、独立原题暴露或计算成本；四轴提示和回答长度不同，U覆盖定位训练而A1/A2不覆盖。', '',
        '### 2.3 优化配置与蒸馏目标', '',
        '| 项目 | 配置 |', '|---|---|',
        '| 教师 / 学生 | Qwen2.5-VL-7B-Instruct，相同固定检查点 |',
        '| 硬件与计算精度 | RTX 5090，BF16；训练和评测均在同卡 |',
        '| 可训练参数 | 语言层注意力q/k/v/o与MLP gate/up/down的LoRA，视觉模块冻结 |',
        '| LoRA | r=16，alpha=32，dropout=0 |',
        '| 优化器 | AdamW，lr=1e-4，weight_decay=0 |',
        '| batch / 累积 | micro-batch=1，gradient_accumulation=1 |',
        '| 其他 | gradient checkpointing，梯度范数裁剪1.0 |',
        '| 长度预算 | 总序列≤2,048，视觉token≤1,280，回答≤128；不静默截断 |',
        '| 保存与终点 | 每50步记录恢复状态；固定第1,000步，不根据评测选最佳检查点 |', '',
        '损失为 **L = 0.5×CE(真实答案) + 0.5×CE(教师回答)**。教师top50 logits虽已保存用于审计，当前KL权重为0，不能将本实验表述为logit-level KL蒸馏。全量3,545条新缓存保存原始生成ID、EOS及张量对齐证据，不以旧缓存未触及长度上限代替EOS证明。', '',
        '## 3. 最小循环与过程验收', '',
        '正式训练前，四轴各125提示、共500，seed42；A1/A2/U各100次更新，N不训练。四模型分别评测800提示、400原题。五项验收全部通过：', '',
        '| 验收项 | 真实证据 |', '|---|---|',
        '| 缓存加载、无截断 | 500条最小缓存可读；随后全量3,545条核对EOS、SHA、top50有限数值及目标对齐 |',
        '| 双CE有限、梯度非零 | 三臂各100个连续真实更新，逐步两路CE及梯度检查通过 |',
        '| 适配器保存/加载 | 导出张量与恢复状态一致，新评测进程从磁盘重载 |',
        '| 四轴评测输出 | 四模型逐题输出与汇总分数独立重算一致 |',
        '| 中断恢复 | A1第50步退出，新PID恢复优化器及CPU/CUDA随机状态，连续写入51–100步 |', '',
        f'恢复后第一步为第51步，损失为`{minimal["resume"]["first_resumed_loss"]:.12g}`；步数无缺失、无重复。详细证据见[最小循环报告](route-b-minimal-report.json)及[独立核验](route-b-minimal-independent-verification.json)。', '',
        '下表是每轴125提示的长度分布，格式为最小/最大/均值；完整分布还包括纯问题文本、视觉token及教师原始生成长度。', '',
        '| 轴 | 含视觉prompt | prompt+真实答案 | prompt+教师答案 |', '|---|---|---|---|',
    ]
    for axis in AXES:
        stats = minimal['token_distribution']['axes'][axis]
        values = []
        for key in ('prompt_including_visual', 'gt_total', 'teacher_total'):
            row = stats[key]
            assert row['n'] == 125
            values.append(f'{row["min"]}/{row["max"]}/{row["mean"]:.2f}')
        lines.append('| ' + axis + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## 4. 训练结果、显存和耗时', '',
              '15次训练均完成固定1,000次真实更新，共15,000次。逐步损失有限、梯度非零；优化器内部步数为1,000，适配器导出与恢复张量一致，LoRA B有真实非零更新。末步损失是最后一个样本的值，不是全数据平均验证损失，不应跨臂直接比较或用于选择检查点。', '',
              '| 臂 | seed | 秒/步 | 纯更新分钟 | reserved GB | 末步双CE |', '|---|---:|---:|---:|---:|---:|']
    for row in training['runs']:
        lines.append(f'| {row["arm"]} | {row["seed"]} | {row["seconds_per_step"]:.3f} | {row["training_step_seconds"]/60:.2f} | {row["reserved_peak_bytes"]/1e9:.3f} | {row["last_loss"]:.6g} |')
    lines += ['', '![15次训练损失曲线](route-b-formal-loss-curves.png)', '',
              '图为20步移动平均双CE，纵轴取对数；不同seed全部展示，没有只挑最好一次。训练损失降低仅说明拟合训练目标，不单独证明能力提升。', '',
              f'纯更新合计{seconds_training:,.3f}秒，即{seconds_training/3600:.3f}小时；训练reserved峰值最高{peak_training/1e9:.3f}GB（{peak_training/1024**3:.3f}GiB）。16模型纯生成合计{seconds_eval:,.3f}秒，即{seconds_eval/3600:.3f}小时；评测reserved峰值最高{peak_eval/1e9:.3f}GB（{peak_eval/1024**3:.3f}GiB）。reserved是PyTorch缓存分配器口径，不等于nvidia-smi整卡占用。', '',
              f'按用户指定2.70元/小时折算，纯更新约{seconds_training/3600*2.7:.2f}元，纯生成约{seconds_eval/3600*2.7:.2f}元，合计约{(seconds_training+seconds_eval)/3600*2.7:.2f}元。该数值不是账户总账单：未含教师缓存构建、模型装载、检查点I/O、上传、校验和其他开机等待。', '',
              '## 5. 全部16模型四轴分数', '',
              '表中均为百分比。N是唯一未训练基线，不虚构五个N重复；训练臂每个seed均从相同基座重新初始化，不接续前一臂适配器。每模型16,900提示、13,700原题，适配器独立重载，无优化器且关闭梯度。', '',
              '| 模型 | pope | grounding | consistency | mmstar |', '|---|---:|---:|---:|---:|']
    for name in ['N-seed42'] + [f'{a}-seed{s}' for s in SEEDS for a in ('A1', 'A2', 'U')]:
        lines.append('| ' + name + ' | ' + ' | '.join(f'{data["scores"][name][a]*100:.3f}%' for a in AXES) + ' |')
    lines += ['', '五seed均值±样本标准差如下；SD描述seed波动，不是置信区间。', '',
              '| 臂 | pope | grounding | consistency | mmstar |', '|---|---:|---:|---:|---:|']
    for arm in ('A1', 'A2', 'U'):
        values = [[100*data['scores'][f'{arm}-seed{s}'][a] for s in SEEDS] for a in AXES]
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{statistics.mean(v):.3f}±{statistics.stdev(v):.3f}' for v in values) + ' |')
    lines += ['', '![四轴分数及全部seed](route-b-final/four-axis-scores.png)', '',
              '彩色点为各seed，黑色菱形为臂均值，灰线为固定N基线。图表可导出[PDF](route-b-final/four-axis-scores.pdf)。', '',
              '## 6. 五seed配对区间与多重比较', '',
              '比较族固定为A1−A2、A1−U、A1−N乘四轴，共12项。使用五个训练seed配对重采样，并按来源图像簇重采样评测单位；一致性五改写先按原题聚合。bootstrap重复10,000次，随机seed为2026101702。表中区间和差值均以百分点表示。', '',
              '每个Bonferroni区间对应名义99.5833%覆盖，以该方法给出12项名义近似95%整体覆盖；保留逐项95%区间用于展示校正前后的区别，不从中挑更有利者作为主结论。', '',
              '| 对比 | 轴 | 平均差值 | 逐项95%区间 | 12项校正区间 | 校正区间结论 |',
              '|---|---|---:|---|---|---|']
    for label in ('A1-A2', 'A1-U', 'A1-N'):
        for axis in AXES:
            row = data['paired_contrasts'][label][axis]
            lo, hi = row['pointwise_ci95']
            low, high = row['bonferroni_familywise_ci95']
            conclusion = '左臂更高' if low > 0 else '右臂更高' if high < 0 else '包含0，证据不足'
            lines.append(f'| {label} | {axis} | {100*row["difference"]:.3f} | [{100*lo:.3f}, {100*hi:.3f}] | [{100*low:.3f}, {100*high:.3f}] | {conclusion} |')
    lines += ['', '![12项校正配对区间](route-b-final/paired-intervals.png)', '',
              '红色表示右臂更高且校正区间排除0，灰色表示包含0；可导出[PDF](route-b-final/paired-intervals.pdf)。原始数值见[multiplicity-results.json](route-b-final/multiplicity-results.json)。', '',
              '## 7. 结果分析', '',
              '**A1没有显示相对A2的稳定优势。** A1−A2在pope、定位、一致性、推理上的均值差分别为+0.331、−1.325、+0.100、+0.227个百分点，但四个校正区间均包含0。不能仅用点估计正负宣布某单轴训练策略胜出，也不能将区间包含0解释成两种训练严格等价。', '',
              '**混合训练相对A1的定位收益最清晰。** U的定位均值为77.658%，A1为66.133%；A1−U为−11.525个百分点，校正区间[-16.157,-5.648]。该结果与U直接接触定位训练、A1只接触推理代理的训练覆盖差异一致，但不是机制的独立识别。U相对N的定位点估计高9.867个百分点，仅为描述性对照；U−N未进入本次12项检验族，不能据现有表格另报它已通过校正检验。', '',
              '**推理代理训练没有给出相对N的明确MMStar收益。** A1的MMStar均值61.840%，N为62.933%；差值−1.093个百分点，校正区间[-4.084,2.002]。当前数据不支持A1改善MMStar的结论，也不足以认定其必然造成负迁移。ScienceQA到MMStar存在训练代理与目标基准差异，相关性不是机制证明。', '',
              '**一致性点估计上升，仍不足以确认收益。** A1−N一致性为+1.200个百分点，校正区间[-1.052,3.646]；A1−U为+1.550个百分点，校正区间[-0.858,4.252]。混合训练的定位优势不意味着所有轴均占优，四轴结果需要分别呈现。', '',
              '**seed波动必须保留。** U的定位在五seed间为69.792%–83.125%，说明单次最好结果不能代表训练分布；末步低损失也不保证最好开发集表现。应依据全部五seed及配对不确定性解释结果。', '',
              '## 8. 测量校准与统计局限', '',
              '本轮7B定位协议明确使用原图像素坐标，严格检查边界并按IoU≥0.5计分；原始生成逐题重评分通过。9B后续发现的0–1000原生坐标差异不回写本轮7B结果。扩展到其他模型时，应独立固定输出坐标契约、转换规则和计分口径，不通过按答案选坐标系获得更高分。', '',
              '本报告有四项主要推断边界：第一，使用已有开发基准，不能将结果称为独立确认；第二，训练轴并非阶段C选择，不能推导ARP选轴有效；第三，只有五个训练seed，bootstrap尾部分位数存在Monte Carlo及小样本不确定性，Bonferroni只调整比较族，不能保证近似区间具有精确有限样本覆盖；第四，多重比较报告是在部分开发结果出现后按用户要求增加，属于事后报告修订，不是事前确认性检验。', '',
              '因此“12项中1项区间排除0”是本方法下的探索性差异结论，不等同于小样本随机化检验已经通过，也不构成普遍训练优势的确认。未做等效性检验，不虚构p值。评测图像簇/原题重采样不能替代增加独立训练seed。', '',
              '## 9. 完整性、复现与数据位置', '',
              '独立核验覆盖全量EOS缓存、15次真实优化器更新/导出/恢复状态、16模型逐题行数与唯一ID、原始生成重新解析与四轴汇总、全部五seed差值及12项区间。原有80个Python源文件与对照版本一致，只新增独立实现；原ARP门槛未改。GitHub资产按实际大小和服务端SHA256复核，不以成功退出、分数标记或上传收据单独证明完成。', '',
              '| 产物 | 位置 |', '|---|---|',
              '| 完整终局数值与区间 | [统计JSON](route-b-final/multiplicity-results.json)、[评测耗时/显存JSON](route-b-final/evaluation-results.json) |',
              '| 15次训练曲线与逐次表 | [训练报告](route-b-formal-training-report.md)、[训练JSON](route-b-formal-training-results.json) |',
              '| 最小循环、EOS、输入预检 | [最小循环](route-b-minimal-report.json)、[全缓存验收](route-b-full-eos-cache-independent-verification.json)、[全量评测输入](route-b-eval-input-preflight.json) |',
              '| 内容与交付证据 | [内容验收](route-b-final/content-verification.json)、[3,738文件交付](route-b-final/delivery-verification.json)、[最终图表交付](route-b-final/delivered.json) |',
              '| 必要过程数据索引 | [配置/日志/缓存/恢复/原始输出索引](route_b_process_artifacts.md) |',
              '| 原始训练与评测大文件 | [本轮GitHub Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-protocol-v2-20261016) |',
              '| 原始图像与数据清单 | [数据交付核验](route-b-data-delivery-audit.md)、[数据Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-formal-20261014) |', '',
              '基座权重按官方revision和SHA固定；大文件由AutoDL直接上传GitHub，不要求在用户Windows下载。增量归档的历史二进制通过清单引用保留，恢复时应按清单定位所有相关批次，不能只取最后一个压缩包。', '',
              '关键SHA256：', '',
              '| 对象 | SHA256 |', '|---|---|',
              '| 冻结数值执行器 | `bb8fb654d254912abaf291b59aeec7f301aaa5437930cfc9e9080cce79de2c30` |',
              '| 路线B协议 | `98b37d61846f4401505107b8b0240c7986dcd484a3ef364c9f901d000690a265` |',
              '| 正式冻结输入配置 | `cd3ad2b931959814b6324b45c484a48e92b1348a699a891d0e043e0473723925` |',
              '| EOS缓存清单 | `0b1547d1e0605da22fce77c0921312d6b391f26bee58bfb3e9e5268ce0321859` |',
              '| 全量评测清单 | `e92d3b13f95e457024b274d0077e917d4fbfc64e1af1221fd75ff30e88506746` |', '',
              '## 10. 结论', '',
              '7B独立路线B的原定流程及随后独立定位复测均已完成并具备可复核产物。实验验证了教师缓存、双CE多臂训练、跨进程恢复、适配器重载和完整评测链路。旧定位显著性受测量契约混杂；新明确提示/共同单位复测未保留旧U优势，独立4项校正中A1高于U，但未确认A1相对A2或N的优势。其他三轴结果保留，新旧统计族分开解释。原ARP科学门槛仍未放行，后续9B+122B的代际、容量、教师和测量变化也不能单独归因为模型大小。', '']
    target = DOC / '7B独立自蒸馏实验结果报告.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    proof = {'status': 'generated_from_verified_final_artifacts', 'report': target.name,
             'report_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
             'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'source_sha256': {name: hashlib.sha256((DOC / name).read_bytes()).hexdigest() for name in (
                 'route-b-final/multiplicity-results.json', 'route-b-final/evaluation-results.json',
                 'route-b-formal-training-results.json', 'route-b-minimal-report.json',
                 'route-b-final/delivery-verification.json', 'route-b-final/sampling-identity-verification.json')},
             'no_new_training_or_inference': True,
             'measurement_amendment': {'previous_report_sha256':'d39e0a259095611165bf4e1c33363e73b3f938779d0468e839b94d6bfaded01e',
                 'old_numeric_artifacts_unchanged':True,'grounding_ability_interpretation_pending':False,
                 'new_explicit_contract_results_sha256':hashlib.sha256((DOC/'7b-processor-final/results.json').read_bytes()).hexdigest(),
                 'posthoc_remeasurement_not_confirmatory':True,
                 'cpu_audit_sha256':hashlib.sha256((DOC/'7b-grounding-cpu-audit.json').read_bytes()).hexdigest()}}
    (DOC / '7b-report-provenance.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'report': str(target), 'bytes': target.stat().st_size, 'sha256': proof['report_sha256']}))


if __name__ == '__main__':
    generate()
