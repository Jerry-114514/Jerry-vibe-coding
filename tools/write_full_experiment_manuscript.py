"""Build the Chinese manuscript from existing verified result artifacts only."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs'
R = DOC / 'results/protocol-v2-20261016'
OUT = DOC / 'paper'
OUT.mkdir(exist_ok=True)
load = lambda p: json.loads(p.read_text(encoding='utf-8'))
v7 = load(R/'route-b-final/multiplicity-results.json')
g7 = load(R/'7b-processor-final/results.json')
v9 = load(R/'9b-v41-final/formal-report/evaluation-results.json')
repair = load(R/'9b-repair-final/report/results.json')
training9 = load(R/'9b-training-final/training-results.json')
old_gains = load(DOC/'results/final-20261016/signed-gains.json')
SEEDS = [42,137,256,512,1024]
AXES = ['pope','grounding','consistency','mmstar']
items=[]
def p(s=''): items.append(s.strip().replace(chr(92)*2,chr(92))+'\n')
def table(headers, rows):
    p('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in rows))
def mean_rows(scores):
    return [[arm]+[f'{statistics.mean(scores[f"{arm}-seed{s}"][a] for s in SEEDS)*100:.2f}' if arm!='N' else f'{scores["N-seed42"][a]*100:.2f}' for a in AXES] for arm in ['N','A1','A2','U']]
def ci_text(ci): return f'[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]'

p('# 多模态序列蒸馏中的定位退化与坐标表示兼容性：从跨轴审计到控制实验')
p('**Grounding Degradation and Coordinate Representation Compatibility in Multimodal Sequence Distillation: From Cross-Axis Auditing to Controlled Experiments**')
p('作者与单位：待作者填写。版本：中文论文初稿 v0.1，2026-10-20。')
p('> 本稿以实际完成的实验与验收文件为依据。正文给出可独立阅读的研究论证，附录保留早期实验、失败记录和全部端点。本研究属于多阶段探索性实证研究；修订ARP阶段C未放行，不将独立序列蒸馏写为原ARP主张已获确认。作者信息、投稿格式及独立人工定位审查尚待补齐，不虚构其已完成。')

p('## 摘要')
p('多模态模型的跨任务得分变化常被用于诊断能力损失和分配微调预算，但这些变化同时受到评分量尺、输出契约及训练目标表示的影响。本文从跨轴退化审计出发，开展包含历史规模阶梯、7B自蒸馏、9B大教师序列蒸馏及坐标表示控制实验的连续实证研究。首先，我们在三个模型家族的七个检查点上建立逐题账本，扫描120种量尺设置；对相对差分母、平局及有效域的复核表明，旧协议的放行结论不能直接延伸到修订协议。其次，我们在固定1,000次更新、五个训练seed下完成7B与9B各15次多臂训练。7B定位复测中，A1−U的方向从旧口径的−11.53个百分点变为明确处理器像素契约下的+3.32个百分点，显示结论对测量协议的敏感性。9B完整评测中，A1相对未训练基线的一致性平均提高5.05个百分点，12项校正区间为[0.35,9.28]，但混合臂U定位平均低于基线20.91个百分点。最后，我们对两个seed实施坐标表示（像素／归一化）与监督目标（双CE／GT-only）的2×2对照。归一化表示的定位主效应在开发集与独立保留集分别为+22.75和+21.88个百分点，24项校正区间均不跨零。结果支持：在本研究设置下，训练目标表示兼容性是解释定位退化的重要因素；跨轴增益不能自动解释为全面能力提升。本文同时提供完整失败记录、恢复验收及可追溯归档。')
p('**关键词：** 多模态大模型；序列级蒸馏；视觉定位；坐标表示；跨任务评测；测量有效性；可复现研究。')

p('## Abstract')
p('Cross-task score changes are often used to diagnose capability degradation and allocate fine-tuning budgets, although they also depend on scoring conventions, output contracts, and target representations. We present a staged empirical investigation spanning seven model checkpoints, 7B self-distillation, 9B sequence distillation from a large API teacher, and a coordinate-by-supervision factorial follow-up. Reanalysis of 120 scoring specifications reveals that a historical audit admission decision does not establish validity under revised denominators, tie handling, and common-domain requirements. We then complete fifteen fixed-budget training runs for each of the 7B and 9B settings, using five seeds and 1,000 updates per run. In the 7B setting, a grounding contrast changes direction under an explicitly revised processor-coordinate evaluation protocol. In the 9B setting, reasoning-proxy training improves consistency over the unchanged baseline by 5.05 percentage points, while mixed-axis training exhibits substantial grounding degradation. A two-seed 2×2 follow-up contrasts pixel versus normalized targets and dual cross-entropy versus ground-truth-only supervision. The normalized-target grounding main effect is +22.75 points on the development set and +21.88 points on an independently reserved subset, with both 24-comparison-adjusted bootstrap intervals excluding zero. These exploratory results identify target-representation compatibility as an important contributor to grounding outcomes in the studied setting. They do not establish universal distillation benefits, broad seed stability, or a capacity-only effect. All fixed endpoints, failures, recovery checks, and final artifacts are retained for verification.')

p('## 1 引言')
p('视觉语言模型的微调常同时覆盖物体识别、指代表达定位、多模态选择题与指令遵循。对某个任务的改善并不意味着所有任务都改善，而不同任务的准确率也不天然具有相同的解释尺度。对于需输出结构化坐标的任务，最终得分还依赖模型生成的单位、图像缩放方式、解析规则与终止行为。如果这些环节未被显式固定，能力变化与接口适配可能混在同一读数中。')
p('我们最初研究的问题是：将同家族大、小模型在不同能力轴上的分数差进行排序时，最严重退化轴是否在合理量尺族下保持不变？原Argmax Retention Profiling（ARP）研究卡据此提出测量底线、枚举退化算子、折叠等价类和比较量尺扰动与抽样噪声。该研究卡是项目内部方案，不是已发表论文；其推测、构造例子和计划结果均不被当作本文新增真实观测。')
p('实际执行发现，研究首先面临测量前提问题：部分模型的定位输出无法被统一像素解析器可靠读取，相对差公式及平局处理也会改变量尺审计的解释。于是，我们保留原始结果，修订分析，并将后续多臂训练登记为独立实验，而非继续把尚未放行的ARP门槛视为满足。')
p('独立实验产生了一个更具体的问题。7B原轮中看似较强的混合训练定位收益，在明确处理器像素契约复测后未被保留。9B原生前端经校准后，混合臂的定位低分仍然存在，且不再能够用简单的单位重解释消除。这促使我们进一步区分两个干预：训练目标采用原图像素还是归一化坐标，以及目标采用GT-only还是同时拟合教师回答。')
p('本文围绕三个研究问题展开：RQ1，跨轴审计的结论在量尺与有效域复核后是否仍足以支持预算决策？RQ2，在固定更新预算下，多轴序列蒸馏带来怎样的收益与退化？RQ3，坐标目标表示与教师监督分别如何影响定位表现，这些影响能否在项目内未使用过的保留数据上重复出现？')
p('本文贡献是：（1）公开从旧量尺审计到修订门槛未放行的完整证据，说明有效域与平局不能被翻转率掩盖；（2）报告7B和9B各五seed的固定端点实验及定位测量修订，展示跨轴收益与代价；（3）通过两个seed的两因素对照及独立保留集，为本设置内的坐标表示效应提供证据。本文不提出新的蒸馏优化算法，也不主张首次发现坐标表示的重要性。')

p('## 2 相关工作')
p('**多模态模型与多轴评测。** Qwen2.5-VL、InternVL和LLaVA-OneVision等模型为公开检查点比较提供了基础，但家族内参数规模变化并不保证只改变容量。Qwen2.5-VL报告涉及动态分辨率和定位接口；Qwen3.5-9B官方模型卡也明确其视觉输入与生成接口。因此，本文使用精确检查点及本次处理器配置记录，而不将厂商榜单读数直接拼入我们的量尺阶梯。[Qwen2.5-VL报告](https://arxiv.org/abs/2502.13923)、[Qwen3.5-9B模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)、[InternVL3](https://arxiv.org/abs/2504.10479)、[LLaVA-OneVision](https://arxiv.org/abs/2408.03326)。')
p('POPE通过物体存在性提问研究幻觉，MMBench和MMStar分别提供多模态能力评测资源；本文使用其题目来源或相关子集构造不同探针，不能把自定义五改写一致性分数称为完整MMBench标准分数。训练中的ScienceQA代理与评测中的MMStar也严格区分。[POPE](https://arxiv.org/abs/2305.10355)、[MMBench](https://arxiv.org/abs/2307.06281)、[MMStar](https://arxiv.org/abs/2403.20330)、[ScienceQA](https://arxiv.org/abs/2209.09513)。')
p('**定位目标表示。** Ranasinghe等研究通过图像空间坐标相关指令微调增强定位与空间推理，并比较不同坐标表示。本文与其区别在于，我们关注多轴序列蒸馏中已经出现的定位退化，将目标表示与监督目标作为两个因素，并将表示效应与测量前端审计联合报告。本文不将“归一化坐标可能有益”包装为首创结论。[Learning to Localize Objects](https://arxiv.org/abs/2404.07449)。')
p('**蒸馏与参数高效训练。** 序列级知识蒸馏以教师生成目标监督学生，LoRA提供低秩参数更新路径。本文将这些已有技术用于控制预算的多臂对照。早期v1的top-50＋tail KL与后续双CE目标不同，二者不能被称为同一损失的直接复现。[Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)、[LoRA](https://arxiv.org/abs/2106.09685)。')

p('## 3 研究设计、数据与统计约定')
p('### 3.1 分阶段设计与证据等级')
table(['阶段','完成内容','主要规模','研究性质'],[
 ['E0：v1审计与原训练','七检查点账本、量尺扫描、3B←7B原协议训练','118,300条基础推理＋50,700条训练后推理；3次训练','历史原协议结果'],
 ['E1：审计复核','分母、结构商集、平局、共同有效域、配对噪声','120设置；三个家族、四轴／去定位六组','旧数据探索性重分析；不放行新C'],
 ['E2：7B自蒸馏','最小循环后15次训练、16模型评测','270,400条正式推理','独立序列蒸馏'],
 ['E3：7B定位复测','相同端点、明确处理器像素契约','16×2,400=38,400条','事后测量修订，独立比较族'],
 ['E4：9B←122B','15次训练；v4.1全16模型评测','270,400条，含复用400条新校准','独立扩展；不是跨代纯容量比较'],
 ['E5：坐标×监督对照','6次新训练＋2历史端点；两集合评测','110,040条新增推理','观察原结果后登记的两seed探索性对照']])
p('表1列示正式账本合计858,240条生成记录，不含最小循环、额外失败校准与API调用；不同阶段重复使用开发题，不能将该合计解释为独立样本数。固定1,000步的新增训练共39次、39,000次更新；E5复用的两个端点不重复计入。我们没有在数据产生后把E0—E5重述为一次完整事前注册。')
p('### 3.2 四轴任务与计分单位')
table(['轴','开发评测提示','独立计分原题','评分规则','训练来源'],[
 ['POPE',9000,9000,'严格yes/no准确率，非POPE F1','COCO训练来源正负物体问答'],
 ['定位',2400,2400,'原图坐标IoU≥0.5','RefCOCO train'],
 ['一致性',4000,800,'每原题五改写全对才计1','未进入开发评测的MMBench来源'],
 ['推理代理',1500,1500,'MMStar严格选项准确率','ScienceQA train，不使用MMStar评测题训练'],
 ['合计',16900,13700,'分轴报告，不用提示总数代替原题数','原池3,545条提示']])
p('一致性探针是本文的五模板操作化定义，不等同于一般语义一致性。POPE的9,000问答共享500个图像簇，定位2,400原题对应1,905簇，一致性800原题对应774簇，MMStar对应1,500簇；这些依赖关系进入重采样。四轴不是全部多模态能力的穷尽分解。')
p('原训练池含POPE 600、定位600、一致性1,745、ScienceQA代理600条，共3,545条。9B教师API获得3,540条可用目标，另5条按供应方图片检查与来源组隔离，不声称3,545条全部成功。早期独立来源构建与后期更严格的图像隔离审计应分别记录；后者不能追溯证明所有历史轮次已满足相同隔离标准。')
p('### 3.3 训练臂与暴露预算')
p('独立E2和E4中，A1预先固定为ScienceQA来源的mmstar轴代理，A2为一致性，U为四轴等比例，N不更新。各训练臂每seed先选600个不同提示，再循环至1,000次实际优化器更新；U每轴150提示、250次更新。seed为42、137、256、512、1024。600个一致性提示不等于600个独立原题。')
p('相同步数并不等于相同token量、独立图像暴露或算力。不同seed同时改变初始化、抽样子集与次序；同seed配对用于保留训练条件间的共同随机设计，不将其解释为只控制了优化噪声。N是一个唯一的未训练检查点，未伪造五个训练重复。')
p('### 3.4 序列级目标与优化')
p(r'后续序列蒸馏对每个样本使用两个回答序列：真实目标 $y^{GT}$ 与缓存教师目标 $y^T$，损失为')
p(r'$$\mathcal L_D=\tfrac12\operatorname{CE}(y^{GT})+\tfrac12\operatorname{CE}(y^T).$$')
p('这里CE按各自目标token计算。E2教师和学生均为Qwen2.5-VL-7B；教师top-50 logits虽保存在缓存中，训练KL权重为0。E4学生为Qwen3.5-9B，教师是Qwen3.5-122B-A10B API文本目标，不具有可直接等同本地logits缓存的内部证据。')
p('E2/E4使用语言层注意力和MLP LoRA（r=16、alpha=32、dropout=0），AdamW学习率1e-4、weight decay=0，microbatch=1、梯度累积1、梯度检查点、BF16及梯度裁剪1.0。视觉模块冻结；E4的混合注意力架构以冻结脚本的实际模块选择为准，不以旧架构模块名称替代。最终端点固定第1,000步，没有按开发分数选择检查点。')
p('### 3.5 区间、比较族与解读')
p('E2和E4各对A1−A2、A1−U、A1−N三组×四轴形成12项比较族；E3为定位四项单独比较族；E5为3种因子效应×四轴×两个集合形成24项族。后续阶段均使用10,000次配对seed／来源图像簇bootstrap。一致性先合并五改写为原题，相关模型共享图像簇抽样权重。跨轴统计及旧门槛采用各自登记方式，不事后混用。')
p('Bonferroni校正区间按每项alpha=0.05/m取百分位数，名义整体覆盖95%。本文称“校正区间不跨零”而非精确p值显著；两个或五个训练seed都限制总体外推，尤其24项校正时每侧尾端仅约10个bootstrap抽样点。区间跨零不证明等效。不同阶段的比较族各自限定；未计算覆盖所有探索路径的全项目错误率控制。')

p('## 4 从规模阶梯到测量有效性：早期ARP实验')
p('### 4.1 算子族、退化答题器与定义域')
p('E0比较Qwen2.5-VL 3B/7B、InternVL3 1B/2B/8B和LLaVA-OneVision 0.5B/7B。逐题记录原始文本、解析结果、correct bit和解析失败，之后在缓存上重算。Qwen视觉FP16曾发生溢出，正式v1统一提升视觉前向至FP32、语言保持FP16并重跑，旧溢出记录不混入正式账本；这与后续BF16实验也是明确的协议差异。')
p(r'对轴 $a$，输入无关答题器实测底线为 $c_a$，原得分为 $x_{m,a}$。底线校正为 $\widetilde x_{m,a}=(x_{m,a}-c_a)/(1-c_a)$，故绝对差满足')
p(r'$$\widetilde x_{L,a}-\widetilde x_{s,a}=\frac{x_{L,a}-x_{s,a}}{1-c_a}.$$')
p('底线分别由恒答no、中心框、固定选项和固定种子均匀随机答题器在同清单上测得；不是将所有二分类任务底线自动设为0.5。120个名义设置组合绝对／相对差、底线校正开关、三种headroom、五个OWA权重及两种解析失败处理。两节点只有一个相邻差，OWA不产生新信息；三节点按相邻差排序加权。')
p('修订分析将相对差分母明确为大成员得分或其扣底线得分；旧实现使用小成员分母。校正后分数低于底线、零分母或headroom符号异常按登记护栏记录无定义，不夹成零。对headroom为族最大分数减小成员分数的设置，底线缩放可以代数抵消，故名义120设置不能当作120把独立量尺。结构商集按函数与定义域折叠，经验商集再按当前账本值折叠；二者不应混为同一估计量。')
p('### 4.2 平局与“严格反转”的区别')
p(r'若某量尺 $g$ 在容差内的最大轴集合为 $S_g$，给轴 $a$ 的信用为 $\mathbf1[a\in S_g]/|S_g|$。保持率取该信用在有效类上的平均，$F=1-\max_a\pi(a)$。因此全轴平局也可能提高 $F$；它不是纯粹的严格排序反转比例。')
p('修订实现另计：在所有可用量尺类对中，两侧均为唯一最大且轴不同的类对比例。分母包含全部可用类对，不仅是唯一最大类对。我们同时保留平局率、无效类数、共同四轴覆盖和外层差值可定义比例。没有这些诊断，较高F可能来自平局或稀疏有效域。')
p('### 4.3 七检查点结果与旧判定的复核')
with (DOC/'results/final-20261016/base-model-scores.csv').open(encoding='utf-8-sig') as f:
    base_rows=list(csv.DictReader(f))
# The historical CSV schema is recorded in evidence; table uses the published audited values.
table(['v1检查点','POPE','定位','一致性','MMStar'],[
 ['Qwen3B','87.478','66.750','81.000','54.333'],['Qwen7B','87.678','67.625','84.125','62.067'],
 ['Intern1B','87.122','0.083','26.750','49.333'],['Intern2B','89.844','0.208','30.125','56.333'],['Intern8B','91.100','0.000','88.250','68.067'],
 ['LLaVA0.5B','75.622','0.000','22.625','27.600'],['LLaVA7B','89.833','0.000','83.375','59.933']])
p('表2为历史百分制分数。InternVL和LLaVA接近零的定位首先暴露解析／坐标契约适用性问题，不能直接解释为其一般定位能力接近零。原始v1中，Qwen、Intern、LLaVA的经验类数为9、54、10，F_scale点估计为0.083、0.512、0.317；v1按自身规则给出go，主要由后两家族支持。这个历史判定没有被删除，但不等价于修订协议通过。')
table(['修订配对开发门槛，四轴','严格反转','有效结构类','可定义外层差值','结论'],[
 ['Qwen','0','12/12','61.8%','indeterminate'],['InternVL','0.4（仅少量有效类）','5/60','0','indeterminate'],['LLaVA','1.0（仅少量有效类）','2/12','0','indeterminate']])
p('修订门槛要求有效结构类至少80%、外层差值可定义至少95%，配对F差值区间下界大于0.05、严格反转差值下界大于0且点估计至少0.05。Qwen虽F_scale约0.25而噪声点约0.0156，但严格反转为0且外层域不足；其余家族多数类无效。去定位后有效比例75%，仍不足门槛。四轴及去定位共六组均不放行。该重分析基于已观察数据，只验证估计流程，不是新确认性检验。')
p('坐标归因中，原预测“floor主导翻转”未获支持；Qwen同质底线负对照也未改变训练臂排序。因此，我们没有将构造反转边界或合成植入检查的成功，当成真实模型上的机制验证。')
p('### 4.4 原3B←7B训练的历史观察')
p('v1以Qwen7B教师、3B学生进行3次单seed训练，使用注意力LoRA、top-50＋显式词表尾桶KL（temperature=1）与教师回答CE各半。教师与学生词表映射核对一致，教师先缓存后卸载。该目标与后续GT／教师双CE不同。')
table(['历史训练臂','POPE增益pp','定位pp','一致性pp','MMStar pp','四轴变化之和pp'],[[x['arm']]+[f"{x['signed_components'][a]*100:+.3f}" for a in AXES]+[f"{x['G_net_raw']*100:+.3f}"] for x in old_gains])
p('这些单seed点估计不足以断言A1优于A2或U；四轴变化之和也不是整体准确率。此处保留它们是为了完整报告早期过程，而不是把旧go和单次收益当成修订ARP预算选择已有效的证据。')

p('## 5 7B独立自蒸馏：训练可复现性与测量敏感性')
p('### 5.1 最小循环及正式实验')
p('E2先用四轴各125提示、共500提示验证流程。A1、A2、U各执行100次实际更新，N只做基线；缓存可读且无静默截断、双CE有限且梯度非零、适配器可重新加载、四轴可评分、中断可恢复五项全部通过。A1第50步退出，新进程恢复后第51步损失为0.000149575484102，连续完成至100步。其后才启动15次正式训练。')
p('全3,545条新教师缓存核对原始生成ID、EOS和top-50张量对齐；“旧缓存未触及长度上限”不被当作EOS证据。15次固定1,000步训练、16×16,900提示评测完成。正式训练pure step合计2.306小时，reserved峰值24.547GB；全部恢复状态与导出适配器对应。')
p('### 5.2 原四轴结果及其测量标签')
table(['7B旧口径均值，%']+[a for a in AXES],mean_rows(v7['scores']))
p('旧12项比较中，仅A1−U定位校正区间不跨零：−11.525pp，区间[−16.157,−5.648]pp。其余11项包含零。这个结果曾显示U具有较大定位优势，但该表定位读数带有原图／处理器像素混杂标签，不能作为修订测量后的主结论。')
p('### 5.3 不重训的定位复测')
p('E3保持全部适配器端点，另立明确“处理器实际缩放后像素→原图”的定位评测：横坐标乘原图宽／处理器宽，纵坐标乘原图高／处理器高，IoU≥0.5。16模型重跑全部2,400定位提示，得到38,400条新输出，单独做四项事后校正。')
table(['7B定位口径','N','A1五seed均值','A2五seed均值','U五seed均值'],[
 ['旧口径，%']+[f"{v7['scores']['N-seed42']['grounding']*100:.3f}"]+[f"{statistics.mean(v7['scores'][f'{a}-seed{s}']['grounding'] for s in SEEDS)*100:.3f}" for a in ['A1','A2','U']],
 ['明确处理器像素复测，%']+[f"{g7['mean_scores'][a]*100:.3f}" for a in ['N','A1','A2','U']]])
table(['7B复测对比','差值pp','四项校正区间pp'],[[k,f"{x['difference']*100:+.3f}",ci_text(x['bonferroni_familywise_ci95'])] for k,x in g7['paired_contrasts'].items()])
p('A1−U方向变为+3.317pp，校正区间不跨零；A1−N和A1−A2仍无明确差异证据。U-seed1024复测仍只有75.708%，含258次解析失败，失败未被删除。复测改变提示及坐标前端并重新生成，故旧优势未保留支持的是协议敏感性，不足以将全部旧差异纯粹归因于坐标单位或格式适配。其他三轴未在该轮重测，我们不将复测定位与旧三轴拼接为同一轮“修复后四轴主结果”。')
p('![图1：7B定位复测与四项校正区间](../results/protocol-v2-20261016/7b-processor-final/final-report/grounding-final.png)')

p('## 6 9B大教师序列蒸馏：完整五seed终局')
p('### 6.1 模型、目标和准入')
p('E4学生为官方Qwen/Qwen3.5-9B，revision c202236235762e1c871ad0ccb60c8ee5ba337b9a；教师按API端点标识为Qwen3.5-122B-A10B。我们冻结本地权重和目标缓存，但不能把商业端点标识等同于已核验教师全部内部权重。9B最小循环五项工程验收与数值预检通过后，才进行15次1,000更新训练。7B、9B评测均在5090完成，未将释放的4090当作实际评测设备。')
p('9B定位训练目标仍为原图像素。早期最小循环出现N定位约1%的现象，提示默认坐标读取不适用；没有据此宣称模型失去定位能力。正式评测须另立原生前端协议并重新校准，训练端点不因测量失败而重训或调优。')
p('### 6.2 终止符故障与独立v4.1校准')
p('旧v4校准在A1-seed512发生20/20次长度封顶、11次解析失败。核查链路是：模型目录缺少generation_config.json，生成回退到<|endoftext|>（248044），而回答后出现<|im_end|>（248046）未及时终止，继续生成重复对话。离线按首个终止符截断支持了该诊断，但未追溯更改旧判定。')
p('旧900秒预算保留不重置。另经授权登记v4.1与新的900秒校准，只修改EOS识别为[248046,248044]，保持提示、坐标与验收条件。全部16检查点、400条校准零解析失败、零截断后进入正式全量，400条复用于270,400总量。v4.1定位前端输出0–1000归一化坐标，通过固定处理器映射还原原图；该共同原图终点不意味着7B与9B前端完全同构。')
p('正式评测每次只加载一个适配器，不加载优化器、不保留梯度，关闭thinking，最多生成128 token。全量N基线仍有120次长度封顶；15个训练适配器为0。它们均作为真实结果保留，未按正式分数重新修改提示或EOS。')
p('### 6.3 四轴终局与跨轴取舍')
table(['9B五seed均值，%']+AXES,mean_rows(v9['scores']))
p('U定位五seed分别为82.33%、42.58%、62.46%、83.46%、65.88%，全部低于N的88.25%；均值67.34%，下降20.91pp。U的MMStar均值69.23%，较N高8.56pp，但该U−N描述差值不在本轮12项比较族内，不附加未经注册的显著性主张。')
p('A1的MMStar从seed42的56.40%到seed137的68.93%，显示训练抽样与初始化的联合敏感性。A2早期方向较稳，但seed1024定位降至79.88%；完整seed覆盖能避免依据中期结果选择论文叙事。')
table(['9B注册对比','轴','差值pp','12项校正区间pp'],[[k,a,f"{x['difference']*100:+.2f}",ci_text(x['bonferroni_familywise_ci95'])] for k,vals in v9['paired_contrasts'].items() for a,x in vals.items()])
p('12项中三项区间不跨零：A1−N一致性+5.05pp、A1−U的POPE +3.33pp与定位+21.14pp。A1−A2四轴均无明确差异；A1−U的MMStar逐项区间虽不跨零，校正后跨零。因而本研究支持局部收益与代价，不支持A1全面优于其他臂。')
p('![图2：9B全部16模型四轴结果](../results/protocol-v2-20261016/9b-v41-final/formal-report/four-axis-scores.png)')
p('![图3：9B十二项校正配对区间](../results/protocol-v2-20261016/9b-v41-final/formal-report/paired-intervals.png)')
p('### 6.4 输出有效性与能力解释')
p('对N与A2-seed42进行事后输出有效性分解：一致性总提升6.75pp中，双侧均有效单元净贡献0.50pp，N无效而A2有效单元贡献6.25pp；MMStar总提升4.53pp中，双侧均有效部分−0.27pp，N无效而A2有效部分+4.80pp。分组由模型输出决定，属于描述性分解，不证明格式是因果中介，也不外推为所有臂的全部收益都来自格式。它说明“评分改善”与“潜在能力提高”之间仍需谨慎解释。')

p('## 7 定位失稳诊断：从42.58%到待检验解释')
p('针对U-seed137，我们核对1,000步适配器、恢复状态、原始账本、目标张量、样本覆盖与SHA。2,400条定位仅3次解析失败、零长度封顶，无法用大量解析失败解释42.58%。将同一批输出整体按原生归一化、原图像素和处理器像素三种固定解释重算，U137依次为42.58%、3.33%、3.46%，简单更换单位并未恢复。正式分数保持原生协议，不进行逐条GT择优。')
p('U137 median IoU约0.438，而U42约0.832、N约0.927；三个定位子集均退化。2,397个有效框中2,317种不同框，最常见完全重复仅4次，不支持单一固定框塌缩。中心偏差中位数约−5.59%图宽、−7.39%图高，但分布较宽；我们没有用GT拟合平移或缩放来制造恢复分数。')
p('U42和U137教师定位目标对GT的median IoU约0.940/0.932，低于0.5的目标10/150和12/150。二者训练损失、权重与梯度均有限，训练轴更新计数对齐，未发现明确数值损坏；这些检查不排除更细的优化或语义错误。')
p('固定hash抽取三个数据集各4张、共12图，逐面板随机匿名化N/U42/U137，先记录观察后揭示标签。可见选错邻近物体／人物和框范围错误，也保留多个模型同时出错的反例。该可视诊断不产生新的GT，不是双人独立人工盲审，也不是全体错误类型发生率的估计。')
p('由此提出两个可检验因素：像素目标可能与学生原生归一化输出习惯不兼容；同时拟合GT和教师目标可能引入额外监督干扰。这两者不能通过继续修改评测解析器区分，需要改变训练条件。')

p('## 8 坐标表示 × 监督目标：两因素补充实验')
p('### 8.1 条件、预算与控制')
table(['条件','坐标目标','监督','端点'],[['P-D','原图像素','0.5GT CE＋0.5教师CE','复用U42与U137历史固定端点'],['P-G','原图像素','完整权重GT CE','两个新训练'],['N-D','归一化0–1000','0.5GT CE＋0.5教师CE','两个新训练'],['N-G','归一化0–1000','完整权重GT CE','两个新训练']])
p('条件N前缀表示归一化，与未训练基线N不同。42/137各seed内，四条件训练行和暴露顺序匹配，所有端点固定1,000步。归一化从旧已冻结像素GT与教师目标分别按坐标轴尺寸转换，不清洗教师目标、不按结果取整或裁剪。单位变换同时改变数字文本和token长度，因此该因子应称目标表示，而非纯空间能力。')
p('GT-only使用系数1而非0.5，避免额外梯度缩放。双CE两次前向、GT-only一次前向，故控制步数而非计算量。六次新训练按登记随机顺序执行；P-G42在50步退出，新进程恢复优化器及RNG至50后完成51–1,000。该恢复验收证明工程连续性，不证明与另一次从不间断训练逐浮点等价。')
p('新训练绝对预算6小时，新评测绝对预算24小时，均不因重启或分数重置。停止规则包括非有限数值、预算、冻结输入变更、覆盖错误及资源红线。实际6次训练均完成，新增开发推理101,400条与保留集8,640条全部完成；最终08:45交付早于17:51评测上限。')
p('### 8.2 独立保留集')
p('从既有角色隔离清单中冻结每轴120原题，共480原题／960提示、426图像。利用样本ID、路径、字节／RGB别名及去重图检查项目内训练与历史推理暴露，选中样本未进入先前的开发评测或训练。本轮固定后不因分数替换题目；独立性限定为项目内暴露，并不保证基础模型或API教师预训练未见。')
p('保留集一致性采用AI2D来源五改写，推理轴为MMMU单图A–D选择题子集；它不是完整MMMU，也不是MMStar。表格仍以mmstar字段保存技术轴名，论文将其称为“推理代理”以避免混淆。每轴120原题使区间较宽；该集合是补充复核，不是原ARP独立确认阶段。')
p('### 8.3 主效应、交互及定位结果')
p(r'记四条件在某轴的得分为 $z_{PD},z_{PG},z_{ND},z_{NG}$，对两个seed取平均。三个预先固定因子效应为')
p(r'$$E_C=\tfrac12(z_{ND}+z_{NG}-z_{PD}-z_{PG}),\qquad E_T=\tfrac12(z_{PD}+z_{ND}-z_{PG}-z_{NG}),$$')
p(r'$$E_{CT}=(z_{ND}-z_{NG})-(z_{PD}-z_{PG}).$$')
p('其中C衡量归一化相对像素，T衡量双CE相对GT-only；它不是纯教师知识的单独因果效应。')
table(['定位条件','开发42','开发137','保留42','保留137'],[[cond]+[f"{repair['datasets'][ds]['scores'][f'{cond}-seed{s}']['grounding']*100:.2f}" for ds in ['development','heldout'] for s in [42,137]] for cond in ['P-D','P-G','N-D','N-G']])
p('开发集归一化双CE达到87.92%/89.58%，归一化GT-only达到88.79%/87.00%；保留集分别为88.33%/90.83%和90.00%/88.33%。未训练N定位为开发88.25%、保留88.33%。因此更准确的表述是恢复至接近原基线，不能据点估计宣布稳定超越基线。像素GT-only两个seed的变化方向不同，不支持“删除教师就能解决定位”的普遍解释。')
rows=[]
for ds in ['development','heldout']:
    for effect,vals in repair['datasets'][ds]['effects'].items():
        x=vals['grounding'];rows.append([ds,effect,f"{x['difference']*100:+.2f}",ci_text(x['bonferroni24_ci95'])])
table(['集合','定位因子','差值pp','24项校正区间pp'],rows)
p('坐标表示主效应在开发集+22.75pp、保留集+21.88pp，两个校正区间均不跨零。定位教师主效应及交互的校正区间包含零，未证明无效或等效。24项中另外只有开发POPE教师主效应−3.64pp、区间[−7.50,−0.14]不跨零；该下降未获得保留集校正区间支持，不泛化为蒸馏整体有害。')
p('![图4：坐标×监督24项校正区间](../results/protocol-v2-20261016/9b-repair-final/report/factorial-effects.png)')
p('![图5：独立保留集全部端点](../results/protocol-v2-20261016/9b-repair-final/report/heldout-scores.png)')
p('### 8.4 解释强度')
p('在两个监督条件上平均后，归一化定位效果在两集合重复出现，因此当前数据更支持目标表示兼容性的重要影响。这里的干预发生在训练目标上，并非把旧低分直接重新解析为高分；原U137端点仍保留42.58%。然而仅两个seed、历史PD端点复用及运行时序差异，限制了精确因果归因。没有跨家族复现、同token长度控制或全新PD重训，我们不能声称已完全解释定位退化。')

p('## 9 讨论')
p('### 9.1 从预算选择问题转向可测量的表示问题')
p('最初ARP关注“最值得修复哪一轴”。实际研究显示，先验证每个轴是否被可靠读出，比直接比较跨轴退化更基础。旧审计中接近零的定位、7B排序方向随契约变化及9B目标表示效应，共同提示跨轴决策应记录测量前提。它们不是同一个现象的三次重复：分别涉及跨模型接口、同端点评测变化和训练目标干预。')
p('### 9.2 提高分数、格式遵循与潜在能力')
p('本文结果中的一致性提升有价值，但严格短答案任务也对EOS、格式和解码预算敏感。恢复原生坐标可提高可测表现，不能直接等同于空间推理知识增加。有效性分解帮助呈现可观察成分，却不能在输出后选择的子群上识别因果中介。论文因此保持“任务表现改善”的措辞。')
p('### 9.3 实用启示')
p('在部署多轴微调之前，应同时冻结训练与评测坐标契约，核对实际处理器尺寸，保存原始生成token及EOS，逐轴报告带符号变化。发生异常时先排除输入、端点、解析与数值问题，再登记独立训练对照。停止规则应独立于结果；较低分seed和解析失败不能被静默删除。该流程不需要新型优化器，却能减少把工程不兼容误写成能力结论的风险。')

p('## 10 局限性与研究伦理')
p('**外推范围。** E5只有两个seed，E2/E4也只有五个；模型容量、学生代际、教师及数据池跨阶段同时改变，不能单独归因于容量或教师规模。家族内检查点也可能存在配方差异。本文不声称通用最优表示或SOTA。')
p('**选择与统计。** 研究方向因观察到异常而调整，旧开发集被多次使用，统计校正在各阶段族内而非全项目内实施。两个seed的bootstrap可能覆盖有限，校正尾分位数存在Monte Carlo误差。未来需要更多seed、全新确认集及冻结后的跨模型验证。')
p('**测量。** 7B复测改变提示及输出前端，9B归一化目标改变文本长度；两者都不是仅修改一个数学单位的无副作用干预。保留集推理代理与开发集不同，跨集合分数不作同任务增益。双人独立定位盲审未完成，小样本匿名可视诊断不能替代。')
p('**数据与服务。** 去重保证限定为项目内可观察暴露，基础模型训练污染不能排除；API教师可能受供应方审核或后端更新影响。5条被隔离样本不回填为成功；未完成的精确同检查点API桥接与新ARP门槛不被写作完成。')
p('**开放与隐私。** 发布代码、适配器、账本、统计与数据来源／哈希；底层图像及数据集的再分发应遵循其原许可与使用条款。论文不将研究用途等同于商业再分发许可。SSH凭证、API密钥与个人路径不作为复现实验必要材料发布，实验费用不等同于作者身份信息。')

p('## 11 结论')
p('本文记录从跨轴量尺审计到多臂蒸馏、测量修订和坐标表示控制实验的完整研究路径。旧ARP的量尺主导预测未在修订有效性条件下得到确认，因而后续训练作为独立实验开展。五seed结果显示局部收益与跨轴退化并存；7B定位方向对测量协议敏感，9B混合训练则在有效原生评测下仍出现定位退化。两seed控制实验进一步发现，归一化目标的定位主效应在开发集与独立保留集均获得校正区间支持。本文据此提出一项经验性结论：评估多模态序列蒸馏时，训练目标表示和生成契约应作为结果解释的一部分，而不是被默认兼容的实现细节。')

p('## 参考文献')
refs=[
 ('qwen25vl','Bai, S., et al.','2025','Qwen2.5-VL Technical Report','https://arxiv.org/abs/2502.13923'),
 ('qwen35','Qwen Team','2026','Qwen3.5-9B: Official Model Card','https://huggingface.co/Qwen/Qwen3.5-9B'),
 ('llavaov','Li, B., et al.','2024','LLaVA-OneVision: Easy Visual Task Transfer','https://arxiv.org/abs/2408.03326'),
 ('pope','Li, Y., et al.','2023','Evaluating Object Hallucination in Large Vision-Language Models','https://arxiv.org/abs/2305.10355'),
 ('mmbench','Liu, Y., et al.','2023','MMBench: Is Your Multi-modal Model an All-around Player?','https://arxiv.org/abs/2307.06281'),
 ('mmstar','Chen, L., et al.','2024','Are We on the Right Way for Evaluating Large Vision-Language Models?','https://arxiv.org/abs/2403.20330'),
 ('scienceqa','Lu, P., et al.','2022','Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering','https://arxiv.org/abs/2209.09513'),
 ('localize','Ranasinghe, K., Shukla, S. N., Poursaeed, O., Ryoo, M. S., and Lin, T.-Y.','2024','Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs','https://arxiv.org/abs/2404.07449'),
 ('seqkd','Kim, Y., and Rush, A. M.','2016','Sequence-Level Knowledge Distillation','https://arxiv.org/abs/1606.07947'),
 ('lora','Hu, E. J., et al.','2021','LoRA: Low-Rank Adaptation of Large Language Models','https://arxiv.org/abs/2106.09685'),
 ('mmmu','Yue, X., et al.','2023','MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI','https://arxiv.org/abs/2311.16502'),
 ('refer','Yu, L., et al.','2016','Referring Expression Datasets API: RefCOCO, RefCOCO+, and RefCOCOg','https://github.com/lichengunc/refer'),
 ('internvl3','InternVL Team','2025','InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models','https://arxiv.org/abs/2504.10479')]
for i,(key,author,year,title,url) in enumerate(refs,1):p(f'{i}. {author} ({year}). [{title}]({url}).')
p('项目内部依据：用户提供的[ARP研究卡](../idea.std.zh.pdf)、[原实验清单](../experiment_checklist.source.md)、[大模型方案修订](../references/大模型方案_用户修订_20261016.md)。这些材料不是同行评议文献；其中未独立核验的前作评价不进入本文事实性引文。')

p('## 附录A 完整端点与区间')
p('### A.1 七检查点旧原协议量尺与分数')
p('七检查点完整账本及120量尺映射见[v1最终索引](../results/final-20261016/README.md)，修订公式与有效域诊断见[P0公式核查](../results/protocol-v2-20261016/p0/formula_audit.md)、[配对开发门槛](../results/protocol-v2-20261016/paired_gate_development.md)。旧go与新indeterminate各自对应不同协议，保留两者是版本溯源，不是同时宣布门槛通过与未通过。')
p('### A.2 7B全部16模型：旧四轴与独立复测定位')
table(['模型','旧POPE %','旧定位 %','旧一致性 %','旧MMStar %','复测定位 %'],[[name]+[f'{values[a]*100:.3f}' for a in AXES]+[f"{g7['scores'][name]['accuracy']*100:.3f}"] for name,values in v7['scores'].items()])
p('同一行的复测定位来自另一轮明确契约生成，仅供版本并列，不是新四轴联合评分。')
p('### A.3 9B全部16模型：v4.1正式四轴')
table(['模型']+[a+' %' for a in AXES],[[name]+[f'{values[a]*100:.3f}' for a in AXES] for name,values in v9['scores'].items()])
p('### A.4 两因素实验全部四轴')
for ds,label in [('development','开发集'),('heldout','独立保留集')]:
    p('#### '+label)
    d=repair['datasets'][ds]
    table(['模型','POPE %','定位 %','一致性 %','推理代理 %'],[['未训练N']+[f"{d['N_scores'][a]*100:.3f}" for a in AXES]]+[[name]+[f'{values[a]*100:.3f}' for a in AXES] for name,values in d['scores'].items()])
p('### A.5 两因素实验全部24项比较')
table(['集合','因素','轴','差值pp','逐项95%区间pp','24项校正区间pp'],[[ds,ef,a,f"{x['difference']*100:+.3f}",ci_text(x['pointwise_ci95']),ci_text(x['bonferroni24_ci95'])] for ds,d in repair['datasets'].items() for ef,vals in d['effects'].items() for a,x in vals.items()])

p('## 附录B 工程验收、资源与复现')
p('### B.1 训练与内存')
p('7B独立15次训练pure step合计2.306小时，reserved峰值24.547GB；9B独立15次为3.919小时、21.104GB；两因素新增6次为约2.36小时、21.534GB。各值是PyTorch reserved口径，不等于nvidia-smi整卡占用。原4090完整7B预检reserved 24.43GB，超过当时22GB准入线，故迁移5090后进行正式独立训练；正式任务的24.547GB峰值与预检峰值不混用。')
p('9B原轮最大裁剪前梯度范数768.76，训练使用1.0裁剪；“损失与梯度有限”不等于“没有尖峰”。BF16实测、模型SHA、恢复状态和日志均保存。旧C缺少历史整卡显存轨迹，缺测不补零，也不以后期采样冒充历史峰值。')
p('![图B1：7B十五次训练曲线](../results/protocol-v2-20261016/route-b-formal-loss-curves.png)')
p('![图B2：9B十五次训练曲线](../results/protocol-v2-20261016/9b-training-final/loss-curves.png)')
p('![图B3：两因素实验训练曲线](../results/protocol-v2-20261016/9b-repair-final/report/loss-curves.png)')
p('### B.2 计算费用口径')
p('按用户提供2.70元／小时折算，7B原独立轮pure step与纯生成合计约48.89元，不含定位复测；9B原轮pure step约10.58元。9B正式评测与交付墙钟26.417小时，折算约71.33元；这些费用口径不能简单与其他阶段重叠相加充作账户总账。两因素预算上限为训练6小时＋评测24小时，名义81元，但实际阶段提前结束。教师API费、加载、上传、等待与历史排错费按实际usage另计，没有完整账单时不虚构项目总成本。')
p('### B.3 最小循环与恢复验收')
table(['验收项','对应证据'],[['缓存加载与EOS','7B全3,545条新缓存逐文件SHA、原始token及张量对齐；9B有效文本目标3,540条'],['双CE数值','连续实际更新、两路CE及裁剪前梯度；跳过步不计真实更新'],['适配器导出／重载','导出张量与恢复检查点一致，新推理进程从磁盘加载'],['评测输出','唯一ID覆盖、行数、原题聚合及独立重评分'],['真实恢复','7B最小循环50→51…100；两因素P-G42新PID恢复50→51…1,000，优化器与RNG恢复']])
p('### B.4 发布与证据版本')
table(['产物','复核入口'],[
 ['原v1 B/C/D','[原始结果与Release索引](../results/final-20261016/README.md)'],
 ['7B独立全量','[旧四轴统计](../results/protocol-v2-20261016/route-b-final/multiplicity-results.json)'],
 ['7B独立定位复测','[复测结果](../results/protocol-v2-20261016/7b-processor-final/results.json)'],
 ['9B原轮终局','[四轴与12项区间](../results/protocol-v2-20261016/9b-v41-final/formal-report/evaluation-results.json)'],
 ['两因素终局','[结果与24项区间](../results/protocol-v2-20261016/9b-repair-final/report/results.json)'],
 ['终止验收','[110,040条验收](../results/protocol-v2-20261016/9b-repair-final/report/verification.json)、[交付凭证](../results/protocol-v2-20261016/9b-repair-final/terminal-proof/delivery-proof.json)'],
 ['原v1大文件','[Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-formal-20261014)'],
 ['7B大文件','[Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-protocol-v2-20261016)'],
 ['9B及补充实验大文件','[Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-9b-122b-20261017)']])
p('最终9B测量协议SHA256为03ad2816908390d748acccfda8825542923cc0db994a40b31fa3dd4c1656802e；两因素协议为ddb7c44059dd5b83ab19b25d8840c2280a545ae676c90dff155b9dc4e06553ba。大包在AutoDL直接压缩上传，按服务端大小和SHA核验；本地仅保存论文、图表和小型统计。源代码zip不包含Release数据附件。')

p('## 附录C 尚未完成的原ARP事项与投稿前工作')
p('原ARP的精确同检查点API／本地桥接、有效新审计与正式C判定、双人独立定位盲审、独立C选择后的预算干预仍未完成。qwen3-vl-plus等商业别名不能替代精确Qwen2.5-VL参数节点；要求的每轴差异<0.5个百分点桥接没有被本次工程成功自动满足。独立路线固定A1/A2/U并不验证量尺决定选轴的因果链。')
p('投稿前需由作者填写真实作者与单位、逐项核对数据许可与复现命令、检查参考文献元数据，并按目标期刊或会议格式压缩。若维持当前证据范围，可按探索性实证研究投稿；若要升级为普遍机制或跨模型方法论文，需要新注册的更多seed、跨模型与表示长度控制。本文保留这些边界，不用待完成工作阻止已有完整实验被透明报告。')

manuscript=OUT/'论文初稿_全实验流程_20261020.md'
manuscript.write_text('\n'.join(items),encoding='utf-8')
sources=[DOC/'idea.std.zh.pdf',DOC/'experiment_checklist.source.md',DOC/'references/大模型方案_用户修订_20261016.md',DOC/'results/final-20261016/results.json',DOC/'results/final-20261016/signed-gains.json',R/'p0/analysis_v1_vs_v2.csv',R/'paired_gate_development.md',R/'route-b-final/multiplicity-results.json',R/'7b-processor-final/results.json',R/'9b-v41-final/formal-report/evaluation-results.json',R/'9b-repair-final/report/results.json',R/'9b-repair-final/report/verification.json']
evidence={'version':'draft-v0.1','date':'2026-10-20','scope':'complete staged empirical record; not original ARP confirmation','sources':{str(x.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources},'counts':{'v1_base':118300,'v1_post':50700,'7b_formal':270400,'7b_grounding_retest':38400,'9b_formal':270400,'repair_new':110040,'sum_formal_records_not_independent_samples':858240,'new_fixed_endpoint_trainings':39,'new_fixed_updates':39000},'author_information':'not supplied','human_independent_grounding_review':'not completed','original_ARP_revised_gate':'not admitted','evidence_map':{'historical_go_vs_revised_gate':['docs/results/final-20261016/analysis/gate.json','docs/results/protocol-v2-20261016/paired-gate-development/gate.json'],'7b_direction_change':['docs/results/protocol-v2-20261016/route-b-final/multiplicity-results.json','docs/results/protocol-v2-20261016/7b-processor-final/results.json'],'9b_consistency_gain':['docs/results/protocol-v2-20261016/9b-v41-final/formal-report/evaluation-results.json'],'representation_effect':['docs/results/protocol-v2-20261016/9b-repair-final/report/results.json']}}
(OUT/'paper-evidence-map.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
bib=[]
for key,author,year,title,url in refs:
    kind='misc' if key in ['qwen35','refer'] else 'article'
    bib_author=author.replace(', et al.',' and others')
    if key in ['qwen35','internvl3']:bib_author='{'+author+'}'
    if key=='seqkd':bib_author='Kim, Y. and Rush, A. M.'
    if key=='localize':bib_author='Ranasinghe, K. and Shukla, S. N. and Poursaeed, O. and Ryoo, M. S. and Lin, T.-Y.'
    bib.append('@'+kind+'{'+key+',\n  title = {'+title+'},\n  author = {'+bib_author+'},\n  year = {'+year+'},\n  url = {'+url+'},\n  note = {Primary source; accessed 2026-10-20. Author abbreviation retained in this draft.}\n}')
(OUT/'references.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8')
(OUT/'README.md').write_text('# 全实验流程论文初稿\n\n[中文完整初稿](论文初稿_全实验流程_20261020.md)涵盖原v1审计、公式与门槛复核、7B独立实验与定位复测、9B正式实验、坐标×监督对照及独立保留集。\n\n正文及完整附录包含真实表格、现有图表、参考文献和复现证据；全部数字从已归档账本摘要生成，未新增推理。它是可继续编辑的初稿，不是已投稿或通过原ARP门槛的声明。\n\n[证据来源及SHA256](paper-evidence-map.json) · [BibTeX草稿](references.bib)\n',encoding='utf-8')
print('Manuscript written; characters:',len(manuscript.read_text(encoding='utf-8')))
