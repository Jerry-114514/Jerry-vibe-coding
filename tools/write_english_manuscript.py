"""Render the full English translation while reusing the Chinese numeric tables."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'docs/paper'
source=PAPER/'论文初稿_全实验流程_20261020.md'
text=source.read_text(encoding='utf-8')
source_tables=[b for b in text.split('\n\n') if b.startswith('|')]
assert len(source_tables)==20
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
def original_rows(index):
    return [[c.strip() for c in line.strip('|').split('|')] for line in source_tables[index].splitlines()[2:]]
headers={
 0:['Stage','Completed work','Principal scale','Status of evidence'],
 1:['Axis','Development prompts','Original scoring units','Scoring rule','Training source'],
 2:['Historical v1 checkpoint','POPE','Grounding','Consistency','MMStar'],
 3:['Revised paired development gate: four axes','Strict reversal rate','Valid structural classes','Defined outer differences','Decision'],
 4:['Historical training arm','POPE gain (pp)','Grounding (pp)','Consistency (pp)','MMStar (pp)','Sum of changes (pp)'],
 5:['7B means under the historical protocol (%)','pope','grounding','consistency','mmstar'],
 6:['7B grounding protocol','N','A1: five-seed mean','A2: five-seed mean','U: five-seed mean'],
 7:['7B grounding reassessment contrast','Difference (pp)','Four-comparison-adjusted interval (pp)'],
 8:['9B five-seed means (%)','pope','grounding','consistency','mmstar'],
 9:['Registered 9B contrast','Axis','Difference (pp)','12-comparison-adjusted interval (pp)'],
 10:['Condition','Coordinate targets','Supervision','Endpoints'],
 11:['Grounding condition','Development: seed 42','Development: seed 137','Held-out: seed 42','Held-out: seed 137'],
 12:['Dataset','Grounding effect','Difference (pp)','24-comparison-adjusted interval (pp)'],
 13:['Model','Historical POPE (%)','Historical grounding (%)','Historical consistency (%)','Historical MMStar (%)','Reassessed grounding (%)'],
 14:['Model','pope (%)','grounding (%)','consistency (%)','mmstar (%)'],
 15:['Model','POPE (%)','Grounding (%)','Consistency (%)','Reasoning proxy (%)'],
 16:['Model','POPE (%)','Grounding (%)','Consistency (%)','Reasoning proxy (%)'],
 17:['Dataset','Effect','Axis','Difference (pp)','Pointwise 95% interval (pp)','24-comparison-adjusted interval (pp)'],
 18:['Acceptance item','Evidence'],
 19:['Artifact','Verification entry point']}
overrides={
0:[
 ['E0: v1 audit and original training','Seven-checkpoint ledgers, scale scan, original 3B-from-7B training','118,300 base generations + 50,700 post-training generations; 3 training runs','Historical protocol results'],
 ['E1: audit reassessment','Denominators, structural quotient, ties, common domains, paired noise','120 settings; three families, six four-axis/grounding-removed analyses','Exploratory reanalysis; revised C not admitted'],
 ['E2: 7B self-distillation','15 training runs and 16-model evaluation after the minimal loop','270,400 formal generations','Independent sequence distillation'],
 ['E3: 7B grounding reassessment','Same endpoints, explicit processor-pixel contract','16 × 2,400 = 38,400 generations','Post hoc measurement revision; separate comparison family'],
 ['E4: 9B from 122B','15 training runs; all 16 models evaluated under v4.1','270,400 generations, including 400 reused calibration predictions','Independent extension; no capacity-only comparison'],
 ['E5: coordinates × supervision','6 new training runs + 2 historical endpoints; two evaluation sets','110,040 new generations','Exploratory two-seed follow-up registered after observing the original results']
],
1:[
 ['POPE',9000,9000,'Strict yes/no accuracy, not POPE F1','Positive/negative object questions from COCO training sources'],
 ['Grounding',2400,2400,'Original-image IoU ≥ 0.5','RefCOCO train'],
 ['Consistency',4000,800,'All five paraphrases must be correct for an original item to pass','MMBench sources excluded from development evaluation'],
 ['Reasoning proxy',1500,1500,'Strict MMStar option accuracy','ScienceQA train; no MMStar evaluation items used for training'],
 ['Total',16900,13700,'Report axes separately; prompts are not original items','Original pool: 3,545 prompts']
],
3:[['Qwen','0','12/12','61.8%','indeterminate'],['InternVL','0.4 (few valid classes only)','5/60','0','indeterminate'],['LLaVA','1.0 (few valid classes only)','2/12','0','indeterminate']],
6:[['Historical protocol (%)']+original_rows(6)[0][1:],['Explicit processor-pixel reassessment (%)']+original_rows(6)[1][1:]],
10:[['P-D','Original-image pixels','0.5 GT CE + 0.5 teacher CE','Reuse the fixed historical U42 and U137 endpoints'],['P-G','Original-image pixels','Full-weight GT CE','Two new runs'],['N-D','Normalized 0–1000','0.5 GT CE + 0.5 teacher CE','Two new runs'],['N-G','Normalized 0–1000','Full-weight GT CE','Two new runs']],
18:[
 ['Cache loading and EOS','All 3,545 new 7B cache entries: file SHA, raw tokens, tensor alignment; 3,540 valid 9B text targets'],
 ['Dual-CE numerical checks','Consecutive actual updates, both CE terms, pre-clipping gradients; skipped steps do not count as updates'],
 ['Adapter export and reload','Exported tensors match recovery checkpoints; a new inference process loads the adapter from disk'],
 ['Evaluation output','Unique-ID coverage, row counts, original-item aggregation, independent rescoring'],
 ['Actual resumption','7B minimal loop: 50 → 51…100; factorial P-G42: a new PID resumes 50 → 51…1,000 with optimizer and RNG restored']
],
19:[
 ['Original v1 B/C/D','[Original results and Release index](../results/final-20261016/README.md)'],
 ['Full independent 7B evaluation','[Historical four-axis statistics](../results/protocol-v2-20261016/route-b-final/multiplicity-results.json)'],
 ['Independent 7B grounding reassessment','[Reassessment results](../results/protocol-v2-20261016/7b-processor-final/results.json)'],
 ['Final original 9B experiment','[Four axes and 12 contrasts](../results/protocol-v2-20261016/9b-v41-final/formal-report/evaluation-results.json)'],
 ['Final factorial experiment','[Results and 24 contrasts](../results/protocol-v2-20261016/9b-repair-final/report/results.json)'],
 ['Terminal acceptance','[110,040-generation verification](../results/protocol-v2-20261016/9b-repair-final/report/verification.json); [delivery proof](../results/protocol-v2-20261016/9b-repair-final/terminal-proof/delivery-proof.json)'],
 ['Original v1 large artifacts','[Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-formal-20261014)'],
 ['7B large artifacts','[Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-protocol-v2-20261016)'],
 ['9B and follow-up large artifacts','[Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-9b-122b-20261017)']
]}
tables={}
for index in range(20):
    rows=overrides.get(index,original_rows(index))
    rows=[[str(c).replace('未训练N','Untrained N') for c in row] for row in rows]
    tables[index]=table(headers[index],rows)
template=(ROOT/'tools/templates/full_experiment_manuscript_en.md.in').read_text(encoding='utf-8')
for index,value in tables.items():
    token='{{TABLE_'+str(index)+'}}'
    assert template.count(token)==1,token
    template=template.replace(token,value)
reference_block=text.split('## 参考文献\n\n',1)[1].split('\n\n项目内部依据',1)[0]
assert '{{REFERENCES}}' in template
template=template.replace('{{REFERENCES}}',reference_block)
assert '{{TABLE_' not in template
destination=PAPER/'paper_draft_full_experiment_20261020_en.md'
destination.write_text(template,encoding='utf-8')
metadata={'translation_of':str(source.relative_to(ROOT)).replace('\\','/'),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'english_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'date':'2026-10-20','scope':'Complete English translation, including all 20 tables, 8 figure references, equations, references and appendices; duplicate Chinese/English abstracts consolidated into one English abstract.','numeric_tables':'Copied from the Chinese manuscript; descriptive cells and headers translated.','no_new_experiments':True}
(PAPER/'english-translation-provenance.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
print('English manuscript written; words:',len(template.split()),'; tables:',len(tables))
