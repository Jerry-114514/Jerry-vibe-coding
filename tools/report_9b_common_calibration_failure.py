"""Describe failed9B calibration from saved outputs only; never add predictions."""
from datetime import datetime,timezone
from experiment_common import ROOT,read_json,read_rows,write_json,write_rows,sha
from evaluate_9b_native import NATIVE,score_native
from processor_pixel_contract import score as processor_score
from processor_pixel_contract_csv import score as csv_score


def run():
    path=NATIVE/'minimal/N-seed42/eval.jsonl'
    lookup={r['item_id']:r for r in read_rows(NATIVE/'minimal-manifest.jsonl')}
    rows=[r for r in read_rows(path) if r['axis']=='grounding'];assert len(rows)==100
    marker=read_json(path.parent/'scores.json');assert marker['axis_scores']['grounding']<.8
    assert marker['ledger_sha256']==sha(path)
    result={};evidence=[]
    functions={'registered_processor_pixels':processor_score,'processor_pixels_csv_schema':csv_score,
               'posthoc_fixed1000_diagnostic_only':score_native}
    for name,fn in functions.items():
        scored=[fn(r['raw_generation'],lookup[r['item_id']]) for r in rows]
        result[name]={'accuracy':sum(r['correct_bit'] for r in scored)/100,
                      'parse_failures':sum(r['parse_fail_flag'] for r in scored)}
    for r in rows:
        item=lookup[r['item_id']]
        evidence.append({'item_id':r['item_id'],'raw_generation':r['raw_generation'],
             'original_size':[item['width'],item['height']],
             'processor_size':[item['processor_width'],item['processor_height']],
             'scorings':{name:fn(r['raw_generation'],item) for name,fn in functions.items()}})
    report={'status':'registered9b_calibration_failed_no_formal_evaluation','timestamp':datetime.now(timezone.utc).isoformat(),
            'rows':100,'registered_threshold':.8,'source_ledger_sha256':sha(path),
            'protocol_sha256':sha(NATIVE/'protocol.json'),'diagnostic_source_sha256':sha(ROOT/'tools/report_9b_common_calibration_failure.py'),
            'scores':result,'new_predictions':0,'posthoc1000_score_not_a_pass':True,'no_threshold_or_prompt_retry':True,
            'scope':'Unit diagnosis only; do not replace the registered processor-pixel score with a higher diagnostic score'}
    folder=NATIVE/'failed-calibration-report';write_json(folder/'diagnostic.json',report);write_rows(folder/'diagnostic.jsonl',evidence)
    text=['# 9B共同处理器像素校准未通过','','15次训练与恢复验收完成；9B正式16模型评测尚未启动。N在新处理器像素提示下的100条定位准确率为1%，解析失败95/100，未达到固定80%及≤5次失败门槛；停止是正确的，不是OOM或磁盘故障。',
      '', '只对同一批已保存输出作CPU诊断：加入严格四数CSV语法后仍为1%（95次解析失败）；固定解释为0–1000的事后诊断为86%（0次解析失败）。这支持模型继续输出千分比单位，但86%不是本协议主分，不能用来放行。没有新增预测，没有降低门槛，也没有为达到80%反复调提示。',
      '', '7B的另一项失败来自四数CSV与JSON解析器不一致，修复语法后原100条达到80%；不能把它与9B的坐标单位失败混为一谈。7B独立复测继续，9B保持停止记录。',
      '', '## 可审阅的后续接口方案（未执行）','',
      '建议将“模型原生输出单位”和“统一计分中间坐标”分开：7B固定读取处理器像素；9B固定要求其原生0–1000，先乘处理器宽高/1000变成统一处理器像素，再使用共同的原图/处理器比例换回原图。所有9B训练臂使用同一个前端规则，不能按臂、样本或GT挑单位。',
      '', '该方案属于另立协议，不能追溯宣布当前失败已修复。U曾学习原图像素，必须在未使用本批结果挑题的固定校准样本上检验其是否遵循明确原生提示；不能仅因N的事后诊断较高就放行其他适配器。',
      '', '如采用下一版，应事先冻结各家族前端、四臂校准样本/图像簇、失败规则和独立预算。可设400条新增定位预测、15分钟GPU上限（2.70元/小时约0.675元），不增加任何学生训练；当前没有启动这笔新预算，原失败900秒绝对截止时间不重置。',
      '', '原ARP阶段C仍未放行。统一中间坐标并不能消除代际、教师、视觉分辨率和指令遵循差异；不能据此作纯容量归因。']
    (folder/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(report['status'],result,flush=True)


if __name__=='__main__':run()
