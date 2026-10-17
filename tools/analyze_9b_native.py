"""Reuse the registered paired-bootstrap routine with explicit native ledgers."""
import json

from experiment_common import ROOT, read_json, write_json, sha
from evaluate_9b_native import NATIVE, verify_formal


def analyze():
    import analyze_9b_formal as registered
    # Dependency substitution is explicit: same registered10000 paired-bootstrap
    # implementation, separate versioned input ledgers and output directory.
    registered.OUT = NATIVE
    registered.verify_formal = verify_formal
    result = registered.analyze(replicates=10000)
    result['measurement_calibration'] = {
        'protocol_sha256': sha(NATIVE/'protocol.json'),
        'registered_statistics_source_sha256': sha(ROOT/'tools/analyze_9b_formal.py'),
        'native_analysis_source_sha256': sha(ROOT/'tools/analyze_9b_native.py'),
        'prompt': 'Explicit normalized0..1000 for every grounding output',
        'scoring': 'x*width/1000,y*height/1000, then pixel-GT IoU>=.5; no per-item unit guessing',
        'old_pixel_minimum_outputs_retained': True,
        'user_amendment_after_legacy_minimum_results': True,
        'training_targets_remain_original_pixel': True,
        'not_original_arp': True}
    write_json(NATIVE/'formal-report/evaluation-results.json',result)
    report=NATIVE/'formal-report/evaluation-report.md'
    with report.open('a',encoding='utf-8') as stream:
        stream.write('\n## 测量校准\n\n9B定位提示统一要求0–1000坐标，再按原图宽高换算成像素，与原始标注计算IoU。所有模型采用同一转换，不按答案选坐标系。原始模型输出保持原样，换算结果另存字段。旧像素口径最小循环和诊断单独保留，不与本版本分数混合。\n\n该修订由用户在旧最小循环之后要求，不能追溯称为训练前登记；训练目标仍为原图像素，因此跨口径指令遵循也是泛化因素。与7B同时改变学生、教师和测量契约，不作纯容量归因。\n')
    return result


if __name__=='__main__':
    print(json.dumps({'status':analyze()['status']}),flush=True)
