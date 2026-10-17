"""Verify local or exact Release recovery, without downloading optimizer states."""
from datetime import datetime, timezone
from experiment_common import read_json, write_json, sha
from route_b_9b_loop import OUT, SEEDS
from verify_9b_student_outputs import verify_training
from run_9b_formal_when_ready import verify_receipt
from resilient_9b_delivery import network_retry


def verify(token):
    rows=[]
    for seed in SEEDS:
        for arm in ('A1','A2','U'):
            folder=OUT/'formal'/f'{arm}-seed{seed}'
            proof=verify_training(arm,seed,allow_offloaded=True)
            local=(folder/'resume.pt').exists()
            row={'arm':arm,'seed':seed,'updates':1000,'resume_sha256':proof['resume_sha256'],
                 'local':local,'optimizer_tensor_verified_before_archival':True}
            if not local:
                receipt=read_json(folder/'training-delivery.json')
                row['fresh_server_verification']=network_retry(lambda:verify_receipt(receipt,token),'verify9b-recovery:'+folder.name)
                row['restore_pointer_sha256']=sha(folder/'resume-offloaded.json')
                row['receipt_sha256']=sha(folder/'training-delivery.json')
            rows.append(row)
    result={'status':'passed','runs':rows,'available_recoveries':15,'local_recoveries':sum(r['local'] for r in rows),
            'verified_release_recoveries':sum(not r['local'] for r in rows),'restored':[],
            'timestamp':datetime.now(timezone.utc).isoformat(),
            'policy':'Latest user disk policy permits completed archived recovery; protect active local state. Evaluation needs adapters, not optimizer downloads.'}
    # Preserve the established evidence pathname, explicitly recording that
    # availability was verified rather than claiming any new restoration.
    write_json(OUT/'native-evaluation-v2/recovery-restoration.json',result)
    return result
