"""Fresh GitHub assetmetadata verification against local verified receipts.

Reads a token only from encryptedstdin; does not download large payloads.
"""
import json
from pathlib import Path
import sys
import requests
from experiment_common import ROOT, read_json, sha, write_json
from route_b_staged import OUT,CACHE


def main():
    token=sys.stdin.readline().strip()
    if not token:raise RuntimeError('GitHub credential required via encryptedstdin')
    session=requests.Session();session.trust_env=False
    session.headers.update({'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
    base='https://api.github.com/repos/Jerry-114514/Jerry-vibe-coding'
    response=session.get(base+'/releases/tags/arp-protocol-v2-20261016',timeout=30);response.raise_for_status()
    release=response.json();assets={}
    for page in range(1,30):
        r=session.get(base+f'/releases/{release["id"]}/assets',params={'per_page':100,'page':page},timeout=30);r.raise_for_status()
        batch=r.json()
        assets.update({a['name']:a for a in batch})
        if len(batch)<100:break
    folders=[OUT,CACHE,*sorted((OUT/'formal').glob('*-seed*'))]
    verified=[]
    for folder in folders:
        receipt=folder/'github-storage.json'
        if not receipt.exists():continue
        for item in read_json(receipt)['assets']:
            remote=assets[item['name']]
            assert remote['size']==item['bytes']
            digest=remote.get('digest')
            if digest:assert digest=='sha256:'+item['sha256']
            verified.append({'name':item['name'],'bytes':remote['size'],'sha256':item['sha256'],
                             'server_sha256_confirmed':bool(digest),'asset_id':remote['id']})
        if (folder/'github-inventory.json').exists():
            inventory=read_json(folder/'github-inventory.json')
            for entry in inventory['files']:
                # Frozen binarycheckpoints, cache and loss ledgers stay unchanged;
                # completion/delivery metadata can legitimately be appended later.
                path=ROOT/entry['path']
                if path.suffix in ('.pt','.safetensors') or path.name=='loss.jsonl':
                    assert sha(path)==entry['sha256']
    result={'status':'passed','release_url':release['html_url'],'fresh_remote_assets_verified':len(verified),
            'all_server_digests_present':all(a['server_sha256_confirmed'] for a in verified),
            'assets':verified,'large_payloads_downloaded':False}
    write_json(OUT/'fresh-github-delivery-verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
