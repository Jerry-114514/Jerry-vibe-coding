"""Small range probes; report host, status and throughput, never signed URLs."""
import time
from urllib.parse import urlsplit
import requests

repo='llava-hf/llava-onevision-qwen2-0.5b-ov-hf'
rev='74dd0bf867a4cda7950c17663794267c60cf4b40'
for endpoint in ['https://huggingface.co','https://hf-mirror.com']:
    source=f'{endpoint}/{repo}/resolve/{rev}/model.safetensors'
    for trust in [True,False]:
        session=requests.Session()
        session.trust_env=trust
        try:
            start=time.monotonic()
            response=session.get(source,headers={'Range':'bytes=0-1048575'},stream=True,timeout=(15,30))
            read=0
            for chunk in response.iter_content(65536):
                read+=len(chunk)
                if read>=1048576:
                    break
            response.close()
            print({'source':urlsplit(endpoint).hostname,'proxy_env':trust,'target':urlsplit(response.url).hostname,
                   'status':response.status_code,'bytes':read,'seconds':round(time.monotonic()-start,2)},flush=True)
        except requests.RequestException as e:
            print({'source':urlsplit(endpoint).hostname,'proxy_env':trust,'error':type(e).__name__},flush=True)
