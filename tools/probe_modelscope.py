import time
from urllib.parse import urlsplit
import requests
files=[('Qwen/Qwen2.5-VL-3B-Instruct','model-00001-of-00002.safetensors'),
       ('OpenGVLab/InternVL3-1B','model.safetensors'),
       ('AI-ModelScope/llava-onevision-qwen2-0.5b-ov-hf','model.safetensors')]
for repo,file in files:
    session=requests.Session()
    session.trust_env=False
    start=time.monotonic()
    url=f'https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={file}'
    try:
        with session.get(url,headers={'Range':'bytes=0-1048575'},stream=True,timeout=(15,30)) as response:
            count=0
            for data in response.iter_content(65536):
                count+=len(data)
                if count>=1048576:
                    break
            print({'repo':repo,'status':response.status_code,'host':urlsplit(response.url).hostname,
                   'range':response.headers.get('Content-Range'),'bytes':count,'seconds':round(time.monotonic()-start,2)},flush=True)
    except requests.RequestException as error:
        print(repo,type(error).__name__,flush=True)
