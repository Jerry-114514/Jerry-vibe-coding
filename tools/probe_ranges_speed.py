import time
import requests
url='https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-VL-3B-Instruct/repo?Revision=master&FilePath=model-00001-of-00002.safetensors'
session=requests.Session()
session.trust_env=False
for length in [1024**2,4*1024**2]:
    start=time.monotonic()
    with session.get(url,headers={'Range':f'bytes=33554432-{33554432+length-1}'},stream=True,timeout=(15,60)) as r:
        size=sum(len(x) for x in r.iter_content(1024**2))
    print({'MiB':length/1024**2,'bytes':size,'seconds':round(time.monotonic()-start,2)},flush=True)
