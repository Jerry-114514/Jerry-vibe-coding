import requests,time
for base in ['http://images.cocodataset.org','https://images.cocodataset.org','https://s3.amazonaws.com/images.cocodataset.org']:
    url=base+'/train2014/COCO_train2014_000000316942.jpg'
    session=requests.Session()
    session.trust_env=False
    start=time.monotonic()
    try:
        response=session.get(url,timeout=(10,20))
        print(base,response.status_code,len(response.content),round(time.monotonic()-start,2),flush=True)
    except requests.RequestException as e:
        print(base,type(e).__name__,round(time.monotonic()-start,2),flush=True)
