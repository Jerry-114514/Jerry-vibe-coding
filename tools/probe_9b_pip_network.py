"""Public package transport probe, no environment/credential output."""
import time
from urllib.parse import urlsplit
import requests

s = requests.Session();s.trust_env = False
r = s.get('https://pypi.org/pypi/transformers/5.20.1/json',timeout=20);r.raise_for_status()
url = next(x['url'] for x in r.json()['urls'] if x['packagetype'] == 'bdist_wheel')
for trust in (False, True):
    session = requests.Session();session.trust_env = trust
    started = time.monotonic()
    try:
        with session.get(url,headers={'Range':'bytes=0-65535'},stream=True,timeout=(10,15)) as response:
            chunk = next(response.iter_content(65536))
            print({'use_environment_proxy':trust,'host':urlsplit(response.url).hostname,'status':response.status_code,
                   'first_bytes':len(chunk),'seconds':time.monotonic()-started},flush=True)
    except requests.RequestException as error:
        print({'use_environment_proxy':trust,'error':type(error).__name__,'seconds':time.monotonic()-started},flush=True)
