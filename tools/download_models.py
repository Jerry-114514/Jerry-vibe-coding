import argparse
from model_backends import model_path
from experiment_common import MODELS

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('checkpoints',nargs='+',choices=MODELS)
    a=p.parse_args()
    for key in a.checkpoints:
        print('Downloading',key,flush=True)
        print(model_path(key),flush=True)
