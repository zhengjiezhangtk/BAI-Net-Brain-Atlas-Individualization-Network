# coding=utf-8
import numpy as np
from pathlib import Path
import time
import nibabel as nib
import os
import argparse
from GCN_model_pytorch.inference import Inference_model
from config_utils import load_config_dict
from pathlib import Path

def pytorch_inference(subdir, hemi, priors, savepath, MSMAll=False):
    _, params = load_config_dict(subdir)
    software_dir = params['softwaredir']
    
    dataset_path = str(Path(subdir).parent)
    subname = Path(subdir).name
    class inference_config:
        def __init__(self):
            self.software_dir = software_dir
            self.dataset_path = dataset_path
            self.sublist_path = ''
            self.single_sub = subname
            self.hemi = hemi
            self.atlas = priors
            self.batch_size = 1
            self.num_workers = 1
            self.MSMAll = MSMAll
            self.model_type = 'GCN'
            self.model_path = f'{software_dir}/pipeline/GCN_pytorch/models'
            self.result_path = savepath
    config = inference_config()
    Inference_model(config.dataset_path, [config.single_sub], config=config)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subdir', type=str, default='')
    parser.add_argument('--hemi', type=str, default='L')
    parser.add_argument('--priors', type=str, default='BN')
    parser.add_argument('--savepath', type=str, default='')
    parser.add_argument('--MSMAll', type=bool, default=False)
    parser.add_argument('--pytorch_GCN', type=bool, default=True, help='using pytorch version GCN to inference')
    args = parser.parse_args()
    subdir = args.subdir
    hemi = args.hemi
    priors = args.priors
    savepath = args.savepath
    MSMAll = args.MSMAll
    if args.pytorch_GCN:
        pytorch_inference( subdir, hemi, priors, savepath, MSMAll)

