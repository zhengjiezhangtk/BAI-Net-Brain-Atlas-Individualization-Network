import numpy as np
from pathlib import Path
from shutil import copyfile
import argparse
import os
from config_utils import update_config, load_config_dict


def bedpost_estimate(subdir, recreation, args):

    print('''______________________________________________________________________________________________
                        Fiber peak estimation using Bedpostx in FSL ''')
    subdir = Path(subdir) 
    _, params = load_config_dict(subdir)
    if (Path(args.bedpost_dir)/'dyads3.nii.gz').exists() and (not recreation):
        params['bedpost_dir'] = args.bedpost_dir
    elif (subdir/'DTI.bedpostX/dyads3.nii.gz').exists() and (not recreation):
        params['bedpost_dir'] = str(subdir/'DTI.bedpostX/')
    else:
        if (subdir/'DTI.bedpostX').exists():
            os.system('rm -r {}/DTI.bedpostX/'.format(str(subdir)))
            print('clearing the residual files left before')
        file_dict = {'dti': 'data.nii.gz', 'bval': 'bvals', 'bvec': 'bvecs', 'nodif_brain_mask': 'nodif_brain_mask.nii.gz'}
        for key in file_dict.keys(): 
            target_file = str(subdir/'DTI'/file_dict[key])
            if params[key] != target_file:
                copyfile(params[key], target_file)
                params[key] = target_file
        dtipath = str(subdir/'DTI')
        n = os.system('export CUDA_VISIBLE_DEVICES={}; bash {}/bedpostx_gpu_local.sh {}'.format(args.card1, params['preprocessdir'],  dtipath))
        params['bedpost_dir'] = str(subdir/'DTI.bedpostX/')

    print(params)
    assert (Path(params['bedpost_dir'])/'dyads3.nii.gz').exists(), 'bedpost files are not completed:'+str(Path(params['bedpost_dir'])/'dyads3.nii.gz')
    update_config(subdir, params)
    return None




if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')default='None',required=True
    parser.add_argument('-s', '--subdir', type=str, default='', help='the dictionary for the transition and final result' )
    parser.add_argument('--bedpost_dir', type=str, default='', help='Bedpost Result Dictionary')

    args = parser.parse_args()

            
            
