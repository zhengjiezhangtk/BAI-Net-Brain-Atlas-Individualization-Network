import numpy as np
from pathlib import Path
import argparse
import os
from config_utils import update_config, load_config_dict


def probtrack_fsl(subdir, hemi, card, args):
    print('''______________________________________________________________________________________________
                                Probatracking tractagraphy using Probtrackx in FSL''')
    subdir = Path(subdir)
    _, params = load_config_dict(subdir)
    sub = subdir.name
    output_file = '{}/{}_{}_probtrackx_omatrix2'.format(str(subdir), sub, hemi)
    signal = 'True' if args.ANTSreg else 'False'
    print('ANTSreg is ', signal)
    os.system('bash {}/probtrack_hemi_gpu.sh {} {} {} {} {} {} {}'.format(params['preprocessdir'], params['subdir'], params['t1'], params['bedpost_dir'], hemi, output_file, signal, card))
    print('bash {}/probtrack_hemi_gpu.sh {} {} {} {} {} {} {}'.format(params['preprocessdir'], params['subdir'], params['t1'], params['bedpost_dir'], hemi, output_file, signal, card))
    update_config(subdir, params)
    return None




if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')default='None',required=True
    parser.add_argument('-s','--subdir', type=str, default='', help='the dictionary for the transition and final result' )
    parser.add_argument('--tractseg_dir', type=str, default='', help='TractSeg Result Dirtionary')
    parser.add_argument('--hemi', type=str, default='L', help='TractSeg Result Dirtionary')
    parser.add_argument('--card', type=str, default='0', help='TractSeg Result Dirtionary')
    parser.add_argument('--HCP', type=bool, default=False, help='TractSeg Result Dirtionary')


    args = parser.parse_args()
    
    probtrack_fsl(args.subdir, args.hemi, args.card, args)