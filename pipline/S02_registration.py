import numpy as np
from pathlib import Path
import argparse
import os
from config_utils import update_config, load_config_dict, load_soft_config
# import configparser

def T12DTI_registration(subdir, args):
    print('''_______________________________________________________________________________________________
                        Trans matric between T1 and DTI space of subjects''')
    subdir = Path(subdir)
    _, params = load_config_dict(subdir)
    recreation = True
    if (not (subdir/'xfms'/'T1_1mm_2_DTI.mat').exists()) or (recreation):
        if args.dti_fa:
            os.system('cp {} {}'.format(args.dti_fa, str(subdir/'DTI'/'dti_FA.nii.gz')))
            params['dti_fa'] = str(subdir/'DTI'/'dti_FA.nii.gz')
        elif (subdir/'DTI'/'dti_FA.nii.gz').exists():
            params['dti_fa'] = str(subdir/'DTI'/'dti_FA.nii.gz')
        else:
            target_format = str(subdir/'DTI'/'dti')
            # mask_file = args['nodif_brain_mask'] if args['nodif_brain_mask'] else params['nodif_brain_mask']
            os.system('dtifit -k {} -m {} -r {} -b {} -o {}'.format(params['dti'], params['nodif_brain_mask'], params['bvec'], params['bval'], target_format))
            params['dti_fa'] = str(subdir/'DTI'/'dti_FA.nii.gz')
        
        os.system('bash {}/space_registration.sh {} {} {}'.format(params['preprocessdir'], params['subdir'], params['dti_fa'], params['t1_brain']))
        if args.ANTSreg:
            antspath = load_soft_config(params['softwaredir'])['ANTSDIR']
            if not ((subdir/'DTI'/'ants_t1Warped.nii.gz').exists()):
                print(antspath)
                os.system('bash {}/ants_space_registration.sh {} {} {} {}'.format(params['preprocessdir'], params['subdir'], params['dti_fa'], params['t1_brain'], antspath))
    

    params['xfms'] = str(subdir/'xfms'/'T1_1mm_2_DTI.mat')
    #assert os.path.exists(params['xfms']), params['xfms']+': xfms file does not exist'       
    update_config(subdir, params)
    return None


if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')default='None',required=True
    parser.add_argument('-s','--subdir', type=str, default='/n20dat01/wyshi1/IndividualBN/NC_05_0071/', help='the dictionary for the transition and final result' )
    parser.add_argument('--dti_fa', type=str, default='', help='FA Value Derived from dtifit in FSL')
    parser.add_argument('--xfms', type=str,  default='', help='transfer matrix from T1 space to DTI space eg. T1_1mm_2_DTI.mat')
    parser.add_argument('--ANTSreg', dest='ANTSreg', action='store_true', help='if using ANTS to registration')
    
    args = parser.parse_args()

    T12DTI_registration(args.subdir, args)
