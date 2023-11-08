import numpy as np
from pathlib import Path
import argparse
import os
from config_utils import update_config, load_config_dict
import time
# import configparser


def miniprocess_T1_DTI(subdir, args):

    print('''_________________________________________________________________________________________________________________
                                    Miniprocessing for T1  data ''')
    subdir = Path(subdir)
    _, params = load_config_dict(subdir)

    if (subdir/'3D'/'T1_1mm.nii.gz').exists() and (subdir/'3D'/'T1_brain.nii.gz').exists():
        params['t1'] = str(subdir/'3D'/'T1_1mm.nii.gz')
        params['t1_brain'] = str(subdir/'3D'/'T1_brain.nii.gz')
    else:
        print('bash {}/t1_preprocessing.sh {} {}'.format(params['preprocessdir'], args.t1, args.subdir))
        os.system('bash {}/t1_preprocessing.sh {} {}'.format(params['preprocessdir'], args.t1, args.subdir))
        params['t1'] = str(subdir/'3D'/'T1_1mm.nii.gz')
        params['t1_brain'] = str(subdir/'3D'/'T1_brain.nii.gz')

    assert os.path.exists(params['t1']), params['t1']+', T1 file does not exist'

    print('''_________________________________________________________________________________________________________________
                                    Miniprocessing for Diffusion MRI data ''')
    # subdir = Path(subdir)
    # _, params = load_config_dict(subdir)

    if args.dti_preprocessed:
        params['dti'] = args.dti    
        params['bvec'] = args.bvec
        params['bval'] = args.bval
        if not args.nodif_brain_mask:
            raise Exception("if dti_preprocessed, nodif_brain_mask should already have.")
        # print('here',args.nodif_brain_mask)
        os.system('cp {} {}'.format(args.nodif_brain_mask, str(subdir/'DTI'/'nodif_brain_mask.nii.gz')))
        params['nodif_brain_mask'] = str(subdir/'DTI'/'nodif_brain_mask.nii.gz')
    else:
        check_out_dti_file(args)
        os.system('bash {}/dti_preprocessing.sh {} {} {} {}'.format(params['preprocessdir'], args.dti, args.subdir, args.bvec, args.bval))
        params['dti'] = str(subdir/'DTI'/'data.nii.gz')  
        params['bvec'] = str(subdir/'DTI'/'bvecs')  
        params['bval'] = str(subdir/'DTI'/'bvals')  
        params['nodif_brain_mask'] = str(subdir/'DTI'/'nodif_brain_mask.nii.gz')

    if not ( 'Lowres_Mask' in params.keys() and (subdir/'DTI'/'LowResMask.nii.gz').exists() ):
        target_format = str((subdir/'DTI'/'LowResMask.nii.gz'))
        os.system('flirt -ref {} -in {} -o {} -applyisoxfm 3 -interp nearestneighbour'.format(params['nodif_brain_mask'], params['nodif_brain_mask'], target_format))
        params['Lowres_Mask'] = str(subdir/'DTI'/'LowResMask.nii.gz')
    assert os.path.exists(params['dti']), params['dti']+' , dti file does not exist'
    assert os.path.exists(params['bvec']), params['bvec']+', bvec file does not exist'
    assert os.path.exists(params['bval']), params['bval']+', bval file does not exist'
    assert os.path.exists(params['nodif_brain_mask']), params['nodif_brain_mask']+', nodif_brain_mask file does not exist'
    assert os.path.exists(params['Lowres_Mask']), params['Lowres_Mask']+', Lowres_Mask file does not exist'
    update_config(subdir, params)
    return None


def check_out_dti_file(args):
    dti_direction = read_fslinfo(args.dti)
    bval_shape = np.loadtxt(args.bval).shape[0]
    bvec_shape = np.loadtxt(args.bvec).shape[:2]
    assert int(bval_shape) == int(dti_direction), 'bval number {} might not correspond to dwi direction {}'.format(bval_shape, dti_direction)
    assert list(bvec_shape) == [3,dti_direction] or list(bvec_shape) == [dti_direction, 3], 'bval shape {} should in 3*N or N*3'.format(list(bvec_shape))
    return None

def read_fslinfo(file):
    result = os.popen('fslinfo {}'.format(file))
    res = result.read()
    filedict = {}
    for line in res.splitlines():
        tmp = line.split()
        filedict[ tmp[0]] = tmp[1] 
    return int(filedict['dim4'])


if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')default='None',required=True
    parser.add_argument('-s','--subdir', type=str, default='', help='the dictionary for the transition and final result' )
    parser.add_argument('--t1', type=str, default='', help='Structure T1 data'  )
    parser.add_argument('--t1_preprocessed', type=bool, default=False, help='if the T1 is in 1mm, preprocessed and produce the relavent surface using Freesurfer software')
    parser.add_argument('--dti', type=str,  default='', help='Diffusion MRI data')
    parser.add_argument('--bval', type=str,  help='Corresponding b values in Diffusion MRI')
    parser.add_argument('--bvec', type=str,  help='Corresponding b vectors in Diffusion MRI')
    parser.add_argument('--dti_preprocessed', type=bool, default=False, help='if the DTI is preprocessed ')
    parser.add_argument('--nodif_brain_mask', type=str, default='', help='the Diffusion brain mask, if you set dti_preprocessed True, this term is required ')

    args = parser.parse_args()

    # root_dir = '/n15dat01/lma/data/MASiVar_prep/anat'
    # for sub in os.listdir(root_dir):
    #     sub_dir = '/n15dat01/lma/data/MASiVar_prep/anat/'+sub
    #     for session in os.listdir(sub_dir):
    #         print(sub,session)
    #         savedir = '{}/{}/anat'.format(sub_dir, session)
    #         t1 = os.path.join(savedir, os.listdir(savedir)[0])
    #         os.system('bash /n15dat01/lma/data/MASiVar_prep/pipline/t1_preprocessing.sh {} {}'.format(t1, savedir))
    
    # miniprocess_T1(args.subdir, args)
    # miniprocess_DTI(args.subdir, args)
