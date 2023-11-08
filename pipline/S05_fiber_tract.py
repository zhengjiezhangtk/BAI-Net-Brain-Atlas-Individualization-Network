import numpy as np
from pathlib import Path
import argparse
import os
from config_utils import update_config, load_config_dict


def fiber_tract_segmentation(subdir, args):
    print('''______________________________________________________________________________________________
                                Segmentation brain fiber tracts using TractSeg''')
    subdir = Path(subdir)
    _, params = load_config_dict(subdir)

    params['tract_MNIspace'] = str(args.tract_MNIspace)
    
    # test if dMRI is in the cubic space
    info = os.popen('fslinfo {}'.format(params['dti']))
    infolist = [line.strip().split() for line in info]
    xres, yres, zres = infolist[6][1], infolist[7][1], infolist[8][1]
    if not (xres == yres and yres == zres and zres == xres):
        params['tract_MNIspace'] = 'True'

    if args.dti and args.bval and args.bvec:
        params['dti'] = args.dti
        params['bval'] = args.bval
        params['bvec'] = args.bvec

    if args.tractseg_dir:
        params['tractseg_dir'] = args.tractseg_dir
    else:
        params['tractseg_dir'] = str(subdir/'tractseg_output')

    in_file = os.path.join(params['tractseg_dir'], 'bundle_segmentations.nii.gz')
    # mni_file = os.path.join(params['tractseg_dir'],'bundle_segmentations_dti.nii.gz')
    if not os.path.exists(in_file):
        os.system('bash {}/fibertract.sh {} {} {} {} {} {} {} {}'.format(params['preprocessdir'], params['subdir'], params['dti'], params['bval'], params['bvec'], params['nodif_brain_mask'], str(args.tract_MNIspace), params['softwaredir'],args.card1))
        
    if (subdir/'DTI'/'LowRes_Fibers.nii.gz').exists():
        params['Lowres_tract'] = str(subdir/'DTI'/'LowRes_Fibers.nii.gz')
    else:
        print('producing 3mm segmentation')
        target_path = str(subdir/'DTI'/'LowRes_Fibers.nii.gz')
        # print('in',in_file, ' out' , target_path) #params['preprocessdir'],
        os.system('flirt -ref {} -in {} -out {} -applyisoxfm 3 -interp nearestneighbour'.format( in_file, in_file, target_path))
        params['Lowres_tract'] = str(subdir/'DTI'/'LowRes_Fibers.nii.gz')

    update_config(subdir, params)
    return None


if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')default='None',required=True
    parser.add_argument('-s','--subdir', type=str, default='', help='the dictionary for the transition and final result' )
    parser.add_argument('--tractseg_dir', type=str, default='', help='TractSeg Result Dirtionary')
    parser.add_argument('--HCP', type=bool, default=False, help='TractSeg Result Dirtionary')
    parser.add_argument('--dti', type=str, default='', help='TractSeg Result Dirtionary')
    parser.add_argument('--bval', type=bool, default='', help='TractSeg Result Dirtionary')
    parser.add_argument('--bvec', type=bool, default='', help='TractSeg Result Dirtionary')
    args = parser.parse_args()
    
    fiber_tract_segmentation(args.subdir, args)