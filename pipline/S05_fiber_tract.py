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
    # info = os.popen('fslinfo {}'.format(params['dti']))
    # infolist = [line.strip().split() for line in info]
    # xres, yres, zres = infolist[6][1], infolist[7][1], infolist[8][1]
    # if not (xres == yres and yres == zres and zres == xres):
    #     params['tract_MNIspace'] = 'True'

    if args.dti and args.bval and args.bvec:
        params['dti'] = args.dti
        params['bval'] = args.bval
        params['bvec'] = args.bvec

    if args.tractseg_dir:
        params['tractseg_dir'] = args.tractseg_dir
    else:
        params['tractseg_dir'] = str(subdir/'tractseg_output')

    # in_file = os.path.join(params['tractseg_dir'], 'bundle_segmentations.nii.gz')
    
    #if not os.path.exists(in_file):
        #params['nodif_brain_mask']='/n04dat01/atlas_group/lma/populationGCN/BAI_Net/sub-N0001/DTI/nodif_brain_mask.nii.gz'
        params['nodif_brain_mask']=str(subdir/'DTI'/'nodif_brain_mask.nii.gz')
        #params['subdir'] = '/n04dat01/atlas_group/lma/populationGCN/BAI_Net/sub-N0001'
        #print('>>>',params['subdir'])

    in_file = os.path.join(params['tractseg_dir'], 'bundle_segmentations.nii.gz')
    if not os.path.exists(in_file):
        print('>>>',params['subdir'])
        #os.system('bash {}/fibertract.sh {} {} {} {} {} {} {} {}'.format(params['preprocessdir'], str(subdir), params['dti'], params['bval'], params['bvec'], params['nodif_brain_mask'], params['tract_MNIspace'], params['softwaredir'],args.card1))
        cmd = 'bash {}/fibertract.sh {} {} {} {} {} {} {} {}'.format(params['preprocessdir'], str(subdir), params['dti'], params['bval'], params['bvec'], params['nodif_brain_mask'], params['tract_MNIspace'], params['softwaredir'],args.card1)
        print("Running shell command:\n", cmd)
        os.system(cmd)

    if (subdir/'DTI'/'LowRes_Fibers.nii.gz').exists():
        params['Lowres_tract'] = str(subdir/'DTI'/'LowRes_Fibers.nii.gz')
    else:
        print('producing 3mm segmentation')
        target_path = str(subdir/'DTI'/'LowRes_Fibers.nii.gz')
        
        os.system('flirt -ref {} -in {} -out {} -applyisoxfm 3 -interp nearestneighbour'.format( in_file, in_file, target_path))
        params['Lowres_tract'] = str(subdir/'DTI'/'LowRes_Fibers.nii.gz')

    update_config(subdir, params)
    return None


if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--subdir', type=str, default='', help='the dictionary for the transition and final result' )
    parser.add_argument('--tractseg_dir', type=str, default='', help='TractSeg Result Dirtionary')
    parser.add_argument('--tract_MNIspace', type=str, default='False', help='Whether to perform tract segmentation in MNI space')
    parser.add_argument('--HCP', type=bool, default=False, help='TractSeg Result Dirtionary')
    parser.add_argument('--dti', type=str, default='', help='TractSeg Result Dirtionary')
    parser.add_argument('--bval', type=bool, default='', help='TractSeg Result Dirtionary')
    parser.add_argument('--bvec', type=bool, default='', help='TractSeg Result Dirtionary')
    parser.add_argument('--card1', type=str, default='0', help='CUDA device to use')
    args = parser.parse_args()
    
    fiber_tract_segmentation(args.subdir, args)