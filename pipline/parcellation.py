import numpy as np
from pathlib import Path
import argparse
import multiprocessing
import os
import configparser
import time

from config_utils import build_subdir, build_sub_config, load_config_dict, save_config_dict, update_config, update_config_all
from S01_miniprocess import miniprocess_T1_DTI
from S02_registration import T12DTI_registration
from S03_build_surface import produce_32k_surface, produce_weighted_adj_matrix
from S04_fiber_orientation import bedpost_estimate
from S05_fiber_tract import fiber_tract_segmentation
from S06_probtrack import probtrack_fsl
from S07_postprobtrack import PostProbtrack, get_fiber_fingerprint
from S08_gcn_inference import pytorch_inference



def main():
    ''' T1 , d-MRI, bval and bvec must be provided from beginning '''
    subdir = Path(args.subdir)
    build_subdir(subdir)
    sub = subdir.name
    dataset_dir = subdir.parent
    step_range = np.arange(int(args.begin_step), int(args.end_step)+1, 1).astype('int32')
    print(step_range)
    if not (subdir/(sub+'_config.ini')).exists():
        _, params = build_sub_config(subdir)
        update_config_all(subdir, args) # tempory
    else:
        update_config_all(subdir, args)
        _, params = load_config_dict(subdir)

    t_begin = time.time()
    if 1 in step_range:
        miniprocess_T1_DTI(subdir, args)
        
    if 2 in step_range:
        p2 = multiprocessing.Process(target = T12DTI_registration, args = (subdir, args))
        p2.start() 
        p2.join()

    if 3 in step_range:
        recreation = False
        string = '_MSMAll' if args.MSMAll=='True' else ''
        p3 = multiprocessing.Process(target = produce_32k_surface, args = (subdir, recreation, args))
        p3.start()
        p3.join()
        p3 = multiprocessing.Process(target = produce_weighted_adj_matrix, args = (subdir, string, recreation, args))
        p3.start()
        
    if 4 in step_range:
        recreation = False
        p4 = multiprocessing.Process(target = bedpost_estimate, args = (subdir, recreation, args))
        p4.start()

    if 5 in step_range:
        p5 = multiprocessing.Process(target = fiber_tract_segmentation, args = (subdir, args))
        p5.start()

    if 3 in step_range:
        p3.join()
    if 4 in step_range:    
        p4.join()
    if 5 in step_range: 
        p5.join()

    if 6 in step_range:
        recreation = True
        p6_L = multiprocessing.Process(target = probtrack_fsl, args = (subdir, 'L', args.card1, args))
        p6_L.start()
        if not args.card2:
            p6_L.join()
            time.sleep(3)
            p6_R = multiprocessing.Process(target = probtrack_fsl, args = (subdir, 'R', args.card1, args))
            p6_R.start()
        else:
            p6_R = multiprocessing.Process(target = probtrack_fsl, args = (subdir, 'R', args.card2, args))
            p6_R.start()

    if 7 in step_range:
        recreation = False
        if 6 in step_range:
            p6_L.join()
        PostProbtrack(subdir.parent, subdir.name, "L")
        get_fiber_fingerprint(subdir.parent, subdir.name, "L", recreation)

        if 6 in step_range:
            p6_R.join()
        PostProbtrack(subdir.parent, subdir.name, "R")
        get_fiber_fingerprint(subdir.parent, subdir.name, "R", recreation)

    if 8 in step_range:
        _, params = load_config_dict(subdir) 

        if args.pytorch_GCN:
            savepath = str(subdir)
            print('subdir: ', subdir, 'savepath:', savepath)
            pytorch_inference(subdir, 'L', args.priors, savepath, args.MSMAll)
            pytorch_inference(subdir, 'R', args.priors, savepath, args.MSMAll)

    print('Total time:', time.time()-t_begin)

    return None


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subdir', type=str, default='', help='the dictionary for the transition and final result', required=True )
    parser.add_argument('-p', '--priors', type=str, default='BN', help='the group atlas prior' )
    parser.add_argument('--begin_step', type=str, default='1', help='the begin step ')
    parser.add_argument('--end_step', type=str, default='8', help='the end step' )
    parser.add_argument('--t1', type=str, default='', help='Structure T1 data'  )
    parser.add_argument('--t1_preprocessed', dest='t1_preprocessed', action='store_true', help='if the T1 is in 1mm, preprocessed for producing the relavent surface using Freesurfer software')
    parser.add_argument('--dti', type=str,  default='', help='Diffusion MRI data')
    parser.add_argument('--bval', type=str,  help='Corresponding b values in Diffusion MRI')
    parser.add_argument('--bvec', type=str,  help='Corresponding b vectors in Diffusion MRI')
    parser.add_argument('--dti_preprocessed', dest='dti_preprocessed', action='store_true', help='if the DTI is preprocessed ')
    parser.add_argument('--nodif_brain_mask', type=str, default='', help='the Diffusion brain mask, if you set dti_preprocessed True, this term is required ')
    parser.add_argument('--xfms', type=str,  default='',help='transfer matrix from T1 space to DTI space eg. T1_1mm_2_DTI.mat')
    parser.add_argument('--dti_fa', type=str, default='', help='FA Value Derived from dtifit in FSL')
    parser.add_argument('--ANTSreg', dest='ANTSreg', action='store_true', help='if using ANTS to registration')
    parser.add_argument('--bedpost_dir', type=str, default='', help='Bedpost Result Dictionary')
    parser.add_argument('--fsaverage_LR32k', type=str, default='', help='Freesurfer Result Dictionary')
    parser.add_argument('--surface_begin_name', type=str, default='', help='Surface begin name in Freesurfer Result Dictionary, if fsaverage_LR32k provided, this terms is required')
    parser.add_argument('--MSMAll', type=bool, default=False, help='if the surface type is MSMAll')
    parser.add_argument('--tractseg_dir', type=str, default='', help='TractSeg Result Dirtionary')
    parser.add_argument('--tract_MNIspace', type=bool, default=True,  help='using MNI TractSeg results back into individual due to unisotropical dwi resolution')
    parser.add_argument('--pytorch_GCN', type=bool, default=True, help='using pytorch version GCN to inference')
    parser.add_argument('--card1', type=str, default='0', help='First Cuda Card Number')
    parser.add_argument('--card2', type=str, default='', help='Second Cuda Card Number')

    args = parser.parse_args()
    main()
