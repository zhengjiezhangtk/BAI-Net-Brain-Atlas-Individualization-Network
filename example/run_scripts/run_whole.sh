#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=out_0.%j
#SBATCH --error=err_0.%j
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n18

soft=/share/soft/BAI_Net
. $soft/pipline/soft_path.sh

megtensity=3T
datadir=/test/mri/${megtensity}
pardir=/project_subject/${megtensity}

sub=sub-04
echo ${sub}
subdir=${pardir}/${sub}
echo ${subdir}

t1=/test/mri/${megtensity}/${sub}/anat/${sub}_desc-preproc_T1w.nii.gz
dti=/test/mri/${megtensity}/${sub}/dwi/${sub}_dwi.nii.gz
bval=/test/mri/${megtensity}/${sub}/dwi/${sub}_dwi.bval
bvec=/test/mri/${megtensity}/${sub}/dwi/${sub}_dwi.bvec

nodifmask=/project_subject/${megtensity}/${sub}/DTI/nodif_brain_mask.nii.gz
bedpostdir=/project_subject/${megtensity}/${sub}/T1w/Diffusion.bedpostX
surfname=${sub}
fsaverage_dir=/project_subject/${megtensity}/${sub}/${sub}/fsaverage_LR32k

# step 1-8
GPU_card=0
python $soft/pipline/parcellation.py -s ${subdir} --t1 ${t1} --t1_preprocessed  --dti ${dti} --bval ${bval} --bvec ${bvec} --ANTSreg --begin_step 1 --end_step 8 --card1 $GPU_card
