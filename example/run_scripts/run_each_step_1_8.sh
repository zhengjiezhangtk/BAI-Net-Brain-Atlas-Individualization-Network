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
pardir=/project_subjects/${megtensity}

sub=sub-01
echo ${sub}
subdir=${pardir}/${sub}

t1=/test/mri/${megtensity}/${sub}/anat/${sub}_desc-preproc_T1w.nii.gz
dti=/test/mri/${megtensity}/${sub}/dwi/${sub}_dwi.nii.gz
bval=/test/mri/${megtensity}/${sub}/dwi/${sub}_dwi.bval
bvec=/test/mri/${megtensity}/${sub}/dwi/${sub}_dwi.bvec

# not used maybe
nodifmask=/project_subjects/${megtensity}/${sub}/DTI/nodif_brain_mask.nii.gz
bedpostdir=/project_subjects/${megtensity}/${sub}/DTI.bedpostX
FA=/project_subjects/${megtensity}/${sub}/DTI/dti_FA.nii.gz
surfacedir=/project_subjects/${megtensity}/${sub}/${sub}/fsaverage_LR32k
surfname=${sub}


mkdir -p ${subdir}/tractseg_output


# model configuration
preprocessdir=${soft}/pipline


# step 1-2
python $soft/pipline/parcellation.py -s ${subdir} --t1 ${t1} --t1_preprocessed  --dti ${dti} --bval ${bval} --bvec ${bvec}  --ANTSreg --tract_MNIspace --MSMAll False --begin_step 1 --end_step 2

# step 3
bash ${soft}/pipline/FS2WB ${pardir} ${sub}
python $soft/pipline/parcellation.py -s ${subdir} --t1 ${t1} --t1_preprocessed  --dti_preprocessed --dti ${dti} --bval ${bval} --bvec ${bvec} --nodif_brain_mask ${nodifmask} --ANTSreg --tract_MNIspace --MSMALL False --begin_step 3 --end_step 3

# step 4
dtipath=${subdir}/DTI
bash ${preprocessdir}/bedpostx_gpu_local.sh ${dtipath}

#step 5
MNI=False
bash ${preprocessdir}/fibertract.sh ${subdir} ${dti} ${bval} ${bvec} ${nodifmask} ${MNI} 3
flirt -ref flirt -ref $subdir/DTI/LowResMask.nii.gz -in $subdir/tractseg_output/bundle_segmentations.nii.gz -o $subdir/DTI/LowRes_Fibers.nii.gz -applyisoxfm 3 -interp nearestneighbour 

# step 6-7 probtracking and fingerprint
for hemi in L R ;
do
    echo ${hemi}

    output_file=${subdir}/${sub}_${hemi}_probtrackx_omatrix2

    # choice 1： set GPU card version
    card=3
    bash ${preprocessdir}/probtrack_hemi_gpu.sh ${subdir} ${t1} ${bedpostdir} ${hemi} ${output_file} True ${card}
    
    # choice 2: sbtach version
    # bash ${preprocessdir}/probtrack_hemi_gpu2.sh ${subdir} ${t1} ${bedpostdir} ${hemi} ${output_file} True

    python ${preprocessdir}/S07_postprobtrack.py -s ${subdir} -p ${hemi} -s ${select_ROI}
done

# step 8
for hemi in L ;
do
    for atlas in Yeo7Network ; # Yeo17Network HCPparcellation ;
    do
        echo ${atlas}  ${hemi}
        modeldir=${soft}/pipline
        export CUDA_VISIBLE_DEVICES=1
        # if atlas=='BN':
        # python ${modeldir}/inference.py -subdir ${subdir} -hemisphere ${hemi} -atlas ${atlas} -MSMAll 'False' -hidden1 64
        # else:
        # python ${modeldir}/inference.py -subdir ${subdir} -hemisphere ${hemi} -atlas ${atlas} -MSMAll 'False'
        # cd ${modeldir}
        # python ${modeldir}/inference.py -subdir ${subdir} -hemisphere ${hemi} -atlas ${atlas} -MSMAll 'False'
        
        savepath=${subdir}
        string='False'
        modeldir=${soft}/GCN_model_pytorch/model/
        cd ${modeldir}
        python ${modeldir}/validation_test.py -modeldir ${modeldir} -subdir ${subdir} -hemisphere ${hemisphere} -MSMAll ${sign}
    done
done

