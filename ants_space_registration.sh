#!/bin/bash

subdir=$1
FA=$2
T1=$3
antspath=$4
# export ANTSPATH=/share/soft/ants
echo $ANTSPATH, $FA, $T1
if [ ! -f ${subdir}/DTI/ants_t10GenericAffine.mat ] ; then 
    mkdir -p ${subdir}/xfms
    bash ${antspath}/antsRegistrationSyN.sh -d 3 -f ${FA} -m ${T1} -o ${subdir}/DTI/ants_t1 -t r
    bet ${subdir}/DTI/ants_t1Warped.nii.gz ${subdir}/DTI/ants_t1 -f 0.3 -m -R
    fslmaths ${subdir}/DTI/nodif_brain_mask.nii.gz -add ${subdir}/DTI/ants_t1_mask.nii.gz -bin ${subdir}/DTI/nodif_brain_mask.nii.gz
    # flirt -in ${subdir}/DTI/ants_t1Warped.nii.gz -ref ${T1} -omat ${subdir}/xfms/DTI_2_T1_1mm.mat -cost leastsq
    # convert_xfm  -inverse ${subdir}/xfms/DTI_2_T1_1mm.mat -omat ${subdir}/xfms/T1_1mm_2_DTI.mat
fi
