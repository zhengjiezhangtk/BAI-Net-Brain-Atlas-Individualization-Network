#!/bin/bash
dti_data=$1
subdir=$2
bvec=$3
bval=$4
preprocessdir=$5

echo  $dti_data $bvec $bval

if [ ! -f ${subdir}/DTI/nodif_brain_mask.nii.gz  ] ; then
 echo correct DTI
 eddy_correct $dti_data ${subdir}/DTI/data.nii.gz  0
 fslroi ${subdir}/DTI/data.nii.gz ${subdir}/DTI/nodif 0 1
 bet ${subdir}/DTI/nodif.nii.gz ${subdir}/DTI/nodif_brain -f 0.3 -m -R
 dtifit -k ${subdir}/DTI/data.nii.gz -m ${subdir}/DTI/nodif_brain_mask.nii.gz -r ${subdir}/DTI/bvecs -b ${subdir}/DTI/bvals -o ${subdir}/DTI/dti
fi

if [ ! -f  ${subdir}/DTI/bvecs ]; then
cp $bvec ${subdir}/DTI/bvecs
fi

if [ ! -f  ${subdir}/DTI/bvals ]; then
cp $bval ${subdir}/DTI/bvals
fi
