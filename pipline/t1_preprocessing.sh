#!/bin/bash
t1_data=$1
subdir=$2


if [ ! -f ${subdir}/3D/T1_brain.nii.gz  ] ; then
 echo T1 to 1mm, $subdir
 flirt -ref ${t1_data}  -in ${t1_data}  -o ${subdir}/3D/T1_1mm.nii.gz  -applyisoxfm 1 -interp nearestneighbour
 robustfov -i ${subdir}/3D/T1_1mm.nii.gz -r ${subdir}/3D/T1_crop.nii.gz
 bet ${subdir}/3D/T1_crop.nii ${subdir}/3D/T1_brain.nii.gz -f 0.3 -R -m
 
fi
