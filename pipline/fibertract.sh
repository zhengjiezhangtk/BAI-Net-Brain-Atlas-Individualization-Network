#!/bin/bash

subdir=$1
dti=$2
bval=$3
bvec=$4
mask=$5
MNI=$6
soft=$7
card=$8

export CUDA_VISIBLE_DEVICES=$card
# . ${soft}/pipline/soft_path.sh
# which mrconvert

if [ ! -d $subdir/DTI ] ; then
   mkdir -p $subdir/DTI
fi


if [ $MNI = "False" ] ; then
	echo 'MNI' $MNI 
  if [ ! -f $subdir/tractseg_output/bundle_segmentations.nii.gz ];then
    TractSeg -i $dti -o $subdir/tractseg_output/ \
    --bvals $bval --bvecs $bvec --brain_mask $mask\
    --raw_diffusion_input --single_output_file
    flirt -ref $subdir/DTI/LowResMask.nii.gz -in $subdir/tractseg_output/bundle_segmentations.nii.gz -o $subdir/DTI/LowRes_Fibers.nii.gz -applyisoxfm 3 -interp nearestneighbour
  fi
elif [ $MNI = "True" ] ; then
  echo  'MNI' $MNI 
  if [ ! -f $subdir/tractseg_output/bundle_segmentations.nii.gz ];then

    rm -rf $subdir/tractseg_output
    mkdir -p $subdir/tractseg_output

    TractSeg -i $dti -o $subdir/tractseg_output/ \
    --bvals $bval --bvecs $bvec --brain_mask $mask\
    --raw_diffusion_input --preprocess --single_output_file
    
    echo 'mask registration'
    #rm $subdir/tractseg_output/Diffusion_MNI.nii.gz
    #flirt -ref $subdir/DTI/dti_FA.nii.gz -in $subdir/tractseg_output/FA_MNI.nii.gz -omat $subdir/xfms/MNI_2_DTI.mat
    #flirt -in $subdir/tractseg_output/bundle_segmentations.nii.gz -ref $subdir/DTI/dti_FA.nii.gz -out $subdir/tractseg_output/bundle_segmentations_tmp.nii.gz -init $subdir/xfms/MNI_2_DTI.mat -applyxfm -interp nearestneighbour 
    flirt -ref $subdir/tractseg_output/bundle_segmentations.nii.gz -in $subdir/tractseg_output/bundle_segmentations.nii.gz -o $subdir/DTI/LowRes_Fibers.nii.gz -applyisoxfm 3 -interp nearestneighbour
    # rm $subdir/tractseg_output/bundle_segmentations_MNI.nii.gz
  fi

fi





