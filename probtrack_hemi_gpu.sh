#!/bin/bash

subdir=$1
t1=$2
bedpost_dir=$3
Hemisphere=$4
output_file=$5
ANTS=$6
card=$7
# echo cuda $CUDA_VISIBEL_DEVICES
export CUDA_VISIBLE_DEVICES=$card


echo $ANTS

if [ ! -f $output_file/fdt_matrix2.dot ] && [ ! -f $output_file/fdt_matrix2.npz ] ; then

    if [ $ANTS == False  ] ;  then
        seeds=white.${Hemisphere}.asc 
        seedref=$t1
        echo 'fsl_mode'
        echo $seedref
        echo $subdir/surf/${seeds}
        probtrackx2_gpu --samples=$bedpost_dir/merged \
        --mask=$bedpost_dir/nodif_brain_mask \
        --xfm=$subdir/xfms/T1_1mm_2_DTI.mat \
        --seedref=$seedref \
        -P 5000 --loopcheck --forcedir -c 0.2 --sampvox=2 --randfib=1 \
        --stop=$subdir/surf/stop --forcefirststep  \
        -x $subdir/surf/${seeds} \
        --omatrix2  --target2=$subdir/DTI/LowResMask   --wtstop=$subdir/surf/wtstop\
        --dir=$output_file --opd -o ${Hemisphere}
    elif [ $ANTS == True ] ; then
        seeds=white_DTI_${Hemisphere}.asc 
        seedref=$subdir/DTI/dti_FA.nii.gz
        echo 'ants_mode'
        echo $seedref
        echo $subdir/surf/${seeds}
        probtrackx2_gpu --samples=$bedpost_dir/merged \
        --mask=$bedpost_dir/nodif_brain_mask \
        --xfm=/n04dat01/atlas_group/lma/DP_MDD_dataset/pipline/eye.mat \
        --seedref=$seedref \
        -P 5000 --loopcheck --forcedir -c 0.2 --sampvox=2 --randfib=1 \
        --stop=$subdir/surf/stop_ants --forcefirststep  \
        -x $subdir/surf/${seeds} \
        --omatrix2  --target2=$subdir/DTI/LowResMask  --wtstop=$subdir/surf/wtstop_ants \
        --dir=$output_file --opd -o ${Hemisphere} 
    else 
        echo 'error in probtracking'  
    fi
fi
