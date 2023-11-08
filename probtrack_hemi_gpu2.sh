#!/bin/bash

subdir=$1
t1=$2
bedpost_dir=$3
Hemisphere=$4
output_file=$5
card=$6
# echo cuda $CUDA_VISIBEL_DEVICES
# export CUDA_VISIBLE_DEVICES=$card


seeds=white.${Hemisphere}.asc 
seedref=$t1

echo seed $seeds
echo seedref $seedref 
#/DATA/232/lma/envs/fsl/bin/
if [ ! -f $output_file/fdt_matrix2.dot ] && [ ! -f $output_file/fdt_matrix2.npz ] ; then
    echo $subdir/surf/${seeds}
    if [ -f $subdir/surf/${seeds} ] ;  then
        /DATA/232/lma/envs/fsl_6.0/fsl/bin/probtrackx2_gpu --samples=$bedpost_dir/merged \
        --mask=$bedpost_dir/nodif_brain_mask \
        --xfm=$subdir/xfms/T1_1mm_2_DTI.mat \
        --seedref=$seedref \
        -P 5000 --loopcheck --forcedir -c 0.2 --sampvox=2 --randfib=1 \
        --stop=$subdir/surf/stop --forcefirststep  \
        -x $subdir/surf/${seeds} \
        --omatrix2  --target2=$subdir/951457_mask_2mm   --wtstop=$subdir/surf/wtstop\
        --dir=$output_file --opd -o ${Hemisphere}
    fi
fi
