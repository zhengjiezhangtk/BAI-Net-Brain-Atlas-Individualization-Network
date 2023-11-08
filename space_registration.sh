#!/bin/bash

subdir=$1
FA=$2
T1=$3

flirt -ref ${T1} -in ${FA}  -omat ${subdir}/xfms/DTI_2_T1_1mm.mat
convert_xfm  -inverse ${subdir}/xfms/DTI_2_T1_1mm.mat -omat ${subdir}/xfms/T1_1mm_2_DTI.mat
