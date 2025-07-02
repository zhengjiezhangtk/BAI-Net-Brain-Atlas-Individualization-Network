#!/bin/bash
# the root directory for BAI_Net pipline
softwaredir=/mnt/host/BAI-Net-Brain-Atlas-Individualization-Network-master

FSLDIR=/opt/fsl/bin/fsl

ANTSDIR=/opt/ants-2.5.2/bin

FREESURFER_HOME=/opt/freesurfer

MRTRIX3=/opt/conda/envs/BAInetenv/bin

WORKBENCH=/opt/workbench

ANACONDA=/opt/conda

export PATH=$WORKBENCH/bin_rh_linux64:$MRTRIX3/release/bin:$MRTRIX3/scripts:$ANTSDIR:$PATH
export PATH=$FREESURFER_HOME/tktools:$FREESURFER_HOME/fsfast/bin:$FREESURFER_HOME/mni/bin:$FSLDIR/bin:$PATH
export LD_LIBRARY_PATH=$WORKBENCH/lib_rh_linux64:$LD_LIBRARY_PATH
export PATH=$softwaredir/pipline:$PATH

FS_FREESURFERENV_NO_OUTPUT=1
source $FREESURFER_HOME/SetUpFreeSurfer.sh

. $ANACONDA/etc/profile.d/conda.sh
# conda activate pytorch_env
. $FSLDIR/etc/fslconf/fsl.sh
