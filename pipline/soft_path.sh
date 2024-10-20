#!/bin/bash
# the root directory for BAI_Net pipline
softwaredir=D:/Code_collect/BAI_Net_new/BAI_Net

FSLDIR=/share/soft/fsl-6.0.4

ANTSDIR=/share/soft/ants

FREESURFER_HOME=/share/soft/freesurfer6.0

MRTRIX3=/share/soft/mrtrix3

WORKBENCH=/share/soft/workbench

ANACONDA=/share/soft/anaconda3

export PATH=$WORKBENCH/bin_rh_linux64:$MRTRIX3/release/bin:$MRTRIX3/scripts:$ANTSDIR:$PATH
export PATH=$FREESURFER_HOME/tktools:$FREESURFER_HOME/fsfast/bin:$FREESURFER_HOME/mni/bin:$FSLDIR/bin:$PATH
export LD_LIBRARY_PATH=$WORKBENCH/lib_rh_linux64:$LD_LIBRARY_PATH
export PATH=$softwaredir/pipline:$PATH

FS_FREESURFERENV_NO_OUTPUT=1
source $FREESURFER_HOME/SetUpFreeSurfer.sh

. $ANACONDA/etc/profile.d/conda.sh
# conda activate pytorch_env
. $FSLDIR/etc/fslconf/fsl.sh
