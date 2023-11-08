#!/bin/bash
softwaredir=/n04dat01/atlas_group/lma/DP_MDD_dataset
FSLDIR=/share/soft/fsl-6.0.4
ANTSDIR=/share/soft/ants
FREESURFER_HOME=/share/soft/freesurfer6.0
MRTRIX3=/share/soft/mrtrix3
WORKBENCH=/share/soft/workbench
ANACONDA=/n14dat01/lma/envs/anaconda3


export PATH=$WORKBENCH/bin_rh_linux64:$MRTRIX3/release/bin:$MRTRIX3/scripts:$ANACONDA/bin:$ANTSDIR:$PATH
export PATH=$FREESURFER_HOME/tktools:$FREESURFER_HOME/fsfast/bin:$FREESURFER_HOME/mni/bin:$FSLDIR/bin:$PATH
export LD_LIBRARY_PATH=$WORKBENCH/lib_rh_linux64:$ANACONDA/lib:$LD_LIBRARY_PATH
export PATH=$softwaredir/pipline:$PATH

FS_FREESURFERENV_NO_OUTPUT=1
source $FREESURFER_HOME/SetUpFreeSurfer.sh

. $ANACONDA/etc/profile.d/conda.sh
. $FSLDIR/etc/fslconf/fsl.sh
