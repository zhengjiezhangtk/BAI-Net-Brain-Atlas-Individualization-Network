#!/bin/bash
surf_dir=$1
surf_name=$2
subdir=$3
ATLAS_DIR=$4
MSMAll=$5

echo $surf_dir $surf_name 
echo $subdir $ATLAS_DIR
echo MSMAll $5
## Loop through left and right hemispheres
for Hemisphere in  L R ; do
	## convert white pial gii to ASCII
	surf2surf -i  $surf_dir/$surf_name.${Hemisphere}.white${MSMAll}.32k_fs_LR.surf.gii -o $subdir/surf/white.${Hemisphere}.asc --outputtype=ASCII --values=$ATLAS_DIR/${Hemisphere}.atlasroi.32k_fs_LR.shape.gii
	surf2surf -i  $surf_dir/$surf_name.${Hemisphere}.pial${MSMAll}.32k_fs_LR.surf.gii -o $subdir/surf/pial.${Hemisphere}.asc --outputtype=ASCII --values=$ATLAS_DIR/${Hemisphere}.atlasroi.32k_fs_LR.shape.gii
done	

## create targets for probtrack
echo "$subdir/surf/pial.L.asc" > $subdir/surf/stop
echo "$subdir/surf/pial.R.asc" >> $subdir/surf/stop

echo "$subdir/surf/white.L.asc" > $subdir/surf/wtstop
echo "$subdir/surf/white.R.asc" >> $subdir/surf/wtstop