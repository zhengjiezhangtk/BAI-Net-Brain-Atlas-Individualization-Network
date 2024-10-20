
## Data Structure

We provide a flexible data structure for data storage, that data pre-processed in different locations can be directly calculated. If the data is not pre-processed in any location, its complete date storage format of final result is below:

```
---workdir--------------------------------
     |
     |----{$sub_name}--------------------------------
     |      |
     |      |------3D--------------------------------
     |      |       |---T1w.nii.gz
     |      |       |---T1_1mm.nii.gz
     |      |      
     |      |------DTI--------------------------------
     |      |       |---data.nii.gz
     |      |       |---bval
     |      |       |---bvec
     |      |       |---nodif_brain_mask.nii.gz          
     |      |       |---dti_FA.nii.gz
     |      |       |---LowRes_Fibers.nii.gz
     |      |       |---LowResMask.nii.gz
     |      |
     |      |------xfms-------------------------------------
     |      |       |---DTI_2_T1_1mm.mat
     |      |       |---T1_1mm_2_DTI.mat
     |      |     
     |      |------{$sub_name}-------------------------------
     |      |       |---fsaverage_LR32k
     |      |                 |---{$sub_name}.R.white.32k_fs_LR.surf.gii
     |      |                 |---{$sub_name}.L.white.32k_fs_LR.surf.gii
     |      |                 |---{$sub_name}.R.pial.32k_fs_LR.surf.gii
     |      |                 |---{$sub_name}.L.pial.32k_fs_LR.surf.gii             
     |      |     
     |      |------surf--------------------------------     
     |      |       |---stop
     |      |       |---wtstop
     |      |       |---white.R.asc
     |      |       |---white.L.asc
     |      |       |---pial.R.asc
     |      |       |---pial.L.asc
     |      |       |---adj_matrix_seed_R.npz
     |      |       |---adj_matrix_seed_L.npz
     |      |
     |      |------DTI.bedpostX--------------------------------
     |      |       |---dyads1.nii.gz
     |      |       |---dyads2.nii.gz
     |      |       |---dyads3.nii.gz 
     |      |    
     |      |------tractseg_output--------------------------------
     |      |       |---bundle_segmentations.nii.gz
     |      |
     |      |------{$sub_name}_R_probtrackx_omatrix2--------------------------------
     |      |       |---fdt_matrix2.npz
     |      |       |---finger_print_fiber.npz
     |      |
     |      |------{$sub_name}_R_probtrackx_omatrix2--------------------------------
     |      |       |---fdt_matrix2.npz
     |      |       |---finger_print_fiber.npz
     |      |
     |      |------{$sub_name}_config.ini
     |      |------{$sub_name}.indi.R.label.gii
     |      |------{$sub_name}.indi.L.label.gii
     |      |
     |----{$sub_name}--------------------------------
     .......
     |----{$sub_name}--------------------------------
     .......
```