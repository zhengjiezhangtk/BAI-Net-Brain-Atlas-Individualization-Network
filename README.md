# BAI-Net: Individualized Anatomical Cerebral Cartography using Graph Neural Network

by: Ma Liang at 2020/4/30

## Introduction
Each functional region has its specific anatomical basis. The BAI-Net model utilizes fiber-density fingerprints as anatomical reference of individualized brain parcellation, and integrate the graph neural network to locate function networks. Specifically, the fiber-density fingerprints are first extracted using the individual brain tractography and whole-brain main fiber tracts.  The calculation of fiber-density fingerprint for an individual is shown below:

$F = Norm(T ×F )$

$T ∈R^{N_v×N_b}$ is the tractography matrix for a subject, indicating the connection densities between cortical vertices and brain areas. $N_v$ represents the number of vertices in the cortical surface of the $S$ subject and $N_bs$ the number of brain voxels of the $S$ subject. The number of brain voxels $N_b$ is different across subjects. $F∈R^{N_b×N_F}$ represents fiber-tract masks with value of 1 or 0 indicating the existence of each fiber tract. In this study, fiber tract masks are extracted individually using [TractSeg](https://github.com/MIC-DKFZ/TractSeg), and the number of fiber tracts $N_F$ is 72. After the multiplication of two matrix, fiber-density fingerprint is obtained and then a normalization will be applied on this matrix to ensure the sum of fiber-density from a vertex is equal to 1.

Then the BAI-Net method trains a Chebyshev graph neural network to achieve the robust detection of the anatomical reference of each function network on population and inference their cortical location individually. The structure of this network consists of two stacked Chebyshev graph convolutional layers with the input of individual tract-density fingerprint ($F$) and normalized graph Laplacian $(L)$ calculated from adjacent matrix which is weighted by the inverse of the geometric distance between near six vertices. 
The pipeline use the fiber connection fingerprint derived from group brainnetome atlas to parcellate the individual cerebral cortex. This pipline is trained on the 32kLR surface, could also be applied into other HCP-styled surface (.surf.gii). Currently, The BAI-Net support pytorch packages to inference individual brain networks or parcellation. Suppoted brain network includes Yeo's 7 networks, 17 networks. Supported brain parcellation on surface includes: Brainnetome Atlas, HCP-parcellation. 
![Fiber-density framework](figure1.png)
![parcellation visualization](figure3.png)
The parcellation result shows high reproducibility within the same subjects, high inter-subject specifity, high robustness across multi-scanner image, and is enable to parcellate brain locally but maintains high similarity with whole-brain parcellation.
![High reproducibility](figure2.png)



## Requirement
```
the pipline dependency: 
    Python----> pytorch >=2.0
          ----> torch_geometric >= 2.6.0
          ----> TractSeg >=2.1
    FSL >= 5.0
    FreeSurfer >= 5.3
    workbench 
```
```
the TractSeg dependency:
    Python---> pytorch >= 1.0 
    FSL
    Mrtrix 3
    ANTS
```
Before implementation, you should revise the software path on the pipline/soft_path.sh



## Model Pipline:

the pipline contain 8 steps, corresponding to the file begin with 'S'. Each step file can be run sepretely aftering running the previous steps. The time cost of pipline is around 12 hours for two GPUs to work. 
```
S01_miniprocess.py           ( CPU )
S02_registration.py          ( CPU )
S03_build_surface.py         ( CPU, long time-consumption ) 
S04_fiber_orientation.py     ( GPU )
S05_fiber_tract.py           ( CPU or GPU)
S06_probtrack.py             ( GPU)
S07_postprobtrack.py         ( CPU )
S08 in folder ../GCN_model   ( CPU or GPU )
```
It is suggested that run the pipline of **'parcellation.py'** in the pipeline folder that has been parllatally structured. The file path of pipeline is record in **'{sub_name}_config.ini'** for the checking, and it will update every steps. The final parcellation result is  {sub_name}.indi.R.label.gii and {sub_name}.indi.R.label.gii 

## Some Examples of Running Pipline:
we also includes some run_scripts and results in the example/ folder.

## docker
docker pull zhengjiezhangtk/bainet-centos7:v2
（
docker load -i bainet-centos7-v2.tar
docker run -it bainet-centos7 /bin/bash
（docker run -it -v ~/BAInet:/mnt/host:ro bainet-centos7:v2 /bin/bash）
docker commit b66a701c223d bainet-centos7:v2
docker save -o bainet-centos7-v2.tar bainet-centos7:v2
）

## detail of the pipline
```
S01_miniprocess.py: first step. You should make sure subdir/3D & subdir/DTI are existed.(mkdir -p subdir/3D subdir/DTI)
python S01_miniprocess.py -s /mnt/host/result1 \
     --t1 /mnt/host/HCP_data/HCP_Retest/103818/MNINonLinear/T1w.nii.gz \
     --dti /mnt/host/HCP_data/HCP_Retest/103818/T1w/Diffusion/data.nii.gz \
     --bval /mnt/host/HCP_data/HCP_Retest/103818/T1w/Diffusion/bvals \
     --bvec /mnt/host/HCP_data/HCP_Retest/103818/T1w/Diffusion/bvecs \
     --dti_preprocessed 1 \
     --nodif_brain_mask /mnt/host/HCP_data/HCP_Retest/103818/T1w/Diffusion/nodif_brain_mask.nii.gz

S02_registration.py: second step. You should make sure subdir/xfms is existed.
python S02_registration.py -s /mnt/host/result1 \

S03_build_surface.py: third step. You should make sure subdir/fsaverage_LR32k & subdir/surf is existed.
and less the L.atlasroi.label.gii.
python S03_build_surface.py -s /mnt/host/result1 --fsaverage_LR32k /mnt/host/HCP_data/HCP_Retest/103818/MNINonLinear/fsaverage_LR32k --surface_begin_name 103818

S04_build_atlas.py: fourth step. 
python S04_fiber_orientation.py -s /mnt/host/result1
for bedpostx

S05_fiber_tract.py: fifth step. 
python S05_fiber_tract.py -s /mnt/host/result1
make sure you have tract_pretrain_parm in /root/.tract And change the soft_path.sh

S06_probtrack.py: sixth step. 
python S06_probtrack.py -s /mnt/host/result1

S07_postprobtrack.py: seventh step.
python S07_postprobtrack.py -s /mnt/host/result1 -p L
Because I run this code in my local computer. the computer's cpu is lower than sever. So I changed the method to read martrixs.

S08_parcellation.py: eighth step.
python S08_gcn_inference.py -s /mnt/host/result1 --savepath /mnt/host/result1/output_result
```



## Publications
L. Ma et al., "BAI-Net: Individualized Anatomical Cerebral Cartography Using Graph Neural Network," in IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 6, pp. 7446-7457, June 2024, doi: 10.1109/TNNLS.2022.3213581.

