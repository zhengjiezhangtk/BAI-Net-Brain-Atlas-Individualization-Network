import numpy as np 
from pathlib import Path
import nibabel as nib
import argparse
from nilearn import surface
import os

def MNI2Grid(affine, MNIList):
    grid_list = []
    label_list = []
    length = len(MNIList)
    MNI_coord = np.concatenate([MNIList, np.ones([length,1])],axis=1)
    grid_coord = MNI_coord.dot(np.linalg.inv(affine.T))      
    grid_coord = grid_coord.round(0).astype('int')    
    return grid_coord[:,:3]

def points_within_areas(vertex, triangles, labels, new):
    for area in triangles:
        if not (labels[area[0]] == labels[area[1]] and labels[area[1]] == labels[area[2]] ):
            continue
        arealabel = labels[area[0]]
        pointbase = np.array([vertex[area[0]], vertex[area[1]], vertex[area[2]] ])
        # max_dis = max(((pointbase[1]-pointbase[0])**2).sum(),((pointbase[2]-pointbase[0])**2).sum(),((pointbase[1]-pointbase[2])**2).sum())
        # max_dis = np.sqrt(max_dis)
        # transarray = produce_array(max_dis)
        # transarray = np.array([[0.167,0.167,0.66], [0.66,0.167,0.167], [0.167, 0.66, 0.167]])
        transarray = np.array([ [0, 0.25, 0.75],
                                [0, 0.5, 0.5],
                                [0.125, 0.125, 0.75],
                                [0.125, 0.75, 0.125],
                                [0.125, 0.5, 0.375],
                                [0.25, 0, 0.75],
                                [0.25, 0.25, 0.5],
                                [0.25, 0.5, 0.25],
                                [0.25, 0.75, 0],
                                [0.5, 0, 0.5],
                                [0.5, 0.25, 0.25],
                                [0.5, 0.5, 0],
                                [0.75, 0.125, 0.125],
                                [0.333, 0.333, 0.333]])
        areapoint = (transarray.dot(pointbase)).round(0).astype('int')
        new[(areapoint[:, 0], areapoint[:, 1], areapoint[:, 2])] = [arealabel] * len(transarray)
    return new


def main( SubjectID, SubjectDir, SurfaceDir, SavePath ):
    """
    required : label, T1w and fsaverage_LR32k
    """
    sub_surface = {}

    # individual label
    for hemi in ['R','L']:
        sub_surface['label'+hemi] = surface.load_surf_data('{}/{}.indi.{}.label.gii'.format(SubjectDir, SubjectID, hemi))
    tmp  = sub_surface['labelR']
    # tmp[tmp>0] = tmp[tmp>0]+180
    sub_surface['labelR'] = tmp
    T1_file = nib.load('{}/3D/T1_1mm.nii.gz'.format(SubjectDir))
    T1_data = T1_file.get_data()
    affine = T1_file.affine
    new = np.zeros_like(T1_data)

    # pia_surface, mid_surface, white_surface, coord_trans
    for hemi in ['R','L']:
        for stat in ['pial', 'white']:
            tmp = surface.load_surf_data('{}/{}.{}.{}.32k_fs_LR.surf.gii'.format( SurfaceDir,SubjectID,hemi,stat))
            grid = MNI2Grid(affine, tmp[0])
            area_structure = tmp[1]
            sub_surface[stat+hemi] = grid
            new[(grid[:, 0], grid[:, 1], grid[:, 2])] = sub_surface['label'+ hemi]
            new = points_within_areas( grid, area_structure, sub_surface['label'+hemi], new)
        sub_surface['max_dis_'+hemi] = np.sqrt(((sub_surface['pial'+hemi]-sub_surface['white'+hemi])**2).sum(axis=1).max())
        print( 'Max_Distence: ',sub_surface['max_dis_'+hemi] )
    # 
    
    # resample the vortex in the line.
    for hemi in ['R','L']:
        sample_number = int(sub_surface['max_dis_' + hemi])
        if sample_number <= 2:
            continue
        resample_inteval = (sub_surface['white'+hemi] - sub_surface['pial' + hemi]) / (sample_number + 1)
        for i in range(sample_number):
            print('Sruface Resample:', i, 'Hemi: ', hemi)
            resample_grid = (sub_surface['pial'+hemi] + resample_inteval * i).round(0).astype('int')
            new[(resample_grid[:, 0], resample_grid[:, 1], resample_grid[:, 2])] = sub_surface['label'+ hemi]
            new = points_within_areas( resample_grid, area_structure, sub_surface['label'+hemi], new)
    def fill_hole(new):
        new2 = np.zeros_like(new)
        #fill the holes
        for i in range(1,new.shape[0]-1):
            for j in range(1,new.shape[1]-1):
                for k in range(1,new.shape[2]-1):
                    cube = new[i-1:i+2,j-1:j+2,k-1:k+2].flatten()
                    new2[i,j,k] = new[i,j,k]
                    # print((cube>0).sum())
                    if ((cube>0).sum() >= 24) and (new[i,j,k]==0):
                        print('fill',i,j,k)
                        label = list(set(cube))
                        flat = list(cube)
                        ll, maxcounts = 0, 0
                        for tt in label:
                            counts = flat.count(tt)
                            if counts>maxcounts:
                                ll = tt
                        new2[i,j,k] = ll
        return new2
    new = fill_hole(new)
    new = fill_hole(new)
    
    #save_image
    NewImage = nib.Nifti1Image(new, affine = T1_file.affine, header = T1_file.header)
    nib.save(NewImage, SavePath)

    return None



if __name__ == "__main__":
    
    setting parameters
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True, type=str, metavar="SujectID")
    ap.add_argument("--sub_dir", required=True, type=float, metavar="SubjectDir")
    ap.add_argument("--surface_dir", required=True, type=float, metavar="SurfaceDir")
    ap.add_argument("--save_path", required=True, type=str, default='', metavar="save_path")
    args = vars(ap.parse_args())

    main(SubjectID=args['s'], SubjectDir=args['sub_dir'], SurfaceDIR=args['surface_dir'], SavePath= args['save_path'])
