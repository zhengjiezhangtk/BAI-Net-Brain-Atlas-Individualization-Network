# coding=utf-8
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import scipy.sparse as sparse
from pathlib import Path
import time
import nibabel as nib
import os
import argparse
from sklearn.preprocessing import normalize
# from nibabel.cifti2 import cifti2


def read_coomat(file):
    with open(str(file), 'r') as f:
        mat = [line.rstrip().split('  ') for line in f]
    print('finish loading')
    i = [int(l[0])-1 for l in mat]
    j = [int(l[1])-1 for l in mat]
    v = [int(l[2]) for l in mat]
    print('i,j,v got')
    return i, j, v


# def readParcel(file):
#     pconn = cifti2.load(file)
#     header = pconn.header
#     res = header.matrix.get_index_map(1)
#     res = list(res.parcels)
#     f = Path('/DATA/234/htzhang/DATA/HCP_parcellation/template/adj_tabel.txt')
#     with f.open('w') as f:
#         for i, parcel in enumerate(res):
#             f.write(f'{i+1}\n')
#             f.write(f'{parcel.name}\n')


def compress_sparse(file):
    sub = str(file).split('/')[-2].split('_')[0]
    if file.exists():
        if  ~(file.parent/'fdt_matrix2.npz').exists():
            # print(f'convert fdt matrix for {sub}')
            i, j, v = read_coomat(file)
            coo_mat = sparse.csr_matrix((v, (i,j)))
            assert coo_mat.shape == (i[-1]+1, j[-1]+1)
            sparse.save_npz(str(file.parent/'fdt_matrix2'), coo_mat)
        print(f'remove {str(file)}')
        os.remove(str(file))
        return sub+' is finished'
    else:
        return sub + ' is already done'


# def compress_sparse_name(file, name):
#     # sub = str(file).split('/')[-2].split('_')[0]
#     file = Path(file)
#     if file.exists():
#         if  ~(file.parent/(name+'.npz')).exists():

#             i, j, v = read_coomat(file)
#             coo_mat = sparse.csr_matrix((v, (i,j)))
#             assert coo_mat.shape == (i[-1]+1, j[-1]+1)
#             sparse.save_npz(str(file.parent/name), coo_mat)
#         print(f'remove {str(file)}')
#         os.remove(str(file))
#         return ' finished'
#     else:
#         return ' already done'


def PostProbtrack(work_dir,sub,hemi):
    print('postprobtract:', work_dir, sub, hemi)
    work_dir = Path(work_dir)
    file = work_dir/sub/('{}_{}_probtrackx_omatrix2'.format(sub,hemi))/'fdt_matrix2.dot'
    fdt_path = file.parent
    if file.exists() and (not (fdt_path/'fdt_matrix2.npz').exists()) :
        print('convert to sparse for {} hemi: {}'.format(sub,hemi))
        compress_sparse(file)
    else:
        print(sub+' is already in sprase format or dot file does not exist',file )
    return None



def label2target(target_coords, atlas, sub):
    labels = []
    img = nib.load(str(atlas)).get_fdata()
    roi_size = [np.sum(img==i) for i in range(247)]
    print('min area size: ',min(roi_size),'max area size: ',max(roi_size))
    with open(str(target_coords), 'r') as f:
        for line in f:
            x, y, z = [int(i) for i in line.strip().split('  ')]
            labels.append(img[x, y, z])
    assert np.unique(labels).shape[0] == 247, f'{sub} label does not consist of 247 units'
    labels = np.array(labels)
    onehot_encoder = LabelBinarizer()
    onehot_encoder.fit(list(range(247)))
    mat = sparse.csr_matrix(onehot_encoder.transform(labels))
    return mat[:,1:],roi_size[1:]


def getFingerPrint(conn_features, vertex_labels):
    fp = conn_features.dot(vertex_labels)
    return sparse.csr_matrix(fp)


def normFp(finger_print):
    return normalize(finger_print, norm='l1', axis=1)

def normal(fp,roi_size):
    for i in range(len(roi_size)):
        fp[:,i] = fp[:,i]/roi_size[i]
    return fp

# def get_prior_area_fingerprint(workpath, sub, hemi, recreation):
#     #get finger_print from .npz file
#     norm_with_size = False
#     target_file = 'finger_print_area.npz'
#     workpath = Path(workpath)
#     # for hemi in tqdm.tqdm(['L','R']):
#     file = workpath/sub/'{}_{}_probtrackx_omatrix2'.format(sub, hemi)/'fdt_matrix2.npz'
#     fdt_path = file.parent
#     if file.exists() and (not (fdt_path/target_file).exists() or recreation):
#         sps_mat = sparse.load_npz(str(file))
#         n_seeds, n_targets = sps_mat.shape
#         print(f'get area fingerprint from {file}')
#         target_coords = fdt_path/'tract_space_coords_for_fdt_matrix2'
#         atlas = workpath/sub/'DTI/LowRes_Atlas.nii.gz'
#         res,roi_size = label2target(target_coords, atlas, sub)
#         assert (res.shape == (n_targets, 246)), f'shape: {res.shape}, got wrong fingerprint'
#         fp = getFingerPrint(sps_mat, res)
#         if norm_with_size:
#             fp = normal(fp,roi_size)
#         sparse.save_npz(str(fdt_path/target_file), fp)
#     else:
#         print("sparse matrix not exists or finger print already exists")
#     return None


def fiber2target(target_coords, fiber, sub):
    labels = []
    img = nib.load(str(fiber)).get_data()
    roi_size = np.array([img[:,:,:,i].sum() for i in range(72)])
    print('min fiber size: ', min(roi_size),'max fiber size: ', max(roi_size))
    mat = []
    with open(str(target_coords), 'r') as f:
        for i, line in enumerate(f):
            try:
                x, y, z = [int(j) for j in line.strip().split('  ')]
                mat.append(img[x, y, z, :].astype('int'))
            except:
                # print('________!!!!!!!!!_______')
                mat.append(np.zeros_like(img[0,0,0,:],dtype='int'))
    mat = sparse.csr_matrix(np.array(mat))
    return mat, roi_size



def shape_fit(fiber_img, mask_img):
    lx, ly, lz = mask_img.shape
    fx, fy, fz, _ = fiber_img.shape
    fiber_img_new = np.zeros([lx, ly, lz, 72])
    tx, ty, tz = min([lx, fx]), min([ly, fy]), min([lz, fz])
    print(tx,ty,tz)
    fiber_img_new[:tx, :ty, :tz, :] = fiber_img[:tx, :ty, :tz, :]
    return fiber_img_new


def fiber2target2(no_diff_path, fiber, sub):
    fiber_img = nib.load(str(fiber)).get_data()
    roi_size = np.array([fiber_img[:,:,:,i].sum() for i in range(72)])
    print('min fiber size: ', min(roi_size), 'max fiber size: ', max(roi_size))
    mask_img = nib.load(str(no_diff_path)).get_data()
    z,y,x = np.nonzero(mask_img.T)
    if (fiber_img.shape)[:3] != mask_img.shape:
        print(fiber_img.shape, mask_img.shape)
        fiber_img = shape_fit(fiber_img, mask_img)
    mat = fiber_img[x, y, z, :]
    mat = sparse.csr_matrix(np.array(mat))
    return mat, roi_size



def get_fiber_fingerprint(workpath, sub, hemi, recreation):
    workpath = Path(workpath)
    target_file = 'finger_print_fiber.npz'
    # for hemi in tqdm.tqdm(['R', 'L']):
    file = workpath/sub/('{}_{}_probtrackx_omatrix2'.format(sub,hemi))/'fdt_matrix2.npz'
    fdt_path = file.parent
    if file.exists() and ( (not (fdt_path/target_file).exists()) or recreation):
        print(f'get fiber fingerprint from {file}')
        target_coords = fdt_path/'tract_space_coords_for_fdt_matrix2'
        no_diff_path = workpath/sub/'DTI'/'LowResMask.nii.gz'
        fiber = workpath/sub/'DTI'/'LowRes_Fibers.nii.gz'
        sps_mat = sparse.load_npz(str(file))
        n_seeds, n_targets = sps_mat.shape
        # mat, roi_size = fiber2target(target_coords, fiber, sub)
        mat, roi_size = fiber2target2(no_diff_path, fiber, sub)
        assert (mat.shape == (n_targets, 72)), f'shape: {mat.shape}, got wrong fingerprint with {n_targets, 72}'
        fp = getFingerPrint(sps_mat, mat)
        fp = normal(fp, roi_size)
        sparse.save_npz(str(fdt_path/target_file), fp)
    else:
        print("sparse matrix not exists or finger print already exists ",file)
    return None




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')default='None',required=True
    parser.add_argument('-c','--cuda', type=str, default='', help='the dictionary for the transition and final result' )
    parser.add_argument('-p','--patch', type=str, default='', help='the dictionary for the transition and final result' )
    
    # args = parser.parse_args()
    # cudanum = args.cuda
    # patchnum = int(args.patch)
    # with open('/n04dat01/atlas_group/lma/MASiVar_prep/patch_adult_all.txt') as f:
    #     pathlist = [ line.strip() for line in f.readlines()]

    # for path in pathlist:
    #     word = path.strip().split('/')
    #     sub = word[-3]
    #     session = word[-2]
    #     aq = word[-1]
    #     subID = sub+'-'+session+'-'+aq
    #     for hemi in ['L','R']:
    #         print(path, hemi)
    #         output_file = '{}/{}_{}_probtrackx_omatrix2'.format(path, subID, hemi)
    #         if not os.path.exists(output_file+'/fdt_matrix2.dot'):
    #             print('dot not exist')
    #             continue
    #         if not os.path.exists(output_file+'/fdt_matrix2.npz'):
    #             # os.system('bash /n04dat01/atlas_group/lma/MASiVar_prep/pipline/probtrack_hemi_gpu.sh {} {} {} {} {} {}'.format(path, t1file, bedpostdir, hemi, output_file, cudanum))
    #             fdt_file = Path('{}/fdt_matrix2.dot'.format(output_file))
    #             compress_sparse(fdt_file)
    #             get_fiber_fingerprint(path, subID, hemi, True)

    # recreation = False
    # Card = '1'
    # with open('/DATA/232/lma/data/HCP_test/sub_list_new.txt') as f:
    #     namelist = [ line.strip() for line in f.readlines()]
    # for workpath, datapath in zip(['/DATA/232/lma/data/HCP_test','DATA/232/lma/data/HCP_retest'],['/DATA/data2/HCP_S1200_Test','/DATA/data2/HCP_Retest']):
    #     atlaspath = '/DATA/232/lma/script/individual-master/individual_surf_pipeline/Brainnetome/'
    #     for hemi in ['L','R']:
    #         for sub in namelist:
    #             os.system('cp {}/{}/T1w/Diffusion/bvals {}/{}/DTI/bvals'.format(datapath, sub, workpath, sub))
    #             os.system('cp {}/{}/T1w/Diffusion/bvecs {}/{}/DTI/bvecs'.format(datapath, sub, workpath, sub))
    #             n = os.system('bash FiberSeg_gpu.sh {} {} {}'.format(datapath, workpath, sub, Card))
    #             n = os.system('bash pre_probtrack.sh {} {} {} {}'.format(datapath, workpath, sub, atlaspath))
    #             get_fiber_fingerprint(workpath, sub, hemi, recreation)
