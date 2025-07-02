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


# def read_coomat(file):
#     with open(str(file), 'r') as f:
#         mat = [line.rstrip().split('  ') for line in f]
#     print('finish loading')
#     i = [int(l[0])-1 for l in mat]
#     j = [int(l[1])-1 for l in mat]
#     v = [int(l[2]) for l in mat]
#     print('i,j,v got')
#     return i, j, v
def read_coomat(file):
    """read file low mem"""
    print(f"Reading coo matrix from: {file}")
    i_list, j_list, v_list = [], [], []
    with open(str(file), 'r') as f:
        for line in f:
            if line.strip():
                try:
                    i_str, j_str, v_str = line.strip().split('  ')
                    i_list.append(int(i_str) - 1)
                    j_list.append(int(j_str) - 1)
                    v_list.append(int(v_str))
                except ValueError:
                    print(f"Invaild line: {line.strip()}")
    print('Finished loading dot file')
    return i_list, j_list, v_list

# def compress_sparse(file):
#     sub = str(file).split('/')[-2].split('_')[0]
#     if file.exists():
#         if  not (file.parent/'fdt_matrix2.npz').exists():
#             # print(f'convert fdt matrix for {sub}')
#             i, j, v = read_coomat(file)
#             coo_mat = sparse.csr_matrix((v, (i,j)))
#             assert coo_mat.shape == (i[-1]+1, j[-1]+1)
#             sparse.save_npz(str(file.parent/'fdt_matrix2'), coo_mat)
#         print(f'remove {str(file)}')
#         os.remove(str(file))
#         return sub+' is finished'
#     else:
#         return sub + ' is already done'
def compress_sparse(file, chunk_size=1000000):

    from scipy.sparse import vstack

    sub = str(file).split('/')[-2].split('_')[0]

    if not file.exists():

        return sub + ' is already done'



    output_npz = file.parent / 'fdt_matrix2.npz'

    if output_npz.exists():

        return sub + ' already converted'



    print(f"[Pass 1] Scanning shape from {file}")

    max_i, max_j = 0, 0

    with open(str(file), 'r') as f:

        for line in f:

            if line.strip():

                try:

                    i_str, j_str, v_str = line.strip().split('  ')

                    i = int(i_str) - 1

                    j = int(j_str) - 1

                    max_i = max(max_i, i)

                    max_j = max(max_j, j)

                except ValueError:

                    continue

    shape = (max_i + 1, max_j + 1)

    print(f"  -> Max shape: {shape}")



    # 第二遍构建矩阵

    print("[Pass 2] Reading and chunking...")

    i_list, j_list, v_list = [], [], []

    chunk_matrices = []

    with open(str(file), 'r') as f:

        for line_num, line in enumerate(f, 1):

            if line.strip():

                try:

                    i_str, j_str, v_str = line.strip().split('  ')

                    i = int(i_str) - 1

                    j = int(j_str) - 1

                    v = int(v_str)

                    i_list.append(i)

                    j_list.append(j)

                    v_list.append(v)

                except ValueError:

                    continue



                if len(i_list) >= chunk_size:

                    print(f"  Writing chunk ending at line {line_num}")

                    chunk = sparse.csr_matrix((v_list, (i_list, j_list)), shape=shape)

                    chunk_matrices.append(chunk)

                    i_list, j_list, v_list = [], [], []



    if i_list:

        print(f"  Writing final chunk with {len(i_list)} entries")

        chunk = sparse.csr_matrix((v_list, (i_list, j_list)), shape=shape)

        chunk_matrices.append(chunk)



    print("Stacking all chunks with consistent shape...")

    final_mat = chunk_matrices[0]

    for cm in chunk_matrices[1:]:

        final_mat += cm  # shape now always same



    print("Saving sparse matrix to", output_npz)

    sparse.save_npz(str(output_npz), final_mat)

    os.remove(str(file))

    return sub + ' is finished'



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
    img = nib.load(str(fiber)).get_fdata()
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
                mat.append(np.zeros_like(img[0,0,0,:], dtype='int'))

    mat = sparse.csr_matrix(np.array(mat))
    return mat, roi_size


def shape_fit(fiber_img, mask_img):
    lx, ly, lz = mask_img.shape
    fx, fy, fz, _ = fiber_img.shape
    fiber_img_new = np.zeros([lx, ly, lz, 72])
    tx, ty, tz = min([lx, fx]), min([ly, fy]), min([lz, fz])
    # print(tx,ty,tz)
    fiber_img_new[:tx, :ty, :tz, :] = fiber_img[:tx, :ty, :tz, :]
    return fiber_img_new


def fiber2target2(no_diff_path, fiber, sub):
    fiber_img = nib.load(str(fiber)).get_fdata()
    roi_size = np.array([fiber_img[:,:,:,i].sum() for i in range(72)])
    print('min fiber size: ', min(roi_size), 'max fiber size: ', max(roi_size))
    mask_img = nib.load(str(no_diff_path)).get_fdata()
    z,y,x = np.nonzero(mask_img.T)
    if (fiber_img.shape)[:3] != mask_img.shape:
        # print(fiber_img.shape, mask_img.shape)
        fiber_img = shape_fit(fiber_img, mask_img)
    mat = fiber_img[x, y, z, :]
    mat = sparse.csr_matrix(np.array(mat))
    return mat, roi_size


def get_fiber_fingerprint(workpath, sub, hemi, recreation):
    workpath = Path(workpath)
    target_file = 'finger_print_fiber.npz'

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

        ##
        print('computing fingerprint in chunks...')
        chunk_size = 1000
        fp_path = str(fdt_path / target_file)
        first_chunk = True

        for i in range(0, n_seeds, chunk_size):
            end = min(i + chunk_size, n_seeds)
            print(f' chunk {i}-{end}')
            chunk_fp = sps_mat[i:end].dot(mat)
            chunk_fp = normal(chunk_fp, roi_size)

            if first_chunk:
                sparse.save_npz(fp_path, chunk_fp)
                first_chunk = False
            else:
                tmp = sparse.load_npz(fp_path)
                tmp = sparse.vstack([tmp, chunk_fp])
                sparse.save_npz(fp_path, tmp)
        ##
        # fp = getFingerPrint(sps_mat, mat)
        # print('normalizing by fiber size...')
        # fp = normal(fp, roi_size)
        # sparse.save_npz(str(fdt_path/target_file), fp)
    else:
        print("sparse matrix not exists or finger print already exists ",file)
    return None




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-s','--subdir', type=str, default='', help='subject dictionary' )
    parser.add_argument('-p','--hemi', type=str, default='', help='cortical hemisphere' )
    args = parser.parse_args()
    subdir = Path(args.subdir)
    hemi = args.hemi
    print(subdir, hemi)
    PostProbtrack(subdir.parent, subdir.name, hemi)
    get_fiber_fingerprint(subdir.parent, subdir.name, hemi, False)

