import numpy as np
# import pickle as pkl
import nibabel as nib
from nilearn import surface
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
# from keras.utils import to_categorical
import sys
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore' )


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
    
def cal_dice(y_a,y_b):
    label_a = np.argmax(y_a, axis=1)    
    label_b = np.argmax(y_b, axis=1)
    inter = (label_a==label_b).sum()
    cross = len(label_a)+len(label_b)
    dice = 2*inter/(cross+0.0001)
    return dice


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def load_data(hemisphere, subdirlist=None, MPM=False, model_dir=None, uniform=True, MSMAll=True, atlas='BNA'):
    """
    funtion for reading the feature, label and mesh files.
    namelist and pathlist should not be empty at the same time. 
    """
    print('load the data in the hemisphere: ', hemisphere)
    #read adj
    adj = load_adj(hemisphere, subdirlist, model_dir, uniform, MSMAll, atlas)
    #read label and mask
    y_train, y_val, y_test, train_mask, val_mask, test_mask = load_label_and_mask(hemisphere, model_dir, MPM)
    #read feature file
    feature = load_feature(hemisphere, subdirlist, MSMAll)

    return adj, feature, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_label_and_mask(hemisphere, model_dir, MPM=False):
    atlas = str(Path(model_dir).name).split('_')[-1]
    print(model_dir)
    label_file = '{}/{}/fsaverage.{}.{}_Atlas.32k_fs_LR.label.gii'.format(model_dir, atlas, hemisphere, atlas)
    print('label_file',label_file)
    label = surface.load_surf_data(label_file)
    select_ind = np.loadtxt('{}/{}/metric_index_{}.txt'.format(model_dir, atlas, hemisphere)).astype('int')
    label = np.array(label)[select_ind]
    if atlas =='BN':
        if hemisphere=='R':
            label = label//2
            label[label<0]=0
        elif hemisphere == 'L':
            label = (label+1)//2
            label[label<0]=0
    elif atlas =='HCPparcellation':
        if hemisphere == 'L':
            label = label - 180
            label[label<0]=0

    select_label = np.sort(list(set(list(label)+[0])))
    print('label set:', select_label, len(label) )
    index = np.arange(len(label))
    label_arr = to_categorical(label, len(select_label))
    print('model classes:', len(select_label))
    y_val, y_test = label_arr.copy(), label_arr.copy()
    
    y_train = label_arr.copy()

    train_mask = np.array([True]*len(index)).astype('int32')
    val_mask = np.array([True]*len(index)).astype('int32')
    test_mask = np.array([True]*len(index)).astype('int32')

    # if MPM:
        # workdir = '/n04dat01/atlas_group/lma/populationGCN/BAI_Net'
        # train_mask = np.load('{}/mpm_{}.npy'.format(str(workdir), hemisphere))
        # train_mask = train_mask/100

    print('Label shape:', label_arr.shape, 'Mask shape: ', train_mask.shape)
    
    return y_train, y_val, y_test, train_mask, val_mask, test_mask



def load_adj(hemisphere, subdirlist=None, modeldir=None, uniform=True, MSMAll=False, atlas='BN'):
    if uniform:
        target = '{}/{}/adj_matrix_seed_{}.npz'.format(modeldir, atlas, hemisphere)
        adj = sp.load_npz(str(target))
        print('adj shape:', adj.shape)
        return adj
    else:
        adjlist = []
        # sign = 'weighted_' if MSMAll else ''
        # sign = ''
        for subdir in subdirlist:
            subdir = Path(subdir)
            target = subdir/'surf'/'weighted_adj_matrix_seed_{}.npz'.format( hemisphere)
            print(target)
            if target.exists():
                adjlist.append(sp.load_npz(str(target)))
                # print('loading adj shape:', sp.load_npz(str(target)).shape)
            else:
                print(str(target), ' do not exist or fail to read!!')
                continue
        print('adj shape:', adjlist[0].shape, 'sample number:', len(adjlist))
        return adjlist


def load_feature(hemisphere, subdirlist=None, MSMAll=True):
    feature = []
    for subdir in subdirlist:
        subdir = Path(subdir)
        sub = subdir.name
        sign = '_MSMALL' if MSMAll else ''
        # sign=''
        target = subdir/(sub+'_'+hemisphere+'_probtrackx_omatrix2')/'finger_print_fiber{}.npz'.format(sign)
        print('features', target)
        if target.exists():
            feature.append(sp.load_npz(str(target)))
        else:
            print(str(target), ' do not exist or fail to read!!')
            continue
    if not len(feature)==1:
        print('feature shape:', feature[0].shape, 'sample number:', len(feature))

    return feature



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)).astype('float')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # print(adj.shape, d_mat_inv_sqrt.shape)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders ):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})

    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized

    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])
    
    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def chebyshev_polynomials_test(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized

    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])
    
    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return t_k


def meshto32klabel(data, hemisphere):
    label = np.zeros(32492)
    path = 'metric_index_{}.txt'.format(hemisphere)
    select_ind = np.loadtxt(path).astype(int)
    label[select_ind] = data
    return label

def saveGiiLabel(data, hemisphere, template_path, savepath):
    '''
    save data as gii format
    template_path: the path template
    the length of data is not required same to the template but should match the relevent surface points' size
    path and save_name is the saving location of gii file
    '''
    atlas = template_path.split('.')[-3]
    if atlas == 'HCPparcellation':
        if hemisphere == 'L':
            data = data+180
            data[data==180]=0

    data = meshto32klabel(data, hemisphere).astype('int32')
    # template_path = '/n04dat01/atlas_group/lma/populationGCN/BAI_Net/Brainnetome/fsaverage.{}.BN_Atlas.32k_fs_LR.label.gii'.format(hemisphere)
    original_label = nib.gifti.giftiio.read(template_path)
    a = nib.gifti.gifti.GiftiDataArray(data.astype('int32'), intent='NIFTI_INTENT_LABEL')
    new_label = nib.gifti.gifti.GiftiImage( meta = original_label.meta, labeltable = original_label.labeltable)
    new_label.add_gifti_data_array(a)
    nib.gifti.giftiio.write(new_label, savepath)
    return None

def random_transefer(feature):
    if type(feature) == sp.csr_matrix:
        feature = feature.toarray()

    index = np.arange(feature.shape[0])
    np.random.shuffle(index)
    data_new = np.zeros_like(feature)
    for i,j in enumerate(index):
        data_new[j] = feature[i]

    data_new = sp.csr_matrix(data_new)
    return data_new

def numpy_softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return softmax_x 

def postparcellation(label, hemi):
    """limit label range, in case of misallignment"""
    mask_hemi = np.load('/n14dat01/lma/envs/gcn-master/gcn/neighbor_mask_{}.npy'.format(hemi))
    label[:, 1:] = label[:, 1:]*mask_hemi
    return label


if __name__ == '__main__':
    print('haha')
    # main()
