from __future__ import division
from __future__ import print_function
import time

from utils import *
from models import GCN
import os
from pathlib import Path
import tensorflow as tf
import numpy as np
# import nibabel as nib
from nilearn import surface
import warnings
import scipy.sparse as sp

warnings.filterwarnings('ignore', category = ResourceWarning , module='nibabel')
warnings.filterwarnings('ignore', category = DeprecationWarning )




def main(hemisphere, sub, subdir_path, modelpath):
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    hemi = FLAGS.hemisphere
    MSMAll = True if FLAGS.MSMAll=='True' else False
    print('MSM', MSMAll)
    #load data
    adj, featurelist, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.hemisphere, subdirlist= subdir_path, modeldir=FLAGS.modeldir, uniform=False, MSMAll=MSMAll)
    adj = adj[0]
    # adj = sp.load_npz('/n04dat01/atlas_group/lma/HCP_S1200_individual_MSM_atlas/994273/surf/adj_matrix_seed_L_new.npz')
    # Some preprocessing
    features = preprocess_features(featurelist[0])
    num_supports = 1 + FLAGS.max_degree
    
    print("Calculating Chebyshev polynomials up to order {}...".format(FLAGS.max_degree))
    supports = chebyshev_polynomials(adj, FLAGS.max_degree)
    
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(FLAGS.dropout, shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    }


    def evaluate(features, supports, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, supports, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.dice, model.outputs], feed_dict = feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


    #loading trained model
    model_name = 'glasser_model_{}_{}_{}'.format( FLAGS.depth, FLAGS.max_degree, FLAGS.hemisphere)
    # model_name = 'glasser_model_0_3_R'
    sess = tf.Session()
    model = GCN(placeholders, input_dim=features[2][1], layer_num=FLAGS.depth, logging=True, name=model_name)
    sess.run(tf.global_variables_initializer())
    model.load(sess = sess, path = modelpath)

    #individual parcellation
    cost, dice, prediction, tt = evaluate(features, supports, y_val, val_mask, placeholders )
    print("Test set results:", "cost=", "{:.5f}".format(cost), "dice=", "{:.3f}".format(dice), "time=", "{:.5f}".format(tt))

    prob = numpy_softmax(prediction)
    # savepath = '{}/{}.indi.{}.prob.npy'.format(FLAGS.savepath, sub, FLAGS.hemisphere)
    # np.save(savepath, prediction)
    #saving the result
    # prediction = postparcellation(prob, hemi)
    print('Final dice:', cal_dice(prediction, y_val))
    savepath = '{}/{}.glasser.indi.MSM_weighted.{}.label.gii'.format(subdir_path[0], sub, FLAGS.hemisphere)
    saveGiiLabel(np.argmax(prediction, axis=1), FLAGS.hemisphere, savepath )

    return None

# def postparcellation2(label, hemi):
#     """limit label range, in case of misallignment"""
#     mask_hemi = np.load('/DATA/232/lma/envs/gcn-master/gcn/neighbor_mask_{}_1.npy'.format(hemi))
#     label[:,1:] = label[:,1:]*mask_hemi
#     return label



if __name__=='__main__':

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', '/n04dat01/atlas_group/lma/HCP_S1200_individual_MSM_atlas', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('modeldir', '/n04dat01/atlas_group/lma/populationGCN/BAI_Net/glasser_360/', 'Dataset string.')
    flags.DEFINE_string('model', 'gcn_cheby', 'Model llstring.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_float('basic_learning_rate', 0.01, 'basic learning rate.')
    flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
    flags.DEFINE_integer('depth', 0, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('train_num', 100, 'sample size for training ')
    flags.DEFINE_integer('validate_num', 20, 'sample size for training ')
    flags.DEFINE_string('hemisphere', 'L', 'cerebral cortex part')
    flags.DEFINE_string('mode', 'train', 'cerebral cortex part')
    flags.DEFINE_string('MSMAll', 'True', 'cerebral cortex part')
    flags.DEFINE_string('subdir', '/n04dat01/atlas_group/lma/HCP_S1200_individual_MSM_atlas/151627', 'cerebral cortex part')
    # os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.card


    
    modelpath = '/n04dat01/atlas_group/lma/populationGCN/BAI_Net/glasser_360/tf_glasser_0_3_{}'.format(FLAGS.hemisphere)
    if FLAGS.subdir != 'None':
        print( 'modeldir', FLAGS.modeldir, ' for sub:', FLAGS.subdir)
        sub = Path(FLAGS.subdir).name
        subdir_path = [ FLAGS.subdir ]
        main(FLAGS.hemisphere, sub, subdir_path, modelpath)

    # elif FLAGS.sublist != 'None':
    #     with open(FLAGS.sublist, 'r') as f:
    #         sublist = [ line.strip() for line in f.readlines()]
    #     print('parcellation in workpath:', FLAGS.workpath, ' for sublist:', FLAGS.sublist)
    #     feature_path_list =[ '{}/{}/{}_{}_probtrackx_omatrix2/finger_print_fiber.npz'.format(FLAGS.workpath, sub, sub, FLAGS.hemisphere) for sub in sublist]
    #     main_sublist(FLAGS.hemisphere, sublist, feature_path_list, modelpath)
    
