from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
from utils import *
# from gcn.utils import *
from models import GCN
import scipy.sparse as sp
import os
import pandas as pd
# plt.switch_backend('agg')
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags() 
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list: 
        FLAGS.__delattr__(keys) 
        

def train_step(dirname, modeldir, datasetpath, FLAGS):
    
    # Set random seed
    seed = 233
    np.random.seed(seed)
    tf.set_random_seed(seed)
    print(dirname, modeldir, datasetpath)

    # Load training data
    namelist_path = './List_HCP100.txt'
    namelist1 = [ name.strip() for name in open(namelist_path, 'r').readlines() ]
    namelist_path = '/n14dat01/lma/data/HCPwork/sub_list.txt'
    namelist2 = [ name.strip() for name in open(namelist_path, 'r').readlines() ]
    namelist = namelist1 + namelist2
    subdir_path_2 = [ '{}/{}'.format(datasetpath, str(name).strip()) for name in namelist]
    subdir_path = []
    for path in subdir_path_2:
        if os.path.exists('{}/surf/adj_matrix_seed_L.npz'.format(path)):
            subdir_path.append(path)
    subdir_path = subdir_path[:170]

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.hemisphere, subdirlist=subdir_path, modeldir=modeldir, MPM=False, uniform=True)
    
    # Load retest data
    f = open('./HCP_test/sub_list.txt', 'r')
    subdir_path = [ '{}/{}'.format('/n14dat01/lma/data/HCP_test', str(name).strip()) for name in f.readlines()][:42]
    adj2, features2, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.hemisphere, subdirlist=subdir_path, modeldir=modeldir, MPM=False, uniform=True)

    # #Load Resesting data
    f = open('./HCP_retest/sub_list.txt', 'r')
    subdir_path = [ '{}/{}'.format('/n14dat01/lma/data/HCP_retest', str(name).strip()) for name in f.readlines()][:42]
    adj3, features3, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.hemisphere, subdirlist=subdir_path, modeldir=modeldir, MPM=False, uniform=True)
    

    feature_tmp = [ ff for ff in features]
    num_supports = 1 + FLAGS.max_degree
    print("Calculating Chebyshev polynomials up to order {}...".format(FLAGS.max_degree))
    
    
    # supports = []
    # Some preprocessing
    for i in range(len(features)):
        features[i] = preprocess_features(features[i])
        # supports.append(chebyshev_polynomials(adj[i], FLAGS.max_degree))
    print('feature shape:', features[0][2], 'ytrain:', y_train.shape[1])

    for i in range(len(features2)):
        features2[i] = preprocess_features(features2[i])
        
    for i in range(len(features3)):
        features3[i] = preprocess_features(features3[i])

    supports = chebyshev_polynomials(adj, FLAGS.max_degree)

    supports2 = chebyshev_polynomials(adj2, FLAGS.max_degree)


    print(len(supports))
    print(num_supports, features[0][2], y_train.shape[1], features[0][2][1])

    print('Construct model')
    # Define placeholders/
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[0][2], dtype=tf.int64)),
        'phase_train': tf.placeholder_with_default(False, shape=()),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.float32), 
        'dropout': tf.placeholder_with_default(0., shape=()),  
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        }

    # Initialize session
    sess = tf.Session()
    # Create model
    model_name = 'model_{}_{}_{}'.format(str(FLAGS.depth), str(FLAGS.max_degree), str(FLAGS.hemisphere))
    print('model name:', model_name)
    model = GCN(placeholders, input_dim=features[0][2][1], layer_num = FLAGS.depth, logging=True, name=model_name)
    
    # Init variables
    sess.run(tf.global_variables_initializer())
    # model.load(sess=sess, path = dirname)
     

    # Define model evaluation function
    def evaluate(features, supports, labels, mask,  placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, supports, labels, mask,  placeholders)
        outs_val = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


    def dynamic_learning_rate(epoch):

        if epoch < 5:
            learning_rate = FLAGS.basic_learning_rate * np.exp( -2*np.log(10) * epoch/5)
        else:
            learning_rate = FLAGS.basic_learning_rate * np.exp( -2*np.log(10))
        return learning_rate


    mean_best_dice = 0
    record_train_loss, record_train_dice = [], []
    record_val_loss, record_val_dice, record_val_prediction = [], [], []
    numlist = np.arange(0, FLAGS.train_num+FLAGS.validate_num, 1)
    np.random.shuffle(numlist)
    train_numlist = numlist[:FLAGS.train_num]
    val_numlist = numlist[FLAGS.train_num:FLAGS.train_num+FLAGS.validate_num]
    test_numlist = np.arange(FLAGS.train_num+FLAGS.validate_num, 170, 1)
    # np.save('{}/validation_number.npy'.format(dirname), np.array(val_numlist))

    random_train = np.zeros_like(y_train)
    random_train[:, 0] = 1
    # Train model
    print('length', len(features))
    for epoch in range(FLAGS.epochs):
        cost_train,  dice_train = [],  []
        cost_val,  dice_val = [],  []
        np.random.shuffle(train_numlist)
        FLAGS.learning_rate = dynamic_learning_rate(epoch)

        print('_____leaning rate____', FLAGS.learning_rate, 'epoch:', epoch)
        print('Begin Training')
        for i in train_numlist:

            # Construct feed dictionary
            feed_dict = construct_feed_dict(features[i], supports, y_train, train_mask,  placeholders )

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict()], feed_dict=feed_dict)
            cost_train.append(outs[1])
            dice_train.append(outs[2])
            record_train_loss.append(outs[1])
            record_train_dice.append(outs[2])
            print("Epoch:", '%04d' % (epoch + 1), 'training sample: ', i, "train_loss=", "{:.5f}".format(outs[1]),
                "train_dice=", "{:.3f}".format(outs[2]))

            feature_random = random_transefer(feature_tmp[i])
            feature_random = preprocess_features(feature_random )
            random_mask = np.ones_like(train_mask)
            feed_dict = construct_feed_dict(feature_random, supports, random_train, random_mask, placeholders )
            outs22 = sess.run([model.opt_op, model.loss, model.accuracy, model.predict()], feed_dict=feed_dict)

        # mean_train_loss = np.array(cost_train).mean()
        # mean_train_dice = np.array(dice_train).mean()
        # print('training average', mean_train_dice, mean_train_loss)

        # Validation
        record_val_prediction = []
        print('Begin Validation')
        for i, validnum in enumerate(val_numlist):
            tmpcost, tmpdice, prediction, tt = evaluate(features[validnum], supports, y_val, val_mask, placeholders )
            cost_val.append(tmpcost)
            dice_val.append(tmpdice)
            record_val_prediction.append(np.argmax( prediction, axis=1))

            # Print results
            print( "Epoch:", '%04d' % (epoch + 1), 'sample: ', validnum, "val_loss=", "{:.5f}".format(tmpcost), "val_dice=", "{:.3f}".format(tmpdice), "time=", "{:.5f}".format(tt))
        
        val_loss = np.array(cost_val)
        val_dice = np.array(dice_val)

        test_retest_dice = []
        for i in range(len(features2)):
            _, _, output1, _ = evaluate(features2[i],  supports2, y_val, val_mask,  placeholders )
            _, _, output2, _ = evaluate(features3[i],  supports2, y_val, val_mask,  placeholders )
            dice = cal_dice( output1, output2)
            test_retest_dice.append(dice)
            print('epoch: ', epoch, 'sample: ', i, 'dice: ', dice)
        tmp_dice = np.array(test_retest_dice)
        print('test_retest_dice:', tmp_dice.mean())

        mean_val_dice = val_dice.mean()
        if (mean_best_dice < mean_val_dice) and ( epoch >= 4): 
            mean_best_dice = mean_val_dice
            best_epoch = epoch
            print("model saved", 'best dice is ', mean_best_dice, 'epoch', best_epoch )      
            model.save(sess=sess, path=dirname)
            val_inter_dice = []
            for i in range(len(record_val_prediction)):
                for j in range(i+1, len(record_val_prediction)):
                    acc = (record_val_prediction[i] == record_val_prediction[j]).sum()/len(record_val_prediction[j])
                    val_inter_dice.append(acc)
            val_inter_dice = np.array(val_inter_dice)
            repr_dice = tmp_dice
            best_loss = val_loss
            best_dice = val_dice

    # print("model saved", 'best dice is ', best_dice, 'epoch', best_epoch)      
    # print("Optimization Finished!", 'best dice is ', best_dice)
    return best_loss, best_dice, val_inter_dice, repr_dice

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags() 
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list: 
        FLAGS.__delattr__(keys) 
        

if __name__ == '__main__':
    del_all_flags(tf.flags.FLAGS)
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    t_test = time.time()


    del_all_flags(tf.flags.FLAGS)
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', './HCP_S1200_individual_MSM_atlas', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('modeldir', './BAI_Net/BAI_Net_BN', 'Dataset string.')
    flags.DEFINE_string('model', 'gcn_cheby', 'Model llstring.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_float('basic_learning_rate', 0.01, 'basic learning rate.')
    flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
    flags.DEFINE_integer('depth', depth, 'GNN depth')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', max_degree, 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('train_num', 100, 'sample size for training ')
    flags.DEFINE_integer('validate_num', 20, 'sample size for validation ')
    flags.DEFINE_string('hemisphere', hemi, 'cerebral cortex part')
    flags.DEFINE_string('mode', 'train', 'model mode')
    print('hemisphere: ', FLAGS.hemisphere)
    print('_____layer_num:', FLAGS.depth, 'model:', FLAGS.model, 'max_degree:', FLAGS.max_degree, 'hidden:', FLAGS.hidden1)
    
    mean_val_loss, mean_val_dice, mean_val_inter_dice, mean_tmp_dice = train_step(model_result, FLAGS.modeldir, FLAGS.dataset, FLAGS)
    resutls = np.array([depth, max_degree, hemi, mean_val_loss, mean_val_dice, mean_val_inter_dice, mean_tmp_dice])
