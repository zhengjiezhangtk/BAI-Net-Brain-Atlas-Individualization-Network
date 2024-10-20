import argparse
import os
from solver import Solver
from data_loader import get_data_loader, get_evaluation_loader, HCP_data, Inference_data
from torch.backends import cudnn
import random

def main(config):
    cudnn.benchmark = True
    # if config.model_type not in ['ST_V1']:
    #     print('ERROR!! model_type should be selected in ')
    #     print('Your input for model_type was %s' % config.model_type)
    #     return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    batch = config.batch_size
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    decay_ratio = 0.8
    num_epochs_decay = int(config.num_epochs*decay_ratio)
    config.num_epochs_decay = num_epochs_decay
    

    dataset_train = HCP_data( mode='train', hemi=config.hemi, atlas=config.atlas)
    train_loader = get_data_loader(dataset_train, batch, config.num_workers)
    dataset_val = HCP_data( mode='val', hemi=config.hemi, atlas=config.atlas)
    val_loader = get_data_loader(dataset_val, batch, config.num_workers)
    dataset_all = HCP_data( mode='all', hemi=config.hemi, atlas=config.atlas)
    all_loader = get_evaluation_loader(dataset_all, batch,  config.num_workers)

    # train_loader = get_data_loader(batch, config.num_workers, 'train', config.fold )
    # val_loader = get_data_loader(batch, config.num_workers, 'val', config.fold )
    # test_loader, image_paths = get_evaluation_loader(batch, config.num_workers, 'test', config.fold )

    

    # Train and sample the images
    if config.mode == 'train':
        solver = Solver(config, train_loader, val_loader, all_loader)
        solver.train()
    elif config.mode == 'eval':
        solver = Solver(config, train_loader, val_loader, all_loader)
        solver.test()
    elif config.mode  == 'repli':
        dataset_path = '/n04dat01/atlas_group/lma/HCP_test_retest/MSM/HCP_test'
        sublist_path = '/n04dat01/atlas_group/lma/HCP_test_retest/sub_list.txt'
        scan1_dataset = Inference_data( hemi=config.hemi, atlas=config.atlas, dataset_path=dataset_path, sublist_path=sublist_path )
        scan1_loader = get_evaluation_loader(scan1_dataset, config.batch_size, config.num_workers)

        dataset_path = '/n04dat01/atlas_group/lma/HCP_test_retest/MSM/HCP_retest'
        sublist_path = '/n04dat01/atlas_group/lma/HCP_test_retest/sub_list.txt'
        scan2_dataset = Inference_data( hemi=config.hemi, atlas=config.atlas, dataset_path=dataset_path, sublist_path=sublist_path )
        scan2_loader = get_evaluation_loader(scan2_dataset, config.batch_size, config.num_workers)

        solver = Solver(config, train_loader, val_loader, all_loader, scan1_loader, scan2_loader)
        solver.replication()
    # elif config.mode == 'eval':
    #     solver.val()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_ch', type=int, default=72)
    parser.add_argument('--output_ch', type=int, default=18)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--hemi', type=str, default='L')
    parser.add_argument('--atlas', type=str, default='Yeo17Network')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--dis', type=bool, default=False)
    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)
    parser.add_argument('--historyhour', type=int, default=2)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='GCN', help='model type')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
