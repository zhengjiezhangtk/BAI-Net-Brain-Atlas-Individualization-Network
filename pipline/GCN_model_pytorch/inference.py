
import torch
from GCN_model_pytorch.data_loader import get_evaluation_loader, Inference_data
from GCN_model_pytorch.network import ChenNet
import argparse
import os
import numpy as np
import nibabel as nib
from nilearn import surface


def meshto32klabel(label, hemi, atlas, software_dir):
    file = f'{software_dir}/priors/BAI_Net_{atlas}/{atlas}/fsaverage.{hemi}.{atlas}_Atlas.32k_fs_LR.label.gii'
    label_model = surface.load_surf_data(file)
    label_new = np.zeros(len(label_model))
    path = f'{software_dir}/priors/BAI_Net_{atlas}/{atlas}/metric_index_{hemi}.txt'
    select_ind = np.loadtxt(path).astype(int)
    label_new[select_ind] = label
    return label_new


def saveLabelGii( label,  hemi,  outputdir, filename, template_path, atlas, software_dir, dtype='int32'):
    '''
    save data as gii label format
    template_path: the path template
    the length of data is not required same to the template but should match the relevent surface points' size
    path and save_name is the saving location of gii file
    '''
    label = np.array(label)
    label = meshto32klabel(label, hemi, atlas, software_dir).astype('int32')
    print(hemi, label)
    template = nib.load(template_path)

    labelarray = nib.gifti.gifti.GiftiDataArray(label.astype(dtype))
    new_label = nib.gifti.gifti.GiftiImage( meta = template.meta, labeltable = template.labeltable)
    new_label.add_gifti_data_array(labelarray)
    nib.loadsave.save(new_label, os.path.join( outputdir, filename))
    return None


def get_atlas_information(atlas, hemi, software_dir):
    """build and load model"""
    if atlas == 'BN':
        output_number = 106
    elif atlas == 'Yeo17Network':
        output_number = 18
    elif atlas == 'Yeo7Network':
        output_number = 8
    elif atlas == 'HCPparcellation':
        output_number = 181
    else:
        print('not have selected atlas')
    template_path = f'{software_dir}/priors/BAI_Net_{atlas}/{atlas}/fsaverage.{hemi}.{atlas}_Atlas.32k_fs_LR.label.gii'
    return template_path, output_number


def Inference_model(dataset_path, sublist_path, config=None):
    software_dir = config.software_dir
    print('BAI_Net_dir:', software_dir)
    result_path = os.path.join(config.result_path, config.atlas)
    MSMAll_signal = '.MSMAll' if bool(config.MSMAll) else ''
    print("MSMAll signal:", MSMAll_signal, config.MSMAll,  bool(config.MSMAll))
    # MSMAll_signal2 = '_MSMALL' if config.MSMAll else ''
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('model:', config.model_type, 'batch_size:', config.batch_size)

    template_path, output_number = get_atlas_information(config.atlas, config.hemi, software_dir)

    # load dataset information
    if type(sublist_path) == str:
        sublist = [line.strip() for line in open(sublist_path).readlines()]
    elif type(sublist_path) == list:
        sublist = sublist_path

    # sublist_path type:  str type or list type
    dataset = Inference_data( hemi=config.hemi, atlas=config.atlas, dataset_path=dataset_path, sublist_path=sublist_path,  software_dir= software_dir)
    eval_dataloader = get_evaluation_loader(dataset, config.batch_size, config.num_workers)
    
    # load model parameters:
    gcn_model = ChenNet(72, output_number)
    gcn_model.to(device)    
    model_path = f'{software_dir}/pipline/GCN_model_pytorch/models'
    gcn_model_path = os.path.join(model_path, '%s-%s-%s.pkl' %(config.model_type, config.atlas, config.hemi))
    
    if os.path.isfile(gcn_model_path):
        gcn_model.load_state_dict(torch.load(gcn_model_path))
        print('%s is Successfully Loaded from %s'%(config.model_type, gcn_model_path))
    else:
        print('pytorch model with priors is not exists', gcn_model_path)

    # inference each sample
    for i, (fingerprint, graph_indice, graph_weights, target, lambda_max) in enumerate(eval_dataloader):     

        fingerprint = fingerprint[0].to(device)
        graph_indice = graph_indice[0].to(device)
        graph_weights = graph_weights[0].to(device)
        target = target[0].to(device)
        lambda_max = lambda_max[0].to(device)

        parcellation = gcn_model(fingerprint, graph_indice, graph_weights, lambda_max)
        
        parcellation = parcellation.data.cpu().numpy()
        target = target.data.cpu().numpy()
        print('parcellation shape', parcellation.shape,  target.shape)
        individual_label = np.argmax(parcellation, axis=1)
        acc = (individual_label == target).sum()/len(individual_label)
        print( 'sub number: ', sublist[i], 'overlap with group atlas', acc)
        filename = f'{sublist[i]}.indi.{config.hemi}.{config.atlas}{MSMAll_signal}.label.gii'
        if config.atlas =='BN':
            if config.hemi =='R':
                individual_label = individual_label*2
                individual_label[individual_label<2]=0
            elif config.hemi == 'L':
                individual_label = individual_label*2-1
                individual_label[individual_label<1]=0
        elif config.atlas =='HCPparcellation':
            if config.hemi == 'L':
                individual_label = individual_label + 180
                individual_label[individual_label<180]=0

        saveLabelGii( individual_label,  config.hemi, result_path, filename, template_path, atlas=config.atlas, software_dir=software_dir, dtype='int32')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--software_dir', type=str, default='',  help='BAI-Net dictionary')
    parser.add_argument('--dataset_path', type=str, default='',  help='data dictionary')
    parser.add_argument('--sublist_path', type=str, default='',  help='optional 1: sublist file path')
    parser.add_argument('--single_sub', type=str, default='', help='optional 2: subject string')
    parser.add_argument('--hemi', type=str, default='L')
    parser.add_argument('--atlas', type=str, default='BN')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--MSMAll', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default='GCN', help='model type')
    # parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--result_path', type=str, default='./label/')

    config = parser.parse_args()
    software_dir = config.software_dir

    if config.single_sub != '':
        Inference_model(config.dataset_path, [config.single_sub], args=config)
    else:
        Inference_model(config.dataset_path, config.sublist_path, args=config)

