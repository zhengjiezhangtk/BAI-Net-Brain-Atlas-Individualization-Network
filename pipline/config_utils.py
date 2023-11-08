import configparser
import os
from pathlib import Path
import sys


def load_soft_config(soft_path):
    
    configpath = os.path.join(soft_path, 'pipline/soft_path.sh')
    pathdict = {}
    with open(configpath, 'r') as f:
        pathtxt = [line.strip().split('=') for line in f.readlines()]
    for path in pathtxt:
        if path[0] in ['softwaredir', 'FSLDIR', 'ANTSDIR', 'FREESURFER_HOME', 'MRTRIX3', 'WORKBENCH', 'ANACONDA']:
            pathdict.update({path[0]: path[1]})
    return pathdict


def build_subdir(subdir):
    subdir = Path(subdir)
    '''Make working dictionary for a subject'''
    if not subdir.exists():
        os.mkdir(subdir)
    for file in ['xfms', 'surf', '3D', 'DTI']:
        if not (subdir/file).exists():
            os.mkdir(str(subdir/file))
    return None

def build_sub_config( subdir):
    
    softwaredir = os.path.split(os.path.realpath(__file__))[0][:-7]
    print('softwaredir:' , softwaredir)
    subdir = Path(subdir)
    """Make configure for subject for the use of individual pipline"""
    config = configparser.ConfigParser()
    config.add_section("Basic")
    config.set('Basic','subdir', str(subdir))
    config.set('Basic','sub', subdir.name)
    config.set('Basic','softwaredir', softwaredir)
    config.set('Basic','atlasdir', os.path.join(softwaredir, 'Brainnetome'))
    config.set('Basic','gcndir', os.path.join(softwaredir, 'GCN_model'))
    config.set('Basic','preprocessdir', os.path.join(softwaredir, 'pipline'))
    sub = subdir.name
    params = {'sub': config['Basic']['sub'], 'subdir': config['Basic']['subdir'], 'atlasdir': config['Basic']['atlasdir'], \
        'softwaredir': config['Basic']['softwaredir'], 'gcndir': config['Basic']['gcndir'], 'preprocessdir': config['Basic']['preprocessdir'] }
    with open(subdir/(sub+'_config.ini'), "w") as f:
        config.write(f)
    return config, params


def load_config_dict(subdir):
    '''read subject configture and convert to dict'''
    subdir = Path(subdir)
    sub = subdir.name
    if not (subdir/(sub+'_config.ini')).exists():
        config, params = build_sub_config(subdir)
    else:
        config = configparser.ConfigParser()
        config.read(subdir/(sub+'_config.ini'), encoding='utf-8')
        params = dict(dict(config._sections)['Basic'])
    # print(params)
    return config, params


def save_config_dict(subdir, config, params):
    subdir = Path(subdir)
    for key in params:
        config.set('Basic', key, params[key])
    sub = subdir.name
    with open(subdir/(sub+'_config.ini'), "w") as f:
        config.write(f)
    return None


def update_config(subdir, params):
    subdir = Path(subdir)
    config = configparser.ConfigParser()
    sub = subdir.name
    config.read(subdir/(sub+'_config.ini'), encoding='utf-8')
    for key in params:
        config.set('Basic', key, params[key])
    with open(subdir/(sub+'_config.ini'), "w") as f:
        config.write(f)

    return None

def update_config_all(subdir, args):
    softwaredir = os.path.split(os.path.realpath(__file__))[0][:-7]
    print('softwaredir:' , softwaredir)
    subdir = Path(subdir)
    config = configparser.ConfigParser()
    sub = subdir.name
    config.read(subdir/(sub+'_config.ini'), encoding='utf-8')
    config.set('Basic', 'subdir', str(subdir))
    config.set('Basic', 'sub', subdir.name)
    config.set('Basic', 'softwaredir', softwaredir)
    config.set('Basic', 'atlasdir', os.path.join(softwaredir, 'Brainnetome'))
    config.set('Basic', 'gcndir', os.path.join(softwaredir, 'GCN_model'))
    config.set('Basic', 'preprocessdir', os.path.join(softwaredir, 'pipline'))
    config.set('Basic', 'msmall', str(args.MSMALL))
    config.set('Basic', 'tratseg_mnispace', str(args.tract_MNIspace))
    config.set('Basic', 'ants_registration', str(args.ANTSreg))

    with open(subdir/(sub+'_config.ini'), "w") as f:
        config.write(f)
    return None

# def add_args_to_params(params, args):
#     for arg in args:
#         params[arg] = args.arg
#     print('123')
#     return params


if __name__ == '__main__': 
    subdir = '/n04dat01/atlas_group/lma'
    # build_sub_config( subdir)
    # soft_path = '/n04dat01/atlas_group/lma/DP_MDD_dataset'
    # load_soft_config(soft_path)
    # config = configparser.ConfigParser()
    # config.read('/DATA/232/lma/script/individual-master/individual_surf_pipeline/config.ini',encoding='utf-8')
    # dic = dict(dict(config._sections)['process_part'])
    # # for i in dic:
    # #     print(i)
    # #     dic[i] = dict(dic[i])
    # print(dic)
