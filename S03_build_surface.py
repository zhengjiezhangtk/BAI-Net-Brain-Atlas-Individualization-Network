import numpy as np
from pathlib import Path
import argparse
import os
from config_utils import update_config, load_config_dict, load_soft_config
from nilearn import surface 
import scipy.sparse as sp 
import pandas as pd



def produce_32k_surface(subdir, recreation, args):
    print('''_____________________________________________________________________________________________
                            Surface produce using freesurfer''')
    subdir = Path(subdir)
    sub = subdir.name
    print(sub)
    _, params = load_config_dict(subdir)
    string = '_MSMAll' if params['msmall']=='True' else ''
    if args.fsaverage_LR32k  and (not recreation):
        params['surfdir'] = args.fsaverage_LR32k
        if args.surface_begin_name:
            params['surfname'] = args.surface_begin_name
        else:
            params['surfname'] = sub
    else:
        target_format = subdir/sub/'label'/'rh.entorhinal_exvivo.label'
        if ( not (target_format).exists() ) or recreation :
            print( target_format, (target_format).exists(), recreation)
            if (subdir/sub).exists():
                print('___clear old recon-all files__')  
                m = os.system('rm -r {}'.format(str(subdir/sub)))
            subinput = params['t1']
            multipr = 4
            logfile = str(subdir/'reconlog.txt')
            n = os.system('recon-all -i {} -subjid {} -sd {} -all -openmp {} > {} 2>&1'.format(subinput, sub, str(subdir), multipr, logfile))
            # n = os.system('recon-all -i {} -subjid {} -sd {} -all -openmp {}'.format(subinput, sub, str(subdir), multipr))
            print('sub {} reon-all return is {}'.format(sub, n))

        target_format = subdir/sub/'MNINonLinear'/'fsaverage_LR32k'/(sub+'.R.pial.32k_fs_LR.surf.gii')
        if (not target_format.exists()) or recreation:
            logfile = str(subdir/'post-reconlog.txt')
            print('logfile is ', logfile)
            n = os.system('bash {}/FS2WB.sh {} {} > {} 2>&1'.format(params['preprocessdir'], str(subdir.parent), sub, logfile))
            params['msmall']='False'
            print('sub {} post reon-all return is {}'.format(sub, n))
        params['surfdir'] = str(subdir/sub/'fsaverage_LR32k')
        params['surfname'] = sub

    if not (subdir/'surf'/'white.R.asc').exists():
        os.system('bash {}/surf_file.sh {} {} {} {} {}'.format(params['preprocessdir'], params['surfdir'], params['surfname'], params['subdir'], params['atlasdir'], string))
    
    if args.ANTSreg : # and (not (subdir/'surf'/'pial_DTI_R.asc').exists()):
        tran_asc(str(subdir), params)

    # params['seed'] = {'L': str(subdir/'surf'/'white.L.asc'), 'R': str(subdir/'surf'/'white.R.asc')}
    # surf_file = '{}/{}.white.32k_fs_LR.surf.gii'.format(params['surdir'], params['surname'])
    assert os.path.exists(params['surfdir']), 'surface files do not exist'
    print(params['surfdir'],params['surfname'])
    update_config(subdir, params)

    return None


def produce_adj_matrix(subdir, MSMALL='_MSMAll', recreation=False, args=None):
    print('''_____________________________________________________________________________________________
                        Surface adjacent matrix''')
    subdir = Path(subdir)
    sub = subdir.name

    _, params = load_config_dict(subdir)

    # surfdir = params['surfdir']
    # brainet = '{}/Brainnetome'.format(params['softwaredir'])
    # surfdir = '{}/{}/fsaverage_LR32k/'.format(params['subdir'], sub)
    surfdir = params['surfdir']
    surfname = params['surfname']
    
    for hemi in ['L', 'R']:
        print('producing adjacent matrix', surfdir, hemi)
        savepath = Path(subdir)/'surf'/'adj_matrix_seed_{}.npz'.format(hemi)
        if savepath.exists():
            continue
        surf_data = surface.load_surf_mesh('{}/{}.{}.white{}.32k_fs_LR.surf.gii'.format(surfdir, surfname, hemi, MSMALL))
        # surf_data = surface.load_surf_mesh('{}/{}/T1w/fsaverage_LR32k/{}.{}.white.32k_fs_LR.surf.gii'.format(HCPdir, sub, surfname, hemi))
        points = np.array(surf_data[0])
        areas = surf_data[1]
        length = len(areas)
        adj_data = sp.lil_matrix(np.zeros([len(points), len(points)]))
        adj_data[areas[:, 0], areas[:, 1]] = 1
        adj_data[areas[:, 1], areas[:, 2]] = 1
        adj_data[areas[:, 2], areas[:, 0]] = 1
        adj_data = adj_data.T+adj_data
        adj_data.data = np.ones(len(adj_data.data))
        index_path = os.path.join(params['atlasdir'], 'metric_index_{}.txt'.format(hemi))
        select_ind = np.loadtxt(index_path)
        select_ind = select_ind.astype(int)
        adj_new = adj_data[:, select_ind]
        adj_new = adj_new[select_ind, :]
        adj_new = sp.csr_matrix(adj_new)
        sp.save_npz(str(savepath), adj_new)
    return None


def produce_weighted_adj_matrix(subdir, MSMALL='_MSMAll', recreation=False, args=None):
    print('''_____________________________________________________________________________________________
                        Surface adjacent matrix''')
    subdir = Path(subdir)
    sub = subdir.name

    _, params = load_config_dict(subdir)


    surfdir = params['surfdir']
    surfname = params['surfname']
    # print(surfname, surfdir)
    for hemi in ['L', 'R']:
        print('producing adjacent matrix', surfdir, hemi)
        savepath = Path(subdir)/'surf'/'weighted_adj_matrix_seed_{}.npz'.format(hemi)
        # print(savepath)
        if savepath.exists():
            continue
        surf_data = surface.load_surf_mesh('{}/{}.{}.white{}.32k_fs_LR.surf.gii'.format(surfdir, surfname, hemi, MSMALL))
        # surf_data = surface.load_surf_mesh('{}/{}/T1w/fsaverage_LR32k/{}.{}.white.32k_fs_LR.surf.gii'.format(HCPdir, sub, surfname, hemi))
        points = np.array(surf_data[0])
        areas = surf_data[1]
        length = len(areas)
        adj_data = sp.lil_matrix(np.zeros([len(points), len(points)]))
        adj_data[areas[:, 0], areas[:, 1]] = 1./np.sqrt(((points[areas[:, 0],:]-points[areas[:, 1],:])**2).sum(axis=1))
        adj_data[areas[:, 1], areas[:, 2]] = 1./np.sqrt(((points[areas[:, 1],:]-points[areas[:, 2],:])**2).sum(axis=1))
        adj_data[areas[:, 2], areas[:, 0]] = 1./np.sqrt(((points[areas[:, 2],:]-points[areas[:, 0],:])**2).sum(axis=1))
        adj_data = adj_data.T+adj_data
        index_path = os.path.join(params['atlasdir'], 'metric_index_{}.txt'.format(hemi))
        select_ind = np.loadtxt(index_path)
        select_ind = select_ind.astype(int)
        adj_new = adj_data[:, select_ind]
        adj_new = adj_new[select_ind, :]
        adj_new = sp.csr_matrix(adj_new)
        sp.save_npz(str(savepath), adj_new)
    return None


def tran_asc(subdir, params):
    for hemi in ['L', 'R']:
        for part in ['white', 'pial']:
            path = '{}/surf/{}.{}.asc'.format(subdir, part, hemi)
            transT1_path = '{}/surf/{}_T1_{}.csv'.format(subdir, part, hemi)
            transDTI_path = '{}/surf/{}_DTI_{}.csv'.format(subdir, part, hemi)
            matpath = '{}/DTI/ants_t10GenericAffine.mat'.format(subdir)
            savepath = '{}/surf/{}_DTI_{}.asc'.format(subdir, part, hemi)
            data = np.loadtxt(path, skiprows=2)
            area = data[32492:32492+64980]
            data = data[:32492]
            data[:, :2] = -data[:, :2]
            data = pd.DataFrame(data, columns=['x', 'y', 'z', 'label'])
            data.to_csv(transT1_path, index=False)
            antspath = load_soft_config(params['softwaredir'])['ANTSDIR']
            command = '{}/antsApplyTransformsToPoints -d 3 -i {} -o {} -t [{}, 1]'.format(antspath, transT1_path, transDTI_path, matpath)
            print(command)
            n = os.system(command)
            print(n)
            data = pd.read_csv(transDTI_path)
            data = data.values
            header1 = '#!ascii from CsvMesh'
            header2 = '32492 64980'
            with open(savepath, 'w') as f:
                f.write(header1)
                f.write('\r\n')
                f.write(header2)
                f.write('\r\n')
                for i in range(len(data)):
                    f.write(str(-data[i,0].round(3))+' '+str(-data[i,1].round(3))+' '+str(data[i,2].round(3))+' '+str(data[i,3]))
                    f.write('\r\n')
                for ii in range(len(area)):
                    f.write(str(int(area[ii,0]))+' '+str(int(area[ii,1]))+' '+str(int(area[ii,2]))+' '+'0')
                    f.write('\r\n')


    for hemi in ['L','R']:
        savepath = '{}/surf/pial_DTI_{}.asc'.format(subdir, hemi)
        os.system('echo {}>>{}/surf/stop_ants'.format(savepath, subdir))
        savepath = '{}/surf/white_DTI_{}.asc'.format(subdir, hemi)
        os.system('echo {}>>{}/surf/wtstop_ants'.format(savepath, subdir))
    
    return None




if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')default='None',required=True
    parser.add_argument('-s', '--subdir', type=str, default='', help='the dictionary for the transition and final result' )
    # parser.add_argument('--fsaverage_LR32k', type=str, default='', help='Freesurfer Result Dictionary')
    # parser.add_argument('--surface_begin_name', type=str, default='', help='Surface begin name in Freesurfer Result Dictionary, if fsaverage_LR32k provided, this terms is required')
    # parser.add_argument('--recreation', type=bool, default=False, help='Freesurfer Result Dictionary')
    args = parser.parse_args()
    # subdir = '{}/{}'.format(workdir, sub)
    subdir = Path(args.subdir)
    print('surface',subdir)
    produce_weighted_adj_matrix(subdir, recreation=False, args=None)
    # workdir = '/n04dat01/atlas_group/lma/HCP_S1200_individual_MSM_atlas'
    # with open(workdir+'/analysis/sublist_motion_remove.txt', 'r') as f:
    #     sublist = [line.strip() for line in f.readlines()]
        
    # for i, sub in enumerate(sublist[::-1]):
    #     print(i, sub)
    #     # sub = '139637'
    #     try:
    #         subdir = '{}/{}'.format(workdir, sub)
    #         produce_weighted_adj_matrix(subdir, recreation=False, args=None)
    #     except:
    #         print(i, sub, 'error')
    # tran_asc('/n04dat01/atlas_group/lma/DP_MDD_d/ataset/NC_05_0252')
    # produce_32k_surface(args.subdir, args.recreation, args)
    # with open('/DATA/232/lma/data/HCPwork/sub_list.txt', 'r') as f:
    #     subdir_list = ['/DATA/232/lma/data/HCPwork/'+str(name).strip() for name in f.readlines()]
    # for subdir in subdir_list[::-1]:
    #     produce_adj_matrix(subdir, True)


    # patchfile = '/n15dat01/lma/data/MASiVar_prep/anat/patch_adult_all.txt'
    # with open(patchfile,'r') as f:
    #     namelist =  [  line.strip() for line in f.readlines()]

    # sessiondir = namelist[int(args.subdir)]
    # sinput = sessiondir+'/anat/T1_1mm.nii.gz'
    # sub = sessiondir.split('/')[-2] + '-' + sessiondir.split('/')[-1] 
    # multipr = 4 
    # subdir = Path(sessiondir)
    # session = Path(sessiondir).name
    # # if not os.path.exists(subdir):
    # #     os.mkdir(subdir)
    # if not os.path.exists('{}/{}/label/rh.entorhinal_exvivo.1abel'.format(str(subdir),sub)):
    #     os.system('rm -r {}/{}/'.format(str(subdir), sub)) 
    #     os.system('recon-all -i {} -subjid {} -sd {} -all -openmp {}'.format(sinput, sub, str(subdir), multipr))
    
    # os.system('bash /n15dat01/lma/data/MASiVar_prep/pipline/FS2WB.sh {} {} {}'.format(str(subdir.parent), session, sub))
            
