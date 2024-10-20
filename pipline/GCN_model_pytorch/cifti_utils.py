import nibabel as nib
from nilearn import surface
import numpy as np
import hdf5storage
import os

template_32kfsLR_dir = '/cbica/home/malia/my_template'
wb_command_dir = '/cbica/home/malia/software/workbench/exe_rh_linux64'

gii_surf_template = {'pial_L': os.path.join(template_32kfsLR_dir, 'fsaverage.L.pial.32k_fs_LR.surf.gii'),  \
                'pial_R':os.path.join(template_32kfsLR_dir, 'fsaverage.R.pial.32k_fs_LR.surf.gii'), \
                'middle_L':os.path.join(template_32kfsLR_dir, 'fsaverage.L.midthickness.32k_fs_LR.surf.gii'), \
                'middle_R':os.path.join(template_32kfsLR_dir, 'fsaverage.R.midthickness.32k_fs_LR.surf.gii'), \
                'white_L':os.path.join(template_32kfsLR_dir, 'fsaverage.L.white.32k_fs_LR.surf.gii'), \
                'white_R':os.path.join(template_32kfsLR_dir, 'fsaverage.R.white.32k_fs_LR.surf.gii') }

gii_func_template = {'L': os.path.join(template_32kfsLR_dir, 'fsaverage.L.Yeo17Network_Atlas.32k_fs_LR.label.gii'), \
                     'R': os.path.join(template_32kfsLR_dir, 'fsaverage.R.Yeo17Network_Atlas.32k_fs_LR.label.gii')}

gii_label_template = {'L':os.path.join(template_32kfsLR_dir, 'fsaverage.L.Yeo17Network_Atlas.32k_fs_LR.label.gii'), \
                      'R':os.path.join(template_32kfsLR_dir, 'fsaverage.R.Yeo17Network_Atlas.32k_fs_LR.label.gii')}

gii_atlasroi_template = {'L':os.path.join(template_32kfsLR_dir, 'L.atlasroi.32k_fs_LR.shape.gii'), \
                        'R':os.path.join(template_32kfsLR_dir, 'R.atlasroi.32k_fs_LR.shape.gii')}

gii_indexroi_template = {'L':os.path.join(template_32kfsLR_dir, 'metric_index_L.txt'), \
                        'R':os.path.join(template_32kfsLR_dir, 'metric_index_R.txt')}

gii_index_mask_remove_template = {'L':os.path.join(template_32kfsLR_dir, 'mask_remove_region_L.txt'), \
                                'R':os.path.join(template_32kfsLR_dir, 'mask_remove_region_L.txt')}

gii_index_mask_remove_front_template = {'L':os.path.join(template_32kfsLR_dir, 'mask_remove_front_L.txt'), \
                                'R':os.path.join(template_32kfsLR_dir, 'mask_remove_front_R.txt')}

gii_vertexarea_template = {'L':os.path.join(template_32kfsLR_dir, 'fsaverage.L.vertexarea.func.gii'), \
                        'R':os.path.join(template_32kfsLR_dir, 'fsaverage.R.vertexarea.func.gii')}

gii_vertexarea_mask = {'L':os.path.join(template_32kfsLR_dir, 'cortical_new.L.atlasroi.32k_fs_LR.shape.gii'), \
                        'R':os.path.join(template_32kfsLR_dir, 'cortical_new.R.atlasroi.32k_fs_LR.shape.gii')}

gii_vertexarea_remove_front_mask = {'L':os.path.join(template_32kfsLR_dir, 'snr.L.shape.gii'), \
                                    'R':os.path.join(template_32kfsLR_dir, 'snr.R.shape.gii')}

gii_group_parcellation = {'L':os.path.join(template_32kfsLR_dir, 'fsLR32k.final_label_s.L.label.gii'), \
                          'R':os.path.join(template_32kfsLR_dir, 'fsLR32k.final_label_s.R.label.gii')}


def saveFuncGii(value,  hemi,  outputdir, filename, dtype='float32',template_path='default'):
    if template_path=='default':
        template_path=gii_label_template[hemi]
    template = nib.load(template_path)
    template.remove_gifti_data_array(0)
    if (value.shape[0]==29696) and hemi=='L':
        print('extending into 32k', hemi)
        if value.ndim==2:
            value_new = np.zeros([32492, value.shape[1]])
        else:
            value_new = np.zeros([32492])
        indice_L = np.loadtxt(gii_indexroi_template['L']).astype('int')
        value_new[indice_L] = value
        value = value_new.copy()
    elif (value.shape[0]==29716) and hemi=='R':
        print('extending into 32k', hemi)
        if value.ndim==2:
            value_new =  np.zeros([32492, value.shape[1]])
        else:
            value_new = np.zeros([32492])
        indice_R = np.loadtxt(gii_indexroi_template['R']).astype('int')
        value_new[indice_R] = value
        value = value_new.copy()

    if value.ndim==2:
        print('cifti dim:',value.ndim)
        for i in range(value.shape[1]):
            template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.array(value[:,i], dtype)))
    else:
        template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.array(value, dtype)))
    nib.loadsave.save(template, os.path.join( outputdir, filename))
    return None


def saveLabelGii( label,  hemi,  outputdir, filename, dtype='int32', template_path='default'):
    '''
    save data as gii label format
    template_path: the path template
    the length of data is not required same to the template but should match the relevent surface points' size
    path and save_name is the saving location of gii file
    '''
    label = np.array(label)
    if template_path=='default':
        template_path=gii_label_template[hemi]
    template = nib.load(template_path)

    labelarray = nib.gifti.gifti.GiftiDataArray(label.astype(dtype))
    new_label = nib.gifti.gifti.GiftiImage( meta = template.meta, labeltable = template.labeltable)
    new_label.add_gifti_data_array(labelarray)
    nib.loadsave.save(new_label, os.path.join( outputdir, filename))
    return None


def save_cifti_dtseries(value, template_path, savepath, tr_l=0.73):
    """
    template should have the same number on second shape of value.
    """

    a = nib.load(template_path)
    b = a.get_fdata()
    h = a.header

    # f_new = value
    ax_0 = nib.cifti2.SeriesAxis(start = 0, step = tr_l, size = value.shape[0]) 
    ax_1 = h.get_axis(1)

    # Create new header and cifti object
    new_h = nib.cifti2.Cifti2Header.from_axes((ax_0, ax_1))
    cii_new = nib.cifti2.Cifti2Image(value.astype('float'), new_h)

    # Write the new cifti
    nib.save(cii_new, savepath)

    return None


def S32kto29k(label, hemi):
    '''
    remove the middle-wall signal for brain surface files 
    label can be int list, float list type
    hemi is in 'L', 'R'
    '''
    # shape = np.array(surface.load_surf_data(gii_atlasroi_template[hemi])).astype('int')
    indices = np.loadtxt(gii_indexroi_template[hemi]).astype('int')
    label = label[indices]
    return label


def S29kto32k(label, hemi):
    '''
    back mapping from the middle-wall-removal signal into whole brain signal for brain surface files 
    label can be int list, float list type
    hemi is in 'L', 'R'
    '''
    label = np.array(label)
    # shape = np.array(surface.load_surf_data(gii_atlasroi_template[hemi])).astype('int')
    indices = np.loadtxt(gii_indexroi_template[hemi]).astype('int')
    if label.ndim==1:
        new = np.zeros(32492)
    elif label.ndim==2:
        new = np.zeros([32492, label.shape[1]])
    new[indices] = label
    return new



def loading2parcellations(UVmat_path, outputdir, filename, var='V', save=True, border=True, mask=False):

    if save:
        length = int(surface.load_surf_data(gii_atlasroi_template['L']).sum())
        label_L = S29kto32k(label[:length], 'L')
        label_R = S29kto32k(label[length:], 'R')
        label = np.concatenate([label_L, label_R])

        saveLabelGii(label_L, 'L', outputdir, 'fsLR32k.{}.L.label.gii'.format(filename) )
        saveLabelGii(label_R, 'R', outputdir, 'fsLR32k.{}.R.label.gii'.format(filename) )

    if border:
        lable2border('L', os.path.join(outputdir, 'fsLR32k.{}.L.label.gii'.format(filename)), outputdir, 'fsLR32k.{}.L.border'.format(filename))
        lable2border('R', os.path.join(outputdir, 'fsLR32k.{}.R.label.gii'.format(filename)), outputdir, 'fsLR32k.{}.R.border'.format(filename))
    
    return label

def value2parcellations(value, outputdir, filename, save=True, border=False):
    '''
    filename should not include the File extension, and file would add automatedly with hemi label.
    value should be in shape [vexter, network] or [vexter]
    ''' 
    if label.ndim==2:
        label = np.argmax(value, axis=1)+1
    elif label.ndim==1:
        label = value
    else:
        print('label is in wrong shape!')
        return None
    
    if save:
        length = int(surface.load_surf_data(gii_atlasroi_template['L']).sum())
        if label.shape[0] == 59412 :
            label_L = S29kto32k(label[:length], 'L')
            label_R = S29kto32k(label[length:], 'R')
        elif label.shape[0] == 64984 :
            label_L = S29kto32k(label[:32492], 'L')
            label_R = S29kto32k(label[32492:], 'R')
            
        label = np.concatenate([label_L, label_R])

        saveLabelGii(label_L, 'L', outputdir, 'fsLR32k.{}.L.label.gii'.format(filename) )
        saveLabelGii(label_R, 'R', outputdir, 'fsLR32k.{}.R.label.gii'.format(filename) )

    if border:
        lable2border('L', os.path.join(outputdir, 'fsLR32k.{}.L.label.gii'.format(filename)), outputdir, 'fsLR32k.{}.L.border'.format(filename))
        lable2border('R', os.path.join(outputdir, 'fsLR32k.{}.R.label.gii'.format(filename)), outputdir, 'fsLR32k.{}.R.border'.format(filename))
    
    return label


def atlas_dice(label1, label2):
    dice = (label1==label2).sum()/len(label1)
    return dice


def loading2funcgii(UVmat_path, outputdir, filename, var='V'):
    '''
    filename should not include the File extension, and file would add automatedly with hemi label.
    '''
    if isinstance(UVmat_path, str ):
        UV_mat = hdf5storage.loadmat(UVmat_path)
        V_mat = UV_mat[var]
        print(V_mat.shape)
        if V_mat.shape == (1,1):
            V_mat = V_mat[0][0]
        if V_mat.ndim == 4:
            V_mat = V_mat[0][0]
    else:
        V_mat = UVmat_path

    length = int(surface.load_surf_data(gii_atlasroi_template['L']).sum())
    # for i in range(V_mat.shape[1]):
    value =  V_mat  #V_mat[:,i]
    value_L = S29kto32k(value[:length], 'L')
    value_R = S29kto32k(value[length:], 'R')
    saveFuncGii(value_L, 'L',  outputdir, 'fsLR32k.{}.L.func.gii'.format(filename), dtype='float32',template_path='default')
    saveFuncGii(value_R, 'R',  outputdir, 'fsLR32k.{}.R.func.gii'.format(filename), dtype='float32',template_path='default')
    return None


def calculate_loading_difference( UV1mat_path, UV2mat_path):
    'calculating  loading difference'
    V1_mat = get_loading_Vmat(UV1mat_path)
    V2_mat = get_loading_Vmat(UV2mat_path)
    diff_value = V1_mat - V2_mat
    return diff_value


def calculate_loading_difference_batch( V1_mat, V2_mat):
    'calculating batch loading difference'
    # V1_mat = get_loading_Vmat(UV1mat_path)
    # V2_mat = get_loading_Vmat(UV2mat_path)
    diff_value = V1_mat - V2_mat
    return diff_value


def calculate_border_difference(UV1mat_path, UV2mat_path):
    "discrete difference: 1 for state , and -1 for rest1"
    V1_mat = get_loading_Vmat(UV1mat_path)
    V2_mat = get_loading_Vmat(UV2mat_path)
    V1_discreted_net = np.argmax(V1_mat, axis=1)
    V2_discreted_net = np.argmax(V2_mat, axis=1) 
    V1_onehot = np.array([ (V1_discreted_net==i).astype('int') for i in range(17) ])
    V2_onehot = np.array([ (V2_discreted_net==i).astype('int') for i in range(17) ])
    diff_value = V1_onehot - V2_onehot 
    return diff_value.T


def calculate_border_difference_batch(V1_mat, V2_mat):
    "discrete difference: 1 for state , and -1 for rest1"
    difflist = []
    for s in range(V1_mat.shape[0]):
        V1_onehot = np.array([ (V1_mat[s,:]==i).astype('int') for i in range(1,18) ])
        V2_onehot = np.array([ (V2_mat[s,:]==i).astype('int') for i in range(1,18) ])
        diff_value = V1_onehot - V2_onehot 
        difflist.append(diff_value.T)
    difflist = np.array(difflist)
    print('border array:', difflist.shape)
    return difflist

def get_loading_Vmat( UV1mat_path, Var='V'):
    UV1_mat = hdf5storage.loadmat(UV1mat_path)
    V_mat = UV1_mat[Var]
    if V_mat.shape == (1,1):
        V_mat = V_mat[0][0]
    if V_mat.ndim == 4:
        V_mat = V_mat[0][0]
    return V_mat


def get_loading_Umat( UV1mat_path):
    UV1_mat = hdf5storage.loadmat(UV1mat_path)
    U_mat = UV1_mat['U']
    if U_mat.shape == (1,1):
        U_mat = U_mat[0][0]
    if U_mat.ndim == 4:
        U_mat = U_mat[0][0]
    return U_mat


def lable2border(hemi, labelpath, output_dir, filename):
    '''
    file name shuold not include the file extension
    '''
    savepath  = os.path.join(output_dir, filename)
    command = '{}/wb_command -label-to-border {} {} {}'.format(wb_command_dir, gii_surf_template['middle_'+hemi], labelpath, savepath)
    os.system(command)
    return None


def mapping_select_label_with_value( select_label, paried_value, hemi, savedir, filename):
    '''
    label should be  in 32492
    '''
    template_path = gii_group_parcellation[hemi]
    group_label = np.array(surface.load_surf_data(template_path))
    label_new = np.zeros_like(group_label).astype('float')

    for label, value in zip(select_label, paried_value):
        indice = group_label==label

        label_new[indice] = value

    print('saving dir', os.path.join(savedir, '{}.{}'.format(hemi, filename)))
    saveFuncGii(label_new,  hemi,  savedir, filename, dtype='float32', template_path= template_path)


    return None


def cal_surf_area(midsurf_L_path, midsurf_R_path, label_L, label_R, labels, filestring, savedir):
    targetfile_L = '{}/{}_areasize.L.func.gii'.format(savedir, filestring)
    if not os.path.exists(targetfile_L):
        command = 'wb_command -surface-vertex-areas {} {}'.format( midsurf_L_path, targetfile_L)
        n = os.system(command)
        print('saving', targetfile_L, 'command state', n)

    targetfile_R = '{}/{}_areasize.R.func.gii'.format(savedir, filestring)
    if not os.path.exists(targetfile_R):
        command = 'wb_command -surface-vertex-areas {} {}'.format( midsurf_R_path, targetfile_R)
        n = os.system(command)
        print('saving', targetfile_R, 'command state', n)

    areasize_L = surface.load_surf_data(targetfile_L)
    areasize_R = surface.load_surf_data(targetfile_R)
    areasize = np.concatenate([areasize_L, areasize_R])
    arealabel = np.concatenate([label_L, label_R])
    arealist = []
    for i in labels:
        tmp = areasize[arealabel == i].sum()
        arealist.append(tmp)
    arealist = np.array(arealist)
    np.save('{}/{}_areasize.npy'.format(savedir, filestring), arealist)
    return arealist




if __name__ == '__main__':
    sub = '100307'
    # values = np.random.randint(10, size=[32492, 17])
    # np
    # aa = hdf5storage.loadmat('/gpfs/fs001/cbica/home/malia/project/HCP_indi_script/Result/HCP_1200_p3/rest/FN_17/init_robust/init.mat')['initV']
    # a1 = S29kto32k(aa[:29696,:],'L')
    # a2 = S29kto32k(aa[29696:,:],'R')
    # for i in range(17):
    #     saveFuncGii(a1[:,i],  'L',  '/gpfs/fs001/cbica/home/malia/project/HCP_indi_script/Result/HCP_1200_p3/rest/FN_17/', 'group_test{}.L.func.gii'.format(i), dtype='float32',template_path='default')
    #     saveFuncGii(a2[:,i],  'R',  '/gpfs/fs001/cbica/home/malia/project/HCP_indi_script/Result/HCP_1200_p3/rest/FN_17/', 'group_test{}.R.func.gii'.format(i), dtype='float32',template_path='default')
    analysis_dir = '/gpfs/fs001/cbica/home/malia/project/HCP_indi_script/Result/HCP_1200_p3/rest/FN_17/loading_analyze/'
    aa = np.load('/gpfs/fs001/cbica/home/malia/project/HCP_indi_script/Result/HCP_1200_p3/rest/FN_17/loading_analyze/network_mask.npy')
    aa = aa + 0.01
    print(aa.shape)
    saveFuncGii(aa[:29696],  'L',  analysis_dir, 'network_mask.L.func.gii', dtype='float32')
    saveFuncGii(aa[29696:],  'R',  analysis_dir, 'network_mask.R.func.gii', dtype='float32')
    # UVsavepath1 = '/gpfs/fs001/cbica/home/malia/project/HCP_indi_script/Result/HCP_1200/rest/FN_17/Individual/{}/task-SOCIAL/HCP_sbj1_comp17_alphaS21_2_alphaL10_vxInfo0_ard1_eta0/final_UV.mat'.format(sub)
    # print(gii_func_template)
    # UVsavepath2 = '/gpfs/fs001/cbica/home/malia/project/HCP_indi_script/Result/HCP_1200/rest/FN_17/Individual/{}/task-REST1/HCP_sbj1_comp17_alphaS21_2_alphaL10_vxInfo0_ard1_eta0/final_UV.mat'.format(sub)
    # calculate_loading_difference( UVsavepath1, UVsavepath2)