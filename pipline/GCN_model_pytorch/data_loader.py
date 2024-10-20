import os
import random
from random import shuffle
import numpy as np
import torch
import nibabel as nib
from nilearn import surface
import scipy.sparse as sp
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric.utils import get_laplacian
from torch_geometric.data import Data

import os
import pandas as pd



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
	# output_shape = input_shape + (num_classes,)
	# categorical = np.reshape(categorical, output_shape)
	return categorical

def load_atlas_label(atlas, hemi,  software_dir):

	label_file = f'{software_dir}/priors/BAI_Net_{atlas}/{atlas}/fsaverage.{hemi}.{atlas}_Atlas.32k_fs_LR.label.gii'
	print('label_file',label_file)
	label = surface.load_surf_data(label_file)
	select_ind = np.loadtxt(f'{software_dir}/priors/BAI_Net_{atlas}/{atlas}/metric_index_{hemi}.txt').astype('int')
	label = np.array(label)[select_ind]
	if atlas =='BN':
		if hemi =='R':
			label = label//2
			label[label<0]=0
		elif hemi == 'L':
			label = (label+1)//2
			label[label<0]=0
	elif atlas =='HCPparcellation':
		if hemi == 'L':
			label = label - 180
			label[label<0]=0


	return label


class HCP_data(data.Dataset):
	def __init__(self, mode, hemi, atlas, MSMAll=True):
		""" Initializes image paths and preprocessing module."""
		self.workdir = '/share/soft/BAI_Net/pipline/GCN_model_pytorch'
		self.dataset = '/HCP_S1200_individual_MSM_atlas'
		self.MSMAll_signal = '_MSMAll' if MSMAll else ''
		self.MSMAll_signal2 = '_MSMALL' if MSMAll else ''
		self.hemi = hemi
		self.mode = mode
		self.random_train = True
		with open(f'{self.dataset}/analysis/sublist_motion_remove.txt') as f:
			self.sublist = [ line.strip() for line in f.readlines() ]

		self.sublist, self.fingerprintlist, self.surfacelist, self.graphlist = self.check_fingerprint(self.sublist)

		print('total', len(self.sublist), len(self.fingerprintlist), len(self.surfacelist), len(self.graphlist))
		length = len(self.sublist)

		self.atlas = load_atlas_label(atlas, hemi)

		if mode == 'train':
			begin_length = 0
			end_length = int(0.8*length*0.8)

		elif mode =='val':
			begin_length = int(0.8*length*0.8)
			end_length = int(0.8*length)

		elif mode == 'test':
			begin_length = int(0.8*length)
			end_length = int(length)

		elif mode == 'all':
			begin_length = 0
			end_length = len(self.sublist)

		index_path = f'{self.workdir}/dataset/sub_index.npy'
		if os.path.exists(index_path):
			sub_index = np.load(index_path)
		else:
			sub_index = np.arange(length)
			np.random.shuffle(sub_index)
			np.save(index_path, sub_index)

		self.sublist = self.sublist[sub_index[begin_length: end_length]]
		self.fingerprintlist = self.fingerprintlist[sub_index[begin_length: end_length]]
		self.surfacelist = self.surfacelist[sub_index[begin_length: end_length]]
		self.graphlist = self.graphlist[sub_index[begin_length: end_length]]
		self.length = len(self.sublist)

		print("{} image count in {}".format(self.mode, len(self.sublist)))

		return None

	def check_fingerprint(self, sublist):
		sublist_new = []
		pathlist = []
		surfacelist = []
		graphlist = []
		for sub in sublist:
			path = f'{self.dataset}/{sub}/{sub}_{self.hemi}_probtrackx_omatrix2/finger_print_fiber{self.MSMAll_signal2}.npz'
			surface  = f'/OpenData/HCP1200/{sub}/T1w/fsaverage_LR32k/{sub}.{self.hemi}.midthickness{self.MSMAll_signal}.32k_fs_LR.surf.gii'
			graph = f'{self.dataset}/{sub}/surf/weighted_adj_matrix_seed_{self.hemi}.npz'
			if os.path.exists(path) and os.path.exists(graph):
				pathlist.append(path)
				sublist_new.append(path)
				surfacelist.append(surface)
				graphlist.append(graph)
			else:
				print(path, graph)
		return np.array(sublist_new), np.array(pathlist), np.array(surfacelist), np.array(graphlist)

	def __getitem__(self, index):
		# sub = self.sublist[index]

		fingerprint = sp.load_npz(self.fingerprintlist[index]).toarray()
		adj = sp.load_npz(self.graphlist[index]).toarray()
		graph_indice = np.array(np.nonzero(adj))
		graph_weights = adj[graph_indice[0,:], graph_indice[1,:]]


		if self.random_train:
			value = np.random.randint(0,1)
			if value>0.8:
				indice = np.arange(fingerprint)
				np.random.shuffle(indice)
				fingerprint = fingerprint[indice]
				target[:] = 0
				# target[:, 1:] = 0

		fingerprint = torch.from_numpy(fingerprint.astype('float32'))
		graph_indice = torch.from_numpy(graph_indice.astype('int64'))
		graph_weights = torch.from_numpy(graph_weights.astype('float32'))
		target =  torch.from_numpy(self.atlas.astype('long'))

		la_edge_index, la_edge_weight = get_laplacian(graph_indice, graph_weights,  normalization='sym')
		x = np.ones(shape=[la_edge_weight.shape[0], 1])
		laplacian = Data(x=x, edge_index=la_edge_index, edge_weight=la_edge_weight)
		lambda_max = LaplacianLambdaMax()(laplacian)
		lambda_max = np.float32(lambda_max.lambda_max)
		# lambda_max =  np.float32(8.999)
		return fingerprint, graph_indice, graph_weights, target, lambda_max

		
	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.sublist)


def produce_weighted_graph(surface_path, save_path, hemi):

	print('producing adjacent matrix', surface_path, hemi)
	if os.path.exists(save_path):
		print('[save path existing]:',save_path)
		return None

	surf_data = surface.load_surf_mesh(surface_path)
	points = np.array(surf_data[0])
	areas = surf_data[1]

	adj_data = sp.lil_matrix(np.zeros([len(points), len(points)]))
	adj_data[areas[:, 0], areas[:, 1]] = 1./np.sqrt(((points[areas[:, 0],:]-points[areas[:, 1],:])**2).sum(axis=1))
	adj_data[areas[:, 1], areas[:, 2]] = 1./np.sqrt(((points[areas[:, 1],:]-points[areas[:, 2],:])**2).sum(axis=1))
	adj_data[areas[:, 2], areas[:, 0]] = 1./np.sqrt(((points[areas[:, 2],:]-points[areas[:, 0],:])**2).sum(axis=1))
	adj_data = adj_data.T+adj_data
	index_path =  'metric_index_{}.txt'.format(hemi)

	select_ind = np.loadtxt(index_path)
	select_ind = select_ind.astype(int)
	adj_new = adj_data[:, select_ind]
	adj_new = adj_new[select_ind, :]
	adj_new = sp.csr_matrix(adj_new)
	sp.save_npz(str(save_path), adj_new)

	return None


class Inference_data(data.Dataset):
	def __init__(self,  hemi, atlas, dataset_path, sublist_path, software_dir, MSMAll=False):
		""" Initializes image paths and preprocessing module."""
		
		self.dataset = dataset_path
		self.sublist_path = sublist_path
		self.workdir = f'{self.dataset}/resutls'
		self.MSMAll_signal = '_MSMAll' if MSMAll else ''
		self.MSMAll_signal2 = '_MSMALL' if MSMAll else ''
		self.hemi = hemi
		self.atlas = load_atlas_label(atlas, hemi, software_dir)

		if type(sublist_path) == str :
			with open(self.sublist_path) as f:
				self.sublist = [ line.strip() for line in f.readlines() ]
		elif type(sublist_path) == list :
			self.sublist = sublist_path
		self.sublist, self.fingerprintlist, self.surfacelist, self.graphlist = self.check_fingerprint(self.sublist)
		print('total', len(self.sublist), len(self.fingerprintlist), len(self.surfacelist), len(self.graphlist))

		self.length = len(self.sublist)
		print("image count in {}".format( len(self.sublist)))

		return None
	
	def check_fingerprint(self, sublist):
		sublist_new = []
		pathlist = []
		surfacelist = []
		graphlist = []
		for sub in sublist:
			path = f'{self.dataset}/{sub}/{sub}_{self.hemi}_probtrackx_omatrix2/finger_print_fiber{self.MSMAll_signal2}.npz'
			surface  = f'{self.dataset}/{sub}/{sub}/fsaverage_LR32k/{sub}.{self.hemi}.midthickness{self.MSMAll_signal}.32k_fs_LR.surf.gii'
			graph = f'{self.dataset}/{sub}/surf/weighted_adj_matrix_seed_{self.hemi}.npz'
			if not os.path.exists(graph) and os.path.exists(surface):
				produce_weighted_graph(surface_path =surface, save_path=graph, hemi=self.hemi)
			if os.path.exists(path) and os.path.exists(graph):
				pathlist.append(path)
				sublist_new.append(path)
				surfacelist.append(surface)
				graphlist.append(graph)
			else:
				print(path, graph)
		return np.array(sublist_new), np.array(pathlist), np.array(surfacelist), np.array(graphlist)
	
	def __getitem__(self, index):

		fingerprint = sp.load_npz(self.fingerprintlist[index]).toarray()
		adj = sp.load_npz(self.graphlist[index]).toarray()
		graph_indice = np.array(np.nonzero(adj))
		graph_weights = adj[graph_indice[0,:], graph_indice[1,:]]

		fingerprint = torch.from_numpy(fingerprint.astype('float32'))
		graph_indice = torch.from_numpy(graph_indice.astype('int64'))
		graph_weights = torch.from_numpy(graph_weights.astype('float32'))
		target =  torch.from_numpy(self.atlas.astype('long'))

		la_edge_index, la_edge_weight = get_laplacian(graph_indice, graph_weights,  normalization='sym')
		x = np.ones(shape=[la_edge_weight.shape[0], 1])
		laplacian = Data(x=x, edge_index=la_edge_index, edge_weight=la_edge_weight)
		lambda_max = LaplacianLambdaMax()(laplacian)
		lambda_max = np.float32(lambda_max.lambda_max)
		# lambda_max =  np.float32(8.999)
		return fingerprint, graph_indice, graph_weights, target, lambda_max

		
	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.sublist)
	


def get_data_loader(dataset, batch_size, num_workers):
	"""Builds and returns Dataloader."""

	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True)
	return data_loader


def get_evaluation_loader(dataset, batch_size, num_workers):
	"""Builds and returns Dataloader."""
	data_loader = data.DataLoader(dataset = dataset,
								  batch_size = batch_size,
								  shuffle = False,
								  num_workers = num_workers,
								  drop_last = False)
	
	# batchnum = len(image_paths)//batch_size
	# image_paths = image_paths[:batchnum*batch_size].reshape(-1, batch_size)

	return data_loader


if __name__ == '__main__':

	prior_list = ['Yeo7Network', 'Yeo17Network', 'BN', 'HCPparcellation']
	for prior in prior_list:
		for hemi in ['L','R']:
			dataset_path = '/dataset'
			sublist_path = '/dataset/ready_sublist.txt'
			dataset = Inference_data( hemi=hemi, atlas=prior, dataset_path=dataset_path, sublist_path=sublist_path )
