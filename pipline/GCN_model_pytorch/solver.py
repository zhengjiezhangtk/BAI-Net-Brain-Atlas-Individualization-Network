import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import ChenNet
import csv
import pandas as pd
# from data_loader import get_data_loader, get_evalutation_loader
import random
import os
import nibabel as nib
from scipy.stats import pearsonr
from data_loader import get_data_loader, get_evaluation_loader

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader, scan1_loader='',scan2_loader=''):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader
		self.scan1_loader = scan1_loader
		self.scan2_loader = scan2_loader
		# Models
		self.gcn_model = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.model_type = config.model_type
		self.hemi = config.hemi
		self.atlas = config.atlas
		# self.criterion = torch.nn.MSELoss(reduction='mean')
		# self.criterion = torch.nn.CrossEntropyLoss()
		self.criterion = torch.nn.NLLLoss()

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2
		
		# Training settings
		self.num_epochs = config.num_epochs 
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		print('model:', self.model_type, 'batch_size:', self.batch_size)
		
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""

		if self.atlas == 'BN':
			output_number = 106
		elif self.atlas == 'Yeo17Network':
			output_number = 18
		elif self.atlas == 'Yeo7Network':
			output_number = 8
		elif self.atlas == 'HCPparcellation':
			output_number = 181

		self.gcn_model = ChenNet(72, output_number)
		self.optimizer = optim.Adam(list(self.gcn_model.parameters()), self.lr, [self.beta1, self.beta2])
		self.gcn_model.to(self.device)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.gcn_model.zero_grad()

	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#

		gcn_model_path = os.path.join(self.model_path, '%s-%s-%s.pkl' %(self.model_type, self.atlas, self.hemi))
		load_path = gcn_model_path

		# gcn_model Train
		if os.path.isfile(load_path):
			# Load the pretrained Encoder
			self.gcn_model.load_state_dict(torch.load(load_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,load_path))
			          
		# Train for Encoder
		lr = self.lr
		best_gcn_model_loss = 100000
		best_epoch = 0
		for epoch in range(self.num_epochs):
			self.gcn_model.train(True)
			epoch_loss = 0
			length = 0
			acc_list = []
			item = 0
			for i, (fingerprint, graph_indice, graph_weights, target, lambda_max) in enumerate(self.train_loader):     
       
				fingerprint = fingerprint[0].to(self.device)
				graph_indice = graph_indice[0].to(self.device)
				graph_weights = graph_weights[0].to(self.device)
				target = target[0].to(self.device)
				length += 1
				parcellation = self.gcn_model(fingerprint, graph_indice, graph_weights, lambda_max)

				loss = self.criterion(parcellation, target)
				epoch_loss += loss.item()
				self.reset_grad()
				loss.backward()
				self.optimizer.step()
			    
				
				parcellation = parcellation.data.cpu().numpy()
				target = target.data.cpu().numpy()
				

				# group_label = np.argmax(target, axis=1)
				group_label = target
				individual_label = np.argmax(parcellation, axis=1)
				if group_label.sum() == 0:
					pass
				else:
					item = item +1
					acc = (individual_label == group_label).sum()/len(individual_label)
					acc_list.append(acc)
				print( 'epoch: ', epoch, 'train batch number: ', i, 'training loss:', loss.item(),'acc:', acc) 
			# # Print the log info
			print('[Train] ', 'Epoch [%d/%d], Loss: %.4f, acc: %.4f' % (epoch+1, self.num_epochs, epoch_loss/length, np.array(acc_list).sum()/item))

			# # Decay learning rate
			if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
				lr -= (self.lr / float(self.num_epochs_decay))
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = lr
				print ('Decay learning rate to lr: {}.'.format(lr))
			
			# ===================================== Validation ====================================#
			print('begin validation')
			self.gcn_model.train(False)
			self.gcn_model.eval()
			
			epoch_loss = 0
			length = 0
			acc_list = []
			item = 0
			for i, (fingerprint, graph_indice, graph_weights, target, lambda_max) in enumerate(self.valid_loader):     
       
				fingerprint = fingerprint[0].to(self.device)
				graph_indice = graph_indice[0].to(self.device)
				graph_weights = graph_weights[0].to(self.device)
				target = target[0].to(self.device)
				lambda_max = lambda_max[0].to(self.device)
				length += 1
				parcellation = self.gcn_model(fingerprint, graph_indice, graph_weights, lambda_max)

				loss = self.criterion(parcellation, target)
				epoch_loss += loss.item()
				self.reset_grad()
				loss.backward()
				self.optimizer.step()
			    
				print( 'epoch: ', epoch, 'val batch number: ', i, 'val loss:', loss.item()) 
				parcellation = parcellation.data.cpu().numpy()
				target = target.data.cpu().numpy()
				

				# group_label = np.argmax(target, axis=1)
				group_label = target
				individual_label = np.argmax(parcellation, axis=1)
				if group_label.sum() == 0:
					pass
				else:
					item = item +1
					acc = (individual_label == group_label).sum()/len(individual_label)
					acc_list.append(acc)

			print('[Validation] ', epoch, 'epoch_loss:', epoch_loss/length, 'acc',  np.array(acc_list).sum()/item, ' best_gcn_model_loss', best_gcn_model_loss)

			# Save Best gcn_model model
			if best_gcn_model_loss > epoch_loss:
				best_gcn_model_loss = epoch_loss
				best_epoch = epoch
				best_gcn_model = self.gcn_model.state_dict()
				print('Best %s model loss : %.4f at epoch %s' % (self.model_type, best_gcn_model_loss, best_epoch))	
				torch.save(best_gcn_model, gcn_model_path)

	def test(self):
		#===================================== Test ====================================#
		print('testing performance')
		
		gcn_model_path = os.path.join(self.model_path, '%s-%s-%s.pkl' %(self.model_type, self.atlas, self.hemi))
	

		self.build_model()
		self.gcn_model.load_state_dict(torch.load(gcn_model_path))
		print('Finish loading model parameters')

		self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
		self.print_network(self.gcn_model, self.model_type)
		self.gcn_model.train(False)
		self.gcn_model.eval()

		epoch_loss = 0
		length = 0
		acc_list = []
		item = 0
		for i, (fingerprint, graph_indice, graph_weights, target, lambda_max) in enumerate(self.train_loader):     
	
			fingerprint = fingerprint[0].to(self.device)
			graph_indice = graph_indice[0].to(self.device)
			graph_weights = graph_weights[0].to(self.device)
			target = target[0].to(self.device)
			lambda_max = lambda_max[0].to(self.device)
			length += 1
			parcellation = self.gcn_model(fingerprint, graph_indice, graph_weights, lambda_max)

			loss = self.criterion(parcellation, target)
			epoch_loss += loss.item()
			self.reset_grad()
			loss.backward()
			self.optimizer.step()
			
			
			parcellation = parcellation.data.cpu().numpy()
			target = target.data.cpu().numpy()
			

			group_label = np.argmax(target, axis=1)
			individual_label = np.argmax(parcellation, axis=1)
			if group_label.sum() == 0:
				pass
			else:
				item = item +1
				acc = (individual_label == group_label).sum()/len(individual_label)
				acc_list.append(acc)
			print( 'val batch number: ', i, 'val loss:', loss.item(),'acc:', acc) 
		# # Print the log info
		print('[Test] ', ' Loss: %.4f, acc: %.4f' % (epoch_loss/length, np.array(acc_list).sum()/item))


	def replication(self):
		gcn_model_path = os.path.join(self.model_path, '%s-%s-%s.pkl' %(self.model_type, self.atlas, self.hemi))
	
		self.build_model()
		self.gcn_model.load_state_dict(torch.load(gcn_model_path))
		print('Finish loading model parameters')

		self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
		self.print_network(self.gcn_model, self.model_type)
		self.gcn_model.train(False)
		self.gcn_model.eval()

		epoch_loss = 0
		length = 0
		
		test_list = []
		retest_list = []
		acc_list = []
		length = 0
		for i, (fingerprint, graph_indice, graph_weights, target, lambda_max) in enumerate(self.scan1_loader):  
			fingerprint = fingerprint[0].to(self.device)
			graph_indice = graph_indice[0].to(self.device)
			graph_weights = graph_weights[0].to(self.device)
			target = target[0].to(self.device)
			lambda_max = lambda_max[0].to(self.device)
			length += 1
			parcellation = self.gcn_model(fingerprint, graph_indice, graph_weights, lambda_max)
			parcellation = parcellation.data.cpu().numpy()
			individual_label = np.argmax(parcellation, axis=1)
			test_list.append(individual_label)

		for i, (fingerprint, graph_indice, graph_weights, target, lambda_max) in enumerate(self.scan2_loader):  
			fingerprint = fingerprint[0].to(self.device)
			graph_indice = graph_indice[0].to(self.device)
			graph_weights = graph_weights[0].to(self.device)
			target = target[0].to(self.device)
			lambda_max = lambda_max[0].to(self.device)

			parcellation = self.gcn_model(fingerprint, graph_indice, graph_weights, lambda_max)
			parcellation = parcellation.data.cpu().numpy()
			individual_label = np.argmax(parcellation, axis=1)
			retest_list.append(individual_label)
		for  i in range(length):
			acc = retest_list[i]==test_list[i]
			acc_list.append(acc)
		acc_list = np.array(acc_list)
		print('reproducibility',  acc_list)
		return None




