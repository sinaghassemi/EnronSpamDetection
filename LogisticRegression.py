import numpy as np
import torch.nn as nn
import torch
from math import ceil
from functions import *

class ClassifierLogisticRegression(object):
	def __init__(self,**kwargs):
		self.batchSize 	= kwargs.get('batchSize',256)
		self.device = kwargs.get('device','cpu')
		self.model = logisticRegressionNet(**kwargs).to(self.device)
		self.criterion = nn.BCELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)

	def saveWeights(self,fileName):
		torch.save(self.model.state_dict(),fileName)

	def loadWeights(self,fileName):
		self.model.load_state_dict(torch.load(fileName))

	def train(self,loader):
		numMiniBatches = ceil(len(loader) / self.batchSize)
		self.model.train()
		for mini_batchNum , (minibatch_data,minibatch_label) in enumerate(loader):
			minibatch_data = minibatch_data.to(self.device)
			minibatch_label = minibatch_label.to(self.device)
			output = self.model(minibatch_data)
			loss = self.criterion(output, minibatch_label.float())
			self.optimizer.zero_grad() 
			loss.backward()
			self.optimizer.step()
			#print("Train : MiniBatch[%3d/%3d]   Train loss:%1.5f"  % (mini_batchNum,numMiniBatches,loss.item()),end="\r")
	def predict(self,loader):
		numMiniBatches = ceil(len(loader) / self.batchSize)
		self.model.eval()
		conf = np.zeros((2,2))
		for mini_batchNum , (minibatch_data,minibatch_label) in enumerate(loader):
			minibatch_data = minibatch_data.to(self.device)
			minibatch_label = minibatch_label.to(self.device)
			output = self.model(minibatch_data)
			loss = self.criterion(output, minibatch_label.float())
			#print("Val : MiniBatch[%3d/%3d]   Val loss:%1.5f"  % (mini_batchNum,numMiniBatches,loss.item()),end="\r")
			predicted = (output.to('cpu')>0.5).numpy().squeeze()
			groundtruth = minibatch_label.to('cpu').numpy()
			groundtruth = groundtruth.astype(np.int32)
			minibatch_conf = computeConfMatrix(predicted,groundtruth)
			conf += minibatch_conf
		hamF1Score,spamF1Score =  performanceMetrics(conf)
		return hamF1Score,spamF1Score


class logisticRegressionNet(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		inputSize = kwargs.get('inputSize',32)
		outputSize = kwargs.get('outputSize',1)
		self.device = kwargs.get('device','cpu')
		# Fully connected layer
		self.fc = nn.Linear(inputSize, outputSize)
		self.sig = nn.Sigmoid()
	def forward(self, x):
		output = self.fc(x).squeeze() 
		output = self.sig(output)
		return output













		
