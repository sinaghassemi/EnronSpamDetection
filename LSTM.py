import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from functions import *
from math import ceil
# LSTM




class ClassifierLSTM(object):
	def __init__(self,**kwargs):
		self.batchSize 	= kwargs.get('batchSize',256)
		train_data 	= kwargs.get('train_data',None)
		val_data 	= kwargs.get('val_data',None)
		test_data 	= kwargs.get('test_data',None)
		train_label 	= kwargs.get('train_label',None)
		val_label 	= kwargs.get('val_label',None)
		test_label 	= kwargs.get('test_label',None)
		train_lengths 	= kwargs.get('train_lengths',None)
		val_lengths 	= kwargs.get('val_lengths',None)
		test_lengths 	= kwargs.get('test_lengths',None)
		self.device = kwargs.get('device','cuda')
		self.model = NetworkLSTM(**kwargs).to(self.device)
		self.criterion = nn.BCELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

	def saveWeights(self,fileName):
		torch.save(self.model.state_dict(),fileName)

	def loadWeights(self,fileName):
		self.model.load_state_dict(torch.load(fileName))

	def train(self,loader):
		numMiniBatches = ceil(len(loader) / self.batchSize)
		self.model.train()
		predicted = []
		for mini_batchNum , (minibatch_data,minibatch_label,minibatch_seqLength) in enumerate(loader):
			minibatch_data = minibatch_data.to(self.device)
			minibatch_label = minibatch_label.to(self.device)
			minibatch_seqLength = minibatch_seqLength.to(self.device)
			# get the output from the model
			output = self.model(minibatch_data, minibatch_seqLength)
			# get the loss and backprop
			loss = self.criterion(output, minibatch_label.float())
			self.optimizer.zero_grad() 
			loss.backward()
			# prevent the exploding gradient
			clip=5 # gradient clipping
			nn.utils.clip_grad_norm_(self.model.parameters(), clip)
			self.optimizer.step()
			print("Train : MiniBatch[%3d/%3d]   Train loss:%1.5f"  % (mini_batchNum,numMiniBatches,loss.item()),end="\r")
		return predicted
	def predict(self,loader):
		numMiniBatches = ceil(len(loader) / self.batchSize)
		self.model.eval()
		outputs = []
		predicted = []
		labels = []
		for mini_batchNum , (minibatch_data,minibatch_label,minibatch_seqLength) in enumerate(loader):
			minibatch_data = minibatch_data.to(self.device)
			minibatch_label = minibatch_label.to(self.device)
			minibatch_seqLength = minibatch_seqLength.to(self.device)
			output = self.model(minibatch_data, minibatch_seqLength)
			loss = self.criterion(output, minibatch_label.float())
			print("Val : MiniBatch[%3d/%3d]   Val loss:%1.5f"  % (mini_batchNum,numMiniBatches,loss.item()),end="\r")
			outputs += output.to('cpu').detach().numpy().squeeze().tolist()
			predicted += (output.to('cpu')>0.5).numpy().squeeze().tolist()
			labels += minibatch_label.to('cpu').numpy().astype(np.uint8).squeeze().tolist()
		return outputs,predicted,labels

class NetworkLSTM(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		outputSize = kwargs.get('outputSize',1)
		numLayers = kwargs.get('numLayers',1)
		self.hiddenSize = kwargs.get('hiddenSize',4)
		embedSize = kwargs.get('embedSize',4)
		vocabSize = kwargs.get('vocabSize',1000)
		dropout = kwargs.get('dropout',0.2)
		dropoutLSTM = kwargs.get('dropoutLSTM',0.2)
		self.device = kwargs.get('device','cpu')
		# embedding layer 
		self.embedding = nn.Embedding(vocabSize, embedSize)
		# LSTM Cells
		self.lstm = nn.LSTM(embedSize, self.hiddenSize, numLayers, dropout = dropoutLSTM, batch_first=True)
		# last layer fully connected and sigmoid layers
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(self.hiddenSize, outputSize)
		self.sig = nn.Sigmoid()
	def forward(self, x, seqLengths):
		# x is a tensor with size of mini-batch size X max Sequence Lenght (other sequences are padded with zeros)
		# Ex: x : 32 X 3000 
		# embedding the input words
		embeddedOutputs = self.embedding(x)
		# embeddedOutputs has size of mini-batch size X max Sequence Lenght X embedding dimensions
		# Ex: embeddedOutputs : 32 X 3000 X 64
		# pack the input ( removing padding for the sequences with length smaller than max length)
		packedInput = pack_padded_sequence(embeddedOutputs, seqLengths.cpu().numpy(), batch_first=True)
		# packedInput includes :
		# 1. the packed input with size of 
		# (sum of all sequence lengths in the batch) X (embedding dimensions) , Ex: 50000 x 64
		# 2. And also it includes the batch size for each time (instance)
		# inputing the following data class to rnn, the rnn will input batches of different sizes, the sequence with maximum element will be in all batches and sequence with one element will only be in one batche therefore allowing training rnn with sequences of different lengths
		# Ex:
		# Hi  	John  	How 	things 	are 	going  
		# Hello	see	you	soon	-	-
		# Fine	-	-	-	-	-
		# [3	2	2	2	1	1]
		packedOutput, (ht, ct) = self.lstm(packedInput, None)
		# packedOutput includes data and batch sizes as well, output size is
		# (sum of all sequence lengths in the batch) X (hidden size)
		# unpacking the output and recover the paddings
		output, inputSizes = pad_packed_sequence(packedOutput, batch_first=True)
		# the output size : batch size X max seq length in the batch X hidden size
		# inputSizes = the length of samples in the batch
		# Get the index for last element for each sample in the batch
		lastWordIndexes = (inputSizes - 1).to(self.device) 
		lastWordIndexes = lastWordIndexes.view(-1, 1).unsqueeze(2).repeat(1, 1, self.hiddenSize)
		# for each sample in batch we want the output at the last element
		output = torch.gather(output, 1, lastWordIndexes).squeeze() # [batch_size, hidden_dim]
		output = self.dropout(output)
		output = self.fc(output).squeeze() 
		output = self.sig(output)
		return output






