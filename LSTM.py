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
		self.__net = NetworkLSTM(**kwargs).to(self.device)
		self.loaderTrain = EnronDataLoader(data=train_data,label=train_label,seqLengths=train_lengths,batchSize=self.batchSize,shuffle=True)
		self.loaderVal   = EnronDataLoader(data=val_data,label=val_label,seqLengths=val_lengths,batchSize=self.batchSize,shuffle=False)
		self.loaderTest  = EnronDataLoader(data=test_data,label=test_label,seqLengths=test_lengths,batchSize=self.batchSize,shuffle=False)
		self.val_hamF1Score = 0
		self.val_spamF1Score = 0
		self.test_hamF1Score = 0
		self.test_spamF1Score = 0
		self.criterion = nn.BCELoss()
		self.optimizer = torch.optim.Adam(self.__net.parameters(), lr=0.005)
	def train(self):
		numMiniBatches = ceil(len(self.loaderTrain) / self.batchSize)
		self.__net.train()
		for mini_batchNum , (minibatch_data,minibatch_label,minibatch_seqLength) in enumerate(self.loaderTrain):
			minibatch_data = minibatch_data.to(self.device)
			minibatch_label = minibatch_label.to(self.device)
			minibatch_seqLength = minibatch_seqLength.to(self.device)
			# get the output from the model
			output = self.__net(minibatch_data, minibatch_seqLength)
			# get the loss and backprop
			loss = self.criterion(output, minibatch_label.float())
			self.optimizer.zero_grad() 
			loss.backward()
			# prevent the exploding gradient
			clip=5 # gradient clipping
			nn.utils.clip_grad_norm_(self.__net.parameters(), clip)
			self.optimizer.step()
			print("Train : MiniBatch[%3d/%3d]   Train loss:%1.5f"  % (mini_batchNum,numMiniBatches,loss.item()))#,end="\r")
	def val(self):
		numMiniBatches = ceil(len(self.loaderVal) / self.batchSize)
		self.__net.eval()
		conf = np.zeros((2,2))
		for mini_batchNum , (minibatch_data,minibatch_label,minibatch_seqLength) in enumerate(self.loaderVal):
			minibatch_data = minibatch_data.to(self.device)
			minibatch_label = minibatch_label.to(self.device)
			minibatch_seqLength = minibatch_seqLength.to(self.device)
			# get the output from the model
			#print(self.__net.lstm.weight_ih_l0)
			output = self.__net(minibatch_data, minibatch_seqLength)
			# get the loss and backprop
			loss = self.criterion(output, minibatch_label.float())
			print("Val : MiniBatch[%3d/%3d]   Val loss:%1.5f"  % (mini_batchNum,numMiniBatches,loss.item()))#,end="\r")
			predicted = (output.to('cpu')>0.5).numpy()
			groundtruth = minibatch_label.to('cpu').numpy()
			groundtruth = groundtruth.astype(np.int32)
			minibatch_conf = computeConfMatrix(predicted,groundtruth)
			conf += minibatch_conf
		print(conf)
		self.val_hamF1Score,self.val_spamF1Score =  performanceMetrics(conf)
	def test(self):
		numMiniBatches = ceil(len(self.loaderTest) / self.batchSize)
		self.__net.eval()
		conf = np.zeros((2,2))	
		for mini_batchNum , (minibatch_data,minibatch_label,minibatch_seqLength) in enumerate(self.loaderTest):
			minibatch_data = minibatch_data.to(self.device)
			minibatch_label = minibatch_label.to(self.device)
			minibatch_seqLength = minibatch_seqLength.to(self.device)
			# get the output from the model
			output = self.__net(minibatch_data, minibatch_seqLength)
			# get the loss and backprop
			loss = self.criterion(output, minibatch_label.float())
			print("Test : MiniBatch[%3d/%3d]   Test loss:%1.5f"  % (mini_batchNum,numMiniBatches,loss.item()))#,end="\r")
			predicted = (output.to('cpu')>0.5).numpy()
			groundtruth = minibatch_label.to('cpu').numpy()
			groundtruth = groundtruth.astype(np.int32)
			minibatch_conf = computeConfMatrix(predicted,groundtruth)
			conf += minibatch_conf
		self.test_hamF1Score,self.test_spamF1Score =  performanceMetrics(conf)




class EnronDataLoader(object):
	def __init__(self, **kwargs):
		self.batchSize 	= kwargs.get('batchSize')
		self.data 	= kwargs.get('data')
		self.label 	= kwargs.get('label')
		self.seqLengths = kwargs.get('seqLengths')
		if not isinstance(self.batchSize, int):
			print(type(self.batchSize))
			raise TypeError('batch size should be an integer')
		if not isinstance(self.data, type(torch.Tensor())):
			raise TypeError('data should be a tensor')
		if not isinstance(self.label, type(torch.Tensor())):
			raise TypeError('label should be a tensor')
		if not isinstance(self.seqLengths, type(torch.Tensor())):
			raise TypeError('seqLengths should be a tensor')
		self.size 	= self.label.size(0)
		self.shuffle	= kwargs.get('shuffle',False)	
	def _batchSampler(self):
		# return a batch indexes using 
		batchIndexes = []
		for idx in self.sampler_index:
			batchIndexes.append(idx)
			if len(batchIndexes) == self.batchSize:
				yield batchIndexes
				batchIndexes = []
		if len(batchIndexes) > 0 :
			yield batchIndexes
	def __iter__(self):
		if self.shuffle:
			self.sampler_index = iter(np.random.permutation(self.size))
		else:
			self.sampler_index = iter(range(self.size))
		self.__batchSampler = self._batchSampler()
		return self
	def __next__(self):
		batchInd = next(self.__batchSampler)
		batch_data = self.data[batchInd]
		batch_label = self.label[batchInd]
		batch_seqLengths = self.seqLengths[batchInd]
		batch_seqLengths, sorted_indexes = batch_seqLengths.sort(0, descending=True)
		batch_data = batch_data[sorted_indexes]
		batch_label = batch_label[sorted_indexes]
		return batch_data, batch_label, batch_seqLengths
	def __len__(self):
		return self.size






class NetworkLSTM(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		outputSize = kwargs.get('outputSize',1)
		numLayers = kwargs.get('numLayers',1)
		self.hiddenSize = kwargs.get('hiddenSize',4)
		embedSize = kwargs.get('embedSize',4)
		vocabSize = kwargs.get('vocabSize',1000)
		dropout = kwargs.get('dropout',0)
		dropoutLSTM = kwargs.get('dropoutLSTM',0)
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
		# Ex: x : 8 X 100 
		# embedding the input words
		embeddedOutputs = self.embedding(x)
		# embeddedOutputs has size of mini-batch size X max Sequence Lenght X embedding dimensions
		# Ex: embeddedOutputs : 8 X 100 X 64
		# pack the input ( removing padding to the sequence with length smaller than max length)
		packedInput = pack_padded_sequence(embeddedOutputs, seqLengths.cpu().numpy(), batch_first=True)
		# packedInput has size of mini-batch size X sum of sequence lengths X embedding dimensions
		# Ex: packedInput : 8 X 80 X 64
		# lstm : input should be the size of miniBatch size X seq lenght X input size
		packedOutput, (ht, ct) = self.lstm(packedInput, None)
		# packedOutput has size of size X seq lenght X hidden size
		# unpack, recover padded sequence
		output, inputSizes = pad_packed_sequence(packedOutput, batch_first=True)
		# collect the last output in each batch
		lastIndexes = (inputSizes - 1).to(self.device) # last_idxs = input_sizes - torch.ones_like(input_sizes)
		output = torch.gather(output, 1, lastIndexes.view(-1, 1).unsqueeze(2).repeat(1, 1, self.hiddenSize)).squeeze() # [batch_size, hidden_dim]
		# dropout and fully-connected layer
		output = self.dropout(output)
		output = self.fc(output).squeeze() 
		# sigmoid function
		output = self.sig(output)
		return output




class EnronDataLoader(object):
	def __init__(self, **kwargs):
		self.batchSize 	= kwargs.get('batchSize')
		self.data 	= kwargs.get('data')
		self.label 	= kwargs.get('label')
		self.seqLengths = kwargs.get('seqLengths')
		if not isinstance(self.batchSize, int):
			print(type(self.batchSize))
			raise TypeError('batch size should be an integer')
		if not isinstance(self.data, type(torch.Tensor())):
			raise TypeError('data should be a tensor')
		if not isinstance(self.label, type(torch.Tensor())):
			raise TypeError('label should be a tensor')
		if not isinstance(self.seqLengths, type(torch.Tensor())):
			raise TypeError('seqLengths should be a tensor')
		self.size 	= self.label.size(0)
		self.shuffle	= kwargs.get('shuffle',False)	
	def _batchSampler(self):
		# return a batch indexes using 
		batchIndexes = []
		for idx in self.sampler_index:
			batchIndexes.append(idx)
			if len(batchIndexes) == self.batchSize:
				yield batchIndexes
				batchIndexes = []
		if len(batchIndexes) > 0 :
			yield batchIndexes
	def __iter__(self):
		if self.shuffle:
			self.sampler_index = iter(np.random.permutation(self.size))
		else:
			self.sampler_index = iter(range(self.size))
		self.__batchSampler = self._batchSampler()
		return self
	def __next__(self):
		batchInd = next(self.__batchSampler)
		batch_data = self.data[batchInd]
		batch_label = self.label[batchInd]
		batch_seqLengths = self.seqLengths[batchInd]
		batch_seqLengths, sorted_indexes = batch_seqLengths.sort(0, descending=True)
		batch_data = batch_data[sorted_indexes]
		batch_label = batch_label[sorted_indexes]
		return batch_data, batch_label, batch_seqLengths
	def __len__(self):
		return self.size






