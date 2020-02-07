import os 
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from random import shuffle
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


# Takes content and return vocab dictionary where keys are words and values are words count
def extractVocab(content):
	dict_vocab = {}
	content_splitted = content.split(" ")
	for word in content_splitted:
		if word in dict_vocab.keys():
			dict_vocab[word] += 1
		else:
			dict_vocab[word] = 1
	return 	dict_vocab

# Takes vocab dictionary and return two list of words and their count which are sorted based on words count
def wordCount(dict_vocab):
	words_sorted = []
	counts_sorted = []
	dict_vocab_counts = list(dict_vocab.values())
	dict_vocab_words  = list(dict_vocab.keys())
	sorted_index = sorted(range(len(dict_vocab_counts)),key = lambda x:dict_vocab_counts[x],reverse=True)
	for i in sorted_index:
		words_sorted  += [dict_vocab_words[i]]
		counts_sorted += [dict_vocab_counts[i]]
	return words_sorted,counts_sorted

def computeConfMatrix(classifierOutputs,groundtruthList):
	confusionMatrix = np.zeros((2,2))
	for i in range(len(classifierOutputs)):
		predicted = classifierOutputs[i]
		groundtruth = groundtruthList[i]
		confusionMatrix[predicted][groundtruth] += 1
	return confusionMatrix

def performanceMetrics(confusionMatrix):
	# For Ham Class
	TP_ham = confusionMatrix[0][0]
	TN_ham = confusionMatrix[1][1]
	FP_ham = confusionMatrix[0][1]
	FN_ham = confusionMatrix[1][0]
	acc_ham = (TP_ham + TN_ham) /(TP_ham + TN_ham + FP_ham + FN_ham)
	precision_ham = TP_ham / (TP_ham + FP_ham)
	recall_ham    = TP_ham / (TP_ham + FN_ham)
	f1Score_ham   = 2 * precision_ham * recall_ham / (precision_ham + recall_ham) 
	FPR_ham =  FP_ham / (FP_ham + TN_ham)
	TPR_ham =  TP_ham / (TP_ham + FN_ham)
	# For Spam Class
	TP_spam = confusionMatrix[1][1]
	TN_spam = confusionMatrix[0][0]
	FP_spam = confusionMatrix[1][0]
	FN_spam = confusionMatrix[0][1]
	acc_spam = (TP_spam + TN_spam) /(TP_spam + TN_spam + FP_spam + FN_spam)
	precision_spam = TP_spam / (TP_spam + FP_spam)
	recall_spam    = TP_spam / (TP_spam + FN_spam)
	f1Score_spam   = 2 * precision_spam * recall_spam / (precision_spam + recall_spam) 
	FPR_spam =  FP_spam / (FP_spam + TN_spam)
	TPR_spam =  TP_spam / (TP_spam + FN_spam)
	
	metrics = {'acc_ham':acc_ham  ,'precision_ham':precision_ham,'recall_ham':recall_ham,'f1Score_ham':f1Score_ham,'FPR_ham':FPR_ham,'TPR_ham':TPR_ham,\
		   'acc_spam':acc_spam,'precision_spam':precision_spam,'recall_spam':recall_spam,'f1Score_spam':f1Score_spam,'FPR_spam':FPR_spam,'TPR_spam':TPR_spam} 

	return metrics

def meanAndStd(data):
	# it takes data as array (n_samples x n_features)
	# it returns mean and std over samples for each feature
	mean = data.mean(axis=0)
	std = data.std(axis=0)
	return mean,std

def scaleData(data,mean,std):
	data_scaled = np.zeros(data.shape,dtype=np.float16)
	for feature_num in range(data.shape[1]):
		data_scaled[:,feature_num] = data[:,feature_num] - mean[feature_num]
		if std[feature_num] != 0:
			data_scaled[:,feature_num] = data_scaled[:,feature_num] / std[feature_num]
	return data_scaled


