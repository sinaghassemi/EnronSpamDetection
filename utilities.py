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


def extractVocab(content):
	'''
	Takes content as arg,
	Returns vocabulary dictionary
	where keys are words and values are words count
	'''
	dict_vocab = {}
	content_splitted = content.split(" ")
	for word in content_splitted:
		if word in dict_vocab.keys():
			dict_vocab[word] += 1
		else:
			dict_vocab[word] = 1
	return 	dict_vocab


def wordCount(dict_vocab):
	'''
	Takes vocabulary dictionary as arg,
	Returns two list : 1. words  2. words count
	which are sorted based on words count
	'''
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
	'''
	Takes classifierOutputs and groundtruthList
	as arg in shape of 1D numpy array with size of num of samples
	Return Confusion matrix
	'''
	confusionMatrix = np.zeros((2,2))
	for i in range(len(classifierOutputs)):
		predicted = classifierOutputs[i]
		groundtruth = groundtruthList[i]
		confusionMatrix[predicted][groundtruth] += 1
	return confusionMatrix


def div(a,b):
	'''
	To prevent division by zero,
	Return 0 instead
	'''
	if b == 0:
		return 0
	else:
		return a/b

def performanceMetrics(confusionMatrix):
	'''
	Takes Confusion Matrix as arg (0: ham , 1:spam),
	Returns some performance matrix:
	accuracy, precision, recall, f1-Score, false positive rate, true positive rate
	and for both spam and ham class
	'''
	# For Ham Class
	TP_ham = confusionMatrix[0][0]
	TN_ham = confusionMatrix[1][1]
	FP_ham = confusionMatrix[0][1]
	FN_ham = confusionMatrix[1][0]
	acc_ham = div ((TP_ham + TN_ham) ,(TP_ham + TN_ham + FP_ham + FN_ham))
	precision_ham = div (TP_ham , (TP_ham + FP_ham))
	recall_ham    = div(TP_ham , (TP_ham + FN_ham))
	f1Score_ham   = div( 2 * precision_ham * recall_ham , (precision_ham + recall_ham) )
	FPR_ham =  div(FP_ham , (FP_ham + TN_ham))
	TPR_ham =  div(TP_ham , (TP_ham + FN_ham))
	# For Spam Class
	TP_spam = confusionMatrix[1][1]
	TN_spam = confusionMatrix[0][0]
	FP_spam = confusionMatrix[1][0]
	FN_spam = confusionMatrix[0][1]
	acc_spam = div((TP_spam + TN_spam) ,(TP_spam + TN_spam + FP_spam + FN_spam))
	precision_spam = div(TP_spam , (TP_spam + FP_spam))
	recall_spam    = div(TP_spam , (TP_spam + FN_spam))
	f1Score_spam   = div(2 * precision_spam * recall_spam , (precision_spam + recall_spam) )
	FPR_spam =  div(FP_spam , (FP_spam + TN_spam))
	TPR_spam =  div(TP_spam , (TP_spam + FN_spam))
	
	metrics = {'acc_ham':acc_ham  ,'precision_ham':precision_ham,'recall_ham':recall_ham,'f1Score_ham':f1Score_ham,'FPR_ham':FPR_ham,'TPR_ham':TPR_ham,\
		   'acc_spam':acc_spam,'precision_spam':precision_spam,'recall_spam':recall_spam,'f1Score_spam':f1Score_spam,'FPR_spam':FPR_spam,'TPR_spam':TPR_spam} 

	return metrics


def RoC(prediction_prob,test_label):
	'''
	Takes prediction probabilities and actual ground truths (for spam class)
	as arg in shape of 1D numpy array with size of num of samples
	Returns false positive and true positive rate for a set of thresholds
	'''

	thresholds = np.arange(0,1,0.02)
	FPR = np.zeros(len(thresholds))
	TPR = np.zeros(len(thresholds))
	for (i,th) in enumerate(thresholds):
		# set threshold on probability along spam column: 0:ham, 1:spam
		prediction_th = (prediction_prob > th)*1 
		conf_th = computeConfMatrix(prediction_th,test_label) 
		metrics_th = performanceMetrics(conf_th)
		FPR[i] = metrics_th['FPR_spam']
		TPR[i] = metrics_th['TPR_spam']
	return FPR,TPR


