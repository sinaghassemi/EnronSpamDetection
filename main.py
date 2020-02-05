import torch
import os 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from random import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import torch.nn as nn
from functions import * 
from math import ceil


# path to spam and ham folder #/home/sina/Downloads/enron/splitted/ham/
ham_dataPath = '/home/sina/Downloads/enron/splitted/ham/'#'data/ham/'
spam_dataPath = '/home/sina/Downloads/enron/splitted/spam/'#'data/spam/'
vocabSize = 1000
# extract the list of files in ham and spam folders
spam_filesList = returnLeavesFiles(spam_dataPath)
ham_filesList = returnLeavesFiles(ham_dataPath)
print("number of spam files: %d" % len(spam_filesList))
print("number of ham files: %d" % len(ham_filesList))

# choosing only a number of files for quick analysis
spam_filesList = spam_filesList[:-1]
ham_filesList  = ham_filesList[:-1]

# extracting content from spam and ham files
print('*'*5 + 'reading spam files' + '*'*5 )
contentList_spam,allContent_spam = readFile(spam_filesList)
print('\n'+'*'*5 + 'reading ham files' + '*'*5)
contentList_ham,allContent_ham = readFile(ham_filesList)

# concatenating contents of spam and ham 
allContent =  allContent_ham + allContent_spam
contentList = contentList_ham + contentList_spam 
numOfSamples = len(contentList)

# labels for [1] spam and [0] ham 
lableList = [0]*len(contentList_ham) + [1]*len(contentList_spam)




# shuffling data and labels 
index_shuffle = list(range(numOfSamples))
shuffle(index_shuffle)
contentList_shuffled = []
lableList_shuffled = []

for i in index_shuffle:
	contentList_shuffled += [contentList[i]]
	lableList_shuffled += [lableList[i]]

contentList = contentList_shuffled
lableList = lableList_shuffled


# extracting vocabulary from spam , ham and all content 

vocab_spam = extractVocab(allContent_spam)
vocab_ham  = extractVocab(allContent_ham)
vocal_all  = extractVocab(allContent)


# sorting words based on their counts
wordSorted_spam, wordSortedCounts_spam  = wordCount(vocab_spam)
wordSorted_ham , wordSortedCounts_ham   = wordCount(vocab_ham)
wordSorted , wordSortedCounts   = wordCount(vocal_all)


numAllWordsInSpam = len(allContent_spam.split())
numAllWordsInHam = len(allContent_ham.split())

print("number of words in spam:%d and ham:%d" % (numAllWordsInSpam,numAllWordsInHam))

set_hamWords = set(wordSorted_ham)
set_spamWords = set(wordSorted_spam)
set_vocabWords = set(wordSorted[:vocabSize])

# words from our vocab that are only in spam e-mails not ham (distinctive words)
VocabWordsOnlyInSpam = (set_vocabWords & set_spamWords) - (set_vocabWords & set_hamWords)
# words from our vocab that are only in ham e-mails not spam (distinctive words)
VocabWordsOnlyInHam = (set_vocabWords & set_hamWords) - (set_vocabWords & set_spamWords)

print("intersection of vocab words with spam words excluding ham words : %d" % len(VocabWordsOnlyInSpam))
print("intersection of vocab words with ham words excluding spam words : %d" % len(VocabWordsOnlyInHam))
print("*"*5 + "vocab words not included in ham files:")
print((set_vocabWords & set_spamWords) - (set_vocabWords & set_hamWords))
print("*"*5 + "vocab words not included in spam files:")
print((set_vocabWords & set_hamWords) - (set_vocabWords & set_spamWords))


print("words in vocabulary ")
for i in range(vocabSize):
	vocabWord = wordSorted[i]
	probWordInSpam = vocab_spam.get(vocabWord,0)
	probWordInHam = vocab_ham.get(vocabWord,0)
	print("%10s\tSpam : %6d\tHam : %6d" %(vocabWord,probWordInSpam,probWordInHam))
	
# spliting data into training / validation and test set

trainLength = int(0.5 * numOfSamples)
valLength   = int(0.25 * numOfSamples)
testLength  = numOfSamples - (trainLength + valLength)


# run naive base and logistic regression over a set of vocab sizes
vocabSize_list = [vocabSize]#[5]#[5,10,20,50,100,500,1000,5000]

for vocabSize in vocabSize_list:
	
	mostCommonWords = wordSorted[:vocabSize]
	print("vocab size used in calssification : %d" % vocabSize)
	# Defining a dictionar whose keys are most common words and values are the indexes
	vocab_indexing = {k:v for (v,k) in enumerate(mostCommonWords)}
	# Tokenization : spliting the content of each file to list of words 
	contentTokenized = []
	for content in contentList:
		contentTokenized += [content.split()]
	# Vectorization : Converting the words to integres using the most common dictionary
	contentVectorized = np.zeros((numOfSamples,vocabSize),dtype=np.uint16)
	for (row,content) in enumerate(contentTokenized):
		for word in content:
			if word in vocab_indexing:
				word_index = vocab_indexing[word]
				contentVectorized[row,word_index]+=1		
	print("total number of samples %d splitted to %d training samples and %d test samples" % (numOfSamples,trainLength,testLength))

	# splitting data into train and test set
	train_data   = contentVectorized[:trainLength]
	train_label  =       lableList  [:trainLength]
	test_data    = contentVectorized[trainLength+valLength:]
	test_label   = lableList        [trainLength+valLength:]

	########## naive bayes classifier
	print("*** Multinomial Naive Bayes Classifer ***")
	classifier = MultinomialNB()
	classifier.fit(train_data, train_label)
	prediction = classifier.predict(test_data) 
	test_conf = computeConfMatrix(prediction,test_label)
	test_hamF1Score, test_spamF1Score = performanceMetrics(test_conf)
	print("Test set, ham f1-score : %1.4f ,  spam f1-score : %1.4f " % (test_hamF1Score,test_spamF1Score))
	print("******")
	########## linear logistic classifier
	print("*** Linear Logistic Regression Classifer ***")
	#mean_train , std_train = meanAndStd(data_trainSet)
	classifier = LogisticRegression(random_state=0,max_iter=1000)
	classifier.fit(train_data, train_label)
	prediction = classifier.predict(test_data) 
	test_conf = computeConfMatrix(prediction,test_label)
	test_hamF1Score, test_spamF1Score = performanceMetrics(test_conf)
	print("Test set, ham f1-score : %1.4f ,  spam f1-score : %1.4f " % (test_hamF1Score,test_spamF1Score))
	print("******")
			

'''
wordcloud = WordCloud().generate(allContent_spam)
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()
wordcloud = WordCloud().generate(allContent_ham)
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()
for n,i in enumerate(contentList_spam):
	print(spam_filesList[n])
	print(i)
	print("\n"*3)
for i in range(1000):
	print('%d	word %s		count %d' % (i,vocab_words_sorted[i],vocab_counts_sorted[i])) 
plt.plot(vocab_words_sorted[:100],vocab_counts_sorted[:100])
plt.xticks(rotation='vertical')
plt.show()
'''

# LSTM
content_lengths = []

for c in contentList:
	content_lengths += [len(c.split())] 
maxLength = max(content_lengths)
print(maxLength)

#plt.hist(content_lengths,bins=40)
#plt.title("contents lengths histogram")
#plt.show()


# tokenization : split each content to a list of words
contentTokenized = [] # list of contents splitted into list of words
for content in contentList:
	contentTokenized += [content.split()]	

# constructing a vocabulary
#vocab = extractVocab(allContent)
#print(len(list(vocab.keys())))

mostCommonWords = wordSorted[:vocabSize]
vocab_indexing = {k:v+2 for (v,k) in enumerate(mostCommonWords)} # 0 : pad, 1: unknown(not in vocab)


# indexing word in tokenized content

contentIndexed = [] # list of content:list of indexes	
for content in contentTokenized:
	c_indexed = []
	for word in content:
		word_index = vocab_indexing.get(word,1)
		c_indexed += [word_index]
	contentIndexed += [c_indexed]

#print(contentIndexed)

# now padding zeros data tensor will have n_samples x maxLength size

content_tensor = torch.zeros((numOfSamples,maxLength)).long()

for index in range(numOfSamples):
	contentLength = content_lengths[index]
	content_tensor[index,:contentLength] = torch.LongTensor(contentIndexed[index])

content_lengths = torch.LongTensor(content_lengths)

# spliting train / val / test

lableList = torch.Tensor(lableList)

train_data   = content_tensor[:trainLength]
train_label  =       lableList[:trainLength]
train_lengths= content_lengths[:trainLength]

val_data  = content_tensor[trainLength : trainLength+valLength]
val_label = lableList      [trainLength : trainLength+valLength]
val_lengths= content_lengths[trainLength : trainLength+valLength]

test_data  = content_tensor[trainLength+valLength:]
test_label = lableList      [trainLength+valLength:]
test_lengths= content_lengths[trainLength+valLength:]	




for (i,l) in enumerate(content_lengths):
	if l <= 0 :
		print(l)
		print(contentTokenized[i])
		print(contentList[i])
		#print(stop)
	

# Loaders 

batchSize = 1024
train_DataLoader = EnronDataLoader(data=train_data,label=train_label,seqLengths=train_lengths,batchSize=batchSize,shuffle=True)
val_DataLoader   = EnronDataLoader(data=val_data,label=val_label,seqLengths=val_lengths,batchSize=batchSize,shuffle=True)
test_DataLoader  = EnronDataLoader(data=test_data,label=test_label,seqLengths=test_lengths,batchSize=batchSize,shuffle=True)
device = 'cuda'

numMiniBatches_train = ceil(len(train_DataLoader) / 1024)
numMiniBatches_val   = ceil(len(val_DataLoader) / 1024)
numMiniBatches_test  = ceil(len(test_DataLoader) / 1024)
# network
net = NetworkLSTM(numLayers=1,outputSize=1,hiddenSize=8,embedSize=4,vocabSize=vocabSize+2,dropout=0,dropoutLSTM=0,device=device)
net = net.to(device)


# Loss

criterion = nn.BCELoss()
lr=0.03
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = 'min',factor = 0.5,patience = 2)


# Training/Validation
numEpoches = 2


import copy
best_spamF1Score = 0
for epoch in range(numEpoches):
	# train
	net.train()
	for mini_batchNum , (minibatch_data,minibatch_label,minibatch_seqLength) in enumerate(train_DataLoader):
		
		minibatch_data = minibatch_data.to(device)
		minibatch_label = minibatch_label.to(device)
		minibatch_seqLength = minibatch_seqLength.to(device)
		# get the output from the model
		output = net(minibatch_data, minibatch_seqLength)
		# get the loss and backprop
		loss = criterion(output, minibatch_label.float())
		optimizer.zero_grad() 
		loss.backward()
		print("Train : Epoch[%3d/%3d] : MiniBatch[%3d]   Train loss:%1.5f"  % (epoch,numEpoches,mini_batchNum,loss.item()),end="\r")
	# val
	net.eval()
	val_conf = np.zeros((2,2))
	for mini_batchNum , (minibatch_data,minibatch_label,minibatch_seqLength) in enumerate(val_DataLoader):
		minibatch_data = minibatch_data.to(device)
		minibatch_label = minibatch_label.to(device)
		minibatch_seqLength = minibatch_seqLength.to(device)
		# get the output from the model
		output = net(minibatch_data, minibatch_seqLength)
		# get the loss and backprop
		loss = criterion(output, minibatch_label.float())
		print("Val : Epoch[%3d/%3d] : MiniBatch[%3d]   Val loss:%1.5f"  % (epoch,numEpoches,mini_batchNum,loss.item()),end="\r")
		predicted = (minibatch_label.to('cpu')>0.5).numpy()
		groundtruth = minibatch_label.to('cpu').numpy()
		groundtruth = groundtruth.astype(np.int32)
		minibatch_conf = computeConfMatrix(predicted,groundtruth)
		val_conf += minibatch_conf
	val_hamF1Score, val_spamF1Score = performanceMetrics(val_conf)
	#print("Validation set, ham f1-score : %1.4f ,  spam f1-score : %1.4f " % (val_hamF1Score,val_spamF1Score))

	if val_spamF1Score > best_spamF1Score:
		#bestModel = copy.deepcopy(net)

		# Test on best model

		#net = bestModel
		#net.eval()
		test_conf = np.zeros((2,2))	
		for mini_batchNum , (minibatch_data,minibatch_label,minibatch_seqLength) in enumerate(test_DataLoader):
			minibatch_data = minibatch_data.to(device)
			minibatch_label = minibatch_label.to(device)
			minibatch_seqLength = minibatch_seqLength.to(device)
			# get the output from the model
			output = net(minibatch_data, minibatch_seqLength)
			# get the loss and backprop
			loss = criterion(output, minibatch_label.float())
			print("Test : Epoch[%3d/%3d] : MiniBatch[%3d]   Test loss:%1.5f"  % (epoch,numEpoches,mini_batchNum,loss.item()),end="\r")
			predicted = (minibatch_label.to('cpu')>0.5).numpy()
			groundtruth = minibatch_label.to('cpu').numpy()
			groundtruth = groundtruth.astype(np.int32)
			minibatch_conf = computeConfMatrix(predicted,groundtruth)
			test_conf += minibatch_conf
		test_hamF1Score, test_spamF1Score = performanceMetrics(test_conf)

print("Test set, ham f1-score : %1.4f ,  spam f1-score : %1.4f " % (test_hamF1Score,test_spamF1Score))
		
print(test_hamF1Score,test_spamF1Score)


