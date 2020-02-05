import torch
import os 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import numpy as np
from random import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import torch.nn as nn
from functions import * 
from math import ceil
from EnronDataset import EnronLoader
from LSTM import ClassifierLSTM

##### path to spam and ham folder 
ham_dataPath = 'data/ham/'#'/home/sina/Downloads/enron/splitted/ham/'#'data/ham/'
spam_dataPath = 'data/spam/'#'/home/sina/Downloads/enron/splitted/spam/'#'data/spam/'
vocabSize = 1000
print("vocabulary size used in calssification : %d" % vocabSize)

##### Loading and preprocessing the data
dataLoader = EnronLoader(hamDir=ham_dataPath,spamDir=spam_dataPath)
print("number of spam files: %d" % len(dataLoader.spamFiles))
print("number of ham files: %d" % len(dataLoader.hamFiles))

contentList_spam = dataLoader.readSpam()
contentList_ham = dataLoader.readHam()

##### Concatenating list of contents to a single string for further analysis
allContent_spam = " ".join([content for content in contentList_spam])
allContent_ham  = " ".join([content for content in contentList_ham])

##### Concatenating contents of spam and ham  
allContent =  allContent_ham + allContent_spam
contentList = contentList_ham + contentList_spam 
numOfSamples = len(contentList)

##### Labels : "1" for Spam, and "0" for Ham 
lableList = [0]*len(contentList_ham) + [1]*len(contentList_spam)


##### Shuffling data and labels 
index_shuffle = list(range(numOfSamples))
shuffle(index_shuffle)
contentList_shuffled = []
lableList_shuffled = []

for i in index_shuffle:
	contentList_shuffled += [contentList[i]]
	lableList_shuffled += [lableList[i]]

contentList = contentList_shuffled
lableList = lableList_shuffled


##### Extracting vocabulary : 
##### ExtractVocab() returns a dictionary in which keys are words and values are words count
vocab_spam = extractVocab(allContent_spam)
vocab_ham  = extractVocab(allContent_ham)
vocal_all  = extractVocab(allContent)

numAllWordsInSpam = len(allContent_spam.split())
numAllWordsInHam = len(allContent_ham.split())

print("number of words in spam:%d and ham:%d" % (numAllWordsInSpam,numAllWordsInHam))


##### Sorting words based on their counts
##### wordCount() return two lists which are words and their count sorted form most common words to least
wordSorted_spam, wordSortedCounts_spam  = wordCount(vocab_spam)
wordSorted_ham , wordSortedCounts_ham   = wordCount(vocab_ham)
wordSorted , wordSortedCounts   = wordCount(vocal_all)


#### Visualization for words in spam and ham E-mails
'''
wordcloud = WordCloud().generate(allContent_spam)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("words in spam")
plt.show()

wordcloud = WordCloud().generate(allContent_ham)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("words in ham")
plt.show()

plt.plot(wordSorted[:100],wordSortedCounts[:100])
plt.title("100 most common words")
plt.xticks(rotation='vertical')
plt.show()
'''
##### To get a unique list of words we use sets
set_hamWords = set(wordSorted_ham)
set_spamWords = set(wordSorted_spam)
set_vocabWords = set(wordSorted[:vocabSize])

##### Words from our vocab that are only in spam E-mails not ham (distinctive words)
VocabWordsOnlyInSpam = (set_vocabWords & set_spamWords) - (set_vocabWords & set_hamWords)

##### Words from our vocab that are only in ham e-mails not spam (distinctive words)

VocabWordsOnlyInHam = (set_vocabWords & set_hamWords) - (set_vocabWords & set_spamWords)

print("intersection of vocab words with spam words excluding ham words : %d" % len(VocabWordsOnlyInSpam))
print("intersection of vocab words with ham words excluding spam words : %d" % len(VocabWordsOnlyInHam))
print("*"*5 + "vocab words not included in ham files:")
print((set_vocabWords & set_spamWords) - (set_vocabWords & set_hamWords))
print("*"*5 + "vocab words not included in spam files:")
print((set_vocabWords & set_hamWords) - (set_vocabWords & set_spamWords))

##### Here we print the words in our vocabulary with thier occurrence counts in spam/ham content 
print("words in vocabulary ")
for i in range(vocabSize):
	vocabWord = wordSorted[i]
	probWordInSpam = vocab_spam.get(vocabWord,0)
	probWordInHam = vocab_ham.get(vocabWord,0)
	print("%10s\tSpam : %6d\tHam : %6d" %(vocabWord,probWordInSpam,probWordInHam))

##### Defining training (50%) validation (25%) and test set (25%) length

trainLength = int(0.5 * numOfSamples)
valLength   = int(0.25 * numOfSamples)
testLength  = numOfSamples - (trainLength + valLength)
print("total number of samples %d splitted to %d training samples and %d test samples" % (numOfSamples,trainLength,testLength))


# Here we define the words to be used as features
# we a number of most common words , it can be thought as bags of words 
mostCommonWords = wordSorted[:vocabSize]

# Defining a dictionar whose keys are most common words and values are the indexes
vocab_indexing = {k:v for (v,k) in enumerate(mostCommonWords)}

# Tokenization : spliting the content of each file to list of words 
contentTokenized = []
for content in contentList:
	contentTokenized += [content.split()]

# Vectorization : Converting the words to integres using the most common words
contentVectorized = np.zeros((numOfSamples,vocabSize),dtype=np.uint16)
for (row,content) in enumerate(contentTokenized):
	for word in content:
		if word in vocab_indexing:
			word_index = vocab_indexing[word]
			contentVectorized[row,word_index]+=1		

##### Splitting data into train, val, and test set
train_data   = contentVectorized[:trainLength]
train_label  = lableList  	[:trainLength]
test_data    = contentVectorized[trainLength+valLength:]
test_label   = lableList        [trainLength+valLength:]

########## Naive Bayes classifier
print("*** Multinomial Naive Bayes Classifer ***")
classifier = MultinomialNB()
classifier.fit(train_data, train_label)
prediction = classifier.predict(test_data) 
test_conf = computeConfMatrix(prediction,test_label)
test_hamF1Score, test_spamF1Score = performanceMetrics(test_conf)
print("Test set, ham f1-score : %1.4f ,  spam f1-score : %1.4f " % (test_hamF1Score,test_spamF1Score))
print("******")
'''
########## Decision Tree Classifier
print("*** Decision Tree Classifier ***")
classifier = DecisionTreeClassifier()			
classifier.fit(train_data, train_label)
prediction = classifier.predict(test_data)
porbs = classifier.predict_proba(test_data) 
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

########## K-Nearest Neighbors
print("*** K-Nearest Neighbors Classifer ***")
classifier = KNeighborsClassifier()			
classifier.fit(train_data, train_label)
prediction = classifier.predict(test_data)
porbs = classifier.predict_proba(test_data) 
test_conf = computeConfMatrix(prediction,test_label)
test_hamF1Score, test_spamF1Score = performanceMetrics(test_conf)
print("Test set, ham f1-score : %1.4f ,  spam f1-score : %1.4f " % (test_hamF1Score,test_spamF1Score))
print("******")
'''

content_lengths = []

for c in contentList:
	content_lengths += [len(c.split())] 
maxLength = max(content_lengths)

# tokenization : split each content to a list of words
contentTokenized = [] # list of contents splitted into list of words
for content in contentList:
	contentTokenized += [content.split()]	

mostCommonWords = wordSorted[:vocabSize-2]
vocab_indexing = {k:v+2 for (v,k) in enumerate(mostCommonWords)} # 0 : pad, 1: unknown(not in vocab)


# indexing word in tokenized content

contentIndexed = [] # list of content:list of indexes	
for content in contentTokenized:
	c_indexed = []
	for word in content:
		word_index = vocab_indexing.get(word,1)
		c_indexed += [word_index]
	contentIndexed += [c_indexed]

# now padding zeros data tensor will have n_samples x maxLength size
content_tensor = torch.zeros((numOfSamples,maxLength)).long()

for index in range(numOfSamples):
	contentLength = content_lengths[index]
	content_tensor[index,:contentLength] = torch.LongTensor(contentIndexed[index])

content_lengths = torch.LongTensor(content_lengths)

print(maxLength)

print(len(content_lengths))
print(numOfSamples)
for (i,l) in enumerate(content_lengths):
	#print(content_tensor[i])
	if l < 2:
		print(l)

#print(stop)

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

# LSTM
classifier = ClassifierLSTM(batchSize = 32,train_data = train_data,val_data = val_data,test_data = test_data,\
		train_label = train_label,val_label = val_label,test_label=test_label,\
		train_lengths = train_lengths, val_lengths = val_lengths, test_lengths = test_lengths,\
		outputSize = 1, numLayers = 1, hiddenSize = 16, embedSize = 32, vocabSize = vocabSize,\
		device = 'cpu')

best_spamF1Score = 0
numEpoches = 10
for epoch in range(numEpoches):
	classifier.train()
	classifier.val()
	print("val f1score : %f "%classifier.val_spamF1Score)
	if classifier.val_spamF1Score > best_spamF1Score:
		best_spamF1Score = classifier.val_spamF1Score 
		classifier.test()

print("Test set, ham f1-score : %1.4f ,  spam f1-score : %1.4f " % (classifier.test_hamF1Score,classifier.test_spamF1Score))







