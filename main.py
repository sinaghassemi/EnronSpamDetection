#import torch
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


ham_dataPath = 'data/ham/'
spam_dataPath = 'data/spam/'

# function to return list of all files in leaves given a root tree 
def returnLeavesFiles(path):
	fileList = []
	for root,dirs,files in os.walk(path):
		if len(dirs) == 0:
			# leaves : containing the files
			for f in files:
				fileList += [os.path.join(root,f)]
	return fileList


def readFile(files_list):
	content_list=[]
	content_concatenated = ""
	punctuation_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
	totalNumOfFiles = len(files_list)
	for (num_file,File) in enumerate(files_list):
		print("Reading and Preprocessing content [% 5d / %d] : %s" % (num_file,totalNumOfFiles,File),end="\r")
		f = open(File,'r',encoding="ISO-8859-1")
		content = f.read()
		f.close()
		# Some Preprocessing over the content : 
		# 1. Converting all letters to lower case
		content = content.lower()

		# 2. Splitting the content based on lines,and removing all fields except body and subject
		special_words = ['message-id:', 'date:', 'from:','sent:', 'to:','cc:','bcc', 'mime-version:', 'content-type:', 'content-transfer-encoding:', 'x-from:', 'x-to:', 'x-cc:', 'x-bcc:', 'x-origin:', 'x-filename:', 'x-priority:', 'x-msmail-priority:', 'x-mimeole:','\tcharset=']
		content = content.split("\n")
		redundant_lines=[]
		for word in special_words:
			for line_num,line in enumerate(content):
				if word in line:
					redundant_lines +=[line_num]
					#break # go for the next word
		cleanedContent=""
		for line_num,line in enumerate(content):
			if line_num not in redundant_lines:
				cleanedContent += (line + " ")
		content = cleanedContent
							
		# 3. Get rid of HTML commands, replacing them with space 
		cleanedContent	= re.sub("<.*?>","",content)
		cleanedContent  = re.sub("&.*?;","",cleanedContent)
		cleanedContent	= re.sub("=[0-9]*","",cleanedContent)
		content = cleanedContent


		# 3. Replace E-mail address with word "emailaddrs"
		cleanedContent=""
		for word in content.split():
			if "@" in word:
				#print(word)
				cleanedContent += "emailaddrs "
			else:
				cleanedContent += word + " "
		content = cleanedContent

		# 3. Replace Website address with word "webaddrs"
		cleanedContent=""
		for word in content.split():
			if (("http:" in word) or (".com" in word)):
				cleanedContent += "webaddrs "
			else:
				cleanedContent += word + " "
		content = cleanedContent

		# Enron 
		'''
		cleanedContent=""
		for word in content.split():
			if "enron" in word:
				pass
				#print("%s ::: %s" % (word,File))
			else:
				pass
		'''


		# 4. Replace tab "\t" with space
		content = content.replace("\t"," ")
		# 5. Replacing punctuation characters with space
		cleanedContent =""
		for char in content:
			if char in punctuation_chars:
				cleanedContent += " "
			else:
				cleanedContent += char
		content = cleanedContent
		# 6. Replace multiple space with single space
		content = re.sub(" +"," ",content)
		# 7. removing number with the text "number"
		cleanedContent=""
		for word in content.split():
			if word.isdigit():
				cleanedContent += "number "
			else:
				cleanedContent += word + " "
		content = cleanedContent

		# 8. Removing stop words
		content = content.split()
		stopwords_indexes=[]
		stopwords_list = stopwords.words('english')
		for stopword in stopwords_list:
			for word_num,word in enumerate(content):
				if word == stopword:
					stopwords_indexes +=[word_num]
		cleanedContent=""
		for word_num,word in enumerate(content):
			if word_num not in stopwords_indexes:
				cleanedContent += (word + " ")
		content = cleanedContent

		
		# 8. Removing E-mails with less than 40 words
		#minNumberOfWords = 30
		#if len(content.split()) < minNumberOfWords:
		#	print("skipping : less than %d words" % minNumberOfWords)
		#	continue

		# 8.remove the words with more than 40 characters
		maxWordLength = 40
		content = "".join([word+" " for word in content.split() if len(word) <= maxWordLength])
		#9. Store and concatenate all content into a single string for analysis
		content_concatenated += content	
		# 7. Splitting the content
		# content = content.split()
		content_list += [content]

	print("\n")
	return content_list,content_concatenated


# takes content and return vocab dictionary where keys are words and values are words count
def extractVocab(content):
	dict_vocab = {}
	content_splitted = content.split(" ")
	for word in content_splitted:
		if word in dict_vocab.keys():
			dict_vocab[word] += 1
		else:
			dict_vocab[word] = 1
	return 	dict_vocab

# takes vocab dictionary and return two list of words and their count which are sorted based on words count
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

def performanceMetrics(classifierOutputs,groundtruthList):
	confusionMatrix = np.zeros((2,2))
	for i in range(len(label_testSet)):
		predicted = classifierOutputs[i]
		groundtruth = groundtruthList[i]
		confusionMatrix[predicted][groundtruth] += 1
	precision_ham = confusionMatrix[0][0] / confusionMatrix.sum(axis=1)[0]
	recall_ham    = confusionMatrix[0][0] / confusionMatrix.sum(axis=0)[0]
	f1Score_ham   = 2 * precision_ham * recall_ham / (precision_ham + recall_ham)  
	precision_spam= confusionMatrix[1][1] / confusionMatrix.sum(axis=1)[1]
	recall_spam   = confusionMatrix[1][1] / confusionMatrix.sum(axis=0)[1]
	f1Score_spam  = 2 * precision_spam * recall_spam / (precision_spam + recall_spam)
	print(confusionMatrix)
	print("Ham : precision:%.3f recall:%.3f f1score:%.3f" % (precision_ham,recall_ham,f1Score_ham))
	print("Spam: precision:%.3f recall:%.3f f1score:%.3f" % (precision_spam,recall_spam,f1Score_spam))  

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





spam_filesList = returnLeavesFiles(spam_dataPath)
ham_filesList = returnLeavesFiles(ham_dataPath)

print("number of spam files: %d" % len(spam_filesList))
print("number of ham files: %d" % len(ham_filesList))

spam_filesList = spam_filesList[:-1]
ham_filesList  = ham_filesList[:-1]

contentList_spam,allContent_spam = readFile(spam_filesList)
contentList_ham,allContent_ham = readFile(ham_filesList)
allContent = allContent_spam + allContent_ham
contentList = contentList_ham + contentList_spam 
lableList = [1]*len(contentList_ham) + [0]*len(contentList_spam)
numOfContents = len(contentList)
# spliting train / test
trainRatio = 0.7
trainLength = int(trainRatio*numOfContents)
testLength  = numOfContents - trainLength

# shuffling content and labels 
index_shuffle = list(range(len(contentList)))
shuffle(index_shuffle)
contentList_shuffled = []
lableList_shuffled = []

for i in index_shuffle:
	contentList_shuffled += [contentList[i]]
	lableList_shuffled += [lableList[i]]

contentList = contentList_shuffled
lableList = lableList_shuffled




vocab_spam = extractVocab(allContent_spam)
vocab_ham  = extractVocab(allContent_ham)
vocal_all  = extractVocab(allContent)

w_sorted_spam, c_sorted_spam  = wordCount(vocab_spam)
w_sorted_ham , c_sorted_ham   = wordCount(vocab_ham)
w_sorted_all , c_sorted_all   = wordCount(vocal_all)


print("words in spam		words in ham")
for i in range(10):
	print('%03d % 20s % 5d    % 20s % 5d' % (i,w_sorted_spam[i],c_sorted_spam[i],w_sorted_ham[i],c_sorted_ham[i])) 

'''
print("all")
for i in range(300):
	print('%03d % 20s % 5d' % (i,w_sorted_all[i],c_sorted_all[i])) 
'''
'''

numMostCommonWords_list = [5,10,20,50,100,500,1000,5000]#[5]#[5,10,20,50,100,500,1000,5000]


for numMostCommon in numMostCommonWords_list:
	mostCommonWords = w_sorted_all[:numMostCommon]
	print("number of featurs (most common words) used in calssification : %d" % numMostCommon)
	# Defining a dictionar whose keys are most common words and values are the indexes
	words_dictionary = {k:v for (v,k) in enumerate(mostCommonWords)}
	# Tokenization : spliting the content of each file to list of words 
	content_tokenized = []
	for content in contentList:
		content_tokenized += [content.split()]
	# Vectorization : Converting the words to integres using the most common dictionary
	content_vectorization = np.zeros((numOfContents,numMostCommon),dtype=np.uint16)
	for (row,content) in enumerate(content_tokenized):
		for word in content:
			if word in words_dictionary:
				word_index = words_dictionary[word]
				content_vectorization[row,word_index]+=1		
	print("total number of samples %d splitted to %d training samples and %d test samples" % (numOfContents,trainLength,testLength))

	data_trainSet = content_vectorization[:trainLength,:]
	data_testSet  = content_vectorization[trainLength:,:]
	label_trainSet = np.array(lableList[:trainLength])
	label_testSet  = np.array(lableList[trainLength:])
	########## naive bayes classifier
	print("*** Multinomial Naive Bayes Classifer ***")
	classifier = MultinomialNB()
	classifier.fit(data_trainSet, label_trainSet)
	prediction = classifier.predict(data_testSet) 
	performanceMetrics(prediction,label_testSet)
	print("******")
	########## linear logistic classifier
	print("*** Linear Logistic Regression Classifer ***")
	mean_train , std_train = meanAndStd(data_trainSet)
	classifier = LogisticRegression(random_state=0,max_iter=1000)
	classifier.fit(data_trainSet, label_trainSet)
	prediction = classifier.predict(data_testSet) 
	performanceMetrics(prediction,label_testSet)
	print("******")
			
'''





'''
wordcloud = WordCloud().generate(allContent_spam)
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()


wordcloud = WordCloud().generate(allContent_ham)
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()
'''


'''
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

content_tokenized = [] # list of contents splitted into list of words

for content in contentList:
	content_tokenized += [content.split()]	



# constructing a vocabalary

vocab = extractVocab(allContent)
print(len(list(vocab.keys())))
w_sorted_all ,c_sorted_all = wordCount(vocal_all)
numMostCommon = 1000
mostCommonWords = w_sorted_all[:numMostCommon]
words_indexing = {k:v for (v,k) in enumerate(mostCommonWords)} # 0 : pad, 1: unknown(not in vocab(


# indexing word in tokenized content

content_indexed = [] # list of content:list of indexes
		
for content in content_tokenized:
	c_indexed = []
	for word in content:
		word_index = words_indexing.get(word,1)
		c_indexed += [word_index]
	content_indexed += [c_indexed]

#print(content_indexed)

# now padding zeros data tensor will have n_samples x maxLength size

content_tensor = torch.zeros((numOfContents,maxLength)).long()

for index in range(numOfContents):
	contentLength = content_lengths[index]
	content_tensor[index,:contentLength] = torch.LongTensor(content_indexed[index])

content_lengths = torch.LongTensor(content_lengths)

# spliting train / val / test
trainLength = int(0.6 * numOfContents)
valLength  = int(0.2 * numOfContents)
testLength = numOfContents - (trainLength + valLength)

train_data   = content_tensor[:trainLength]
train_label  =       lableList[:trainLength]
train_lengths= content_lengths[:trainLength]

val_data  = content_tensor[trainLength : trainLength+valLength]
val_label = lableList      [trainLength : trainLength+valLength]
val_lengths= content_lengths[trainLength : trainLength+valLength]

test_data  = content_tensor[trainLength+valLength:]
test_label = lableList      [trainLength+valLength:]
test_lengths= content_lengths[trainLength+valLength:]		
# module 








# Loaders 

import torch.utils.data.sampler as splr

class CustomDataLoader(object):
	def __init__(self, seq_tensor, seq_lengths, label_tensor, batch_size):
		self.batch_size = batch_size
		self.seq_tensor = seq_tensor
		self.seq_lengths = seq_lengths
		self.label_tensor = label_tensor
		self.sampler = splr.BatchSampler(splr.RandomSampler(self.label_tensor), self.batch_size, False)
		self.sampler_iter = iter(self.sampler)
    
	def __iter__(self):
		self.sampler_iter = iter(self.sampler) # reset sampler iterator
		return self

	def _next_index(self):
		return next(self.sampler_iter) # may raise StopIteration

	def __next__(self):
		index = self._next_index()

		subset_seq_tensor = self.seq_tensor[index]
		subset_seq_lengths = self.seq_lengths[index]
		subset_label_tensor = self.label_tensor[index]

		subset_seq_lengths, perm_idx = subset_seq_lengths.sort(0, descending=True)
		subset_seq_tensor = subset_seq_tensor[perm_idx]
		subset_label_tensor = subset_label_tensor[perm_idx]

		return subset_seq_tensor, subset_seq_lengths, subset_label_tensor
	def __len__(self):
		return len(self.sampler)
'''
'''

















