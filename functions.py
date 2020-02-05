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
		print("Reading and Preprocessing content [% 5d / %d] : %s" % (num_file,totalNumOfFiles-1,File),end="\r")
		f = open(File,'r',encoding="ISO-8859-1")
		content = f.read()
		f.close()
		# Some Preprocessing over the content : 
		# 1. Converting all letters to lower case
		content = content.lower()
		'''
		# 2. Removing everything from first line to subject and from second subject to the end:
		content = content.split("\n")
		firstSubject_lineNumber = 0
		secondSubject_lineNumber = len(content)
		numbOfSubjects = 0
		for line_num,line in enumerate(content):
			if "subject:" in line:
				numbOfSubjects += 1
				if numbOfSubjects == 1:
					firstSubject_lineNumber = line_num
				elif numbOfSubjects == 2:
					secondSubject_lineNumber = line_num
					break

		cleanedContent=""
		for line_num in range(firstSubject_lineNumber,secondSubject_lineNumber):
			cleanedContent += (content[line_num] + "\n ")
		content = cleanedContent
		'''
		# 3. Splitting the content based on lines,and removing all fields except body and subject
		special_words = ['message-id:', 'date:', 'from:','sent:', 'to:','cc:','bcc', 'mime-version:', 'content-type:', 'content-transfer-encoding:', 'x-from:', 'x-to:', 'x-cc:', 'x-bcc:', 'x-origin:', 'x-filename:', 'x-priority:', 'x-msmail-priority:', 'x-mimeole:','return-path:','delivered-to:','received:','x-mailer:','thread-index:','content-class:','x-mimeole:','x-originalarrivaltime:','charset=','http://','by projecthoneypotmailserver','--=','clamdscan:','error:','alias:','=_nextpart_','href=','src=','size=','type=']
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
				cleanedContent += (line + "\n ")
		content = cleanedContent
							
		# 4. Get rid of HTML commands, replacing them with space 
		'''
		cleanedContent	= re.sub(">(.|\n)*?</","",content)
		cleanedContent	= re.sub("{(.|\n)*?}","",cleanedContent)
		cleanedContent	= re.sub("<.*?>","",cleanedContent)
		cleanedContent  = re.sub("&.*?;","",cleanedContent)
		cleanedContent	= re.sub("=[0-9]*","",cleanedContent)
		cleanedContent = re.sub(r"(?is)<(script|style).*?>.*?()", "", cleanedContent)
		cleanedContent = re.sub(r"(?s)[\n]?", "", cleanedContent) 		# Then remove html comments. 
		cleanedContent = re.sub(r"(?s)<.*?>", " ", cleanedContent)		# Next remove the remaining tags:
		content = cleanedContent
		'''
		# 5. Replace E-mail address with word "emailaddrs"
		cleanedContent=""
		for word in content.split():
			if "@" in word:
				#print(word)
				cleanedContent += "emailaddrs "
			else:
				cleanedContent += word + " "
		content = cleanedContent

		# 6. Replace Website address with word "webaddrs"
		cleanedContent=""
		for word in content.split():
			if (("http:" in word) or (".com" in word)):
				cleanedContent += "webaddrs "
			else:
				cleanedContent += word + " "
		content = cleanedContent
		# 7. Replace tab "\t" with space
		content = content.replace("\t"," ")
		# 8. Replacing punctuation characters with space
		cleanedContent =""
		for char in content:
			if char in punctuation_chars:
				cleanedContent += " "
			else:
				cleanedContent += char
		content = cleanedContent
		# 9. Replace multiple space with single space
		content = re.sub(" +"," ",content)
		# 10. removing number with the text "number"
		cleanedContent=""
		for word in content.split():
			if word.isdigit():
				cleanedContent += "number "
			else:
				cleanedContent += word + " "
		content = cleanedContent
		# 11. Removing stop words
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

		# 12. removing one letter words = a,b,c,...
		cleanedContent=""
		for word in content.split():
			if len(word) > 1:
				cleanedContent += (word + " ")
		content = cleanedContent
		# 13. removing words wich are distinctive of class
		content = content.split()
		distinctiveWords_indexes=[]
		distinctiveWords_list = ['hou','kaminski', 'kevin', 'ena','vince', 'enron','stinson','shirley','squirrelmail','ect','smtp','mime','gif','xls','mx','louise','ferc']
		for distinctiveWord in distinctiveWords_list:
			for word_num,word in enumerate(content):
				if word == distinctiveWord:
					distinctiveWords_indexes +=[word_num]
		cleanedContent=""
		for word_num,word in enumerate(content):
			if word_num not in distinctiveWords_indexes:
				cleanedContent += (word + " ")
		content = cleanedContent

		# 14.remove the words with more than 40 characters
		maxWordLength = 40
		content = "".join([word+" " for word in content.split() if len(word) <= maxWordLength])
		#15. Store and concatenate all content into a single string for analysis
		if len(content) > 0:
			content_concatenated += content	
			# 7. Splitting the content
			content_list += [content]
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

def computeConfMatrix(classifierOutputs,groundtruthList):
	confusionMatrix = np.zeros((2,2))
	for i in range(len(classifierOutputs)):
		predicted = classifierOutputs[i]
		groundtruth = groundtruthList[i]
		confusionMatrix[predicted][groundtruth] += 1
	return confusionMatrix


def performanceMetrics(confusionMatrix):
	precision_ham = confusionMatrix[0][0] / confusionMatrix.sum(axis=1)[0]
	recall_ham    = confusionMatrix[0][0] / confusionMatrix.sum(axis=0)[0]
	f1Score_ham   = 2 * precision_ham * recall_ham / (precision_ham + recall_ham)  
	precision_spam= confusionMatrix[1][1] / confusionMatrix.sum(axis=1)[1]
	recall_spam   = confusionMatrix[1][1] / confusionMatrix.sum(axis=0)[1]
	f1Score_spam  = 2 * precision_spam * recall_spam / (precision_spam + recall_spam)
	return f1Score_ham, f1Score_spam


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

# LSTM

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


# LSTM module

class NetworkLSTM(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		outputSize = kwargs.get('outputSize')
		numLayers = kwargs.get('numLayers')
		self.hiddenSize = kwargs.get('hiddenSize')
		embedSize = kwargs.get('embedSize')
		vocabSize = kwargs.get('vocabSize')
		dropout = kwargs.get('dropout')
		dropoutLSTM = kwargs.get('dropoutLSTM')
		self.device = kwargs.get('device')
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





