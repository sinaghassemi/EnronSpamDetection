import os 
import re
import nltk
import torch
import numpy as np
from nltk.corpus import stopwords
nltk.download('stopwords')

class EnronLoader(object):
	def __init__(self,**kwargs):
		spamDir = kwargs.get('spamDir')
		hamDir  = kwargs.get('hamDir')
		if spamDir == None or hamDir == None:
			raise NameError("the directory containing ham and spam should be provided")
		self.spamFiles = self.__filesToBeRead(spamDir)
		self.hamFiles  = self.__filesToBeRead(hamDir)

		self.spamFiles = self.spamFiles[:10000]
		self.hamFiles = self.hamFiles[:10000]

		# Punctuations to be removed
		self.punctuation_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',\
		 '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
		# Lines including the header words will be eliminated
		self.header_words = ['message-id:', 'date:', 'from:','sent:', 'to:','cc:','bcc', 'mime-version:', 'content-type:', \
		'content-transfer-encoding:', 'x-from:', 'x-to:', 'x-cc:', 'x-bcc:', 'x-origin:', 'x-filename:', 'x-priority:', 'x-msmail-priority:',\
		 'x-mimeole:','return-path:','delivered-to:','received:','x-mailer:','thread-index:','content-class:','x-mimeole:','x-originalarrivaltime:',\
		'charset=','http://','by projecthoneypotmailserver','--=','clamdscan:','error:','alias:','=_nextpart_','href=','src=','size=','type=']
		# Words in the following list will be eliminated if presented in data to avoid words which are only presented in one class (ham or spam) (distinctive words) 
		self.distinctive_words = ['hou','kaminski', 'kevin', 'ena','vince', 'enron','stinson','shirley','squirrelmail','ect','smtp','mime','gif',\
		'xls','mx','louise','ferc',	'ppin', 'wysak', 'tras', 'rien', 'saf', 'photoshop', 'viagra', 'cialis', 'xual', 'voip',\
		'dynegy', 'skilling', 'mmbtu', 'westdesk', 'epmi', 'fastow', 'bloomberg']
		# if the number of words exceeded 5000, trunk the content
		self.maxContentLength = 1000

	def __filesToBeRead(self,path):
	# function to return list of all files in leaves given a root tree directory
		fileList = []
		for root,dirs,files in os.walk(path):
			if len(dirs) == 0: 	# leaves : containing the files
				for f in files:
					fileList += [os.path.join(root,f)]
		return fileList

	def readHam(self):
		print('\n'+'*'*5 + 'reading ham files' + '*'*5)
		content = self.__preprocess(self.hamFiles)
		return content

	def readSpam(self):
		print('\n'+'*'*5 + 'reading spam files' + '*'*5)
		content = self.__preprocess(self.spamFiles)
		return content

	def __preprocess(self,files_list):
		content_list=[]
		numOfFiles = len(files_list)
		for (num_file,File) in enumerate(files_list):
			print("Reading and Preprocessing content [% 5d / %d] : %s" % (num_file+1,numOfFiles,File),end="\r")
			f = open(File,'r',encoding="ISO-8859-1")
			content = f.read()
			f.close()
			# Some Preprocessing over the content : 
			# 1. Converting all letters to lower case
			content = content.lower()
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
			# 3. Splitting the content based on lines,and removing all fields except body and subject
			content = content.split("\n")
			redundant_lines=[]
			for word in self.header_words:
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
			cleanedContent	= re.sub(">(.|\n)*?</","",content)
			cleanedContent	= re.sub("{(.|\n)*?}","",cleanedContent)
			cleanedContent	= re.sub("<.*?>","",cleanedContent)
			cleanedContent  = re.sub("&.*?;","",cleanedContent)
			cleanedContent	= re.sub("=[0-9]*","",cleanedContent)		
			content = cleanedContent
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
				if char in self.punctuation_chars:
					cleanedContent += " "
				else:
					cleanedContent += char
			content = cleanedContent
			# 9. Replace multiple space with single space
			content = re.sub(" +"," ",content)
			# 10. Replace number with the text "number"
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
			for distinctiveWord in self.distinctive_words:
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
			content = " ".join([word for word in content.split() if len(word) <= maxWordLength])
			# 15. Trunk the content if number of words exceeded
			if len(content.split()) > self.maxContentLength:
				shorten_content = " ".join(content.split()[:self.maxContentLength])
				content = shorten_content
			# 16. skipping one word content
			if len(content.split()) > 1:
				content_list += [content]
		return content_list

class EnronBatchLoader(object):
	def __init__(self, **kwargs):
		self.batchSize 	= kwargs.get('batchSize')
		self.data 	= kwargs.get('data')
		self.label 	= kwargs.get('label')
		self.seqLengths = kwargs.get('seqLengths')
		self.LSTM       = kwargs.get('LSTM',False)

		# check the inputs
		if not isinstance(self.batchSize, int):
			print(type(self.batchSize))
			raise TypeError('batch size should be an integer')
		if not isinstance(self.data, type(torch.Tensor())):
			raise TypeError('data should be a tensor')
		if not isinstance(self.label, type(torch.Tensor())):
			raise TypeError('label should be a tensor')
		if self.LSTM :
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
		if not self.LSTM: 
			return batch_data, batch_label
		else:
			batch_seqLengths = self.seqLengths[batchInd]
			batch_seqLengths, sorted_indexes = batch_seqLengths.sort(0, descending=True)
			batch_data = batch_data[sorted_indexes]
			batch_label = batch_label[sorted_indexes]
			return batch_data, batch_label, batch_seqLengths
	def __len__(self):
		return self.size

























