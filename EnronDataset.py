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

		self.spamFiles = self.spamFiles[:-1]
		self.hamFiles = self.hamFiles[:-1]

		# Punctuation marks to be removed
		self.punctuation_marks = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',\
		 '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~','\t']
		# list of stop words
		self.stopwords_list = stopwords.words('english')
		# Lines including the header words will be eliminated
		self.header_words = ['message-id:', 'date:', 'from:','sent:', 'to:','cc:','bcc', 'mime-version:', 'content-type:', \
		'content-transfer-encoding:', 'x-from:', 'x-to:', 'x-cc:', 'x-bcc:', 'x-origin:', 'x-filename:', 'x-priority:', 'x-msmail-priority:',\
		 'x-mimeole:','return-path:','delivered-to:','received:','x-mailer:','thread-index:','content-class:','x-mimeole:','x-originalarrivaltime:',\
		'charset=','http://','by projecthoneypotmailserver','--=','clamdscan:','error:','alias:','=_nextpart_','href=','src=','size=','type=']
		# Words in the following list will be removed to avoid words which are only presented in one class (ham or spam) (distinctive words) 
		self.distinctive_words = ['hou','kaminski', 'kevin', 'ena','vince', 'enron','stinson','shirley','squirrelmail','ect','smtp','mime','gif',\
		'xls','mx','louise','ferc','ppin', 'wysak', 'tras', 'rien', 'saf', 'photoshop', 'viagra', 'cialis', 'xual', 'voip',\
		'dynegy', 'skilling', 'mmbtu', 'westdesk', 'epmi', 'fastow', 'bloomberg','ist', 'slashnull', 'xp', 'localhost', 'dogma', 'authenticated','ees','esmtp','john','fw','postfix','xp','3a','steve','cs','mike','macromedia','http','secs', 'futurequest','scheduling']
		# if the number of words exceeded maxContentLength, trunk the content
		self.maxContentLength = kwargs.get('maxWords',1000)

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
			print("Reading and pre-processing content [% 5d / %d] : %s" % (num_file+1,numOfFiles,File),end="\r")
			f = open(File,'r',encoding="ISO-8859-1")
			content = f.read()
			f.close()
			# Some Preprocessing over the content : 
			# 1. Converting all letters to lower case
			content = content.lower()
			# 2. Splitting the content based on lines, and removing fields except body and subject
			cleanedContent=""
			for line_num,line in enumerate(content.split("\n")):
				for word in self.header_words:
					if word in line: # if word in line , we don't want that line
						break
					elif word == self.header_words[-1]: # if word not in line and we check all the words, we  want that line
						cleanedContent += (line + "\n ")		
			content = cleanedContent				
			# 3. Get rid of HTML JAVA script
			#content = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", content)
			#content = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", content)
			content	= re.sub(">(.|\n)*?</","",content)
			content	= re.sub("{(.|\n)*?}","",content)
			content	= re.sub("<.*?>","",content)
			content = re.sub("&.*?;","",content)
			content	= re.sub("=[0-9]*","",content)		

			# 4. Replace E-mail address with word "emailaddrs" 
			#       and Website address with word "webaddrs"
			cleanedContent=""
			for word in content.split():
				if "@" in word:
					cleanedContent += "emailaddrs "
					continue
				if ("http:" in word) or (".com" in word):
					cleanedContent += "webaddrs "
					continue					
				cleanedContent += word + " "
			content = cleanedContent

			# 5. Replace dates and time with 'date' and 'time'
			content = re.sub(" [0-9]{2}/[0-9]{2}/[0-9]{4} " ," date ",content) 
			content	= re.sub(" [0-9]{1,2}:[0-9]{2} ",' time ',content)

			# 6. Replace punctuation characters with space
			cleanedContent =""
			for char in content:
				if char in self.punctuation_marks:
					cleanedContent += " "
				else:
					cleanedContent += char
			content = cleanedContent

			# 7. Replace number with the text "number"
			cleanedContent=""
			for word in content.split():
				if word.isdigit():
					cleanedContent += "number "
					continue					
				cleanedContent += word + " "
			content = cleanedContent

			# 8. Replace multiple consecutive spaces
			content = re.sub(" +"," ",content)

			# 9. Removing stop words and distinctive words
			WordsToBeRemoved = self.stopwords_list + self.distinctive_words
			cleanedContent=""
			for word in content.split():
				#print(word)
				for word_toBRem in WordsToBeRemoved:
					if word == word_toBRem: # we don't want this word
						break
					elif word_toBRem == WordsToBeRemoved[-1]:
						cleanedContent += (word + " ")
			content = cleanedContent
			
			# 10. Removing words with length smaller than 1 and bigger than a max
			cleanedContent = " ".join([word for word in content.split() if ( len(word) > 1 and len(word) < 40)])
			content = cleanedContent
			# 11. Trunk the content if number of words exceeded a max
			if len(content.split()) > self.maxContentLength:
				shorten_content = " ".join(content.split()[:self.maxContentLength])
				content = shorten_content
			# 12. Skipping E-mails with empty content after pre-processing
			if len(content) < 1:
					continue
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

























