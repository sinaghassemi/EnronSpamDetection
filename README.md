# EnronSpamDetection

The repository contains the codes addressing the spam detection over [Enron-Spam](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) dataset. Enron-Spam dataset includes non-spam (ham) messages from six Enron employess who had large mail boxes, and also it includes spam messages from four differnet sources namely: the SpamAssassin corpus, the Honeypot project, the spam collection of Bruce Guenter, and spam collected by the authors of the [paper](http://www2.aueb.gr/users/ion/docs/ceas2006_paper.pdf).

The codes organized as following : 
-  The main file `main.py` used to apply classification and measure the performance.
-  The file `EnronDataset.py` contains:
	- Class `EnronLoader` to read and pre-process E-mail contents.
	- Class `EnronBatchLoader` returns an iterable object to be used for mini-batch loading during training/testing neural networks.
- The file `utilities.py` includes usefull functions for contents analysis and also measuring classification performance.
- The file `LogisticRegression.py` implement `ClassifierLogisticRegression` class for Logistic Regression Classifier.
- The file `LSTM.py` implement `ClassifierLSTM` class for Long Short-Term Memory networks.


The methods used for spam classification are : Decision Tree, Multinomial Naive Bayes, K-Nearest Neighbors classifiers (scikit-learn) and also Logistic Regression and LSTM (PyTorch).
For extracting the features for all classifiers except LSTM, we use bags of words method in which we selected a number of most common words in E-mails (after pre-processing), and for each E-mail in the dataset we count the number of selected words in that E-mail hence our data would be two dimentional array where the number of rows is the number of samples and the number of columns is the number of select words (words in the bag). However, as we will see in the next parts, since LSTM can takes sequence input of different sizes, we use word embedding techniuqe for extracting features for LSTM calssifier.

The classifcation performance is measured using accuracy, precision, recall, and f1-score for both spam and ham classes as well as RoC curves for all classifiers.

But, first of all, to classify E-mails to spam and ham classes, we pre-process the raw E-mails using `EnronLoader` class as following:


# Pre-processing


The goal of this project is to detect whether an E-mail is spam or not (ham) solely based on the content and the subject. Therefore, in pre-processing stage we remove all other parts of an email except subject and the body or the content. For reading and preprocessing the raw Enron-spam dataset, `Class EnronLoader` is provided in `EnronDataset.py`.



```python

class EnronLoader(object):
	def __init__(self,**kwargs):
		spamDir = kwargs.get('spamDir')
		hamDir  = kwargs.get('hamDir')
		if spamDir == None or hamDir == None:
			raise NameError("the directory containing ham and spam should be provided")
		self.spamFiles = self._filesToBeRead(spamDir)
		self.hamFiles  = self._filesToBeRead(hamDir)

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
		'dynegy', 'skilling', 'mmbtu', 'westdesk', 'epmi', 'fastow', 'bloomberg','ist', 'slashnull', 'xp', 'localhost', 'dogma',\ 			'authenticated','ees','esmtp','john','fw','postfix','xp','3a','steve','cs','mike','macromedia','http','secs', 'futurequest','scheduling']
		# if the number of words exceeded maxContentLength, trunk the content
		self.maxContentLength = kwargs.get('maxWords',1000)
```
To initilize the class, two keywords arguments `spamDir` and `hamDir` should be provided which locates the raw files belonging to spam and ham E-mails. Moreover, the punctuation marks is provided which will be removed from the content as they are not usefull for classification and take space in the vocabulary. ` header_words ` is the list of words indicating the lines in the E-mail which are not body and the subject such as 'message-id:' and 'date:'. To challenge the classification methods and their generalization capability, I also remove the most common words which are only present in one of the catagories(ham or spam), some of these words which are only present in ham E-mails are the name of Enron employees which the ham files are originated from,  or the words that are only seen in spam files are the name of the companies which have sent spam E-mails. Therefore, I decided to exclude these words to further challenge the classifiers and prevent overfitting on dataset. In the next part, we will see how these distintive words are chosen.

```python
class EnronLoader(object):
	.
	.
	.

	def _filesToBeRead(self,path):
	# function to return list of all files in leaves given a root tree directory
		fileList = []
		for root,dirs,files in os.walk(path):
			if len(dirs) == 0: 	# leaves : containing the files
				for f in files:
					fileList += [os.path.join(root,f)]
		return fileList

	def readHam(self):
		print('\n'+'*'*5 + 'reading ham files' + '*'*5)
		content = self._preprocess(self.hamFiles)
		return content

	def readSpam(self):
		print('\n'+'*'*5 + 'reading spam files' + '*'*5)
		content = self._preprocess(self.spamFiles)
		return content

```

The class `EnronLoader` also includes a method `_filesToBeRead(self,path)` which returns the list of files given the root directory. Methods `readHam(self)` and `readSpam(self)` are used to read and preprocess all the files in `spamDir` and `hamDir` using  `preprocess` method.


```python
class EnronLoader(object):
	.
	.
	.

	def _preprocess(self,files_list):
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
			# 3. Get rid of HTML commands, replacing them with space 
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
```

The `preprocess` method read and process the raw content from a list of files using in 12 steps: first we convert all letters to lowercase, then lines that contain E-mail header such as message-id:, date:, from:, etc are removed form the content preserving only body and the subject. In third step, regular expression are used to remove HTTP syntax as they will not contribute to the classification. In the following in step 4, 5, and 7 we replace any E-mail or Website address or any time and date or number with the words : 'emailaddrs' , 'webaddrs' , 'time', 'date, and 'number' respectively.
The logic behind this pre-processing is that the actual E-mail or website address or a number or a date will not provide information for detecting spam E-mails, and if it does, it will be limited to this dataset and will not contribute to the genralization of classifier. For instance, if in this dataset spam filters are coming from a certain number of companies, then the classifier will be trained to detect spam based on the web address of these comapanies which in real application it might not include all types of spam. In addition, using generelized term such as 'emailaddrs' for all E-mail addresses will prevent the unnecessary increase of the vocabulary size.
In the next lines of code, the punctuation marks, mutiple cosecutive spaces are eliminated from the content. 
Stop words such as: whom, this, that, ... which can not provide useful information are also removed in pre-processing. 
After these cleaning attempts, there might remained some meaningless one character words or very long words, therefore we remove these words by setting boundary for word length.
In the next part, there might be E-mails that after preprocessing do not include any words which will not be considered. In the other hand, some E-mails contain a huge number of words, as we will see in the next part, to prevent very large input demension for LSTM classifier, we set a upper bpundary for number of words in the E-mail, E-mail which exceed this boundary will be truncated.

The following is the histogram of number of words in both spam and ham folders after applying the preprocessing.
As can be seen the emials which are truncated at 1000 words are very small portion of the data (less than 1 %) hence we will not loss information by setting boundary on number of E-mail words.

![vai](readMe/ContentLenghts.png "The histogram")



# Data loader for LSTM and logistic regression classifier

`EnronBatchLoader` is a iterator provided in `EnronDataset.py` to iterate over pre-processed Enron dataset to be used while training/testing logistic regression and LSTM networks.



```python
class EnronBatchLoader():
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
		# if LSTM then we should return sequence length as well
		if self.LSTM :
			if not isinstance(self.seqLengths, type(torch.Tensor())):
				raise TypeError('seqLengths should be a tensor')
		self.size 	= self.label.size(0)
		self.shuffle	= kwargs.get('shuffle',False)	
	def _batchSampler_generator(self):
		# generator used to return a batch indexes using sampler
		batchIndexes = []
		for idx in self.sampler_index:
			batchIndexes.append(idx)
			if len(batchIndexes) == self.batchSize:
				yield batchIndexes
				batchIndexes = []
		if len(batchIndexes) > 0 :
			yield batchIndexes
	def __iter__(self):
		# when start iteration, this method will be called, reset the iterators
		# if shuffle then return a random permutation, if not just a range for sampler
		# batch sampler will be intialized with a generator 
		if self.shuffle:
			self.sampler_index = iter(np.random.permutation(self.size))
		else:
			self.sampler_index = iter(range(self.size))
		self._batchSampler = self._batchSampler_generator()
		return self
	def __next__(self):
		batchInd = next(self._batchSampler)
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
```

In `EnronBatchLoader`, batch size (integer), data (PyTorch Tensor), and label (PyTorch Tensor) are provided as keyword arguments,
nevertheless, when used as data loader for LSTM, it also takes sequence length as argument. The data loader in each iteration returns a mini-batch of data and labels (shuffled if shuffled is True) and if LSTM is ture also a batch of sequence length. For LSTM, it is required to sort data in each mini-batch based on sequence length such that the longest sequence should be first. This is necessary as we will see when we trained LSTM for sequences with different lengths.


# `utilities.py`

In `utilities.py`, there are some useful functions which will be used in analysing text and also to measure the classification performance.


```python
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
```

`extractVocab` functions takes as argument a string and extract the words and their count of occurence in the string and returns a vocabulary dictionary where keys are words and values are the wrods count. `wordCount` is used to sort the words based on their counts given a vocabulary dictionary, it returns two lists: sorted words, and their count.


```python
def computeConfMatrix(predictions,labels):
	'''
	Takes predictions and labels
	as arg in shape of 1D array with size of num of samples
	Return Confusion matrix
	'''
	confusionMatrix = np.zeros((2,2))
	for i in range(len(predictions)):
		predicted = predictions[i]
		groundtruth = labels[i]
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

def ROC(predictions_prob,labels):
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
		predictions_th = (predictions_prob > th)*1 
		conf_th = computeConfMatrix(predictions_th,labels) 
		metrics_th = performanceMetrics(conf_th)
		FPR[i] = metrics_th['FPR_spam']
		TPR[i] = metrics_th['TPR_spam']
	return FPR,TPR
```

`computeConfMatrix` takes as argument predictions and groundtruth labels and it returns confusion matrix in which rows represent predicted class and columns represent the actual class.
`performanceMetrics` takes as argument the confusion matrix and it computes several metrics for both spam and non-spam classes including: accuracy, precision, recall, f1-score, false positive rate, and true positve rate and it returns the metrics in a python dictionary.
`ROC` takes as argument predictions probabilties (for spam class) and labels, by setting a threshold (from 0 to 1 by a step of 0.02) it computes the false and true positive rates to be used for ploting ROC curves.











