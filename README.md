# EnronSpamDetection

The repository contains the codes addressing the spam detection over [Enron-Spam](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) dataset.

Enron-Spam dataset includes ham messages from six Enron employess who had large mail boxes, and also it includes spam messages from four differnet sources namely: the SpamAssassin corpus, the Honeypot project, the spam collection of Bruce Guenter, and spam collected by the authors of the [paper](http://www2.aueb.gr/users/ion/docs/ceas2006_paper.pdf).

To classify E-mails to spam and non-spam (ham) classes, first we pre-process the raw E-mails then the cleaned and pre-processed data will be splitted into training, validation and test sets then several machine learning approaches are provided with thier corresponding perfromance on the data.


# 1. Pre-processing


The goal of this project is to detect whether an E-mail is spam or ham solely based on content and subject.
Therefore, in pre-processing stage we remove all other parts of an email except subject and the body or the content.

For reading, cleaning, and preprocessing the raw Enron-spam dataset, ```Class EnronLoader``` is provided as following.



```python
class EnronLoader(object):
	def __init__(self,**kwargs):
		spamDir = kwargs.get('spamDir')
		hamDir  = kwargs.get('hamDir')
		if spamDir == None or hamDir == None:
			raise NameError("the directory containing ham and spam should be provided")
		self.spamFiles = self.__filesToBeRead(spamDir)
		self.hamFiles  = self.__filesToBeRead(hamDir)
		# Punctuations to be removed
		self.punctuation_marks = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',\
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
```

To initilize the class, two keywords arguments namely ```spamDir``` and ```hamDir``` should be provided which locates the raw files belonging to spam and ham E-mails.
Moreover, the punctuation marks is provided which will be removed from the content as they are not usefull for distinguishing spam from ham E-mails.
``` header_words ``` is the list of words which indicates lines in the E-mail which are not body and the subject, hence lines containing these words will be removed.
To challenge the classification methods and their generalization capability, I also remove the most common words which are only present in one of the catagories(ham or spam),
some of these words which are only present in ham E-mails are the name of Enron employees which the ham files are originated from, 
or the words that are only seen in spam files are the name of the companies which have sent spam E-mail.
Therefore, I decided to exclude these names to further challenge the classifiers. In the next part, we will see how these distintive words are chosen.

```python
class EnronLoader(object):
	.
	.
	.

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

```

The class ```EnronLoader``` also includes a method ```__filesToBeRead(self,path)``` which returns the list of files given the root directory.
Methods ```readHam(self)``` and ```readSpam(self)``` are used to read and preprocess all the files in ```spamDir``` and ```hamDir```.


```python
class EnronLoader(object):
	.
	.
	.

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
			# 2. Removing everything from first line to subject and from the second subject to the end:
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


```

The ```__preprocess(self,files_list)``` method is used to prepare and process the raw content from a list of files,
the method can be catagorized into 16 steps as following. In the step 1, all the letters in the content are transformed to lower case.
In the second step, to preserve only the subject and the body of the E-mail and also to prevent repetetive content in the case of E-mail reply,
all lines from the first line to the line includes subject (which is header including information such as sender,data) is removed, and also if there is a second subject (in the case of reply) 
everthing after second subject is removed to avoid duplicates in samples, particularly it is not desirable if a duplicate version of a sample is in both training and test sets which will undermine the classifier performance. 




```python
class EnronLoader(object):
	.
	.
	.

	def __preprocess(self,files_list):
		.
		.
		.
		.
			# 3. Splitting the content based on lines, and removing fields except body and subject
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

```



In third step, we reomved the lines including words such as 'date:', 'from:','sent:',... . Since the E-mail contents especially spam ones include alot of HTML syntaxes, regular expression are used to remove these type of syntaxes.
Then, we also subtitute the E-mail addresses with the word "emailaddrs" and also website addresses with the word "webaddrs" to constrain the classifier on the content not clues related to E-mail or website address.


































# 1.1 Lower case

# 1.2 Reserving subject and body

# 1.3 Removing html commands

# 1.4 Replacing E-mail addresses and websites

# 1.5 Removing puncuations

# 1.6 Replacing numbers

# 1.7 Removing stop words

# 1.8 Removing destinctive words


# 2. Feature selection

# 3. Methods


# 3.1 Naive Bayes Classfier

# 3.2 Logistic Regression

# 3.3 K-nearest neighbor

# 3.4 decision tress

# 3.5 LSTM







