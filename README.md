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
Moreover, the punctuation marks is provided which will be removed from the content as there are not usefull for distinguishing spam from ham E-mails.
``` header_words ``` is the list of words which indicates lines in the E-mail which are not body and the subject, hence lines containing these words will be removed.
To challenge the classification methods and their generalization capability, we also remove those most common words which are only present in one of the catagories(ham or spam),
some of these words which are only present in ham E-mails are the name of Enron employees which the ham files are originated from, 
or the words that are only seen in spam files are the name of the companies which have sent spam E-mail.
Therefore, I decided to exclude these names to further challenge the classifers. In the next part, we will see how these distintive words are chosen.






The 


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







