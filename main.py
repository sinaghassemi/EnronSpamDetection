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
	punctuation_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']#[':',"/","-",'"',"_",")","(","*",";","#",">","<"]
	for File in files_list:
		print("reading file:%s" % File,end="\r")
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

spam_filesList = returnLeavesFiles(spam_dataPath)
ham_filesList = returnLeavesFiles(ham_dataPath)

print("number of spam files: %d" % len(spam_filesList))
print("number of ham files: %d" % len(ham_filesList))

spam_filesList = spam_filesList[:50]
ham_filesList  = ham_filesList[:50]

contentList_spam,allContent_spam = readFile(spam_filesList)
contentList_ham,allContent_ham = readFile(ham_filesList)
allContent = allContent_spam + allContent_ham
contentList = contentList_ham + contentList_spam 
lableList = [1]*len(contentList_ham) + [0]*len(contentList_spam)

vocab_spam = extractVocab(allContent_spam)
vocab_ham  = extractVocab(allContent_ham)
vocal_all  = extractVocab(allContent)

w_sorted_spam, c_sorted_spam  = wordCount(vocab_spam)
w_sorted_ham , c_sorted_ham   = wordCount(vocab_ham)
w_sorted_all , c_sorted_all   = wordCount(vocal_all)

'''
print("words in spam		words in ham")
for i in range(300):
	print('%03d % 20s % 5d    % 20s % 5d' % (i,w_sorted_spam[i],c_sorted_spam[i],w_sorted_ham[i],c_sorted_ham[i])) 


print("all")
for i in range(300):
	print('%03d % 20s % 5d' % (i,w_sorted_all[i],c_sorted_all[i])) 
'''

numMostCommon   = 20
mostCommonWords =  w_sorted_all[:numMostCommon]


# Defining a dictionar whose keys are most common words and values are the indexes

words_dictionary = {k:v for (v,k) in enumerate(mostCommonWords)}



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


# Tokenization : spliting the content of each file to list of words 

content_tokenized = []
for content in contentList:
	content_tokenized += [content.split()]


# Vectorization : Converting the words to integres using the most common dictionary

#print(content_tokenized[0])


numOfContents = len(content_tokenized)
content_vectorization = np.zeros((numOfContents,numMostCommon),dtype=np.uint16)

for (row,content) in enumerate(content_tokenized):
	for word in content:
		if word in words_dictionary:
			word_index = words_dictionary[word]
			content_vectorization[row,word_index]+=1



#print(mostCommonWords)
#print(content_vectorization[0])
#print(contentList[0])			

# spliting train / test
trainRatio = 0.7
trainLength = int(trainRatio*numOfContents)
testLength  = numOfContents - trainLength

print("total number of samples %d splitted to %d training samples and %d test samples" % (numOfContents,trainLength,testLength))

data_trainSet = content_vectorization[:trainLength,:]

data_testSet  = content_vectorization[trainLength:,:]

label_trainSet = np.array(lableList[:trainLength])
label_testSet  = np.array(lableList[trainLength:])


########## naive bayes classifier
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()




classifier.fit(data_trainSet, label_trainSet)
prediction = classifier.predict(data_testSet) 

print(prediction)
print(label_testSet)

			






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























