# EnronSpamDetection

The repository contains the codes addressing the spam detection over [Enron-Spam](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) dataset.

Enron-Spam dataset includes ham messages from six Enron employess who had large mail boxes, and also it includes spam messages from four differnet sources namely: the SpamAssassin corpus, the Honeypot project, the spam collection of Bruce Guenter, and spam collected by the authors of the [paper](http://www2.aueb.gr/users/ion/docs/ceas2006_paper.pdf).

To classify E-mails to spam and non-spam (ham) classes, first we pre-process the raw E-mails then the cleaned and pre-processed data will be splitted into training, validation and test sets then several machine learning approaches are provided with thier corresponding perfromance on the data.


# 1. Pre-processing


The goal of this project is to detect whether an E-mail is spam or ham solely based on content and subject.
Therefore, in pre-processing stage we remove all other parts of an email except subject and the body or the content.

For reading, cleaning, and preprocessing the raw Enron-spam dataset, ```python class EnronLoader``` is written.
The input 



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







