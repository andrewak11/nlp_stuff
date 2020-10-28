#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 00:33:01 2020

@author: andrewak11
"""
#importing dataset
import pandas as pd

messages = pd.read_csv('/home/beast/Workspace/nlp_stuff/SpamClassifier/smsspamcollection/SMSSpamCollection',sep='\t',names=["label","message"])

# Data preprocessing
import re
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer

ps = PorterStemmer()
corpus=[]

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review) 

# Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

# train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state=0)

# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)

# predict spam
prediction = spam_detect_model.predict(X_test)

# Creating confusion matrix
from sklearn.metrics import confusion_matrix
confu = confusion_matrix(y_test,prediction)

# Calculating Accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,prediction)





