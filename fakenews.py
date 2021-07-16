pip install numpy pandas sklearn

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#read file
df=pd.read_csv('news.csv')
#shape and head
df.shape
df.head()

#get labels
labels=df.label
labels.head()

#split the dataset
x_train,x_test,y_train,y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#initialize the TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

# fit and transfer tarin set , transfer test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

#passiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#predict on test set
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100, 2)}%')

#confusion matrix
confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])