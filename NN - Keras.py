# https://visualstudiomagazine.com/articles/2020/12/15/pytorch-network.aspx
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import string
import re
from collections import Counter
import matplotlib.pyplot as plt
import random
from sklearn import svm
import torch
import numpy as np

# multi-class classification with Keras
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#load dataset
# dataframe = pandas.read_csv("neuralData.csv", header=None)
# dataset = dataframe.values

location = []
bedroom = []
bathroom = []
propertyType = []
price = []
predictions = []
results = pd.read_csv('neuralData.csv')
print('Number of lines in neuralData CSV: ', len(results))

print('Converting CSV columns into lists:')
address = results['Address'].tolist()
location = results['Location']
bedroom = results['Bedroom'].tolist()
bathroom = results['Bathroom'].tolist()
propertyType = results['PropertyType'].tolist() #Need to convert the three different types into numbers
price = results['Price'].tolist()
predictions = results['Predictions'].tolist()
print('...Done.')

LabelList = [[] for _ in range(len(results))]
DataList = [[] for _ in range(len(results))]

for i in range(len(results)):
    DataList[i].append(bedroom[i])
    DataList[i].append(bathroom[i])
    DataList[i].append(propertyType[i])
    DataList[i].append(price[i])
    DataList[i].append(predictions[i])

dataset = results.values
LabelList = dataset[:,2]

X = np.array(DataList)
Y = LabelList
print(Y)
print('y')
print(X)
print('X')

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print('dummy_y:', dummy_y)

model = Sequential()
# define baseline model
def baseline_model():
	# create model
	model.add(Dense(8, input_dim=5, activation='relu'))
	model.add(Dense(32, activation='softmax'))
	model.add(Dense(22, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# model = Sequential()
# model.add(Dense(8, input_dim=5, activation='relu'))
# model.add(Dense(20, activation='softmax'))
# # Compile model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Starting training')
estimator = KerasClassifier(build_fn=baseline_model, epochs=500, batch_size=16, verbose=0)
#estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=5, shuffle=True)
print('Start cross val score')
results = cross_val_score(estimator, X, dummy_y, cv=kfold, verbose=1)
print(results)

yhat = model.predict(X)
#print(*yhat)
# print('Predictions')
# b = np.zeros_like(yhat)
# b[np.arange(len(yhat)), yhat.argmax(1)] = 1
# print(*b)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))