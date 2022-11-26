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
location = results['Location'].tolist()
bedroom = results['Bedroom'].tolist()
bathroom = results['Bathroom'].tolist()
propertyType = results['PropertyType'].tolist() #Need to convert the three different types into numbers
price = results['Price'].tolist()
predictions = results['Predictions'].tolist()
print('...Done.')

LabelList = [[] for _ in range(len(results))]
DataList = [[] for _ in range(len(results))]

for i in range(len(results)):
    LabelList[i].append(location[i])
    DataList[i].append(bedroom[i])
    DataList[i].append(bathroom[i])
    DataList[i].append(propertyType[i])
    DataList[i].append(price[i])
    DataList[i].append(predictions[i])
X = np.array(DataList)
y = np.array(LabelList)

yLabel = [ [0]*25 for i in range(len(y))]
for index in range(len(LabelList)):
    print(LabelList[int(index)][0])
    val = int(LabelList[int(index)][0])
    yLabel[int(index)][val] = 1
    
yLabel = np.array(yLabel)


#X_train, X_test, y_train, y_test = train_test_split(DataList, LabelList, test_size=.33, random_state=26)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=26)
#X_train, X_test, y_train, y_test = train_test_split(X, yLabel, test_size=.33, random_state=26)
print(X_train)

import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
   
batch_size = 16 #64

# Instantiate training and test data
train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break


"""
Batch: 1
X shape: torch.Size([64, 2])
y shape: torch.Size([64])
"""

import torch
from torch import nn
from torch import tanh
from torch import optim

input_dim = 5
hidden_dim = 16
hidden_dim2 = 32
#output_dim = 1 #24
output_dim = 1

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim2)
        #nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer3 = nn.Linear(hidden_dim2, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.relu(self.layer3(x))

        return x   
    
model = NeuralNetwork(input_dim, hidden_dim, output_dim)
#model = NeuralNetwork()
print(model)


#Train model 
learning_rate = 0.1
#loss_fn = nn.BCELoss() #Binary cross-entropy 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #should probably use adam

num_epochs = 1000
loss_values = []

print("\nbat_size = %3d " % batch_size)
print("loss = " + str(loss_fn))
print("optimizer = SGD")
print("max_epochs = %3d " % num_epochs)
print("lrn_rate = %0.3f " % learning_rate)
# since we're not training, we don't need to calculate the gradients for our outputs
for epoch in range(num_epochs):
    for X, y in train_dataloader: 
        optimizer.zero_grad()
        
        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

print("Training Complete")

X_test = torch.Tensor(X_test)
y_pred = model(X_test)
y_pred = y_pred.detach().numpy()
print('\n\ny_pred: ', y_pred)
y_pred = np.rint(y_pred)
print('\n\y_test: ', y_test)
print('\n\ny_pred: ', y_pred)

print('Print accuracy score:')
print(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
print(classification_report(y_test, y_pred))

print('Done!')

