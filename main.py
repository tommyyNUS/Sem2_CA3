# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:03:01 2019

@author: Tommy Yong
"""
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split

data = pd.read_csv("train.csv")

print("---------------------------------------------")
print("Dataset: "+str(data.shape))

#Handle missing data
data[data.isnull().any(axis=1)]
data = data.dropna(axis='columns')

print("---------------------------------------------")
print("Dataset after dropping columns: "+str(data.shape))

#Drop unncessary columns
print(data.columns)
col = [1,2,3,4,5,6,7]
data.drop(data.columns[col],axis=1,inplace=True)

#Category encoding
#exactly according to the specification (Class A) 
#throwing the elbows to the front (Class B)
#lifting the dumbbell only halfway (Class C)
#lowering the dumbbell only halfway (Class D)
#throwing the hips to the front (Class E)
le = preprocessing.LabelEncoder()
le.fit(data['classe'])
print(le.classes_)
data['classe'] = le.transform(data['classe'])

#Shuffle data
data = data.sample(frac=1)

X = data.iloc[:, 0:52]
Y = data.iloc[:, 52]
Y1 = np.array(Y)
#Split into train and test data
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#Visualize data


#Build and train Random forest model
from sklearn.neighbors import KNeighborsClassifier 
model1 = KNeighborsClassifier(5)
model1.fit(xTrain, yTrain)

#Create predictions
pred = model1.predict(xTest)

#Evaluate model
from sklearn.metrics import classification_report, confusion_matrix

print("=== Confusion Matrix KNeighbors ===")
print(confusion_matrix(yTest, pred))

print("\n=== Classification Report KNeighbors ===")
print(classification_report(yTest, pred))

#Build and train decision tree model
from sklearn.naive_bayes import GaussianNB 
model2 = GaussianNB()
model2.fit(xTrain, yTrain)

#Create predictions
pred2 = model2.predict(xTest)

#Evaluate model
from sklearn.metrics import classification_report, confusion_matrix

print("\n=== Confusion Matrix GuassianNB ===")
print(confusion_matrix(yTest, pred2))

print("\n=== Classification Report GuassianNB ===")
print(classification_report(yTest, pred2))

