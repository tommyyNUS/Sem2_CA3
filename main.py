# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:03:01 2019

@author: Tommy Yong
"""
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
import pandas as pd
from sklearn import preprocessing

#from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Flatten
#from tensorflow.keras.layers import Conv2D
#from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.layers import Activation
#from tensorflow.keras.layers import AveragePooling2D
#from tensorflow.keras.layers import add
#from tensorflow.keras.regularizers import l2
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras import optimizers
#from tensorflow.keras.layers import Dropout 
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.layers import MaxPooling2D
#import tensorflow as tf
#import keras.callbacks
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

#Split into train and test data
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Visualize data
colours = ['Red', 'Blue', 'Green', 'Purple', 'Orange']
#plt.scatter(xTrain['magnet_arm_y'], xTrain['magnet_arm_z'], c=colours[yTrain], labels = colours)

for row in xTrain:
    print("Row index is: "+str(row.index))
    print(row)
    plt.scatter(row['magnet_arm_y'], row['magnet_arm_z'], c=colours[yTrain[row.index]], labels = colours)

plt.title("Similar activity data")
plt.show