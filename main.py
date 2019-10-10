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
#from sklearn.model_selection import train_test_split

train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")

print("---------------------------------------------")
print("Train dataset: "+str(train_set.shape))

print("---------------------------------------------")
print("Test dataset: "+str(test_set.shape))

#Handle missing data
train_set[train_set.isnull().any(axis=1)]
train_set = train_set.dropna(axis='columns')

test_set[test_set.isnull().any(axis=1)]
test_set = test_set.dropna(axis='columns')

print("---------------------------------------------")
print("Train dataset: "+str(train_set.shape))

print("---------------------------------------------")
print("Test dataset: "+str(test_set.shape))

print(train_set.columns)
col = [1,2,3,4,5,6,7]
train_set.drop(train_set.columns[col],axis=1,inplace=True)
test_set.drop(test_set.columns[col],axis=1,inplace=True)