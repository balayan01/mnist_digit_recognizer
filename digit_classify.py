# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 09:35:50 2018

@author: user
"""

import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

image_data = pd.read_csv('train.csv') 

data_labels = image_data.iloc[:,:1]
data_images = image_data.iloc[:,1:]

data_images[data_images>0]=1
train_X,val_X,train_y,val_y =train_test_split(data_images,data_labels,train_size= 0.8,random_state =0)

model = RandomForestClassifier(random_state = 1,n_estimators = 200 )
model.fit(train_X,train_y.values.ravel())
model.score(val_X,val_y)

"""
import matplotlib.pyplot as plt
model.predict(val_X[:])
"""

test_data=pd.read_csv('test.csv')
test_data[test_data>0]=1
results=model.predict(test_data[:])

results
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
