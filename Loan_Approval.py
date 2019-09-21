# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 22:27:21 2019

@author: Prashun
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Training dataset
data_1=pd.read_csv("train_ctrUa4K.csv")
data_1=data_1.replace(to_replace='3+',value=3)
data_1=data_1.replace(to_replace='No',value=0)
data_1=data_1.replace(to_replace='Yes',value=1)
X_train=data_1.iloc[:,1:-1].values
Y_train=data_1.iloc[:,-1].values

#Importing the Test dataset
data_2=pd.read_csv("test_lAUu6dG.csv")
data_2=data_2.replace(to_replace='3+',value=3)
data_2=data_2.replace(to_replace='No',value=0)
data_2=data_2.replace(to_replace='Yes',value=1)
X_test=data_2.iloc[:,1:].values


# Encoding the Independent Variable for Training Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X_train[:,0] = labelencoder_X1.fit_transform(np.array(X_train[:,0]).astype('str'))
onehotencoder1 = OneHotEncoder(categorical_features = [0])
X_train = onehotencoder1.fit_transform(X_train).toarray()
labelencoder_X2 = LabelEncoder()
X_train[:,3] = labelencoder_X2.fit_transform(X_train[:,3].astype('str'))
onehotencoder2 = OneHotEncoder(categorical_features = [3])
X_train = onehotencoder2.fit_transform(X_train).toarray()
labelencoder_X3 = LabelEncoder()
X_train[:,10] = labelencoder_X3.fit_transform(X_train[:,10].astype('str'))
onehotencoder3 = OneHotEncoder(categorical_features = [10])
X_train = onehotencoder3.fit_transform(X_train).toarray()

# Encoding the Independent Variable for Test Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X4 = LabelEncoder()
X_test[:,0] = labelencoder_X4.fit_transform(np.array(X_test[:,0]).astype('str'))
onehotencoder4 = OneHotEncoder(categorical_features = [0])
X_test = onehotencoder4.fit_transform(X_test).toarray()
labelencoder_X5 = LabelEncoder()
X_test[:,3] = labelencoder_X5.fit_transform(X_test[:,3].astype('str'))
onehotencoder5 = OneHotEncoder(categorical_features = [3])
X_test = onehotencoder5.fit_transform(X_test).toarray()
labelencoder_X6 = LabelEncoder()
X_test[:,10] = labelencoder_X6.fit_transform(X_test[:,10].astype('str'))
onehotencoder6 = OneHotEncoder(categorical_features = [10])
X_test = onehotencoder6.fit_transform(X_test).toarray()

# Taking care of missing data(Training Set)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_train[:,0:5])
X_train[:, 0:5] = imputer.transform(X_train[:, 0:5])
imputer = imputer.fit(X_train[:,9:])
X_train[:, 9:] = imputer.transform(X_train[:, 9:])
imputer2 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer2 = imputer2.fit(X_train[:,5:8])
X_train[:, 5:8] = imputer2.transform(X_train[:, 5:8])
imputer3 = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer3= imputer3.fit(X_train[:,8:])
X_train[:, 8:] = imputer3.transform(X_train[:, 8:])
X_train=X_train.astype('float64')

# Taking care of missing data(Test Set)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_test[:,0:5])
X_test[:, 0:5] = imputer.transform(X_test[:, 0:5])
imputer = imputer.fit(X_test[:,9:])
X_test[:, 9:] = imputer.transform(X_test[:, 9:])
imputer2 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer2 = imputer2.fit(X_test[:,5:8])
X_test[:, 5:8] = imputer2.transform(X_test[:, 5:8])
imputer3 = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer3= imputer3.fit(X_test[:,8:])
X_test[:, 8:] = imputer3.transform(X_test[:, 8:])
X_test=X_test.astype('float64')

# Encoding the Dependent Variable
labelencoder_y1 = LabelEncoder()
Y_train = labelencoder_y1.fit_transform(Y_train)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred2 = classifier.predict(X_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
ConfusionMatrix = confusion_matrix(Y_train, Y_pred2)

