
# Artificail Neural Network




# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset

dataset=pd.read_csv('Churn_Modelling.csv')

X=dataset.iloc[:,3:13]
y=dataset.iloc[:,13]


# Create dummy variables

geography=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

# Concate the DataFrame

X=pd.concat([X,geography,gender],axis=1)

# Drop unnecessay columns

X=X.drop(['Geography','Gender'],axis=1)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit(X_test)


# Part 2 --Now Lets make the ANN

# Imorting the Keras libraries and packages

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

# Initialising the ANN

classifier=Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,epochs=100)


