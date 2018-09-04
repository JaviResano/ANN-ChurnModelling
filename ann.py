# Artificial Neural Network

# Installing Theano (from python terminal (anaconda promt))
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow (from python terminal (anaconda promt))
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html
#   conda create -n tensorflow pip python=3.6.4                 #creates a tensorflow environment in python
#   activate tensorflow                                         #activates the environment
#   pip install --ignore-installed --upgrade tensorflow         #Installs Tensorflow (cpu only version)
#   pip install ipykernel cloudpickle                           #Installs these libraries in the new environment
#   select, in spyder->herramientas->preferencias->interprete de python:
#       C:\ProgramData\Anaconda3\envs\tensorflow\python.exe
#   y reiniciar spyder

# Installing Keras (from python terminal (anaconda promt))
# pip install --upgrade keras



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # Needed to ititialize NN
from keras.layers.core import Dense # Needed to build the layers of the NN


# Initialising the ANN
#   Can be done as sequence of layers (like this) or by defining a graph
classifier = Sequential()

# Adding the input layer and the first hidden layer
#   output_dim size "quick rule of thumb": average from input and output nodes (11+1/2 in this case). Best number is an art to find.
classifier.add(Dense(units = 6, kernel_initializer='random_uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer='random_uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer='random_uniform', activation = 'sigmoid'))
    #classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'softmax')) #In case of classifying non-binary output

# Compiling the ANN ( with stochastic gradient descent)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # adam = stochastic gradient descent parameter optimization algorithm
    # binary_crossentropy = loss function, cost funtion used to optimize the parameters (in this case, logaritmic loss function)

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 5, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)