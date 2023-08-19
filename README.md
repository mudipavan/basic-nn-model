# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network with multiple hidden layers and multiple nodes in each hidden layer is known as a deep learning system or a deep neural network. Here the basic neural network model has been created with one input layer, one hidden layer and one output layer.The number of neurons(UNITS) in each layer varies the 1st input layer has 16 units and hidden layer has 8 units and output layer has one unit.

In this basic NN Model, we have used "relu" activation function in input and hidden layer, relu(RECTIFIED LINEAR UNIT) Activation function is a piece-wise linear function that will output the input directly if it is positive and zero if it is negative.



## Neural Network Model

![deep n](https://github.com/mudipavan/basic-nn-model/assets/94828517/8c4adea2-f5c6-470f-bdab-b5f588ea98f0)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

```
## Importing modules

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## Authenticate & Create data frame using data in sheets

from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('deep').sheet1
data = worksheet.get_all_values()
dataset1=pd.DataFrame(data[1:],columns=data[0])
dataset1=dataset1.astype({'input':'float'})
dataset1=dataset1.astype({'output':'float'})
dataset1.head()

## Assign X & Y Values

x = dataset1[['input']].values
y = dataset1[['output']].values
x

## Normalize the values and split the data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
x_train1 = Scaler.transform(x_train)
## Create a neural network and train it.
ai_brain=Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x_train1,y_train,epochs=2000)

## Plot the loss

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

## Predict for some value

ai_brain.evaluate(x_test1,y_test)

x_test1 = Scaler.transform(x_test)
x_n1 = [[30]]
x_n1_1 = Scaler.transform(x_n1)
ai_brain.predict(x_n1_1)
```

## Dataset Information

![d-1-1](https://github.com/mudipavan/basic-nn-model/assets/94828517/96431d5d-2504-4e9f-8eb0-26012d1c69cb)

## OUTPUT

### Training Loss Vs Iteration Plot

![d-1-2](https://github.com/mudipavan/basic-nn-model/assets/94828517/5283dac7-9681-4dca-9839-d89fa0c21ead)

### Test Data Root Mean Squared Error

![d-1-3](https://github.com/mudipavan/basic-nn-model/assets/94828517/d12862cc-5edd-49c5-ab7c-4d779f504039)

### New Sample Data Prediction

![d-1-4](https://github.com/mudipavan/basic-nn-model/assets/94828517/068d9676-42c1-42b8-b53f-634ddfd5a271)

## RESULT

Therefore We successfully developed a neural network regression model for the given dataset.
