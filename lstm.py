import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, Activation

dataset = pd.read_csv('GOOG.csv')
dataset['Date'] =  pd.to_datetime(dataset['Date'])
df = dataset.set_index('Date')

df['Open'].plot(figsize = (10, 6), title = 'Google Stock Price $')

filter_cols = ["Open", "High", "Low", "Close"] # Only these columns will be extracted from the dataset
x_window_size = 14 # timestep for the LSTM model
y_window_size = 1 # The model will predict the price only for the next 1 day
y_column = "Open" # The target variable

counter = 0
nbepochs = 5
batch_size_ = 10
date = pd.DataFrame(columns = ['Date'])

Result = [] # The predicted 'Open' prices
Test = [] # Real 'Open' prices

down = 0
up = 0

l = len(dataset)

while True:
    
    up = down + 400
    if up > l:
        break
    data = dataset[down:up]
    
    print("**down = ", down, "  **up = ", up, '  ** len(dataset) = ', l)
    
    time = data['Date'][-100:]
    date = pd.concat((date, pd.DataFrame(time)), axis = 0)
    
    if (filter_cols):
        # Remove any columns from data that we don't need by getting the difference between cols and filter list
        rm_cols = set(data.columns) - set(filter_cols)
        for col in rm_cols:
            del data[col]
    
    # Convert y-predict column name to numerical index
    y_col = list(data.columns).index(y_column)
    
    x_train = data[:-100]
    x_test = data[-100:]
    y_test = data['Close'][-100:]
    
    x_col = x_train.columns
    
    sc_X = MinMaxScaler(feature_range=(0, 1))
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)
    
    sc_Y = MinMaxScaler(feature_range=(0, 1))
    sc_Y.fit(np.array(y_test).reshape(-1,1))
    
    x_train = pd.DataFrame(data=x_train, columns=x_col)
    x_test = pd.DataFrame(data=x_test, columns=x_col)
    y_test_scaled = x_test['Close'][-100:]
    
    num_rows = len(x_train)
    x_data = []
    y_data = []
    i = 0
    while ((i + x_window_size + y_window_size) <= num_rows):
        x_window_data = x_train[i:(i + x_window_size)]
        y_window_data = x_train[(i + x_window_size):(i + x_window_size + y_window_size)]
        
        y_average = np.average(y_window_data.values[:, y_col])
        x_data.append(x_window_data.values)
        y_data.append(y_average)
        i += 1
        
    x_np_arr = np.array(x_data) # 3D input tensor with shape = (286, 14, 4)
    y_np_arr = np.array(y_data)
    
    
    #-------------------------------------------------------------------------------------
    
    
    model = Sequential()
    model.add(LSTM(input_dim = x_np_arr.shape[2], output_dim=10, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(x_np_arr.shape[1], return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim = 1))
    model.add(Activation("tanh"))
    model.compile(loss = "mse", optimizer = "Nadam")
    counter = 1
    
    history = model.fit(x_np_arr, y_np_arr, validation_split = 0.2, epochs = nbepochs, batch_size = batch_size_)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss']) #RAISE ERROR
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
    
    dataset_total = pd.concat((x_train, x_test), axis = 0)
    test = dataset_total[-100-x_window_size+1:]
    
    num_rows = len(test)
    x_data = []
    i = 0
    while ((i + x_window_size) <= num_rows):
        test_window = test[i:(i + x_window_size)]
        x_data.append(test_window.values)
        i += 1
        
    x_test_arr = np.array(x_data) # 3D tensor with shape = (100, 14, 4)
    
    predicted_price = model.predict(x_test_arr)
    predicted_price = sc_Y.inverse_transform(predicted_price)
    #------------------------------------------------------------------------------
    
    Result = np.concatenate((Result, predicted_price.reshape(-1)), axis = 0)
    Test = np.concatenate((Test, y_test), axis = 0)
    down = down + 100


Predited_Stock_Price = pd.DataFrame(data = Result, columns = ['Predicted_Price'])
Real_Stock_Price = pd.DataFrame(data = Test, columns = ['Real_Price'])
Predited_Stock_Price = pd.concat((Predited_Stock_Price, date.reset_index(drop = True)), axis = 1)
Real_Stock_Price = pd.concat((Real_Stock_Price, date.reset_index(drop = True)), axis = 1)
Predited_Stock_Price.set_index('Date', inplace = True)
Real_Stock_Price.set_index('Date', inplace = True)

plt.figure(figsize = (10,5))
plt.plot(Predited_Stock_Price['Predicted_Price'])
plt.plot(Real_Stock_Price['Real_Price'])
plt.xlabel('Date')
plt.ylabel('Price $')
plt.title('Google Stock Price $')
plt.legend()
plt.show()
