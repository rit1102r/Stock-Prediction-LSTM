import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import math
model = load_model('D:\stock predictor\Stock Price Predict.keras')

st.header('Stock Market Predictor')

from sklearn.preprocessing import MinMaxScaler
stock =st.text_input('Enter Stock Symbol', 'JUBLFOOD.NS')
start = '2010-01-01'
end = datetime.now()

df = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(df)
data= df.filter(['Close'])
dataset= data.values
training_data_len = math.ceil(len(dataset)*0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data [0:training_data_len, :]
test_data= scaled_data[training_data_len-100:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range( 100, len(test_data)):
  x_test.append(test_data[i-100:i, 0])
  
x_test= np.array(x_test)
x_test= np.reshape(x_test, (x_test.shape+(1,)))
predicted_price = yf.download(stock, start, end)
predicted_price = yf.download(stock, start, end)
new_df = predicted_price.filter(['Close'])
last_100_days = new_df[-100:].values
last_100_days_scaled= scaler.transform(last_100_days)

X_test=[]
X_test.append(last_100_days_scaled)
X_test= np.array(X_test)
X_test = np.reshape (X_test, (X_test.shape[0], X_test.shape[1],1))
pred_price=model.predict(X_test)
pred_price= scaler.inverse_transform(pred_price)[0]
st.write('Predicted Price:', pred_price)

st.subheader('Price vs MA40')
ma_40_days = df.Close.rolling(40).mean()
fig1 = plt.figure(figsize=(16,8))
plt.plot(ma_40_days, 'r')
plt.plot(df.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA40 vs MA100')
ma_100_days = df.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(16,8))
plt.plot(ma_40_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(df.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = df.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(16,8))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(df.Close, 'g')
plt.show()
st.pyplot(fig3)


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(model.predict(x_test))

st.subheader('Original Price vs Predicted Price')
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
fig4=plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.plot (train['Close'])
plt.plot (valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val','Predictions'], loc='lower right')
plt.show()
st.pyplot(fig4)