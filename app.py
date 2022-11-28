import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

# Functions
def createDataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)

st.write(
    """
# 📊 Dự đoán xu hướng giá chứng khoán
Chọn ra mã chứng khoán, khoảng thời gian và loại mô hình để có thễ nhận được kết quả dự đoán mong muốn
"""
)

model_name = st.selectbox(
        "Chọn mô hình mà bạn muốn dùng để dự đoán",
        ("FFNN", "RNN", "LSTM", "GRU")
    )
stock_code = st.text_input("Nhập vào mã chứng khoán muốn dự đoán", "AAPL")
start = st.date_input(
    "Nhập vào ngày bắt đầu dự đoán",
    datetime.date(2005, 1, 1))
end = st.date_input(
    "Nhập vào ngày kết thúc dự đoán",
    datetime.date(2019, 12, 31))

# Dữ liệu lấy từ DataReader
raw_data = data.DataReader(stock_code, 'yahoo', start, end)
df = raw_data.dropna();

#Mô tả dữ liệu
st.subheader("Thông tin của mã chứng khoán {stock_code} từ {start_date} đến {end_date}".format(stock_code = stock_code, start_date = start, end_date = end))
st.write(df.describe())

#Trực hóa dữ liệu

st.subheader("Xu hướng chuyển dịch giá đóng cùng với các đường MA100 và MA200 của mã {stock_code} từ {start_date} đến {end_date}".format(stock_code=stock_code,start_date = start, end_date = end))
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r', label="MA100", color="r")
plt.plot(ma200, 'g', label="MA200", color="b")
plt.plot(df.Close, 'b', label="Actual Closing Prices", color="g")
legend = plt.legend(loc="best", fontsize = 12,frameon = True)
legend.get_frame().set_edgecolor('b')
legend.get_frame().set_linewidth(0.4)
st.pyplot(fig)

# Xử lý dữ liệu
dataset = df[['Close']].values.astype('float32');

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

look_back = 10

X, Y = createDataset(dataset, look_back)

#Load model
models_dir = "./h5/"
model_names_map = {
    "FFNN": "ffnn_model.h5",
    "RNN": "rnn_model.h5",
    "LSTM": "lstm_model.h5",
    "GRU" : "gru_model.h5"
}
model = load_model(models_dir + model_names_map[model_name])

predict = model.predict(X)
predict = scaler.inverse_transform(predict)
Y = scaler.inverse_transform([Y])

# Plotting
predictPlot = np.empty_like(dataset)
predictPlot[:, :] = np.nan
predictPlot[look_back:len(predict)+look_back, :] = predict

st.subheader("Kết quả dự giá giá chứng khoáng của mã {stock_code} trên tập train và tập test so với kết quả thực tế sử dụng model {model_name} là".format(stock_code=stock_code, model_name = model_name))
plt.style.use('ggplot')
fig = plt.figure(figsize=(12,6), dpi=110)
plt.xlabel('Observations')
plt.ylabel(stock_code,rotation=90)
plt.plot(scaler.inverse_transform(dataset), label = 'Actual Closing Prices', linewidth = 1.2, color = 'c')
plt.plot(predictPlot, label = 'Predicted Closing Price', linewidth = 0.9, color = 'k')
legend = plt.legend(fontsize = 12,frameon = True)
legend.get_frame().set_edgecolor('b')
legend.get_frame().set_linewidth(0.4)

st.pyplot(fig)

# Get accuracies
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def calculate_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return round(mse, 3), round(mae, 3), round(mape, 3), round(rmse, 3)

mse, mae, mape, rmse = calculate_performance(Y[0],predict[:, 0])

st.subheader("Độ chính xác mô hình dự đoán của mã {stock_code} từ {start_date} đến {end_date}".format(stock_code=stock_code,start_date = start, end_date = end))
d = {'RMSE': [rmse], 'MSE': [mse], 'MAE': [mae], "MAPE": [mape]}
df = pd.DataFrame(data=d, index=[model_name])

st.dataframe(df)  # Same as st.write(df)