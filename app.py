import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Funcitons
def createDataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)
# Dữ liệu người dùng nhập từ giao diện
# start = "2005-01-01"
# end = "2019-12-31"

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
user_input = st.text_input("Nhập vào mã chứng khoán muốn dự đoán", "AAPL")
start = st.date_input(
    "Nhập vào ngày bắt đầu dự đoán",
    datetime.date(2005, 1, 1))
end = st.date_input(
    "Nhập vào ngày kết thúc dự đoán",
    datetime.date(2019, 12, 31))

# Dữ liệu lấy từ DataReader
raw_data = data.DataReader(user_input, 'yahoo', start, end)
df = raw_data.dropna();

#Mô tả dữ liệu
st.subheader("Thông tin của mã chứng khoán {stock_code} từ {start_date} đến {end_date}".format(stock_code = user_input, start_date = start, end_date = end))
st.write(df.describe())

#Trực hóa dữ liệu
st.subheader("Xu hướng chuyển dịch giá đóng của mã {stock_code} từ {start_date} đến {end_date}".format(stock_code=user_input, start_date = start, end_date = end))
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Xu hướng chuyển dịch giá đóng cùng với các đường MA100 và MA200 của mã {stock_code} trong khoảng đầu năm 2005 đến cuối năm 2019".format(stock_code=user_input))
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# Xử lý dữ liệu
dataset = df[['Close']].values.astype('float32');

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Tập test và tập dự đoán được chia theo tỷ lệ 8 - 2 
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size

train_ds, test_ds = dataset[0:train_size,:], dataset[train_size:len(dataset)+1,:]

look_back = 10
trainX, trainY = createDataset(train_ds, look_back)
testX, testY = createDataset(test_ds, look_back)

#Load model
models_dir = "./h5/"
model_names_map = {
    "FFNN": "ffnn_model.h5",
    "RNN": "rnn_model.h5",
    "LSTM": "lstm_model.h5",
    "GRU" : "gru_model.h5"
}
model = load_model(models_dir + model_names_map[model_name])

# Dùng model để dự đoán
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

st.subheader("Kết quả dự giá giá chứng khoáng của mã {stock_code} trên tập train và tập test so với kết quả thực tế sử dụng model {model_name} là".format(stock_code=user_input, model_name = model_name))

plt.style.use('ggplot')
fig = plt.figure(figsize=(12,6), dpi=110)
plt.grid(color='grey', linestyle='dashed')
plt.xlabel('Observations')
plt.ylabel(user_input,rotation=90)
plt.plot(scaler.inverse_transform(dataset), label = 'Actual Closing Prices', linewidth = 1.2, color = 'c')
plt.plot(trainPredictPlot, label = 'A.I. Train Data Price Predictions_After fit', linewidth = 0.9, color = 'k')
plt.plot(testPredictPlot, label = 'A.I. Test Data Price Predictions', linewidth = 0.9, color = 'r')
legend = plt.legend(fontsize = 12,frameon = True)
legend.get_frame().set_edgecolor('b')
legend.get_frame().set_linewidth(0.4)

st.pyplot(fig)