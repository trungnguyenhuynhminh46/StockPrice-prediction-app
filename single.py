import streamlit as st
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
# Functions
from functions import createDataset, calculate_performance

def single():
    st.write(
    """
    # 📊 Dự đoán xu hướng giá chứng khoán
    """
    )

    # Inputs
    model_name = st.sidebar.selectbox(
            "Chọn mô hình mà bạn muốn dùng để dự đoán",
            ("FFNN", "RNN", "LSTM", "GRU")
        )
    stock_code = st.sidebar.text_input("Nhập vào mã chứng khoán muốn dự đoán", "AAPL")
    start = st.sidebar.date_input(
        "Nhập vào ngày bắt đầu dự đoán",
        datetime.date(2021, 1, 1))
    end = st.sidebar.date_input(
        "Nhập vào ngày kết thúc dự đoán",
        datetime.date(2021, 12, 31))

    # Take data from pandas_reader
    raw_data = data.DataReader(stock_code, 'yahoo', start, end)
    df = raw_data.dropna();

    # Describe the data
    st.subheader("Thông tin của mã chứng khoán {stock_code} từ {start_date} đến {end_date}".format(stock_code = stock_code, start_date = start, end_date = end))
    st.write(df.describe())

    # Data visualization

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

    # Prepare data
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

    st.subheader("Kết quả dự giá giá chứng khoáng của mã {stock_code} so với thực tế sử dụng model {model_name} là".format(stock_code=stock_code, model_name = model_name))
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
    mse, mae, mape, rmse = calculate_performance(Y[0],predict[:, 0])

    st.subheader("Độ chính xác mô hình dự đoán của mã {stock_code} từ {start_date} đến {end_date}".format(stock_code=stock_code,start_date = start, end_date = end))
    d = {'RMSE': [rmse], 'MSE': [mse], 'MAE': [mae], "MAPE": [mape]}
    df = pd.DataFrame(data=d, index=[model_name])

    st.dataframe(df)