import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override()
from pandas_datareader import data as pdr
from keras.models import load_model
# Functions
from functions import createDataset, add_Ma, calculate_performance, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,root_mean_squared_error, ar

def single():
    st.write(
    """
    # 📊 Dự đoán xu hướng giá chứng khoán
    """
    )

    # Inputs
    model_name = st.sidebar.selectbox(
            "Chọn mô hình mà bạn muốn dùng để dự đoán",
            ("RNN và FFNN", "LSTM và FFNN", "GRU và FFNN", "BiLSTM và FFNN", "DC BI LSTM và FFNN")
        )
    stock_code = st.sidebar.text_input("Nhập vào mã chứng khoán muốn dự đoán", "AAPL")
    start = st.sidebar.date_input(
        "Nhập vào ngày bắt đầu dự đoán",
        datetime.date(2019, 1, 1))
    end = st.sidebar.date_input(
        "Nhập vào ngày kết thúc dự đoán",
        datetime.date(2021, 12, 31))

    # Take data from pandas_reader
    raw_data = pdr.get_data_yahoo(stock_code, start, end)
    df = raw_data.dropna();

    # Describe the data
    st.subheader("Thông tin của mã chứng khoán {stock_code} từ {start_date} đến {end_date}".format(stock_code = stock_code, start_date = start, end_date = end))
    st.write(df.describe())

    # Data visualization

    st.subheader("Xu hướng chuyển dịch giá đóng cùng với các đường MA100 và MA200 của mã {stock_code} từ {start_date} đến {end_date}".format(stock_code=stock_code,start_date = start, end_date = end))
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'r', label="MA100")
    plt.plot(ma200, 'g', label="MA200")
    plt.plot(df.Close, 'b', label="Actual Closing Prices")
    legend = plt.legend(loc="best", fontsize = 12,frameon = True)
    legend.get_frame().set_edgecolor('b')
    legend.get_frame().set_linewidth(0.4)
    st.pyplot(fig)

    # Prepare data
    look_back=5
    n_features = 7
    df = df.reset_index(level=0)
    df = add_Ma(df, look_back)


    #Load model
    models_dir = "./h5/"
    model_names_map = {
        "RNN và FFNN": "rnn_ffnn.h5",
        "LSTM và FFNN": "lstm_ffnn.h5",
        "GRU và FFNN": "gru_ffnn.h5", 
        "BiLSTM và FFNN": "bilstm_ffnn.h5",
        "DC BI LSTM và FFNN": "dc_bi_lstm_ffnn.h5"
    }
    model = load_model(models_dir + model_names_map[model_name], custom_objects={"mse": mean_squared_error , "mae": mean_absolute_error, "mape": mean_absolute_percentage_error, "rmse": root_mean_squared_error, "ar": ar})

    x, y, mean, std = createDataset(df, look_back, n_features)
    y_pred = model.predict(x)
    y_pred_real = y_pred*std + mean
    y_true = y
    y_true_real = y_true*std + mean

    st.subheader("Kết quả dự giá giá chứng khoáng của mã {stock_code} so với thực tế sử dụng model GAN kết hợp giữa {model_name} là".format(stock_code=stock_code, model_name = model_name))
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(12,6), dpi=110)
    plt.xlabel('Observations')
    plt.ylabel(stock_code,rotation=90)
    plt.plot(y_true_real[:,3], label = 'Actual Closing Prices', linewidth = 1.2, color = 'c')
    plt.plot(y_pred_real[:,3], label = 'Predicted Closing Price', linewidth = 0.9, color = 'k')
    legend = plt.legend(fontsize = 12,frameon = True)
    legend.get_frame().set_edgecolor('b')
    legend.get_frame().set_linewidth(0.4)

    st.pyplot(fig)

    # Get accuracies
    mse, mae, mape, rmse = calculate_performance(y_true_real, y_pred_real)

    st.subheader("Độ chính xác mô hình dự đoán của mã {stock_code} từ {start_date} đến {end_date}".format(stock_code=stock_code,start_date = start, end_date = end))
    d = {'RMSE': [rmse], 'MSE': [mse], 'MAE': [mae], "MAPE": [mape]}
    df = pd.DataFrame(data=d, index=[model_name])

    st.dataframe(df)