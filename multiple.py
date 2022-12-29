import streamlit as st
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override()
from pandas_datareader import data as pdr
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
# Functions
from functions import createDataset, calculate_performance

def multiple():
    st.write(
    """
    # 📊 So sánh các mô hình dự đoán xu hướng giá chứng khoán
    """
    )
    # Inputs
    stock_code = st.sidebar.text_input("Nhập vào mã chứng khoán muốn dự đoán", "AAPL")
    start = st.sidebar.date_input(
        "Nhập vào ngày bắt đầu dự đoán",
        datetime.date(2021, 1, 1))
    end = st.sidebar.date_input(
        "Nhập vào ngày kết thúc dự đoán",
        datetime.date(2021, 12, 31))

    # Take data from pandas_reader
    raw_data = pdr.get_data_yahoo(stock_code, start, end)
    df = raw_data.dropna()

    # Prepare data
    dataset = df[['Close']].values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    look_back = 20

    X, Y = createDataset(dataset, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Load models
    FFNN_model = load_model("./h5/ffnn_model.h5");
    RNN_model = load_model("./h5/rnn_model.h5");
    LSTM_model = load_model("./h5/lstm_model.h5");
    GRU_model = load_model("./h5/gru_model.h5");

    # Make predictions
    Y = scaler.inverse_transform([Y])

    FFNN_predict = FFNN_model.predict(X)
    FFNN_predict = scaler.inverse_transform(FFNN_predict)

    RNN_predict = RNN_model.predict(X)
    RNN_predict = scaler.inverse_transform(RNN_predict)

    LSTM_predict = LSTM_model.predict(X)
    LSTM_predict = scaler.inverse_transform(LSTM_predict)

    GRU_predict = GRU_model.predict(X)
    GRU_predict = scaler.inverse_transform(GRU_predict)

    # Plotting
    FFNN_predictPlot = np.empty_like(dataset)
    FFNN_predictPlot[:, :] = np.nan
    FFNN_predictPlot[look_back:len(FFNN_predict)+look_back, :] = FFNN_predict
    
    RNN_predictPlot = np.empty_like(dataset)
    RNN_predictPlot[:, :] = np.nan
    RNN_predictPlot[look_back:len(RNN_predict)+look_back, :] = RNN_predict
    
    LSTM_predictPlot = np.empty_like(dataset)
    LSTM_predictPlot[:, :] = np.nan
    LSTM_predictPlot[look_back:len(LSTM_predict)+look_back, :] = LSTM_predict
    
    GRU_predictPlot = np.empty_like(dataset)
    GRU_predictPlot[:, :] = np.nan
    GRU_predictPlot[look_back:len(GRU_predict)+look_back, :] = GRU_predict

    st.subheader("Kết quả dự giá giá chứng khoáng của mã {stock_code}".format(stock_code=stock_code))
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(12,6), dpi=110)
    plt.xlabel('Observations')
    plt.ylabel(stock_code,rotation=90)
    plt.plot(scaler.inverse_transform(dataset), label = 'Actual Closing Prices', linewidth = 1.2, color = 'k')

    plt.plot(FFNN_predictPlot, label = 'Predicted Closing Price By FFNN', linewidth = 0.9, color = 'b')
    plt.plot(RNN_predictPlot, label = 'Predicted Closing Price By RNN', linewidth = 0.9, color = 'g')
    plt.plot(LSTM_predictPlot, label = 'Predicted Closing Price By LSTM', linewidth = 0.9, color = 'r')
    plt.plot(GRU_predictPlot, label = 'Predicted Closing Price By GRU', linewidth = 0.9, color = 'c')

    ## Plot legend
    legend = plt.legend(fontsize = 12,frameon = True)
    legend.get_frame().set_edgecolor('b')
    legend.get_frame().set_linewidth(0.4)

    st.pyplot(fig)

    # Get accuracies
    FFNN_mse, FFNN_mae, FFNN_mape, FFNN_rmse = calculate_performance(Y[0],FFNN_predict[:, 0])
    RNN_mse, RNN_mae, RNN_mape, RNN_rmse = calculate_performance(Y[0],RNN_predict[:, 0])
    LSTM_mse, LSTM_mae, LSTM_mape, LSTM_rmse = calculate_performance(Y[0],LSTM_predict[:, 0])
    GRU_mse, GRU_mae, GRU_mape, GRU_rmse = calculate_performance(Y[0],GRU_predict[:, 0])

    st.subheader("So sánh độ chính xác của các mô hình như dự đoán giá đóng của mã {stock_code} từ {start_date} đến {end_date}".format(stock_code=stock_code,start_date = start, end_date = end))
    d = {'RMSE': [FFNN_rmse, RNN_rmse, LSTM_rmse, GRU_rmse], 'MSE': [FFNN_mse, RNN_mse, LSTM_mse, GRU_mse], 'MAE': [FFNN_mae, RNN_mae, LSTM_mae, GRU_mae], "MAPE": [FFNN_mape, RNN_mape, LSTM_mape, GRU_mape]}
    df = pd.DataFrame(data=d, index=["FFNN", "RNN", "LSTM", "GRU"])

    st.dataframe(df)

    

    return False