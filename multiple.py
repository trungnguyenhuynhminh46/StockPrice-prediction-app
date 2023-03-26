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
        datetime.date(2019, 1, 1))
    end = st.sidebar.date_input(
        "Nhập vào ngày kết thúc dự đoán",
        datetime.date(2021, 12, 31))

    # Take data from pandas_reader
    raw_data = pdr.get_data_yahoo(stock_code, start, end)
    df = raw_data.dropna()

    # Prepare data
    look_back=5
    n_features = 7
    df = df.reset_index(level=0)
    df = add_Ma(df, look_back)

    # Load models
    RNN_FFNN_model = load_model("./h5/rnn_ffnn.h5", custom_objects={"mse": mean_squared_error , "mae": mean_absolute_error, "mape": mean_absolute_percentage_error, "rmse": root_mean_squared_error, "ar": ar});
    LSTM_FFNN_model = load_model("./h5/lstm_ffnn.h5", custom_objects={"mse": mean_squared_error , "mae": mean_absolute_error, "mape": mean_absolute_percentage_error, "rmse": root_mean_squared_error, "ar": ar});
    GRU_FFNN_model = load_model("./h5/gru_ffnn.h5", custom_objects={"mse": mean_squared_error , "mae": mean_absolute_error, "mape": mean_absolute_percentage_error, "rmse": root_mean_squared_error, "ar": ar});
    BiLSTM_FFNN_model = load_model("./h5/bilstm_ffnn.h5", custom_objects={"mse": mean_squared_error , "mae": mean_absolute_error, "mape": mean_absolute_percentage_error, "rmse": root_mean_squared_error, "ar": ar});
    DC_BiLSTM_FFNN_model = load_model("./h5/dc_bi_lstm_ffnn.h5", custom_objects={"mse": mean_squared_error , "mae": mean_absolute_error, "mape": mean_absolute_percentage_error, "rmse": root_mean_squared_error, "ar": ar});

    x, y, mean, std = createDataset(df, look_back, n_features)
   
    y_true = y
    y_true_real = y_true*std + mean
    # Make predictions
    RNN_FFNN_predict = RNN_FFNN_model.predict(x)
    RNN_FFNN_predict = RNN_FFNN_predict*std + mean
    
    LSTM_FFNN_predict = LSTM_FFNN_model.predict(x)
    LSTM_FFNN_predict = LSTM_FFNN_predict*std + mean
    
    GRU_FFNN_predict = GRU_FFNN_model.predict(x)
    GRU_FFNN_predict = GRU_FFNN_predict*std + mean
    
    BiLSTM_FFNN_predict = BiLSTM_FFNN_model.predict(x)
    BiLSTM_FFNN_predict = BiLSTM_FFNN_predict*std + mean
    
    DC_BiLSTM_FFNN_predict = DC_BiLSTM_FFNN_model.predict(x)
    DC_BiLSTM_FFNN_predict = DC_BiLSTM_FFNN_predict*std + mean

    st.subheader("Kết quả dự giá giá chứng khoáng của mã {stock_code}".format(stock_code=stock_code))
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(12,6), dpi=110)
    plt.xlabel('Observations')
    plt.ylabel(stock_code,rotation=90)
    plt.plot(y_true_real[:,3], label = 'Actual Closing Prices', linewidth = 1.2, color = 'k')

    plt.plot(RNN_FFNN_predict[:,3], label = 'Predicted Closing Price Using RNN As Generator', linewidth = 0.9, color = 'b')
    plt.plot(LSTM_FFNN_predict[:,3], label = 'Predicted Closing Price Using LSTM As Generator ', linewidth = 0.9, color = 'g')
    plt.plot(GRU_FFNN_predict[:,3], label = 'Predicted Closing Price Using GRU As Generator', linewidth = 0.9, color = 'r')
    plt.plot(BiLSTM_FFNN_predict[:, 3], label = 'Predicted Closing Price BiLSTM As Generator', linewidth = 0.9, color = 'c')
    plt.plot(DC_BiLSTM_FFNN_predict[:, 3], label = 'Predicted Closing Price Using DC-BiLSTM As Generator', linewidth = 0.9, color = 'y')

    ## Plot legend
    legend = plt.legend(fontsize = 12,frameon = True)
    legend.get_frame().set_edgecolor('b')
    legend.get_frame().set_linewidth(0.4)

    st.pyplot(fig)

    # Get accuracies
    RNN_FFNN_mse, RNN_FFNN_mae, RNN_FFNN_mape, RNN_FFNN_rmse = calculate_performance(y_true_real,RNN_FFNN_predict)
    LSTM_FFNN_mse, LSTM_FFNN_mae, LSTM_FFNN_mape, LSTM_FFNN_rmse = calculate_performance(y_true_real,LSTM_FFNN_predict)
    GRU_FFNN_mse, GRU_FFNN_mae, GRU_FFNN_mape, GRU_FFNN_rmse = calculate_performance(y_true_real,GRU_FFNN_predict)
    BiLSTM_FFNN_mse, BiLSTM_FFNN_mae, BiLSTM_FFNN_mape, BiLSTM_FFNN_rmse = calculate_performance(y_true_real,BiLSTM_FFNN_predict)
    DC_BiLSTM_FFNN_mse, DC_BiLSTM_FFNN_mae, DC_BiLSTM_FFNN_mape, DC_BiLSTM_FFNN_rmse = calculate_performance(y_true_real,DC_BiLSTM_FFNN_predict)

    st.subheader("So sánh độ chính xác của các mô hình như dự đoán giá đóng của mã {stock_code} từ {start_date} đến {end_date}".format(stock_code=stock_code,start_date = start, end_date = end))
    d = {'RMSE': [RNN_FFNN_rmse, LSTM_FFNN_rmse, GRU_FFNN_rmse, BiLSTM_FFNN_rmse, DC_BiLSTM_FFNN_rmse], 'MSE': [RNN_FFNN_mse, LSTM_FFNN_mse, GRU_FFNN_mse, BiLSTM_FFNN_mse, DC_BiLSTM_FFNN_mse], 'MAE': [RNN_FFNN_mae, LSTM_FFNN_mae, GRU_FFNN_mae, BiLSTM_FFNN_mae, DC_BiLSTM_FFNN_mae], "MAPE": [RNN_FFNN_mape, LSTM_FFNN_mape, GRU_FFNN_mape, BiLSTM_FFNN_mape, DC_BiLSTM_FFNN_mape]}
    df = pd.DataFrame(data=d, index=["RNN và FFNN", "LSTM và FFNN", "GRU và FFNN", "BiLSTM và FFNN", "DC-BiLSTM và FFNN"])

    st.dataframe(df)

    

    return False