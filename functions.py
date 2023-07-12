import tensorflow as tf
import numpy as np

# Functions
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true[:,3]-y_pred[:,3]))
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true[:,3]-y_pred[:,3])))
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true[:,3]-y_pred[:,3])/y_true[:,3]))
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean(tf.square(y_true[:,3]-y_pred[:,3])))
def ar(y_true, y_pred):
    mask = tf.cast(y_pred[1:,3] > y_true[:-1,3],tf.float32)
    return np.mean((y_true[1:,3]-y_true[:-1,3])*mask)
def accuracy(y_true, y_pred):
    return 1 - np.mean(np.abs(y_pred[:,3]-y_true[:,3])/y_true[:,3])
def calculate_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    acuracy = accuracy(y_true, y_pred)
    return round(mse, 3), round(mae, 3), round(mape, 3), round(rmse, 3), round(acuracy*100, 3)
def add_Ma(df, window=5):
    for i in range(window, df.shape[0]):
        sum = 0.0
        for k in range(1, window+1):
            sum += df.iloc[i-k, 4]
        df.loc[df.index[i], 'Ma'] = np.round(sum/window, 6)
    return df[window:]
def createDataset(dataset, look_back, n_features):
    dataX = np.zeros([len(dataset)-look_back, look_back, n_features])
    dataY = np.zeros([len(dataset)-look_back, n_features])
    data = dataset.drop(columns='Date').to_numpy()
    print(data.shape)
    for i in range(0, len(data)-look_back):
        dataX[i] = data[i:i+look_back]
        dataY[i] = data[i+look_back]
    mean = dataX.mean(axis=1)
    std = dataX.std(axis=1)
    dataX = (dataX - mean[:,None,:])/std[:,None,:]
    dataY = (dataY - mean)/std
    return dataX, dataY, mean, std