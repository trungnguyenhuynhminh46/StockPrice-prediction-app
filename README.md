# 1. Tập dữ liệu

Tất cả dữ liệu trong phạm vi bài tiểu luận này đều được lấy từ trang yahoo_finance thông qua thư viện pandas_reader. Thư viện ngày hỗ trợ người sử dụng truy cập các mã chứng khoán khác nhau với khoản thời gian mong muốn. Dữ liệu chúng tôi dùng phục vụ cho bài tiểu luận là của các mã chứng khoáng AAPL, AMZN, IBM, TSLA trong khoản thời gian từ 1/1/2000 đến 31/12/2020 (20 năm kể từ năm 2000).

- High: Giá trị cao nhất trong ngày
- Low: Giá trị thấp nhất trong ngày
- Open: Giá mở cửa của ngày hôm đó
- Cloes: Giá đóng cửa của ngày hôm đó
- Volume: Khối lượng giao dịch ngày hôm đó
- Adj close: Giá trị hiệu chỉnh đóng cửa ngày hôm đó

# 2. Thư viện

Sau đây là giới hiệu sơ lược về các thư viện được sử dụng trong đồ án tốt nghiệp.

- numpy: Thư viện dùng để xử lý mảng và ma trận số học, cũng như các phép tính toán liên quan đến đại số tuyến tính.
- pandas: Thư viện mã nguồn mở được sử dụng để làm việc với dữ liệu, cung cấp các công cụ để đọc và ghi các tệp dữ liệu trong nhiều định dạng khác nhau, và cho phép xử lý và phân tích dữ liệu.
- os: Thư viện Python cung cấp các phương thức để tương tác với hệ điều hành, cho phép thực hiện các thao tác như tạo, xóa, di chuyển hoặc đổi tên tệp và thư mục.
- tensorflow: Thư viện mã nguồn mở được sử dụng để xây dựng và huấn luyện các mô hình trí tuệ nhân tạo, cung cấp các công cụ để xử lý dữ liệu, xây dựng các mô hình neural network và các thuật toán học máy.
- Keras: Thư viện mã nguồn mở được xây dựng trên nền tảng TensorFlow, cung cấp một cách đơn giản và dễ dàng để xây dựng và huấn luyện các mô hình neural network.
- TimeseriesGenerator: Là một lớp được định nghĩa trong thư viện Keras, dùng để tạo ra dữ liệu tuần tự dùng để huấn luyện các mô hình neural network nhằm giải quyết các bài toán như dự báo theo chuỗi thời gian thực.
- yfinance : Thư viện cho phép lấy dữ liệu từ trang web Yahoo Finance.

# 3. Chuẩn bị dữ liệu, lớp TimeseriesGenerator

- Lớp này mục đích là để tạo ra loại dữ liệu chuỗi thời gian phù hợp với mô hình huấn luyện. Ý tưởng của nó là từ dữ liệu tuần tự đầu vào sẽ tạo ra các batch dữ liệu các nhau phù hợp với mô hình huấn luyện. Lớp này thường tạo ra các dữ liệu phục vụ cho bài toán dự đoán giá cổ phiếu hoặc dự báo thời tiết, nó cũng cung cấp nhiều đối số đầu vào nhằm quyết định cấu trúc của dữ liệu được tạo ra như độ lớn của sổ (length), biên độ trượt (stride), tần suất dữ liệu được lấy (sampling_rate), số lượng sample có trong một batch dữ liệu (batch_size).
- Lớp Standarized_TimeseriesGenerator được kế thừa từ lớp TimeseriesGenerator, lớp này đơn giản là chuẩn hóa lại dữ liệu được trả về từ lớp Standarized_TimeseriesGenerator.

```python
class Standarized_TimeseriesGenerator(tf.keras.preprocessing.sequence.TimeseriesGenerator):
  def __getitem__(self, index):
    samples, targets  = super(Standarized_TimeseriesGenerator, self).__getitem__(index)
    mean = samples.mean(axis=1)
    std = samples.std(axis=1)
    samples = (samples - mean[:,None,:])/std[:,None,:]
    targets = (targets - mean)/std
    return samples, targets
```

## 4. Xây dựng Generator và Discriminator

- Trong phạm vi khóa luận tốt nghiệp, chúng tôi chọn ra 5 mô hình dùng làm Generator đó là RNN, LSTM, GRU, BiLSTM, DCBiLSTM.
![1](https://github.com/trungnguyenhuynhminh46/StockPrice-prediction-app/assets/58035150/d69c55d0-aa69-4be8-a06b-420a051f584c)
![2](https://github.com/trungnguyenhuynhminh46/StockPrice-prediction-app/assets/58035150/a0a66a04-5e91-48c8-aff3-37a4e0aa73f9)
![3](https://github.com/trungnguyenhuynhminh46/StockPrice-prediction-app/assets/58035150/081b9252-42e2-449b-bc39-77b361ab04cb)
![4](https://github.com/trungnguyenhuynhminh46/StockPrice-prediction-app/assets/58035150/02eaf7ee-3f37-47ea-a180-61a759cc6bba)
![5](https://github.com/trungnguyenhuynhminh46/StockPrice-prediction-app/assets/58035150/1fc280c1-3240-4929-acc0-25b7978c49c9)
- Vì đặc tính đầu vào khá đơn giản là dữ liệu tuần tự theo thời gian nên chúng tôi quyết định sử dụng mô hình FFNN làm discriminator. Cấu trúc mô hình như sau:
![6](https://github.com/trungnguyenhuynhminh46/StockPrice-prediction-app/assets/58035150/5b33b0a6-8bdf-4e1f-a149-e1c46070e204)

## 5. Quá trình training

- Sau bước sử dụng lớp Standarized_TimeseriesGenerator để tạo ra các bộ dữ liệu được chuẩn hóa để training mô hình, chúng ta sẽ có một tập các batch dữ liệu khác nhau dùng để train dữ liệu, với mỗi một batch thì hàm sau sẽ được chạy:

```python
def train_step_def(sequences, sequences_end):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_prediction = generator(sequences, training=True)

        sequences_true = tf.concat((sequences, sequences_end[:, None, :]), axis=1)
        sequences_fake = tf.concat((sequences, generated_prediction[:, None, :]), axis=1)

        real_output = discriminator(sequences_true, training=True)
        fake_output = discriminator(sequences_fake, training=True)

        gen_loss, gen_mse_loss = generator_loss(generated_prediction,
                                                sequences_end,
                                                fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return tf.reduce_mean(gen_loss), tf.reduce_mean(disc_loss), tf.reduce_mean(gen_mse_loss)
```

- Đầu tiên là giải thích về vai trò của GradientTape trong hàm trên: Trong quá trình huấn luyện GAN, GradientTape được sử dụng để tính gradient của hàm mất mát của cả bộ sinh (generator) và bộ phân biệt (discriminator). Trong quá trình huấn luyện, bộ sinh cố gắng sinh ra các mẫu giống như thật, trong khi bộ phân biệt cố gắng phân biệt giữa các mẫu thật và giả. GradientTape giúp tính toán gradient của hàm mất mát theo các tham số của bộ sinh và bộ phân biệt, sau đó sử dụng gradient này để cập nhật các tham số của mô hình.
- Trong hàm train_step_def, đầu tiên chúng ta sẽ tạo ra dữ liệu được tạo bời generator gọi là generated_prediction, gôp vào sequences là đầu vào để dự đoán đê tạo ra dữ liệu giả. Từ hai tập dữ liệu giả và thật, thông qua hai hàm generator_loss và discriminator_loss thì sẽ tính được giá trị loss_function của hai hàm này. Từ giá trị loss trả về, thông qua GradientTape, các trọng số trong hai mô hình được cập nhật, generator sẹ tạo ra những dữ liệu giả giống thật hơn và discriminator sẽ có khả năng phân biệt dữ liệu giả và thật tốt hơn. Quá trị train trên một epoch kết thúc khi tất cả các batch trong tập dữ liệu được sử dụng hết.

```python
def test_step_def(sequences, sequences_end):
    generated_prediction = generator(sequences, training=False)

    sequences_true = tf.concat((sequences, sequences_end[:,None,:]), axis=1)
    sequences_fake = tf.concat((sequences, generated_prediction[:,None,:]), axis=1)

    real_output = discriminator(sequences_true, training=False)
    fake_output = discriminator(sequences_fake, training=False)

    gen_loss, gen_mse_loss = generator_loss(generated_prediction, sequences_end, fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
    return tf.reduce_mean(gen_loss), tf.reduce_mean(disc_loss), tf.reduce_mean(gen_mse_loss)
```

- Đối với hàm test_step_def, trong mỗi epoch, sau khi quá trình train diễn ra thành công thì chúng ta sẽ thư được generator có khả năng tạo ra dữ liệu giả tốt nhất trong lần epoch đó. Dùng generator này để dự đoán giá trị của tập dữ liệu test. Giá trị của hàm mất mát của generator và discriminator được trả về để ghi nhận kết quả dự đoán.
- Toàn bộ quá trình train và test giá trị được mô tả ở trên được thể hiện cụ thể trong hàm sau, đầu ra của hàm này là hai mô hình generator và discriminator được huấn luyện kỹ qua một số lượng các epochs.

```python
def train(dataset, dataset_val, epochs):
    history = np.empty(shape = (8, epochs))
    history_val = np.empty(shape = (8, epochs))
    len_dataset = len(dataset)
    len_dataset_val = len(dataset_val)
    for epoch in range(epochs):
        start = time.time()

        cur_dis_loss = 0
        cur_gen_loss = 0
        cur_gen_mse_loss = 0
        for sequence_batch, sequence_end_batch in dataset:
            aux_cur_losses = train_step(tf.cast(sequence_batch, tf.float32),
                                      tf.cast(sequence_end_batch, tf.float32))
            cur_gen_loss += aux_cur_losses[0]/len_dataset
            cur_dis_loss += aux_cur_losses[1]/len_dataset
            cur_gen_mse_loss += aux_cur_losses[2]/len_dataset
        cur_gen_metrics = generator.evaluate(dataset,verbose=False)[1:]

        history[:, epoch] = cur_gen_loss, cur_dis_loss, cur_gen_mse_loss, *cur_gen_metrics

        cur_gen_metrics_val = generator.evaluate(dataset_val,verbose=False)[1: ]

        cur_gen_loss_val = 0
        cur_dis_loss_val = 0
        cur_gen_mse_loss_val = 0
        for sequence_batch, sequence_end_batch in dataset_val:
            aux_cur_losses_val = test_step(tf.cast(sequence_batch, tf.float32),
                                         tf.cast(sequence_end_batch, tf.float32))
            cur_gen_loss_val += aux_cur_losses_val[0]/len_dataset_val
            cur_dis_loss_val += aux_cur_losses_val[1]/len_dataset_val
            cur_gen_mse_loss_val += aux_cur_losses_val[2]/len_dataset_val



        history_val[:, epoch] = cur_gen_loss_val, cur_dis_loss_val, cur_gen_mse_loss_val, *cur_gen_metrics_val

        print ('Time for epoch {} is {} sec Generator Loss: {},  Discriminator_loss: {}'
               .format(epoch + 1, time.time()-start, cur_gen_loss, cur_dis_loss))

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

    return history, history_val
```

## 6. Training

Trên đây là giới thiệu chi tiết về các hàm phục vụ cho quá trình khởi tạo và huấn luyện mô hình GAN. Toàn bộ quá trình sử dụng các hàm trên để thực thi như sau:

- Chuẩn bị tham số đầu vào và tập dữ liệu:
- Chuẩn bị mô hình và các thông số mô hình

```python
# Variables
window = 5
n_sequence = window
n_features = 7
n_batch = 50
# Data Loading
stock_code = "AAPL"
start = dt.datetime(2000, 1, 1)
end = dt.datetime(2020, 12, 31)
raw_data = pdr.get_data_yahoo(stock_code, start, end,threads=False, proxy="http://127.0.0.1:7890")
df = raw_data.dropna();
df = df.reset_index(level=0)
df = add_Ma(df, window)
df
```

- Chuẩn bị mô hình và các thông số mô hình:

```python
start_time = datetime.datetime.now()
df = add_Ma(df)
data_gen_train, data_gen_test = get_gen_train_test(df, n_sequence, n_batch)

generator = make_generator_model(n_sequence, n_features)
discriminator=make_discriminator_model(n_features)

learning_rate=1e-4
generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
```

- Bắt đầu quá trình huấn luyện mô hình và lưu lại các Checkpoint, cuối cùng là lưu lại generator tốt nhất sau quá trình huấn luyện.

```python
@tf.function
def train_step(sequences, sequences_end):
return train_step_def(sequences, sequences_end)

@tf.function
def test_step(sequences, sequences_end):
return test_step_def(sequences, sequences_end)

checkpoint_dir = './training_checkpoints'+stock_code
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)

history, history_val = train(data_gen_train, data_gen_test, epochs)

plot_history(history, history_val)
plot_frame(*data_gen_test[0], generator)

print("[MSE Baseline] train:",mean_squared_error(data_gen_train)," test:", mean_squared_error(data_gen_test))
now = datetime.datetime.now()
delta = now - start_time
print("Delta time with epochs = {0}:".format(epochs), delta)
generator.save("bilstm_ffnn_epochs_{0}.h5".format(epochs))
```

## 7. Testing

Hàm tạo ra các dataset thích hợp dùng làm đầu vào cho generator

```python
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
```

Hàm vẽ ra mô hình dự đoán được tạo bởi generator

```python
def plot_frame(sequence, target, mean, std, model):
    y_pred = model.predict(sequence)
    y_pred_real = y_pred*std + mean
    y_true = target
    y_true_real = y_true*std + mean

    plt.figure()
    plt.title("closing price")
    plt.plot(y_true_real[...,3], label="true")
    plt.plot(y_pred_real[...,3], label="prediction")
    plt.legend()
    plt.show()
    return y_pred_real, y_true_real
```

Kết quả thực nghiệm <br>
![download](https://github.com/trungnguyenhuynhminh46/StockPrice-prediction-app/assets/58035150/0f8f9ce5-c5f6-4d6d-a074-d0f1bb3a694f)
