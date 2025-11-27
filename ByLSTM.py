import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# ======== 1. Load dữ liệu ========
data = pd.read_csv(r"D:\HocPython\NCKH\Vingroup_4y.csv")

# Chỉ dùng cột "close" để dự đoán giá đóng cửa
df = data[["close"]].astype(float).values  # Chuyển thành numpy array

# ======== 2. Chia dữ liệu theo thời gian ========
SEQ_LEN = 60  # Số ngày dùng để dự đoán
TEST_SIZE = 0.2  # Tỷ lệ dữ liệu kiểm tra

train_size = int(len(df) * (1 - TEST_SIZE))
train_data, test_data = df[:train_size], df[train_size:]

# ======== 3. Scale dữ liệu ========
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)  # Fit chỉ trên tập huấn luyện
test_scaled = scaler.transform(test_data)        # Transform tập kiểm tra

# ======== 4. Tạo chuỗi dữ liệu dạng time-series ========
def create_sequences(dataset, seq_len=SEQ_LEN):
    x, y = [], []
    for i in range(seq_len, len(dataset)):
        x.append(dataset[i-seq_len:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(train_scaled)
x_test, y_test = create_sequences(test_scaled)

# Reshape cho LSTM: (samples, timesteps, features)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# ======== 5. Xây dựng mô hình LSTM ========
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# ======== 6. Train mô hình ========
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ======== 7. Dự đoán ========
y_pred_scaled = model.predict(x_test)

# Đảo ngược scaler về giá thật
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))

# ======== 8. Đánh giá ========
mae = mean_absolute_error(y_test_true, y_pred)
mse = mean_squared_error(y_test_true, y_pred)
r2 = r2_score(y_test_true, y_pred)

print("===== LSTM Evaluation =====")
print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)

# ======== 9. In giá trị dự đoán ========
for i, j in zip(y_pred, y_test_true):
    print("Predicted value: {} . Actual value: {}".format(i[0], j[0]))
