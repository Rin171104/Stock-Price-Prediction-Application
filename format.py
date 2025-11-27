import pandas as pd

# Đọc file CSV
df = pd.read_csv("Vingroup_4y.csv", sep=";")

# 1️⃣ Chuyển cột 'time' sang datetime
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# 2️⃣ Đảm bảo các cột số là numeric
cols = ['open', 'high', 'low', 'close']
for c in cols:
    # Nếu dữ liệu có dấu phẩy, thay bằng dấu chấm rồi chuyển sang float
    df[c] = df[c].astype(str).str.replace(',', '.').astype(float)

# 3️⃣ Lưu lại file mới (chuẩn cho mô hình học)
df.to_csv("Vingroup_4y_numeric.csv", index=False)

print("✅ Đã chuyển các cột về dạng numeric để mô hình có thể học được.")
print(df.dtypes)
