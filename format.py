import pandas as pd

# Đọc file CSV
df = pd.read_csv("Vingroup.csv", sep=";")

#  Chuyển cột 'time' sang datetime
df['time'] = pd.to_datetime(df['time'], errors='coerce')

#  Đảm bảo các cột số là numeric
cols = ['open', 'high', 'low', 'close']
for c in cols:
    # Nếu dữ liệu có dấu phẩy, thay bằng dấu chấm rồi chuyển sang float
    df[c] = df[c].astype(str).str.replace(',', '.').astype(float)

#  Lưu lại file mới (chuẩn cho mô hình học)
df.to_csv("Vingroup_format.csv", index=False)

print(df.dtypes)
