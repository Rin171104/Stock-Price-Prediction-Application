from vnstock import Vnstock

def get_vic_data():
    # Khởi tạo đối tượng cho mã VIC
    stock = Vnstock().stock(symbol="VIC", source="VCI")

    # Lấy dữ liệu 2 năm gần nhất (2023-01-01 → 2025-01-01)
    data = stock.quote.history(
        start="2021-01-01",
        end="2025-09-19",
        interval="1D"   # phải viết hoa
    )

    # Hiển thị vài dòng đầu và cuối
    print("5 dòng đầu tiên:")
    print(data.head())
    print("\n5 dòng cuối cùng:")
    print(data.tail())

    # Lưu ra CSV
    data.to_csv("Vingroup_4y.csv", index=False, encoding="utf-8-sig",sep=";")
    print("\n✅ Đã lưu dữ liệu vào file 'Vingroup_2y.csv'")

if __name__ == "__main__":
    get_vic_data()
