from vnstock import Vnstock

def get_vic_data():
    # Khởi tạo đối tượng cho mã VIC
    stock = Vnstock().stock(symbol="VIC", source="VCI")

    # Lấy dữ liệu
    data = stock.quote.history(
        start="2015-01-01",
        end="2025-12-31",
        interval="1D"
    )

    # Hiển thị vài dòng đầu và cuối
    print("5 dòng đầu tiên:")
    print(data.head())
    print("\n5 dòng cuối cùng:")
    print(data.tail())

    # Lưu ra CSV
    data.to_csv("Vingroup.csv", index=False, encoding="utf-8-sig",sep=";")
    print("\n✅ Đã lưu dữ liệu vào file 'Vingroup.csv'")

if __name__ == "__main__":
    get_vic_data()
