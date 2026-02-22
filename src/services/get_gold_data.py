import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta

# Hàm chuyển đổi thời gian với nhiều định dạng
def convert_to_datetime(time_str):
    formats = [
        '%Y-%m-%d %H:%M:%S',  # Định dạng 2016-01-05 08:21:01
        '%d/%m/%Y %H:%M:%S',  # Định dạng 05/01/2016 08:21:01
        '%Y/%m/%d %H:%M:%S'   # Định dạng 2016/01/05 08:21:01
    ]
    for fmt in formats:
        try:
            return pd.to_datetime(time_str, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"Không thể chuyển đổi thời gian '{time_str}' với bất kỳ định dạng nào đã thử.")

# Đặt ngày bắt đầu và ngày kết thúc
start_date = datetime(2016, 1, 1)
end_date = datetime(2016, 1, 31)

# Tạo hoặc tải DataFrame
csv_file = 'gia_vang_pnj_sjc.csv'
if pd.io.common.file_exists(csv_file):
    df = pd.read_csv(csv_file, encoding='utf-8')
    # Chuyển đổi cột 'Thời gian cập nhật' sang datetime
    try:
        df['Thời gian cập nhật'] = df['Thời gian cập nhật'].apply(convert_to_datetime)
        # Tìm ngày cuối cùng đã xử lý
        last_processed_date = df['Thời gian cập nhật'].max().replace(hour=0, minute=0, second=0, microsecond=0)
        current_date = last_processed_date + timedelta(days=1)
    except (ValueError, TypeError) as e:
        print(f"Lỗi khi xử lý thời gian trong DataFrame: {e}. Bắt đầu từ ngày {start_date.date()}.")
        current_date = start_date
else:
    # Nếu không có file CSV, khởi tạo DataFrame mới
    columns = ['Loại vàng', 'Giá mua', 'Giá bán', 'Thời gian cập nhật']
    df = pd.DataFrame(columns=columns)
    current_date = start_date

# Khởi tạo trình duyệt Chrome
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Chạy ở chế độ không giao diện
options.add_argument('--no-sandbox')  # Tăng tính ổn định
options.add_argument('--disable-dev-shm-usage')  # Giảm sử dụng bộ nhớ
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

try:
    while current_date <= end_date:
        # Tạo URL theo định dạng
        day = current_date.strftime('%d')
        month = current_date.strftime('%m')
        year = current_date.strftime('%Y')
        print(f"Đang xử lý ngày {day}-{month}-{year}")

        url = f"https://www.giavang.pnj.com.vn/history?gold_history_day={day}&gold_history_month={month}&gold_history_year={year}"

        # Tải trang web
        try:
            driver.get(url)
            time.sleep(0.5)  # Đợi JavaScript tải
            html = driver.page_source
        except Exception as e:
            print(f"Lỗi khi tải trang {url}: {e}")
            current_date += timedelta(days=1)
            continue

        # Phân tích HTML
        try:
            soup = BeautifulSoup(html, 'html.parser')
            tables = soup.find_all('table', class_='w-full border-collapse border border-gray-400 font-sans')
            
            if not tables:
                print(f"Không tìm thấy bảng dữ liệu cho ngày {day}-{month}-{year}")
                current_date += timedelta(days=1)
                continue

            soup_table = BeautifulSoup(str(tables[0]), 'html.parser')
            cells = soup_table.find_all('td', class_='text-center align-middle border border-gray-400 text-[14px] h-[20px] leading-[40px]')
            data = [cell.text.strip() for cell in cells]
        except Exception as e:
            print(f"Lỗi khi phân tích HTML cho ngày {day}-{month}-{year}: {e}")
            current_date += timedelta(days=1)
            continue

        # Xử lý dữ liệu
        rows = []
        current_type = None
        i = 0

        while i < len(data):
            if data[i] in ['PNJ', 'SJC']:
                current_type = data[i]
                i += 1
            else:
                if i + 2 < len(data):
                    try:
                        bid = float(data[i])  # Giá mua
                        ask = float(data[i + 1])  # Giá bán
                        timestamp = data[i + 2]  # Thời gian
                        rows.append({
                            'Loại vàng': current_type,
                            'Giá mua': bid,
                            'Giá bán': ask,
                            'Thời gian cập nhật': timestamp
                        })
                        i += 3
                    except ValueError:
                        print(f"Lỗi tại vị trí {i} ngày {day}-{month}-{year}: Không thể chuyển đổi dữ liệu.")
                        i += 1
                else:
                    print(f"Dữ liệu không đủ tại vị trí {i} ngày {day}-{month}-{year}")
                    i += 1

        # Thêm dữ liệu vào DataFrame
        if rows:
            new_df = pd.DataFrame(rows)
            # Chuyển đổi thời gian cho dữ liệu mới
            try:
                new_df['Thời gian cập nhật'] = new_df['Thời gian cập nhật'].apply(convert_to_datetime)
                df = pd.concat([df, new_df], ignore_index=True)
            except ValueError as e:
                print(f"Lỗi khi chuyển đổi thời gian cho dữ liệu mới ngày {day}-{month}-{year}: {e}")
                current_date += timedelta(days=1)
                continue

        # Lưu DataFrame vào CSV sau mỗi ngày
        try:
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"Đã lưu dữ liệu vào {csv_file} cho ngày {day}-{month}-{year}")
        except Exception as e:
            print(f"Lỗi khi lưu CSV: {e}")

        # Chuyển sang ngày tiếp theo
        current_date += timedelta(days=1)

except KeyboardInterrupt:
    print("Quá trình bị gián đoạn. Đã lưu dữ liệu vào CSV.")
except Exception as e:
    print(f"Lỗi không mong muốn: {e}")
finally:
    # Đóng trình duyệt
    driver.quit()


# Tính giá trung bình của Giá bán theo Loại vàng
avg_price = df.groupby('Loại vàng')['Giá bán'].mean().reset_index()
avg_price = avg_price.rename(columns={'Giá bán': 'Giá trung bình bán'})

# Gộp giá trung bình vào DataFrame gốc
df = df.merge(avg_price, on='Loại vàng')

df.to_csv(csv_file, index=False, encoding='utf-8')
# In DataFrame cuối cùng
print("Dữ liệu cuối cùng:")
print(df)