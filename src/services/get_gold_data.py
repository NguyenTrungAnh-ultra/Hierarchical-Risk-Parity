import os
import pandas as pd
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
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
end_date = datetime(2026, 2, 22)

csv_file = '/Users/nguyentrunganhonichan/Documents/Hierarchical-Risk-Parity/datasets/gold/sjc.csv'

async def fetch_day(context, date, semaphore, df_list):
    day = date.strftime('%d')
    month = date.strftime('%m')
    year = date.strftime('%Y')
    url = f"https://www.giavang.pnj.com.vn/history?gold_history_day={day}&gold_history_month={month}&gold_history_year={year}"

    async with semaphore:
        try:
            page = await context.new_page()
            # Chặn tải hình ảnh, css, font để tăng tốc
            await page.route("**/*", lambda route: route.continue_() if route.request.resource_type in ["document", "script", "fetch", "xhr"] else route.abort())
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(500) # Tuỳ chỉnh thời gian chờ js load
            html = await page.content()
            await page.close()
        except Exception as e:
            print(f"Lỗi tải trang {day}-{month}-{year}: {e}")
            try:
                await page.close()
            except:
                pass
            return
            
    try:
        soup = BeautifulSoup(html, 'html.parser')
        tables = soup.find_all('table', class_='w-full border-collapse border border-gray-400 font-sans')
        
        if not tables:
            print(f"Không có dữ liệu ngày {day}-{month}-{year}")
            return

        soup_table = BeautifulSoup(str(tables[0]), 'html.parser')
        cells = soup_table.find_all('td', class_='text-center align-middle border border-gray-400 text-[14px] h-[20px] leading-[40px]')
        data = [cell.text.strip() for cell in cells]
    except Exception as e:
        print(f"Lỗi phân tích HTML {day}-{month}-{year}: {e}")
        return

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
                    
                    if current_type == 'SJC':
                        rows.append({
                            'Loại vàng': current_type,
                            'Giá mua': bid,
                            'Giá bán': ask,
                            'Thời gian cập nhật': timestamp
                        })
                    i += 3
                except ValueError:
                    i += 1
            else:
                i += 1

    if rows:
        new_df = pd.DataFrame(rows)
        try:
            new_df['Thời gian cập nhật'] = new_df['Thời gian cập nhật'].apply(convert_to_datetime)
            latest_idx = new_df['Thời gian cập nhật'].idxmax() # Lấy cuối phiên
            new_df = new_df.loc[[latest_idx]]
            
            df_list.append(new_df)
            print(f"Thành công: {day}-{month}-{year}")
        except ValueError as e:
            print(f"Lỗi quá trình chuyển thời gian {day}-{month}-{year}: {e}")

async def main():
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    if os.path.exists(csv_file):
        df_main = pd.read_csv(csv_file, encoding='utf-8')
        try:
            df_main['Thời gian cập nhật'] = df_main['Thời gian cập nhật'].apply(convert_to_datetime)
            last_processed = df_main['Thời gian cập nhật'].max().replace(hour=0, minute=0, second=0, microsecond=0)
            current_date = last_processed + timedelta(days=1)
        except Exception as e:
            print(f"Lỗi load csv {e}, bắt đầu lại từ {start_date}")
            df_main = pd.DataFrame(columns=['Loại vàng', 'Giá mua', 'Giá bán', 'Thời gian cập nhật'])
            current_date = start_date
    else:
        df_main = pd.DataFrame(columns=['Loại vàng', 'Giá mua', 'Giá bán', 'Thời gian cập nhật'])
        current_date = start_date

    dates = []
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
        
    if not dates:
        print("Đã cập nhật đủ dữ liệu!")
        return

    print(f"Cần cào {len(dates)} ngày...")
    
    # Số luồng (tabs) đồng thời: để 10-15 để không bị trang web chặn quá nhiều request
    semaphore = asyncio.Semaphore(15) 
    df_list = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
        context = await browser.new_context()
        
        # Chia nhỏ ra từng khoảng 30 ngày để nếu có lỗi cũng không mất hết công sức
        chunk_size = 30
        for i in range(0, len(dates), chunk_size):
            chunk = dates[i:i + chunk_size]
            print(f"\n--- Đang xử lý từ {chunk[0].strftime('%d/%m/%Y')} đến {chunk[-1].strftime('%d/%m/%Y')} ---")
            
            tasks = [fetch_day(context, d, semaphore, df_list) for d in chunk]
            await asyncio.gather(*tasks)
            
            if df_list:
                df_main = pd.concat([df_main] + df_list, ignore_index=True)
                df_main = df_main.drop_duplicates(subset=['Thời gian cập nhật'])
                
                # Tính lại giá trung bình
                if 'Giá trung bình bán' in df_main.columns:
                    df_main = df_main.drop(columns=['Giá trung bình bán'])
                avg_price = df_main.groupby('Loại vàng')['Giá bán'].mean().reset_index()
                avg_price = avg_price.rename(columns={'Giá bán': 'Giá trung bình bán'})
                df_main = df_main.merge(avg_price, on='Loại vàng')
                
                df_main = df_main.sort_values('Thời gian cập nhật')
                df_main.to_csv(csv_file, index=False, encoding='utf-8')
                df_list.clear() # Xoá cache để chuẩn bị cho chu trình lưu tiếp theo
                print(f"-> Đã lưu checkpoint đợt này. Tổng record hiện tại: {len(df_main)}\n")
                
        await browser.close()
    
    print("Hoàn thành quá trình cào dữ liệu!")

if __name__ == "__main__":
    asyncio.run(main())