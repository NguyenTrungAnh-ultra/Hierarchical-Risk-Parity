import os
import pandas as pd
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import random

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

# Tự động xác định đường dẫn tương đối từ vị trí file script
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
csv_file = os.path.join(project_root, 'datasets', 'gold', 'sjc.csv')

async def fetch_day(context, date, semaphore, df_list):
    day = date.strftime('%d')
    month = date.strftime('%m')
    year = date.strftime('%Y')
    url = f"https://www.giavang.pnj.com.vn/history?gold_history_day={day}&gold_history_month={month}&gold_history_year={year}"

    async with semaphore:
        try:
            # Random delay 1-3s để tránh bị block do request quá đều và nhanh
            await asyncio.sleep(random.uniform(1.0, 3.0))
            page = await context.new_page()
            # Chặn tải hình ảnh, css, font để tăng tốc
            await page.route("**/*", lambda route: route.continue_() if route.request.resource_type in ["document", "script", "fetch", "xhr"] else route.abort())
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(random.randint(1000, 2000)) # Tuỳ chỉnh thời gian chờ js load ngẫu nhiên 1-2s
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
        
        # Tìm bảng có chứa thẻ th 'Lịch sử giá vàng TPHCM'
        target_table = None
        for th in soup.find_all('th'):
            if "Lịch sử giá vàng TPHCM" in th.get_text(strip=True):
                target_table = th.find_parent('table')
                break
                
        if not target_table:
            print(f"Không tìm thấy bảng dữ liệu 'Lịch sử giá vàng TPHCM' ngày {day}-{month}-{year}")
            return
            
        tbody = target_table.find('tbody')
        if not tbody:
            print(f"Không có dữ liệu (tbody) ngày {day}-{month}-{year}")
            return

        rows = []
        current_gold_type_group = None
        
        for tr in tbody.find_all('tr'):
            tds = tr.find_all('td')
            if not tds:
                continue
                
            # Bỏ qua dòng tiêu đề "Loại vàng" bên trong khối dữ liệu
            if tds[0].text.strip() == "Loại vàng":
                continue
                
            if tds[0].has_attr('rowspan'):
                current_gold_type_group = tds[0].text.strip()
                gold_type = current_gold_type_group
                bid_idx = 1
                ask_idx = 2
            else:
                gold_type = current_gold_type_group
                bid_idx = 0
                ask_idx = 1
                
            if len(tds) > ask_idx:
                if gold_type == "SJC":
                    try:
                        # Thay thế dấu chấm và phẩy để lấy ra số đúng (ví dụ 168.500 -> 168500)
                        bid_str = tds[bid_idx].text.strip().replace('.', '').replace(',', '')
                        ask_str = tds[ask_idx].text.strip().replace('.', '').replace(',', '')
                        
                        bid = float(bid_str)
                        ask = float(ask_str)
                        
                        # Fix cứng timestamp vào lúc 23:59:59 của ngày cào
                        timestamp = date.strftime('%Y-%m-%d 23:59:59')
                        
                        rows.append({
                            'Loại vàng': gold_type,
                            'Giá mua': bid,
                            'Giá bán': ask,
                            'Thời gian cập nhật': timestamp
                        })
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Lỗi phân tích HTML {day}-{month}-{year}: {e}")
        return

    if rows:
        # Nhóm SJC sẽ trả về nhiều rows trong ngày đó
        # T lấy giá của hàng cuối cùng tương ứng mức giá chốt cuối phiên
        last_sjc_row = rows[-1]
        
        new_df = pd.DataFrame([last_sjc_row])
        try:
            new_df['Thời gian cập nhật'] = new_df['Thời gian cập nhật'].apply(convert_to_datetime)
            
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
    
    # Lấy danh sách các User-Agent phổ biến
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0"
    ]
    
    # Số luồng (tabs) đồng thời: giảm xuống 3-5 để tránh bị block IP chặn request
    semaphore = asyncio.Semaphore(10) 
    df_list = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-blink-features=AutomationControlled'])
        context = await browser.new_context(
            user_agent=random.choice(user_agents),
            viewport={'width': 1920, 'height': 1080}
        )
        
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
                
                # Không tính giá trung bình, chỉ giữ giá đóng (giá bán cuối phiên)
                if 'Giá trung bình bán' in df_main.columns:
                    df_main = df_main.drop(columns=['Giá trung bình bán'])
                
                df_main = df_main.sort_values('Thời gian cập nhật')
                df_main.to_csv(csv_file, index=False, encoding='utf-8')
                df_list.clear() # Xoá cache để chuẩn bị cho chu trình lưu tiếp theo
                print(f"-> Đã lưu checkpoint đợt này. Tổng record hiện tại: {len(df_main)}\n")
                
        await browser.close()
    
    print("Hoàn thành quá trình cào dữ liệu!")

if __name__ == "__main__":
    asyncio.run(main())