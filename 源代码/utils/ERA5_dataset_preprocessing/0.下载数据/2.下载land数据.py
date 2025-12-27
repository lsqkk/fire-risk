import cdsapi
import calendar
from subprocess import call
import time
import threading

def idmDownloader(task_url, folder_path, file_name):
    idm_engine = "C:\\Program Files (x86)\\Internet Download Manager\\IDMan.exe"
    call([idm_engine, '/d', task_url, '/p', folder_path, '/f', file_name, '/a'])
    call([idm_engine, '/s'])

def submit_era5_land_request(year, month):
    try:
        c = cdsapi.Client()  # 每个线程创建自己的客户端
        # 获取当月的天数
        day_num = calendar.monthrange(year, month)[1]
        days = [str(d).zfill(2) for d in range(1, day_num + 1)]
        # ERA5-Land请求参数
        request = {
            "variable": [
                "2m_dewpoint_temperature",
                "skin_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_pressure",
                "leaf_area_index_high_vegetation",
                "leaf_area_index_low_vegetation"
            ],
            "year": str(year),
            "month": str(month).zfill(2),
            "day": days,
            "daily_statistic": "daily_maximum",
            "time_zone": "utc+08:00",
            "frequency": "1_hourly",
            "area": [48, 124, 43, 129]  # 保持不变
        }
        
        print(f"提交ERA5-Land请求: {year}年{month}月")
        r = c.retrieve('derived-era5-land-daily-statistics', request)
        url = r.location  # 获取文件下载地址
        
        # 设置存储路径和文件名
        path = 'D:\\ERA5_LAND'
        filename = f"ERA5_LAND_{year}{str(month).zfill(2)}.nc"
        
        print(f"添加到IDM: {filename}")
        idmDownloader(url, path, filename)
        
    except Exception as e:
        print(f"错误: {year}年{month}月 - {str(e)}")

def generate_monthly_requests(start_year, start_month, end_year, end_month):
    """
    生成月份请求列表
    """
    requests = []
    current_year = start_year
    current_month = start_month
    
    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        requests.append((current_year, current_month))
        
        # 移动到下一个月
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    
    return requests

if __name__ == '__main__':
    monthly_requests = generate_monthly_requests(2024, 3, 2025, 2)
    print(f"开始提交ERA5-Land下载请求: 2024年3月到2025年2月")
    print(f"总共 {len(monthly_requests)} 个月份")
    # 创建并启动所有线程
    threads = []
    for year, month in monthly_requests:
        thread = threading.Thread(
            target=submit_era5_land_request, 
            args=(year, month)
        )
        thread.start()
        threads.append(thread)
        time.sleep(2)  # 每隔2秒提交一个请求，避免过于密集
    # 等待所有线程完成
    for thread in threads:
        thread.join()