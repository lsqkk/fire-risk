import cdsapi
import calendar
from subprocess import call
import time
import threading

def idmDownloader(task_url, folder_path, file_name):
    idm_engine = "C:\\Program Files (x86)\\Internet Download Manager\\IDMan.exe"
    call([idm_engine, '/d', task_url, '/p', folder_path, '/f', file_name, '/a'])
    call([idm_engine, '/s'])

def submit_request(month, pressure_group, group_name):
    try:
        c = cdsapi.Client()  # 每个线程创建自己的客户端
        # 获取当月的天数
        day_num = calendar.monthrange(2025, month)[1]
        days = [str(d).zfill(2) for d in range(1, day_num + 1)]
        # 基础请求参数
        request = {
            "product_type": "reanalysis",
            "variable": [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind"
            ],
            "year": "2025",
            "month": str(month).zfill(2),
            "day": days,
            "pressure_level": pressure_group,
            "daily_statistic": "daily_maximum",
            "time_zone": "utc+08:00",
            "frequency": "1_hourly",
            "area": [48, 124, 43, 129]
        }
        
        print(f"提交请求: 2025年{month}月 {group_name}组")
        r = c.retrieve('derived-era5-pressure-levels-daily-statistics', request)
        url = r.location  # 获取文件下载地址
        # 设置存储路径和文件名
        path = 'D:\\ERA5'
        filename = f"ERA5_2025{str(month).zfill(2)}_{group_name}.nc"
        print(f"添加到IDM: {filename}")
        idmDownloader(url, path, filename)
    except Exception as e:
        print(f"错误: 2025年{month}月 {group_name}组 - {str(e)}")

if __name__ == '__main__':
    pressure_group = ["500", "850"]
    group_name = "pressure_500_850"
    months = [1, 2]
    thread_feb = threading.Thread(
        target=submit_request, 
        args=(months[1], pressure_group, group_name)
    )
    thread_feb.start()
    thread_feb.join()