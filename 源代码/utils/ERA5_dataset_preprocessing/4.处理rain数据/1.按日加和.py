import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime

def process_era5_daily_precipitation(file_path, output_path=None):
    """
    处理ERA5小时降水数据为日总降水量
    
    Parameters:
    file_path: 输入NetCDF文件路径
    output_path: 输出文件路径（可选）
    """
    
    print("正在读取ERA5降水数据...")
    
    # 读取数据
    ds = xr.open_dataset(file_path)
    
    # 数据信息
    print("\n=== 原始数据信息 ===")
    print(f"数据维度: {dict(ds.dims)}")
    print(f"时间范围: {ds.valid_time.min().item()} 到 {ds.valid_time.max().item()}")
    print(f"时间长度: {len(ds.valid_time)} 小时")
    print(f"经纬度范围: {ds.latitude.min().item()}°N 到 {ds.latitude.max().item()}°N, "
          f"{ds.longitude.min().item()}°E 到 {ds.longitude.max().item()}°E")
    
    # 方法1: 使用resample方法（推荐）
    print("\n使用方法1: resample进行日重采样...")
    
    # 将valid_time设置为坐标（如果还没有的话）
    if 'valid_time' not in ds.coords:
        ds = ds.set_coords('valid_time')
    
    # 使用resample将小时数据重采样为日数据并求和
    daily_ds = ds.resample(valid_time='1D').sum()
    
    # 重命名变量以明确这是日数据
    daily_ds = daily_ds.rename({'tp': 'tp_daily'})
    
    # 重命名时间坐标
    daily_ds = daily_ds.rename({'valid_time': 'time'})
    
    # 方法2: 使用groupby的替代方法（如果resample不工作）
    # print("\n使用方法2: 手动分组进行日聚合...")
    # # 创建日期分组器
    # dates = xr.DataArray(
    #     pd.to_datetime(ds.valid_time.values).date,
    #     dims=['valid_time'],
    #     coords={'valid_time': ds.valid_time}
    # )
    # daily_precip = ds['tp'].groupby(dates).sum(dim='valid_time')
    # 
    # # 创建新的数据集
    # daily_ds = xr.Dataset({
    #     'tp_daily': (['date', 'latitude', 'longitude'], daily_precip.values)
    # }, 
    # coords={
    #     'date': daily_precip.date.values,
    #     'latitude': ds.latitude.values,
    #     'longitude': ds.longitude.values
    # })
    
    # 设置属性
    daily_ds['tp_daily'].attrs = {
        'long_name': 'Daily Total Precipitation',
        'units': 'm',
        'standard_name': 'precipitation_amount',
        'description': 'Accumulated daily precipitation from ERA5 hourly data'
    }
    
    # 添加全局属性
    daily_ds.attrs.update({
        'title': 'ERA5 Daily Total Precipitation',
        'source': 'ECMWF ERA5 reanalysis',
        'institution': 'European Centre for Medium-Range Weather Forecasts',
        'processing': 'Hourly to daily accumulation by summation',
        'processing_date': datetime.now().isoformat(),
        'original_file': file_path,
        'contact': 'Processed using xarray'
    })
    
    # 验证结果
    print("\n=== 处理结果 ===")
    print(f"日数据维度: {dict(daily_ds.dims)}")
    print(f"日期范围: {daily_ds.time.min().item()} 到 {daily_ds.time.max().item()}")
    print(f"总天数: {len(daily_ds.time)}")
    print(f"日降水量范围: {daily_ds['tp_daily'].min().item():.6f} 到 "
          f"{daily_ds['tp_daily'].max().item():.6f} m")
    
    # 保存文件
    if output_path:
        encoding = {
            'tp_daily': {
                'zlib': True,
                'complevel': 5,
                'dtype': 'float32'
            }
        }
        daily_ds.to_netcdf(output_path, encoding=encoding)
        print(f"\n日降水数据已保存: {output_path}")
    
    # 关闭原始数据集
    ds.close()
    
    return daily_ds

# 更简单的版本，专门针对您的情况
def simple_daily_precipitation(file_path, output_path=None):
    """
    简化的日降水处理函数
    """
    print("读取ERA5数据...")
    ds = xr.open_dataset(file_path)
    
    print(f"原始数据: {len(ds.valid_time)} 小时数据点")
    
    # 使用resample方法 - 这是最直接的方法
    print("使用resample进行日聚合...")
    daily_data = ds.resample(valid_time='D').sum()
    
    # 重命名变量
    daily_data = daily_data.rename({'tp': 'precipitation_daily'})
    daily_data = daily_data.rename({'valid_time': 'time'})
    
    # 添加单位属性
    daily_data['precipitation_daily'].attrs['units'] = 'm'
    daily_data['precipitation_daily'].attrs['long_name'] = 'Daily total precipitation'
    
    print(f"处理后: {len(daily_data.time)} 天数据点")
    
    if output_path:
        daily_data.to_netcdf(output_path)
        print(f"已保存到: {output_path}")
    
    ds.close()
    return daily_data

# 使用示例
if __name__ == "__main__":
    # 替换为您的实际文件路径
    input_file = "rain-25.nc"  # 替换为您的文件路径
    output_file = "era5_daily_precip_2025.nc"
    
    try:
        # 使用简化版本
        daily_data = simple_daily_precipitation(input_file, output_file)
        
        # 或者使用完整版本
        # daily_data = process_era5_daily_precipitation(input_file, output_file)
        
        print("\n处理完成！")
        print(daily_data)
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

# 如果上述方法仍有问题，这里提供一个最基础的手动方法
def manual_daily_aggregation(file_path, output_path=None):
    """
    手动日聚合方法 - 最可靠的方法
    """
    ds = xr.open_dataset(file_path)
    
    # 将时间转换为pandas DatetimeIndex
    times = pd.to_datetime(ds.valid_time.values)
    
    # 获取唯一的日期
    unique_dates = sorted(set(times.date))
    
    print(f"从 {min(unique_dates)} 到 {max(unique_dates)} 共 {len(unique_dates)} 天")
    
    # 为每天创建掩码并求和
    daily_precip = []
    dates_list = []
    
    for date in unique_dates:
        # 创建当天的掩码
        mask = times.date == date
        if mask.sum() == 24:  # 确保有完整的24小时数据
            # 对当天的所有小时数据求和
            day_sum = ds['tp'].isel(valid_time=mask).sum(dim='valid_time')
            daily_precip.append(day_sum)
            dates_list.append(date)
    
    # 合并结果
    daily_precip_array = xr.concat(daily_precip, dim='time')
    daily_precip_array = daily_precip_array.assign_coords(time=pd.DatetimeIndex(dates_list))
    
    # 创建数据集
    daily_ds = xr.Dataset({
        'precipitation_daily': daily_precip_array
    })
    
    if output_path:
        daily_ds.to_netcdf(output_path)
        print(f"手动聚合数据已保存: {output_path}")
    
    ds.close()
    return daily_ds