import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime

def merge_precipitation_files(file_2024, file_2025, output_file):
    """
    合并2024年和2025年的日降水数据文件
    
    Parameters:
    file_2024: 2024年日降水数据文件路径
    file_2025: 2025年日降水数据文件路径  
    output_file: 合并后的输出文件路径
    """
    
    print("开始合并降水数据文件...")
    
    # 读取两个文件
    print(f"读取文件: {file_2024}")
    ds_2024 = xr.open_dataset(file_2024)
    
    print(f"读取文件: {file_2025}") 
    ds_2025 = xr.open_dataset(file_2025)
    
    # 检查数据集结构
    print("\n=== 数据集结构检查 ===")
    print(f"2024年数据变量: {list(ds_2024.data_vars)}")
    print(f"2025年数据变量: {list(ds_2025.data_vars)}")
    print(f"2024年时间维度: {ds_2024.dims}")
    print(f"2025年时间维度: {ds_2025.dims}")
    
    # 检查时间范围
    print(f"2024年时间范围: {ds_2024.time.min().item()} 到 {ds_2024.time.max().item()}")
    print(f"2025年时间范围: {ds_2025.time.min().item()} 到 {ds_2025.time.max().item()}")
    
    # 方法1: 使用xarray的concat函数合并
    print("\n使用方法1: concat合并数据集...")
    
    # 检查变量名是否一致，如果不一致需要重命名
    var_name_2024 = list(ds_2024.data_vars)[0]
    var_name_2025 = list(ds_2025.data_vars)[0]
    
    # 如果变量名不同，统一变量名
    if var_name_2024 != var_name_2025:
        print(f"变量名不一致: {var_name_2024} vs {var_name_2025}")
        # 将2025年的变量重命名为与2024年相同
        ds_2025 = ds_2025.rename({var_name_2025: var_name_2024})
        print(f"已将2025年变量重命名为: {var_name_2024}")
    
    # 合并数据集
    try:
        # 按时间维度合并
        merged_ds = xr.concat([ds_2024, ds_2025], dim='time')
        
        # 按时间排序，确保时间顺序正确
        merged_ds = merged_ds.sortby('time')
        
    except Exception as e:
        print(f"concat方法失败: {e}")
        print("尝试使用combine_by_coords方法...")
        
        # 方法2: 使用combine_by_coords
        merged_ds = xr.combine_by_coords([ds_2024, ds_2025])
    
    # 检查合并结果
    print("\n=== 合并结果 ===")
    print(f"合并后时间范围: {merged_ds.time.min().item()} 到 {merged_ds.time.max().item()}")
    print(f"合并后数据维度: {dict(merged_ds.dims)}")
    print(f"总天数: {len(merged_ds.time)}")
    
    # 更新全局属性
    merged_ds.attrs.update({
        'title': 'ERA5 Daily Total Precipitation (2024-2025)',
        'source': 'ECMWF ERA5 reanalysis',
        'institution': 'European Centre for Medium-Range Weather Forecasts',
        'processing': 'Merged from separate yearly files',
        'processing_date': datetime.now().isoformat(),
        'time_period': '2024-03-01 to 2025-02-28',
        'contact': 'Processed using xarray'
    })
    
    # 保存合并后的文件
    print(f"\n保存合并文件: {output_file}")
    
    # 设置编码参数
    var_name = list(merged_ds.data_vars)[0]
    encoding = {
        var_name: {
            'zlib': True,
            'complevel': 5,
            'dtype': 'float32'
        },
        'time': {
            'dtype': 'double',
            'calendar': 'gregorian'
        }
    }
    
    merged_ds.to_netcdf(output_file, encoding=encoding)
    print("文件合并完成!")
    
    # 关闭数据集
    ds_2024.close()
    ds_2025.close()
    
    return merged_ds

# 更稳健的合并函数，处理可能的问题
def robust_merge_precipitation_files(file_2024, file_2025, output_file):
    """
    更稳健的合并方法，处理各种可能的情况
    """
    
    print("使用稳健方法合并文件...")
    
    # 读取文件
    ds_2024 = xr.open_dataset(file_2024)
    ds_2025 = xr.open_dataset(file_2025)
    
    # 获取变量名
    var_2024 = list(ds_2024.data_vars)[0]
    var_2025 = list(ds_2025.data_vars)[0]
    
    print(f"2024年变量: {var_2024}, 2025年变量: {var_2025}")
    
    # 如果变量名不同，统一变量名
    if var_2024 != var_2025:
        ds_2025 = ds_2025.rename({var_2025: var_2024})
        print(f"统一变量名为: {var_2024}")
    
    # 方法1: 尝试直接合并
    try:
        print("尝试直接合并...")
        merged = xr.concat([ds_2024, ds_2025], dim='time')
        merged = merged.sortby('time')
        
    except Exception as e:
        print(f"直接合并失败: {e}")
        
        # 方法2: 手动合并数据数组
        print("尝试手动合并数据数组...")
        
        # 提取数据数组
        data_2024 = ds_2024[var_2024]
        data_2025 = ds_2025[var_2024]
        
        # 合并数据数组
        merged_data = xr.concat([data_2024, data_2025], dim='time')
        merged_data = merged_data.sortby('time')
        
        # 创建新数据集
        merged = xr.Dataset({
            var_2024: merged_data
        })
        
        # 添加坐标
        merged.coords['latitude'] = ds_2024.latitude
        merged.coords['longitude'] = ds_2024.longitude
        
        # 检查坐标一致性
        if not np.array_equal(ds_2024.latitude.values, ds_2025.latitude.values):
            print("警告: 纬度坐标不一致!")
        if not np.array_equal(ds_2024.longitude.values, ds_2025.longitude.values):
            print("警告: 经度坐标不一致!")
    
    # 验证合并结果
    print(f"\n合并验证:")
    print(f"时间范围: {merged.time.min().item()} 到 {merged.time.max().item()}")
    print(f"总时间步长: {len(merged.time)}")
    print(f"数据形状: {merged[var_2024].shape}")
    
    # 保存文件
    print(f"\n保存到: {output_file}")
    merged.to_netcdf(output_file)
    
    # 关闭数据集
    ds_2024.close()
    ds_2025.close()
    
    return merged

# 使用示例
if __name__ == "__main__":
    # 文件路径
    file_2024 = "rain-daily-24.nc"  # 2024年3-12月数据
    file_2025 = "rain-daily-25.nc"  # 2025年1-2月数据
    output_file = "rain-daily-2024-2025-combined.nc"
    
    try:
        # 使用标准合并方法
        merged_data = merge_precipitation_files(file_2024, file_2025, output_file)
        
        print("\n合并成功!")
        print(merged_data)
        
    except Exception as e:
        print(f"标准合并方法失败: {e}")
        print("尝试使用稳健方法...")
        
        # 使用稳健方法
        merged_data = robust_merge_precipitation_files(file_2024, file_2025, output_file)
        
        print("\n稳健方法合并成功!")
        print(merged_data)

# 额外功能：检查文件内容的函数
def inspect_netcdf_files(file1, file2):
    """
    检查两个NetCDF文件的内容和结构
    """
    print("=== 文件检查 ===")
    
    ds1 = xr.open_dataset(file1)
    ds2 = xr.open_dataset(file2)
    
    print(f"文件1: {file1}")
    print(f"  变量: {list(ds1.data_vars)}")
    print(f"  坐标: {list(ds1.coords)}")
    print(f"  维度: {dict(ds1.dims)}")
    print(f"  时间范围: {ds1.time.min().item()} 到 {ds1.time.max().item()}")
    
    print(f"\n文件2: {file2}")
    print(f"  变量: {list(ds2.data_vars)}")
    print(f"  坐标: {list(ds2.coords)}")
    print(f"  维度: {dict(ds2.dims)}")
    print(f"  时间范围: {ds2.time.min().item()} 到 {ds2.time.max().item()}")
    
    # 检查变量名是否一致
    var1 = list(ds1.data_vars)[0]
    var2 = list(ds2.data_vars)[0]
    print(f"\n变量名一致性: {var1 == var2}")
    
    # 检查坐标是否一致
    lat_equal = np.array_equal(ds1.latitude.values, ds2.latitude.values)
    lon_equal = np.array_equal(ds1.longitude.values, ds2.longitude.values)
    print(f"纬度坐标一致性: {lat_equal}")
    print(f"经度坐标一致性: {lon_equal}")
    
    ds1.close()
    ds2.close()

# 运行前可以先检查文件
if __name__ == "__main__":
    file_2024 = "rain-daily-24.nc"
    file_2025 = "rain-daily-25.nc"
    
    # 先检查文件内容
    inspect_netcdf_files(file_2024, file_2025)
    
    # 然后合并
    output_file = "rain-daily-2024-2025-combined.nc"
    merged_data = merge_precipitation_files(file_2024, file_2025, output_file)