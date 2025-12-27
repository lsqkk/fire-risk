import xarray as xr
import numpy as np
import os

def crop_land_data(input_file, output_file, lat_range, lon_range):
    """
    裁切land数据到指定的经纬度范围
    
    Parameters:
    input_file: 输入NetCDF文件路径
    output_file: 输出NetCDF文件路径
    lat_range: 纬度范围 (min, max)
    lon_range: 经度范围 (min, max)
    """
    print(f"开始裁切land数据文件: {input_file}")
    
    # 读取数据
    ds = xr.open_dataset(input_file)
    
    print("原始数据信息:")
    print(f"  纬度范围: {ds.latitude.min().values:.2f} ~ {ds.latitude.max().values:.2f}")
    print(f"  经度范围: {ds.longitude.min().values:.2f} ~ {ds.longitude.max().values:.2f}")
    print(f"  空间网格: {len(ds.latitude)} × {len(ds.longitude)}")
    print(f"  变量数量: {len(ds.data_vars)}")
    print(f"  时间步数: {len(ds.valid_time)}")
    
    # 使用sel方法进行裁切
    # 纬度从大到小 (47.40~43.00)，经度从小到大 (124.00~128.08)
    cropped_ds = ds.sel(
        latitude=slice(lat_range[1], lat_range[0]),  # 纬度从大到小
        longitude=slice(lon_range[0], lon_range[1])   # 经度从小到大
    )
    
    print(f"\n裁切后数据信息:")
    print(f"  纬度范围: {cropped_ds.latitude.min().values:.2f} ~ {cropped_ds.latitude.max().values:.2f}")
    print(f"  经度范围: {cropped_ds.longitude.min().values:.2f} ~ {cropped_ds.longitude.max().values:.2f}")
    print(f"  空间网格: {len(cropped_ds.latitude)} × {len(cropped_ds.longitude)}")
    
    # 验证裁切结果
    expected_lat_points = 440
    expected_lon_points = 408
    actual_lat_points = len(cropped_ds.latitude)
    actual_lon_points = len(cropped_ds.longitude)
    
    print(f"\n裁切结果验证:")
    print(f"  预期纬度点数: {expected_lat_points}, 实际纬度点数: {actual_lat_points}")
    print(f"  预期经度点数: {expected_lon_points}, 实际经度点数: {actual_lon_points}")
    
    if actual_lat_points == expected_lat_points and actual_lon_points == expected_lon_points:
        print("  ✅ 裁切维度符合预期")
    else:
        print("  ⚠️ 裁切维度与预期不符")
        print(f"  实际网格大小: {actual_lat_points} × {actual_lon_points}")
    
    # 保存裁切后的文件
    cropped_ds.to_netcdf(output_file)
    print(f"\n✅ 裁切完成! 输出文件: {output_file}")
    
    # 验证输出文件
    verify_cropped_file(output_file, expected_lat_points, expected_lon_points)
    
    return cropped_ds

def verify_cropped_file(file_path, expected_lat, expected_lon):
    """
    验证裁切后的文件
    """
    if not os.path.exists(file_path):
        print(f"❌ 输出文件不存在: {file_path}")
        return
    
    ds = xr.open_dataset(file_path)
    
    actual_lat = len(ds.latitude)
    actual_lon = len(ds.longitude)
    
    print(f"\n🔍 输出文件验证:")
    print(f"  文件大小: {os.path.getsize(file_path) / 1024 / 1024:.1f} MB")
    print(f"  实际网格: {actual_lat} × {actual_lon}")
    print(f"  变量数量: {len(ds.data_vars)}")
    print(f"  时间步数: {len(ds.valid_time)}")
    
    if actual_lat == expected_lat and actual_lon == expected_lon:
        print("  ✅ 文件验证通过")
    else:
        print("  ⚠️ 文件验证未通过预期")
    
    # 打印变量信息
    print(f"  变量列表: {list(ds.data_vars.keys())}")
    
    ds.close()

def analyze_cropped_land_data(file_path):
    """
    分析裁切后的land数据
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    ds = xr.open_dataset(file_path)
    
    print("\n🌿 裁切后land数据分析:")
    
    # 植被变量
    vegetation_vars = [var for var in ds.data_vars if 'lai' in var]
    for var in vegetation_vars:
        data = ds[var].mean(dim=['latitude', 'longitude'])
        print(f"  {var}: {float(data.mean().values):.4f}")
    
    # 温度变量
    temp_vars = ['skt', 'd2m']
    for var in temp_vars:
        if var in ds.data_vars:
            data = ds[var].mean(dim=['latitude', 'longitude'])
            print(f"  {var}: {float(data.mean().values):.2f}°K")
    
    # 风场变量
    wind_vars = ['u10', 'v10']
    for var in wind_vars:
        if var in ds.data_vars:
            data = ds[var].mean(dim=['latitude', 'longitude'])
            print(f"  {var}: {float(data.mean().values):.2f} m/s")
    
    # 气压变量
    if 'sp' in ds.data_vars:
        data = ds['sp'].mean(dim=['latitude', 'longitude'])
        print(f"  sp: {float(data.mean().values):.1f} Pa")
    
    ds.close()

# 主执行程序
if __name__ == "__main__":
    # 文件路径
    input_file = "land_interp_0.01deg.nc"  # 假设这是插值后的land数据文件
    output_file = "land_cropped_440x408.nc"
    
    # 定义裁切范围
    # 纬度: 43.00°N ~ 47.40°N (440个点)
    # 经度: 124.00°E ~ 128.08°E (408个点)
    lat_range = (43.00, 47.40)  # (min, max)
    lon_range = (124.00, 128.08)  # (min, max)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        print("请确保已运行插值代码并生成 land_interp_0.01deg.nc 文件")
        exit(1)
    
    try:
        # 执行裁切
        cropped_data = crop_land_data(input_file, output_file, lat_range, lon_range)
        
        if cropped_data is not None:
            # 分析裁切后的数据
            analyze_cropped_land_data(output_file)
            
            print(f"\n🎯 最终裁切结果:")
            print(f"  文件: {output_file}")
            print(f"  网格大小: {len(cropped_data.latitude)} × {len(cropped_data.longitude)}")
            print(f"  变量数量: {len(cropped_data.data_vars)}")
            print(f"  时间步数: {len(cropped_data.valid_time)}")
            
        else:
            print("❌ 裁切失败")
        
    except Exception as e:
        print(f"❌ 裁切过程中出现错误: {e}")
        import traceback
        traceback.print_exc()