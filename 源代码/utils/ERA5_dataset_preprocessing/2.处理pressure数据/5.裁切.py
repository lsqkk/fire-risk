import xarray as xr
import numpy as np

def crop_netcdf_file(input_file, output_file, lat_range, lon_range):
    """
    裁切NetCDF文件到指定的经纬度范围
    
    Parameters:
    input_file: 输入NetCDF文件路径
    output_file: 输出NetCDF文件路径
    lat_range: 纬度范围 (min, max)
    lon_range: 经度范围 (min, max)
    """
    print(f"开始裁切文件: {input_file}")
    
    # 读取数据
    ds = xr.open_dataset(input_file)
    
    print("原始数据信息:")
    print(f"  纬度范围: {ds.latitude.min().values:.2f} ~ {ds.latitude.max().values:.2f}")
    print(f"  经度范围: {ds.longitude.min().values:.2f} ~ {ds.longitude.max().values:.2f}")
    print(f"  空间网格: {len(ds.latitude)} × {len(ds.longitude)}")
    
    # 使用sel方法进行裁切 [citation:2][citation:5]
    # slice函数用于指定连续的坐标范围
    cropped_ds = ds.sel(
        latitude=slice(lat_range[1], lat_range[0]),  # 纬度从大到小 (47.4~43.0)
        longitude=slice(lon_range[0], lon_range[1])   # 经度从小到大 (124.0~128.08)
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
    
    # 保存裁切后的文件
    cropped_ds.to_netcdf(output_file)
    print(f"\n✅ 裁切完成! 输出文件: {output_file}")
    
    return cropped_ds

# 主执行程序
if __name__ == "__main__":
    # 文件路径
    input_file = "pressure_xarray_interp.nc"
    output_file = "pressure_cropped_440x408.nc"
    
    # 定义裁切范围 [citation:5]
    # 纬度: 43.00°N ~ 47.40°N (440个点)
    # 经度: 124.00°E ~ 128.08°E (408个点)
    lat_range = (43.00, 47.40)  # (min, max)
    lon_range = (124.00, 128.08)  # (min, max)
    
    try:
        # 执行裁切
        cropped_data = crop_netcdf_file(input_file, output_file, lat_range, lon_range)
        
        print(f"\n🎯 最终裁切结果:")
        print(f"  文件: {output_file}")
        print(f"  网格大小: {len(cropped_data.latitude)} × {len(cropped_data.longitude)}")
        print(f"  变量数量: {len(cropped_data.data_vars)}")
        print(f"  时间步数: {len(cropped_data.valid_time)}")
        
    except Exception as e:
        print(f"❌ 裁切过程中出现错误: {e}")
        import traceback
        traceback.print_exc()