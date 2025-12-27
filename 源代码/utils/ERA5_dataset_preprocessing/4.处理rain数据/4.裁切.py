import xarray as xr
import numpy as np

def crop_precipitation_data(input_file, output_file, lat_range, lon_range):
    """
    裁切降雨量NetCDF文件到指定的经纬度范围
    
    Parameters:
    input_file: 输入NetCDF文件路径
    output_file: 输出NetCDF文件路径
    lat_range: 纬度范围 (min, max)
    lon_range: 经度范围 (min, max)
    """
    print(f"开始裁切降雨量文件: {input_file}")
    
    # 读取数据
    ds = xr.open_dataset(input_file)
    
    # 自动检测降水变量名
    precip_vars = ['tp', 'precipitation_daily', 'precip', 'precipitation', 'rainfall']
    var_name = None
    for var in precip_vars:
        if var in ds.data_vars:
            var_name = var
            break
    
    if var_name is None:
        print(f"可用的变量: {list(ds.data_vars.keys())}")
        raise ValueError("无法自动识别降水变量")
    
    print(f"找到降水变量: {var_name}")
    print("原始数据信息:")
    print(f"  纬度范围: {ds.latitude.min().values:.2f} ~ {ds.latitude.max().values:.2f}")
    print(f"  经度范围: {ds.longitude.min().values:.2f} ~ {ds.longitude.max().values:.2f}")
    print(f"  空间网格: {len(ds.latitude)} × {len(ds.longitude)}")
    
    # 使用sel方法进行裁切
    # 注意：纬度从大到小 (47.39~43.00)，经度从小到大 (124.00~128.07)
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
    
    # 计算实际分辨率
    lat_res = abs(cropped_ds.latitude.values[1] - cropped_ds.latitude.values[0])
    lon_res = abs(cropped_ds.longitude.values[1] - cropped_ds.longitude.values[0])
    print(f"  实际纬度分辨率: {lat_res:.3f}°")
    print(f"  实际经度分辨率: {lon_res:.3f}°")
    
    if actual_lat_points == expected_lat_points and actual_lon_points == expected_lon_points:
        print("  ✅ 裁切维度符合预期")
    else:
        print("  ⚠️ 裁切维度与预期不符")
    
    # 检查数据质量
    precip_data = cropped_ds[var_name]
    print(f"\n数据质量检查:")
    print(f"  最小值: {precip_data.min().values:.6f} m")
    print(f"  最大值: {precip_data.max().values:.6f} m")
    print(f"  平均值: {precip_data.mean().values:.6f} m")
    
    # 更新全局属性
    cropped_ds.attrs.update({
        'cropped_lat_range': f"{lat_range[0]:.2f} ~ {lat_range[1]:.2f}",
        'cropped_lon_range': f"{lon_range[0]:.2f} ~ {lon_range[1]:.2f}",
        'cropped_grid_size': f"{actual_lat_points} × {actual_lon_points}",
        'processing_step': 'Cropped after interpolation'
    })
    
    # 保存裁切后的文件
    encoding = {
        var_name: {
            'zlib': True,
            'complevel': 5,
            'dtype': 'float32'
        }
    }
    cropped_ds.to_netcdf(output_file, encoding=encoding)
    print(f"\n✅ 裁切完成! 输出文件: {output_file}")
    
    return cropped_ds

# 主执行程序
if __name__ == "__main__":
    # 文件路径
    input_file = "rain-daily-0.01deg-kriging.nc"  # 插值后的文件
    output_file = "rain-daily-cropped-440x408.nc"
    
    # 定义裁切范围
    # 纬度: 43.00°N ~ 47.39°N (440个点)
    # 经度: 124.00°E ~ 128.07°E (408个点)
    lat_range = (43.00, 47.40)  # (min, max)
    lon_range = (124.00, 128.08)  # (min, max)
    
    try:
        # 执行裁切
        cropped_data = crop_precipitation_data(input_file, output_file, lat_range, lon_range)
        
        print(f"\n🎯 最终裁切结果:")
        print(f"  文件: {output_file}")
        print(f"  网格大小: {len(cropped_data.latitude)} × {len(cropped_data.longitude)}")
        print(f"  变量: {list(cropped_data.data_vars)}")
        if 'time' in cropped_data.dims:
            print(f"  时间步数: {len(cropped_data.time)}")
        
    except Exception as e:
        print(f"❌ 裁切过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

# 可选：批量裁切多个文件
def batch_crop_precipitation_files(file_list, output_dir, lat_range, lon_range):
    """
    批量裁切多个降雨量文件
    
    Parameters:
    file_list: 文件路径列表
    output_dir: 输出目录
    lat_range: 纬度范围
    lon_range: 经度范围
    """
    import os
    
    cropped_datasets = []
    
    for file_path in file_list:
        print(f"\n处理文件: {file_path}")
        try:
            # 生成输出文件名
            basename = os.path.basename(file_path)
            name, ext = os.path.splitext(basename)
            output_file = os.path.join(output_dir, f"{name}_cropped{ext}")
            
            # 执行裁切
            cropped_ds = crop_precipitation_data(file_path, output_file, lat_range, lon_range)
            cropped_datasets.append(cropped_ds)
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    return cropped_datasets

# 如果需要批量处理，可以使用以下代码
if __name__ == "__main__" and False:  # 设为False不执行，需要时改为True
    # 批量处理示例
    file_list = [
        "rain-daily-0.01deg-kriging.nc",
        # 可以添加更多文件
    ]
    output_dir = "cropped_results"
    
    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 批量裁切
    batch_crop_precipitation_files(file_list, output_dir, lat_range, lon_range)