import xarray as xr
import numpy as np
import os

def merge_precipitation_to_land(land_file, rain_file, output_file):
    """
    将降水数据合并到land数据中作为第8个通道
    
    Parameters:
    land_file: land数据文件路径
    rain_file: 降水数据文件路径
    output_file: 输出文件路径
    """
    print("开始合并降水数据到land数据...")
    
    # 读取land数据
    print(f"读取land数据: {land_file}")
    ds_land = xr.open_dataset(land_file)
    
    # 读取降水数据
    print(f"读取降水数据: {rain_file}")
    ds_rain = xr.open_dataset(rain_file)
    
    print("\n数据维度对比:")
    print(f"Land数据 - 时间: {len(ds_land.valid_time)}, 纬度: {len(ds_land.latitude)}, 经度: {len(ds_land.longitude)}")
    print(f"Rain数据 - 时间: {len(ds_rain.time)}, 纬度: {len(ds_rain.latitude)}, 经度: {len(ds_rain.longitude)}")
    
    # 检查维度一致性
    if (len(ds_land.latitude) != len(ds_rain.latitude) or 
        len(ds_land.longitude) != len(ds_rain.longitude)):
        print("❌ 空间维度不匹配!")
        return None
    
    print("✅ 空间维度一致")
    
    # 重命名降水数据的时间维度以匹配land数据
    if 'time' in ds_rain.dims and 'valid_time' in ds_land.dims:
        print("重命名时间维度: time -> valid_time")
        ds_rain = ds_rain.rename({'time': 'valid_time'})
    
    # 检查时间维度一致性
    if len(ds_land.valid_time) != len(ds_rain.valid_time):
        print(f"⚠️ 时间维度不一致: land={len(ds_land.valid_time)}, rain={len(ds_rain.valid_time)}")
        # 如果时间维度不一致，我们需要对齐时间
        print("尝试对齐时间维度...")
        
        # 找到共同的时间范围
        common_times = np.intersect1d(ds_land.valid_time.values, ds_rain.valid_time.values)
        if len(common_times) == 0:
            print("❌ 没有共同的时间点")
            return None
        
        print(f"找到 {len(common_times)} 个共同时间点")
        
        # 选择共同的时间点
        ds_land = ds_land.sel(valid_time=common_times)
        ds_rain = ds_rain.sel(valid_time=common_times)
    
    # 重命名降水变量为更简洁的名称
    if 'precipitation_daily' in ds_rain.data_vars:
        print("重命名降水变量: precipitation_daily -> precipitation")
        ds_rain = ds_rain.rename({'precipitation_daily': 'precipitation'})
    
    # 合并数据集
    print("合并数据集...")
    merged_ds = xr.merge([ds_land, ds_rain])
    
    # 输出合并后的信息
    print(f"\n✅ 合并完成!")
    print(f"合并后变量数量: {len(merged_ds.data_vars)}")
    print(f"合并后变量列表: {list(merged_ds.data_vars.keys())}")
    print(f"时间步数: {len(merged_ds.valid_time)}")
    print(f"空间网格: {len(merged_ds.latitude)} × {len(merged_ds.longitude)}")
    
    # 添加处理历史记录
    if 'history' in merged_ds.attrs:
        merged_ds.attrs['history'] += f"; Merged with precipitation data on {np.datetime64('now')}"
    else:
        merged_ds.attrs['history'] = f"Merged with precipitation data on {np.datetime64('now')}"
    
    # 保存合并后的文件
    merged_ds.to_netcdf(output_file)
    print(f"输出文件: {output_file}")
    
    # 关闭数据集
    ds_land.close()
    ds_rain.close()
    
    return merged_ds

def verify_merged_data(file_path):
    """
    验证合并后的数据
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    ds = xr.open_dataset(file_path)
    
    print("\n🔍 合并数据验证:")
    print(f"文件: {file_path}")
    print(f"文件大小: {os.path.getsize(file_path) / 1024 / 1024:.1f} MB")
    print(f"时间步数: {len(ds.valid_time)}")
    print(f"空间网格: {len(ds.latitude)} × {len(ds.longitude)}")
    print(f"变量数量: {len(ds.data_vars)}")
    print(f"变量列表: {list(ds.data_vars.keys())}")
    
    # 检查每个变量的数据范围
    print("\n📊 变量数据范围:")
    for var_name in ds.data_vars:
        data = ds[var_name]
        if len(data.shape) == 3:  # 时间, 纬度, 经度
            # 计算整个时间序列的平均值
            mean_val = float(data.mean().values)
            min_val = float(data.min().values)
            max_val = float(data.max().values)
            print(f"  {var_name}: [{min_val:.4f}, {max_val:.4f}], 平均值: {mean_val:.4f}")
    
    ds.close()

def analyze_fire_risk_with_precipitation(file_path):
    """
    分析包含降水数据的山火风险变量
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    ds = xr.open_dataset(file_path)
    
    print("\n🔥 山火风险变量分析 (包含降水):")
    
    # 植被相关变量
    vegetation_vars = [var for var in ds.data_vars if 'lai' in var]
    for var in vegetation_vars:
        data = ds[var].mean(dim=['valid_time', 'latitude', 'longitude'])
        print(f"  {var} (叶面积指数): {float(data.values):.4f}")
        print(f"    - 重要性: 反映植被密度，高值区域可能增加火险")
    
    # 温度相关变量
    temp_vars = ['skt', 'd2m']
    for var in temp_vars:
        if var in ds.data_vars:
            data = ds[var].mean(dim=['valid_time', 'latitude', 'longitude'])
            if var == 'skt':
                print(f"  {var} (地表温度): {float(data.values):.2f}°K")
                print(f"    - 重要性: 高温增加可燃物干燥度")
            else:
                print(f"  {var} (露点温度): {float(data.values):.2f}°K")
                print(f"    - 重要性: 低值表示干燥空气，增加火险")
    
    # 风场相关变量
    wind_vars = ['u10', 'v10']
    for var in wind_vars:
        if var in ds.data_vars:
            data = ds[var].mean(dim=['valid_time', 'latitude', 'longitude'])
            print(f"  {var} (10米风场): {float(data.values):.2f} m/s")
            print(f"    - 重要性: 影响火势蔓延方向和速度")
    
    # 气压变量
    if 'sp' in ds.data_vars:
        data = ds['sp'].mean(dim=['valid_time', 'latitude', 'longitude'])
        print(f"  sp (地表气压): {float(data.values):.1f} Pa")
        print(f"    - 重要性: 与天气系统相关，影响火险气象条件")
    
    # 降水变量
    if 'precipitation' in ds.data_vars:
        data = ds['precipitation'].mean(dim=['valid_time', 'latitude', 'longitude'])
        print(f"  precipitation (降水): {float(data.values):.4f} mm/day")
        print(f"    - 重要性: 降水减少火险，干旱增加火险")
        print(f"    - 关键指标: 连续无降水天数对火险影响更大")
    
    ds.close()

# 主执行程序
if __name__ == "__main__":
    # 文件路径
    land_file = "land_cropped_440x408.nc"
    rain_file = "rain-daily-cropped-440x408.nc"
    output_file = "land_with_precipitation_8channels.nc"
    
    # 检查输入文件是否存在
    if not os.path.exists(land_file):
        print(f"❌ land文件不存在: {land_file}")
        exit(1)
    
    if not os.path.exists(rain_file):
        print(f"❌ 降水文件不存在: {rain_file}")
        exit(1)
    
    try:
        # 执行合并
        merged_data = merge_precipitation_to_land(land_file, rain_file, output_file)
        
        if merged_data is not None:
            # 验证合并结果
            verify_merged_data(output_file)
            
            # 分析山火风险变量
            analyze_fire_risk_with_precipitation(output_file)
            
            print(f"\n✅ 合并完成!")
            print(f"最终文件: {output_file}")
            print(f"总通道数: {len(merged_data.data_vars)}")
            
        else:
            print("❌ 合并失败")
            
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()