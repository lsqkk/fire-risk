import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

def merge_land_data(month_files, output_filename):
    """
    合并陆地数据文件为单个NetCDF文件
    
    Parameters:
    month_files: list of str, 月份文件列表
    output_filename: str, 输出文件名
    """
    # 文件到变量名的映射
    file_var_mapping = {
        'leaf_area_index_low_vegetation_0_daily-max.nc': 'lai_lv',
        'leaf_area_index_high_vegetation_0_daily-max.nc': 'lai_hv',
        'surface_pressure_0_daily-max.nc': 'sp',
        '10m_v_component_of_wind_0_daily-max.nc': 'v10',
        '10m_u_component_of_wind_0_daily-max.nc': 'u10',
        'skin_temperature_0_daily-max.nc': 'skt',
        '2m_dewpoint_temperature_0_daily-max.nc': 'd2m'
    }
    
    print("开始合并陆地数据...")
    print(f"处理 {len(month_files)} 个文件")
    
    # 存储处理后的数据集
    processed_datasets = []
    
    for file in month_files:
        if not os.path.exists(file):
            print(f"❌ 文件不存在: {file}")
            continue
            
        print(f"处理文件: {file}")
        
        try:
            # 读取NetCDF文件
            ds = xr.open_dataset(file)
            
            # 获取变量名
            var_name = file_var_mapping.get(file, list(ds.data_vars.keys())[0])
            
            if var_name not in ds.data_vars:
                print(f"  ⚠️ 变量 {var_name} 不存在，使用第一个可用变量")
                var_name = list(ds.data_vars.keys())[0]
            
            print(f"  变量: {var_name}, 形状: {ds[var_name].shape}")
            
            # 直接使用变量数据
            data_array = ds[var_name]
            
            # 创建新的数据集
            ds_single = data_array.to_dataset(name=var_name)
            processed_datasets.append(ds_single)
            
            ds.close()
            
        except Exception as e:
            print(f"  ❌ 处理文件 {file} 时出错: {e}")
            continue
    
    if not processed_datasets:
        print("❌ 没有成功处理任何数据集")
        return None
    
    # 合并所有数据集
    print("\n合并所有数据集...")
    try:
        merged_ds = xr.merge(processed_datasets, compat='override')
        
        # 输出合并后的信息
        print(f"✅ 合并完成!")
        print(f"时间步数: {len(merged_ds.valid_time)}")
        print(f"空间网格: {len(merged_ds.latitude)} × {len(merged_ds.longitude)}")
        print(f"变量数量: {len(merged_ds.data_vars)}")
        print(f"变量列表: {list(merged_ds.data_vars.keys())}")
        
        # 保存合并后的文件
        merged_ds.to_netcdf(output_filename)
        print(f"输出文件: {output_filename}")
        
        return merged_ds
        
    except Exception as e:
        print(f"❌ 合并过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_land_variables(ds):
    """
    分析陆地变量对山火预测的重要性
    """
    print("\n🔥 陆地变量山火相关性分析:")
    
    # 植被相关变量
    vegetation_vars = ['lai_lv', 'lai_hv']
    for var in vegetation_vars:
        if var in ds.data_vars:
            data = ds[var].mean(dim=['latitude', 'longitude'])
            print(f"  {var} (叶面积指数): {float(data.mean().values):.3f}")
            print(f"    - 重要性: 反映植被密度，高值区域可能增加火险")
    
    # 温度相关变量
    temp_vars = ['skt', 'd2m']
    for var in temp_vars:
        if var in ds.data_vars:
            data = ds[var].mean(dim=['latitude', 'longitude'])
            if var == 'skt':
                print(f"  {var} (地表温度): {float(data.mean().values):.2f}°K")
                print(f"    - 重要性: 高温增加可燃物干燥度")
            else:
                print(f"  {var} (露点温度): {float(data.mean().values):.2f}°K")
                print(f"    - 重要性: 低值表示干燥空气，增加火险")
    
    # 风场相关变量
    wind_vars = ['u10', 'v10']
    for var in wind_vars:
        if var in ds.data_vars:
            data = ds[var].mean(dim=['latitude', 'longitude'])
            print(f"  {var} (10米风场): {float(data.mean().values):.2f} m/s")
            print(f"    - 重要性: 影响火势蔓延方向和速度")
    
    # 气压变量
    if 'sp' in ds.data_vars:
        data = ds['sp'].mean(dim=['latitude', 'longitude'])
        print(f"  sp (地表气压): {float(data.mean().values):.1f} Pa")
        print(f"    - 重要性: 与天气系统相关，影响火险气象条件")

def visualize_land_data(file_path, time_index=0):
    """
    可视化陆地数据
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    # 读取数据
    ds = xr.open_dataset(file_path)
    
    # 获取所有数据变量
    data_vars = list(ds.data_vars.keys())
    
    print(f"可用的变量 ({len(data_vars)}个): {data_vars}")
    
    # 创建子图
    n_vars = len(data_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if n_cols == 1 else axes
    
    # 获取时间信息
    time_str = str(ds.valid_time.values[time_index])[:10]
    
    for i, var_name in enumerate(data_vars):
        if i >= len(axes):
            break
            
        try:
            # 选择数据
            data = ds[var_name].isel(valid_time=time_index)
            
            # 根据变量类型选择颜色映射
            if 'lai' in var_name:
                cmap = 'YlGn'  # 植被用绿色系
            elif 'temp' in var_name or 'skt' in var_name:
                cmap = 'hot'   # 温度用热力图
            elif 'd2m' in var_name:
                cmap = 'Blues' # 湿度用蓝色系
            elif 'sp' in var_name:
                cmap = 'viridis' # 气压用viridis
            else:
                cmap = 'RdBu_r' # 风场用红蓝系
            
            # 绘图
            im = axes[i].imshow(data, cmap=cmap, aspect='auto',
                              extent=[ds.longitude.min(), ds.longitude.max(),
                                     ds.latitude.min(), ds.latitude.max()])
            axes[i].set_title(f'{var_name}\n{time_str}', fontsize=10)
            axes[i].set_xlabel('Longitude')
            axes[i].set_ylabel('Latitude')
            plt.colorbar(im, ax=axes[i], shrink=0.8)
            
        except Exception as e:
            print(f"绘制变量 {var_name} 时出错: {e}")
            axes[i].set_title(f'{var_name}\n(无法显示)', fontsize=10)
    
    # 隐藏多余的子图
    for i in range(len(data_vars), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'陆地数据可视化 - 时间: {time_str}', y=1.02, fontsize=16)
    plt.show()
    
    ds.close()

def process_monthly_land_data(month, year=2024):
    """
    处理单个月的陆地数据
    
    Parameters:
    month: int, 月份
    year: int, 年份
    """
    # 构建文件列表
    base_files = [
        'leaf_area_index_low_vegetation_0_daily-max.nc',
        'leaf_area_index_high_vegetation_0_daily-max.nc',
        'surface_pressure_0_daily-max.nc',
        '10m_v_component_of_wind_0_daily-max.nc',
        '10m_u_component_of_wind_0_daily-max.nc',
        'skin_temperature_0_daily-max.nc',
        '2m_dewpoint_temperature_0_daily-max.nc'
    ]
    
    # 如果文件有月份标识，可以根据需要调整文件名
    # 这里假设文件已经在当前目录，且名称如诊断所示
    
    output_file = f"land_merged_{year}-{month:02d}.nc"
    
    print(f"\n处理 {year}年{month}月 陆地数据...")
    merged_data = merge_land_data(base_files, output_file)
    
    if merged_data is not None:
        # 分析变量
        analyze_land_variables(merged_data)
        
        # 可视化结果
        print("\n生成数据可视化...")
        visualize_land_data(output_file)
        
        print(f"\n✅ {year}年{month}月陆地数据处理完成!")
        return merged_data, output_file
    else:
        print(f"❌ {year}年{month}月陆地数据处理失败")
        return None, None

# 主执行程序 - 处理单个月份
if __name__ == "__main__":
    # 示例：处理3月数据
    month_to_process = 2  # 3月
    year_to_process = 2025  # 2024年
    
    try:
        result = process_monthly_land_data(month_to_process, year_to_process)
        
        if result[0] is not None:
            print(f"\n🎯 最终处理结果:")
            print(f"  文件: {result[1]}")
            print(f"  网格大小: {len(result[0].latitude)} × {len(result[0].longitude)}")
            print(f"  变量数量: {len(result[0].data_vars)}")
            print(f"  时间步数: {len(result[0].valid_time)}")
            
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()