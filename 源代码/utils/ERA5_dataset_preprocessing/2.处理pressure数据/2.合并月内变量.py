import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

def merge_era5_data():
    """
    合并ERA5数据，将5个文件*2个气压层合并为1个文件10个通道
    """
    files = {
        'v_component_of_wind_0_daily-max.nc': 'v',
        'u_component_of_wind_0_daily-max.nc': 'u',
        'temperature_0_daily-max.nc': 't',
        'specific_humidity_0_daily-max.nc': 'q',
        'geopotential_stream-oper_daily-max.nc': 'z'
    }
    
    # 存储所有数据变量
    all_data_vars = {}
    
    # 首先收集所有变量数据
    for filename, var_name in files.items():
        if not os.path.exists(filename):
            print(f"❌ 文件不存在: {filename}")
            continue
            
        print(f"处理文件: {filename}")
        
        try:
            ds = xr.open_dataset(filename)
            
            if var_name not in ds.data_vars:
                print(f"  ⚠️ 变量 {var_name} 不存在，使用第一个可用变量")
                var_name = list(ds.data_vars.keys())[0]
            
            # 获取数据变量
            data_var = ds[var_name]
            print(f"  数据形状: {data_var.shape}")
            
            # 处理每个压力层
            for plev in [850, 500]:
                try:
                    # 选择压力层数据
                    data_at_level = data_var.sel(pressure_level=plev)
                    
                    # 创建新变量名
                    new_var_name = f"{var_name}_{plev}"
                    
                    # 存储数据数组（不包含坐标）
                    all_data_vars[new_var_name] = data_at_level
                    print(f"    提取变量: {new_var_name}")
                    
                except Exception as e:
                    print(f"    处理压力层 {plev} 时出错: {e}")
            
            ds.close()
            
        except Exception as e:
            print(f"  ❌ 处理文件 {filename} 时出错: {e}")
            continue
    
    if not all_data_vars:
        print("❌ 没有成功提取任何数据")
        return None, None
    
    # 创建一个新的数据集，使用第一个文件的坐标
    print("\n创建合并数据集...")
    
    # 使用第一个有效文件作为坐标模板
    template_file = list(files.keys())[0]
    template_ds = xr.open_dataset(template_file)
    
    # 创建新的数据集，只包含时间、纬度、经度坐标
    merged_ds = xr.Dataset(
        coords={
            'valid_time': template_ds.valid_time,
            'latitude': template_ds.latitude,
            'longitude': template_ds.longitude
        }
    )
    
    # 添加所有数据变量
    for var_name, data_array in all_data_vars.items():
        merged_ds[var_name] = data_array
    
    template_ds.close()
    
    # 输出合并后的信息
    print(f"合并后的数据维度: {dict(merged_ds.dims)}")
    print(f"合并后的变量 ({len(merged_ds.data_vars)}个):")
    for var in merged_ds.data_vars:
        print(f"  {var}: {merged_ds[var].shape}")
    
    # 保存合并后的文件
    output_file = 'era5_merged_10channels.nc'
    merged_ds.to_netcdf(output_file)
    print(f"\n✅ 合并完成! 输出文件: {output_file}")
    
    return merged_ds, output_file

def visualize_merged_data(file_path, time_index=0):
    """
    可视化合并后的数据
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
    n_cols = 4
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
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
            
            # 绘图
            im = axes[i].imshow(data, cmap='viridis', aspect='auto', 
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
    plt.suptitle(f'ERA5数据可视化 - 时间: {time_str}', y=1.02, fontsize=16)
    plt.show()
    
    ds.close()

def check_data_quality(merged_ds):
    """
    检查数据质量
    """
    if merged_ds is None:
        return
        
    print("\n📊 数据质量检查:")
    for var_name in merged_ds.data_vars:
        data = merged_ds[var_name]
        print(f"\n{var_name}:")
        print(f"  形状: {data.shape}")
        print(f"  范围: {float(data.min().values):.4f} ~ {float(data.max().values):.4f}")
        print(f"  均值: {float(data.mean().values):.4f}")
        missing_count = data.isnull().sum().values
        print(f"  缺失值: {missing_count}")

# 主执行程序
if __name__ == "__main__":
    try:
        print("开始处理ERA5数据...")
        
        # 处理并合并数据
        merged_data, output_file = merge_era5_data()
        
        if merged_data is not None:
            # 检查数据质量
            check_data_quality(merged_data)
            
            # 可视化结果
            print("\n生成数据可视化...")
            visualize_merged_data(output_file, time_index=0)
            
            print("\n✅ 处理完成!")
            print(f"最终输出文件: {output_file}")
            print(f"数据形状: {len(merged_data.valid_time)} (时间) × {len(merged_data.data_vars)} (通道) × {len(merged_data.latitude)} (纬度) × {len(merged_data.longitude)} (经度)")
        else:
            print("❌ 数据处理失败")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()