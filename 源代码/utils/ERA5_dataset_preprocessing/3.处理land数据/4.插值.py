import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

def interpolate_land_data(input_file, output_file, target_resolution=0.01):
    """
    对陆地数据进行经纬度插值，从0.25度插值到目标分辨率
    
    Parameters:
    input_file: 输入NetCDF文件路径
    output_file: 输出NetCDF文件路径  
    target_resolution: 目标分辨率（度）
    """
    print(f"开始处理陆地数据文件: {input_file}")
    
    # 读取数据
    ds = xr.open_dataset(input_file)
    
    print("原始数据信息:")
    print(f"  纬度范围: {ds.latitude.min().values:.2f} ~ {ds.latitude.max().values:.2f}")
    print(f"  经度范围: {ds.longitude.min().values:.2f} ~ {ds.longitude.max().values:.2f}")
    print(f"  原始分辨率: 0.25度")
    print(f"  目标分辨率: {target_resolution}度")
    print(f"  变量数量: {len(ds.data_vars)}")
    print(f"  时间步数: {len(ds.valid_time)}")
    
    # 创建目标经纬度网格
    lat_min, lat_max = float(ds.latitude.min().values), float(ds.latitude.max().values)
    lon_min, lon_max = float(ds.longitude.min().values), float(ds.longitude.max().values)
    
    # 创建新的经纬度坐标（0.01度间隔）
    new_lat = np.arange(lat_max, lat_min - target_resolution, -target_resolution)
    new_lon = np.arange(lon_min, lon_max + target_resolution, target_resolution)
    
    print(f"  新纬度网格: {len(new_lat)}个点 ({new_lat[0]:.2f} ~ {new_lat[-1]:.2f})")
    print(f"  新经度网格: {len(new_lon)}个点 ({new_lon[0]:.2f} ~ {new_lon[-1]:.2f})")
    print(f"  新网格大小: {len(new_lat)} × {len(new_lon)}")
    
    # 使用xarray的interp方法进行插值
    # 对于陆地数据，使用线性插值是合适的
    print("开始插值...")
    
    try:
        # 使用线性插值，对连续场效果良好
        ds_interp = ds.interp(
            latitude=new_lat,
            longitude=new_lon,
            method='linear',
            kwargs={'fill_value': None}  # 不填充，保持NaN
        )
        
        print("✅ 插值完成!")
        
        # 保存结果
        ds_interp.to_netcdf(output_file)
        print(f"输出文件: {output_file}")
        
        return ds_interp
        
    except Exception as e:
        print(f"❌ 插值过程中出现错误: {e}")
        # 尝试备选方案
        return alternative_interpolation(ds, new_lat, new_lon, output_file)

def alternative_interpolation(ds, new_lat, new_lon, output_file):
    """
    备选插值方案：逐个变量处理
    """
    print("尝试备选插值方案...")
    
    # 创建新的数据集
    interp_data = {}
    
    for var_name in ds.data_vars:
        print(f"  插值变量: {var_name}")
        
        try:
            # 对每个变量单独插值
            var_data = ds[var_name]
            
            # 使用线性插值
            var_interp = var_data.interp(
                latitude=new_lat,
                longitude=new_lon,
                method='linear',
                kwargs={'fill_value': None}
            )
            
            interp_data[var_name] = var_interp
            
        except Exception as e:
            print(f"    ❌ 变量 {var_name} 插值失败: {e}")
    
    # 创建新的数据集
    if interp_data:
        ds_interp = xr.Dataset(
            interp_data,
            coords={
                'valid_time': ds.valid_time,
                'latitude': new_lat,
                'longitude': new_lon
            }
        )
        
        # 复制全局属性
        ds_interp.attrs = ds.attrs
        
        # 保存结果
        ds_interp.to_netcdf(output_file)
        print(f"备选方案完成! 输出文件: {output_file}")
        
        return ds_interp
    else:
        print("❌ 所有变量插值都失败了")
        return None

def compare_land_resolution(original_file, interpolated_file, time_index=0):
    """
    比较原始分辨率和插值后的分辨率
    """
    if not os.path.exists(original_file) or not os.path.exists(interpolated_file):
        print("❌ 比较文件不存在")
        return
    
    # 读取数据
    ds_orig = xr.open_dataset(original_file)
    ds_interp = xr.open_dataset(interpolated_file)
    
    # 选择一个变量进行比较（例如地表温度）
    sample_var = 'skt' if 'skt' in ds_orig.data_vars else list(ds_orig.data_vars.keys())[0]
    
    # 创建对比图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始数据
    orig_data = ds_orig[sample_var].isel(valid_time=time_index)
    im1 = ax1.imshow(orig_data, extent=[ds_orig.longitude.min(), ds_orig.longitude.max(), 
                                       ds_orig.latitude.min(), ds_orig.latitude.max()],
                    cmap='viridis', aspect='auto')
    ax1.set_title(f'原始数据 (0.25°)\n{ds_orig.latitude.size}×{ds_orig.longitude.size}')
    ax1.set_xlabel('经度')
    ax1.set_ylabel('纬度')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 插值后数据
    interp_data = ds_interp[sample_var].isel(valid_time=time_index)
    im2 = ax2.imshow(interp_data, extent=[ds_interp.longitude.min(), ds_interp.longitude.max(), 
                                         ds_interp.latitude.min(), ds_interp.latitude.max()],
                    cmap='viridis', aspect='auto')
    ax2.set_title(f'插值后数据 ({0.01}°)\n{ds_interp.latitude.size}×{ds_interp.longitude.size}')
    ax2.set_xlabel('经度')
    ax2.set_ylabel('纬度')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 计算分辨率提升比例
    lat_ratio = ds_interp.latitude.size / ds_orig.latitude.size
    lon_ratio = ds_interp.longitude.size / ds_orig.longitude.size
    total_ratio = lat_ratio * lon_ratio
    
    # 差异图（可选）
    try:
        # 将原始数据插值到新网格以计算差异
        from scipy.interpolate import griddata
        
        # 创建原始网格点
        orig_lon, orig_lat = np.meshgrid(ds_orig.longitude.values, ds_orig.latitude.values)
        orig_points = np.column_stack((orig_lat.ravel(), orig_lon.ravel()))
        orig_values = orig_data.values.ravel()
        
        # 创建新网格点
        interp_lon, interp_lat = np.meshgrid(ds_interp.longitude.values, ds_interp.latitude.values)
        interp_points = np.column_stack((interp_lat.ravel(), interp_lon.ravel()))
        
        # 将原始数据插值到新网格
        orig_on_new_grid = griddata(orig_points, orig_values, interp_points, method='linear')
        orig_on_new_grid = orig_on_new_grid.reshape(interp_data.shape)
        
        # 计算差异
        diff = interp_data - orig_on_new_grid
        
        im3 = ax3.imshow(diff, extent=[ds_interp.longitude.min(), ds_interp.longitude.max(), 
                                      ds_interp.latitude.min(), ds_interp.latitude.max()],
                        cmap='RdBu_r', aspect='auto')
        ax3.set_title('差异 (插值 - 原始)')
        ax3.set_xlabel('经度')
        ax3.set_ylabel('纬度')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
    except Exception as e:
        print(f"无法创建差异图: {e}")
        ax3.text(0.5, 0.5, '无法计算差异', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('差异图')
    
    plt.tight_layout()
    plt.suptitle(f'陆地数据分辨率对比 - {sample_var}', y=1.02, fontsize=16)
    plt.show()
    
    # 打印统计信息
    print(f"\n📊 分辨率对比统计:")
    print(f"原始网格: {ds_orig.latitude.size} × {ds_orig.longitude.size}")
    print(f"插值网格: {ds_interp.latitude.size} × {ds_interp.longitude.size}")
    print(f"纬度点数增加: {lat_ratio:.1f}倍")
    print(f"经度点数增加: {lon_ratio:.1f}倍")
    print(f"总网格点数增加: {total_ratio:.1f}倍")
    
    # 计算数据大小估算
    n_times = len(ds_orig.valid_time)
    n_vars = len(ds_orig.data_vars)
    orig_size = n_times * ds_orig.latitude.size * ds_orig.longitude.size * n_vars * 4  # 假设float32
    interp_size = n_times * ds_interp.latitude.size * ds_interp.longitude.size * n_vars * 4
    
    print(f"原始数据估算大小: {orig_size / 1024 / 1024:.1f} MB")
    print(f"插值后估算大小: {interp_size / 1024 / 1024:.1f} MB")
    
    ds_orig.close()
    ds_interp.close()

def analyze_land_variables_after_interp(ds):
    """
    分析插值后的陆地变量
    """
    print("\n🌿 插值后陆地变量分析:")
    
    # 植被相关变量
    vegetation_vars = [var for var in ds.data_vars if 'lai' in var]
    for var in vegetation_vars:
        data = ds[var].mean(dim=['latitude', 'longitude'])
        print(f"  {var}: {float(data.mean().values):.3f}")
    
    # 温度相关变量
    temp_vars = ['skt', 'd2m']
    for var in temp_vars:
        if var in ds.data_vars:
            data = ds[var].mean(dim=['latitude', 'longitude'])
            unit = "°K"
            print(f"  {var}: {float(data.mean().values):.2f}{unit}")
    
    # 风场相关变量
    wind_vars = ['u10', 'v10']
    for var in wind_vars:
        if var in ds.data_vars:
            data = ds[var].mean(dim=['latitude', 'longitude'])
            print(f"  {var}: {float(data.mean().values):.2f} m/s")
    
    # 气压变量
    if 'sp' in ds.data_vars:
        data = ds['sp'].mean(dim=['latitude', 'longitude'])
        print(f"  sp: {float(data.mean().values):.1f} Pa")

# 主执行程序
if __name__ == "__main__":
    input_file = "land_merged_whole.nc"
    output_file = "land_interp_0.01deg.nc"
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        exit(1)
    
    try:
        print("开始陆地数据插值处理...")
        
        # 执行插值
        interpolated_ds = interpolate_land_data(input_file, output_file, target_resolution=0.01)
        
        if interpolated_ds is not None:
            # 分析插值后的变量
            analyze_land_variables_after_interp(interpolated_ds)
            
            # 比较分辨率
            print("\n生成分辨率对比图...")
            compare_land_resolution(input_file, output_file)
            
            print("\n✅ 陆地数据插值处理完成!")
            print(f"输入文件: {input_file}")
            print(f"输出文件: {output_file}")
            print(f"最终网格大小: {len(interpolated_ds.latitude)} × {len(interpolated_ds.longitude)}")
            
        else:
            print("❌ 陆地数据插值处理失败")
            
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()