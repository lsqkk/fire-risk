import xarray as xr
import numpy as np
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import griddata
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def auto_detect_variable(ds):
    """
    自动检测数据集中的降水变量
    """
    # 可能的降水变量名
    precip_vars = ['precipitation_daily', 'tp', 'precip', 'precipitation', 'rainfall', 'pr']
    
    for var in precip_vars:
        if var in ds.data_vars:
            print(f"找到降水变量: {var}")
            return var
    
    # 如果没有找到预定义的变量名，列出所有变量让用户选择
    print(f"可用的变量: {list(ds.data_vars.keys())}")
    raise ValueError("无法自动识别降水变量，请检查数据集中的变量名")

def kriging_interpolate_precipitation(input_file, output_file, target_res=0.01, var_name=None):
    """
    将降雨量数据从0.25°插值到0.01°分辨率（普通克里金法）
    """
    # 读取数据
    ds = xr.open_dataset(input_file)
    
    # 自动检测或使用指定的变量名
    if var_name is None:
        var_name = auto_detect_variable(ds)
    
    precip = ds[var_name]
    
    print(f"原始数据分辨率: {ds.latitude.values[1] - ds.latitude.values[0]:.3f}°")
    print(f"数据维度: {precip.dims}")
    print(f"数据形状: {precip.shape}")
    
    # 定义目标网格（0.01°分辨率）
    lat_min, lat_max = ds.latitude.min().item(), ds.latitude.max().item()
    lon_min, lon_max = ds.longitude.min().item(), ds.longitude.max().item()
    
    # 创建目标经纬度网格
    new_lats = np.arange(lat_max, lat_min - target_res, -target_res)
    new_lons = np.arange(lon_min, lon_max + target_res, target_res)
    
    print(f"目标网格纬度范围: {new_lats[0]:.2f}°N ~ {new_lats[-1]:.2f}°N, 点数: {len(new_lats)}")
    print(f"目标网格经度范围: {new_lons[0]:.2f}°E ~ {new_lons[-1]:.2f}°E, 点数: {len(new_lons)}")
    
    # 准备插值结果数组
    n_time = len(precip.time) if 'time' in precip.dims else 1
    interpolated_data = np.zeros((n_time, len(new_lats), len(new_lons)))
    
    # 对每个时间步进行克里金插值
    for t in tqdm(range(n_time), desc="时间步插值进度"):
        # 获取当前时间步的数据
        if n_time > 1:
            data_slice = precip.isel(time=t).values
        else:
            data_slice = precip.values
        
        # 创建源网格坐标
        src_lats = ds.latitude.values
        src_lons = ds.longitude.values
        
        # 重塑为点数据格式
        xx, yy = np.meshgrid(src_lons, src_lats)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        values = data_slice.ravel()
        
        # 移除NaN值
        mask = ~np.isnan(values)
        points_valid = points[mask]
        values_valid = values[mask]
        
        if len(values_valid) < 5:
            interpolated_data[t] = np.nan
            continue
        
        try:
            # 创建普通克里金对象
            ok = OrdinaryKriging(
                points_valid[:, 0],  # 经度
                points_valid[:, 1],  # 纬度  
                values_valid,
                variogram_model='spherical',
                nlags=20,
                weight=True,
                enable_plotting=False
            )
            
            # 执行插值
            z, ss = ok.execute('grid', new_lons, new_lats)
            interpolated_data[t] = z
            
        except Exception as e:
            print(f"时间步 {t} 克里金插值失败: {e}")
            print("尝试使用反距离权重法...")
            # 备用方法：反距离权重法
            try:
                grid_lon, grid_lat = np.meshgrid(new_lons, new_lats)
                target_points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
                interp_flat = griddata(points_valid, values_valid, target_points, 
                                      method='linear', fill_value=0)
                interpolated_data[t] = interp_flat.reshape(grid_lat.shape)
            except Exception as e2:
                print(f"反距离权重法也失败: {e2}")
                interpolated_data[t] = np.nan
    
    # 创建新的数据集
    coords = {
        'latitude': new_lats,
        'longitude': new_lons
    }
    
    if n_time > 1:
        coords['time'] = precip.time.values
        dims = ['time', 'latitude', 'longitude']
    else:
        dims = ['latitude', 'longitude']
    
    interp_ds = xr.Dataset({
        var_name: (dims, interpolated_data)
    }, coords=coords)
    
    # 添加属性
    interp_ds[var_name].attrs = {
        'long_name': 'Total precipitation',
        'units': 'm',
        'description': f'Interpolated from 0.25° to {target_res}° using Ordinary Kriging',
        'interpolation_method': 'Ordinary Kriging with spherical variogram'
    }
    
    interp_ds.attrs = {
        'title': f'ERA5 Daily Precipitation ({target_res}° resolution)',
        'source': 'ECMWF ERA5 reanalysis',
        'original_resolution': '0.25°',
        'interpolated_resolution': f'{target_res}°',
        'interpolation_method': 'Ordinary Kriging',
        'variogram_model': 'spherical',
        'history': f'Interpolated on {np.datetime64("now")}'
    }
    
    # 保存结果
    encoding = {
        var_name: {
            'zlib': True,
            'complevel': 5,
            'dtype': 'float32'
        }
    }
    interp_ds.to_netcdf(output_file, encoding=encoding)
    print(f"插值完成！结果已保存至: {output_file}")
    
    # 关闭数据集
    ds.close()
    
    return interp_ds

def idw_interpolate_precipitation(input_file, output_file, target_res=0.01, var_name=None):
    """
    使用反距离权重法插值降雨量数据（备用方法）
    """
    from scipy.interpolate import griddata
    
    # 读取数据
    ds = xr.open_dataset(input_file)
    
    # 自动检测或使用指定的变量名
    if var_name is None:
        var_name = auto_detect_variable(ds)
    
    precip = ds[var_name]
    
    print(f"使用IDW方法插值变量: {var_name}")
    
    # 定义目标网格
    lat_min, lat_max = ds.latitude.min().item(), ds.latitude.max().item()
    lon_min, lon_max = ds.longitude.min().item(), ds.longitude.max().item()
    
    new_lats = np.arange(lat_max, lat_min - target_res, -target_res)
    new_lons = np.arange(lon_min, lon_max + target_res, target_res)
    grid_lon, grid_lat = np.meshgrid(new_lons, new_lats)
    
    # 准备插值结果
    n_time = len(precip.time) if 'time' in precip.dims else 1
    interpolated_data = np.zeros((n_time, len(new_lats), len(new_lons)))
    
    # 源网格点
    src_lats = ds.latitude.values
    src_lons = ds.longitude.values
    src_points = np.column_stack([np.meshgrid(src_lons, src_lats)[0].ravel(), 
                                 np.meshgrid(src_lons, src_lats)[1].ravel()])
    
    for t in tqdm(range(n_time), desc="IDW插值进度"):
        if n_time > 1:
            data_slice = precip.isel(time=t).values
        else:
            data_slice = precip.values
            
        values = data_slice.ravel()
        
        # 移除NaN
        mask = ~np.isnan(values)
        points_valid = src_points[mask]
        values_valid = values[mask]
        
        if len(values_valid) == 0:
            interpolated_data[t] = np.nan
            continue
            
        # IDW插值
        target_points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
        interp_flat = griddata(points_valid, values_valid, target_points, 
                              method='linear', fill_value=0)
        
        interpolated_data[t] = interp_flat.reshape(grid_lat.shape)
    
    # 创建数据集
    coords = {
        'latitude': new_lats,
        'longitude': new_lons
    }
    
    if n_time > 1:
        coords['time'] = precip.time.values
        dims = ['time', 'latitude', 'longitude']
    else:
        dims = ['latitude', 'longitude']
    
    interp_ds = xr.Dataset({
        var_name: (dims, interpolated_data)
    }, coords=coords)
    
    interp_ds[var_name].attrs.update({
        'interpolation_method': 'Inverse Distance Weighting (linear)'
    })
    
    interp_ds.to_netcdf(output_file)
    print(f"IDW插值完成！结果保存至: {output_file}")
    ds.close()
    return interp_ds

# 使用示例
if __name__ == "__main__":
    input_file = "rain-daily-2024-2025-combined.nc"  # 您的合并文件
    output_file_kriging = "rain-daily-0.01deg-kriging.nc"
    output_file_idw = "rain-daily-0.01deg-idw.nc"
    
    # 首先尝试克里金法
    try:
        print("尝试克里金插值法...")
        result_kriging = kriging_interpolate_precipitation(input_file, output_file_kriging, target_res=0.01)
        
        print(f"\n克里金插值后数据信息:")
        print(f"纬度分辨率: {result_kriging.latitude.values[1] - result_kriging.latitude.values[0]:.3f}°")
        print(f"经度分辨率: {result_kriging.longitude.values[1] - result_kriging.longitude.values[0]:.3f}°")
        
    except Exception as e:
        print(f"克里金法失败: {e}")
        print("尝试使用IDW方法...")
        
        # 如果克里金失败，使用IDW方法
        result_idw = idw_interpolate_precipitation(input_file, output_file_idw, target_res=0.01)
        
        print(f"\nIDW插值后数据信息:")
        print(f"纬度分辨率: {result_idw.latitude.values[1] - result_idw.latitude.values[0]:.3f}°")
        print(f"经度分辨率: {result_idw.longitude.values[1] - result_idw.longitude.values[0]:.3f}°")