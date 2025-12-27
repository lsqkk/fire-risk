import xarray as xr
import numpy as np

def interpolate_with_xarray(input_file, output_file, target_resolution=0.01):
    """
    使用xarray内置方法进行插值
    """
    print(f"使用xarray插值方法处理: {input_file}")
    
    # 读取数据
    ds = xr.open_dataset(input_file)
    
    # 创建新的经纬度坐标
    lat_min, lat_max = ds.latitude.min().values, ds.latitude.max().values
    lon_min, lon_max = ds.longitude.min().values, ds.longitude.max().values
    
    new_lat = np.arange(lat_max, lat_min - target_resolution, -target_resolution)
    new_lon = np.arange(lon_min, lon_max + target_resolution, target_resolution)
    
    print(f"新网格: {len(new_lat)} × {len(new_lon)}")
    
    # 使用xarray的插值方法
    # 注意：这里使用线性插值，适合连续的气象场
    ds_interp = ds.interp(
        latitude=new_lat,
        longitude=new_lon,
        method='linear',
        kwargs={'fill_value': 'extrapolate'}
    )
    
    # 保存结果
    ds_interp.to_netcdf(output_file)
    print(f"✅ xarray插值完成! 输出文件: {output_file}")
    
    return ds_interp

# 如果您想尝试xarray方法，取消下面的注释
interpolate_with_xarray("pressure_beforeChazhi.nc", "pressure_xarray_interp.nc")