import xarray as xr
import os

def inspect_netcdf_structure():
    """
    诊断NetCDF文件结构
    """
    files = [

'land_with_precipitation_8channels.nc'

    ]
    
    for file in files:
        if not os.path.exists(file):
            print(f"❌ 文件不存在: {file}")
            continue
            
        print(f"\n{'='*60}")
        print(f"分析文件: {file}")
        print(f"{'='*60}")
        
        try:
            # 读取文件
            ds = xr.open_dataset(file)
            
            # 基本信息
            print("📊 数据集维度:")
            for dim, size in ds.dims.items():
                print(f"  {dim}: {size}")
            
            print("\n📋 数据集坐标:")
            for coord in ds.coords:
                print(f"  {coord}: {ds.coords[coord].values}")
            
            print("\n🔧 数据变量:")
            for var in ds.data_vars:
                print(f"  {var}:")
                print(f"    维度: {ds[var].dims}")
                print(f"    形状: {ds[var].shape}")
                if hasattr(ds[var], 'attributes'):
                    print(f"    属性: {ds[var].attributes}")
            
            print("\n📝 全局属性:")
            for attr in ds.attrs:
                print(f"  {attr}: {ds.attrs[attr]}")
                
            ds.close()
            
        except Exception as e:
            print(f"❌ 读取文件时出错: {e}")
            import traceback
            traceback.print_exc()

def get_detailed_variable_info(file_path):
    """
    获取文件的详细变量信息
    """
    print(f"\n🔎 详细分析: {file_path}")
    ds = xr.open_dataset(file_path)
    
    # 打印所有变量及其完整信息
    for var_name in ds.data_vars:
        var = ds[var_name]
        print(f"\n变量: {var_name}")
        print(f"  维度: {var.dims}")
        print(f"  形状: {var.shape}")
        print(f"  数据类型: {var.dtype}")
        
        # 打印每个维度的具体值
        for dim in var.dims:
            if dim in ds.coords:
                coord_vals = ds.coords[dim].values
                if len(coord_vals) <= 10:  # 如果值不多，全部打印
                    print(f"    {dim}: {coord_vals}")
                else:
                    print(f"    {dim}: [{coord_vals[0]}, ..., {coord_vals[-1]}] (共{len(coord_vals)}个)")
        
        # 打印属性
        if var.attrs:
            print(f"  属性: {var.attrs}")
    
    ds.close()

# 执行诊断
if __name__ == "__main__":
    print("开始诊断NetCDF文件结构...")
    inspect_netcdf_structure()
    
    # 可选：详细分析第一个文件
    files = [
        'v_component_of_wind_0_daily-max.nc',
        'u_component_of_wind_0_daily-max.nc', 
        'temperature_0_daily-max.nc',
        'specific_humidity_0_daily-max.nc',
        'geopotential_stream-oper_daily-max.nc'
    ]
    
    for file in files:
        if os.path.exists(file):
            get_detailed_variable_info(file)
            break  # 只分析第一个作为示例