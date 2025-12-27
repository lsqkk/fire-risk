import xarray as xr
import numpy as np
import os

def merge_land_data_safe():
    """
    安全合并land数据文件 - 逐个文件处理确保数据完整性
    """
    # 明确列出所有要合并的文件
    file_list = [
        'land_merged_2024-03.nc',
        'land_merged_2024-04.nc', 
        'land_merged_2024-05.nc',
        'land_merged_2024-06.nc',
        'land_merged_2024-07.nc',
        'land_merged_2024-08.nc',
        'land_merged_2024-09.nc',
        'land_merged_2024-10.nc',
        'land_merged_2024-11.nc',
        'land_merged_2024-12.nc',
        'land_merged_2025-01.nc',
        'land_merged_2025-02.nc'
    ]
    
    # 检查文件是否存在并获取基本信息
    existing_files = []
    file_sizes = []
    
    for file in file_list:
        if os.path.exists(file):
            file_size = os.path.getsize(file) / 1024  # KB
            existing_files.append(file)
            file_sizes.append(file_size)
            print(f"✅ {file} ({file_size:.1f} KB)")
        else:
            print(f"❌ {file} (缺失)")
    
    if not existing_files:
        print("❌ 没有找到任何可用的文件")
        return None
    
    print(f"\n找到 {len(existing_files)} 个文件，总大小: {sum(file_sizes):.1f} KB")
    
    # 逐个读取文件并收集数据
    all_datasets = []
    
    for i, file in enumerate(existing_files):
        print(f"读取文件 {i+1}/{len(existing_files)}: {file}")
        
        try:
            ds = xr.open_dataset(file)
            
            # 检查数据维度
            print(f"  时间步: {len(ds.valid_time)}, 网格: {len(ds.latitude)}×{len(ds.longitude)}, 变量: {len(ds.data_vars)}")
            
            # 确保数据被实际加载
            # 通过计算一个变量的均值来强制加载数据
            sample_var = list(ds.data_vars.keys())[0]
            _ = ds[sample_var].mean().values  # 强制计算
            
            all_datasets.append(ds)
            
        except Exception as e:
            print(f"  ❌ 读取文件 {file} 时出错: {e}")
            continue
    
    if not all_datasets:
        print("❌ 没有成功读取任何数据集")
        return None
    
    print(f"\n成功读取 {len(all_datasets)} 个数据集，开始合并...")
    
    try:
        # 方法1: 使用concat合并（更可靠）
        print("使用concat方法合并...")
        merged_ds = xr.concat(all_datasets, dim='valid_time')
        
        # 确保时间顺序正确
        merged_ds = merged_ds.sortby('valid_time')
        
        # 输出合并后的详细信息
        print(f"\n✅ 合并完成!")
        print(f"时间范围: {merged_ds.valid_time.values[0]} 到 {merged_ds.valid_time.values[-1]}")
        print(f"总时间步数: {len(merged_ds.valid_time)}")
        print(f"空间网格: {len(merged_ds.latitude)} × {len(merged_ds.longitude)}")
        print(f"变量数量: {len(merged_ds.data_vars)}")
        print(f"变量列表: {list(merged_ds.data_vars.keys())}")
        
        # 计算预期的数据大小
        n_times = len(merged_ds.valid_time)
        n_lats = len(merged_ds.latitude)
        n_lons = len(merged_ds.longitude)
        n_vars = len(merged_ds.data_vars)
        # 假设float32数据 (4字节)
        expected_size = n_times * n_lats * n_lons * n_vars * 4 / 1024 / 1024  # MB
        
        print(f"预期数据大小: ~{expected_size:.1f} MB")
        
        # 保存合并后的文件
        output_file = "land_merged_2024_03-12.nc"
        
        # 使用更安全的保存选项
        encoding = {var: {'dtype': 'float32', 'zlib': True} for var in merged_ds.data_vars}
        merged_ds.to_netcdf(output_file, encoding=encoding)
        
        # 检查输出文件大小
        output_size = os.path.getsize(output_file) / 1024 / 1024  # MB
        print(f"输出文件: {output_file} ({output_size:.1f} MB)")
        
        # 验证输出文件
        verify_output_file(output_file)
        
        # 关闭所有数据集
        for ds in all_datasets:
            ds.close()
            
        return merged_ds
        
    except Exception as e:
        print(f"❌ 合并过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 关闭所有数据集
        for ds in all_datasets:
            ds.close()
            
        return None

def verify_output_file(file_path):
    """
    验证输出文件的完整性
    """
    if not os.path.exists(file_path):
        print(f"❌ 输出文件不存在: {file_path}")
        return
    
    try:
        ds = xr.open_dataset(file_path)
        
        print(f"\n🔍 输出文件验证:")
        print(f"文件大小: {os.path.getsize(file_path) / 1024 / 1024:.1f} MB")
        print(f"时间步数: {len(ds.valid_time)}")
        print(f"空间网格: {len(ds.latitude)} × {len(ds.longitude)}")
        print(f"变量数量: {len(ds.data_vars)}")
        
        # 检查数据范围
        for var_name in ds.data_vars:
            data = ds[var_name]
            print(f"  {var_name}: {data.shape}, 范围: [{float(data.min().values):.3f}, {float(data.max().values):.3f}]")
        
        ds.close()
        print("✅ 文件验证通过")
        
    except Exception as e:
        print(f"❌ 文件验证失败: {e}")

def alternative_merge_method():
    """
    备选合并方法：逐个变量合并
    """
    print("\n尝试备选合并方法...")
    
    file_list = [
        'land_merged_2024-03.nc',
        'land_merged_2024-04.nc', 
        'land_merged_2024-05.nc',
        'land_merged_2024-06.nc',
        'land_merged_2024-07.nc',
        'land_merged_2024-08.nc',
        'land_merged_2024-09.nc',
        'land_merged_2024-10.nc',
        'land_merged_2024-11.nc',
        'land_merged_2024-12.nc'
    ]
    
    existing_files = [f for f in file_list if os.path.exists(f)]
    
    if not existing_files:
        return None
    
    # 使用第一个文件作为模板
    template = xr.open_dataset(existing_files[0])
    var_names = list(template.data_vars.keys())
    
    # 为每个变量创建空列表
    var_data = {var: [] for var in var_names}
    
    # 收集所有数据
    for file in existing_files:
        ds = xr.open_dataset(file)
        for var in var_names:
            var_data[var].append(ds[var])
        ds.close()
    
    template.close()
    
    # 合并每个变量
    merged_vars = {}
    for var in var_names:
        merged_vars[var] = xr.concat(var_data[var], dim='valid_time')
    
    # 创建新数据集
    merged_ds = xr.Dataset(
        merged_vars,
        coords={
            'valid_time': merged_vars[var_names[0]].valid_time,
            'latitude': merged_vars[var_names[0]].latitude,
            'longitude': merged_vars[var_names[0]].longitude
        }
    )
    
    # 按时间排序
    merged_ds = merged_ds.sortby('valid_time')
    
    # 保存
    output_file = "land_merged_2024_03-12_alt.nc"
    merged_ds.to_netcdf(output_file)
    
    print(f"备选方法完成: {output_file}")
    return merged_ds

# 主执行程序
if __name__ == "__main__":
    print("开始合并land数据文件 (2024年3月-12月)...")
    
    # 首先尝试主要方法
    merged_data = merge_land_data_safe()
    
    if merged_data is None:
        print("\n主要方法失败，尝试备选方法...")
        merged_data = alternative_merge_method()
    
    if merged_data is not None:
        print(f"\n✅ land数据合并成功完成!")
        print(f"最终文件大小: {os.path.getsize('land_merged_2024_03-12.nc') / 1024 / 1024:.1f} MB")
    else:
        print("❌ land数据合并失败")