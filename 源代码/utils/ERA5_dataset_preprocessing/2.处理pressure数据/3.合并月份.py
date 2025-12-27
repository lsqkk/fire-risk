import xarray as xr
import os
import pandas as pd

def flexible_merge_months(file_patterns, output_filename):
    """
    灵活合并多个月份/年份的数据
    
    Parameters:
    file_patterns: list of str, 文件模式列表，如 ['2024-3.nc', '2024-4.nc', '2024-5.nc']
                  或使用通配符 ['2024-*.nc']
    output_filename: str, 输出文件名
    """
    # 展开通配符
    file_list = []
    for pattern in file_patterns:
        if '*' in pattern or '?' in pattern:
            matched_files = glob.glob(pattern)
            file_list.extend(matched_files)
        else:
            if os.path.exists(pattern):
                file_list.append(pattern)
    
    # 去重并排序
    file_list = sorted(list(set(file_list)))
    
    if not file_list:
        print("❌ 没有找到匹配的文件")
        return None
    
    print("找到以下文件:")
    for f in file_list:
        print(f"  {f}")
    
    try:
        # 合并文件
        merged_ds = xr.open_mfdataset(
            file_list, 
            combine='nested', 
            concat_dim='valid_time',
            compat='no_conflicts',
            parallel=True
        )
        
        # 按时间排序
        merged_ds = merged_ds.sortby('valid_time')
        
        # 输出信息
        times = merged_ds.valid_time.values
        print(f"\n✅ 合并完成!")
        print(f"时间范围: {times[0]} 到 {times[-1]}")
        print(f"总时间步数: {len(times)}")
        print(f"空间网格: {len(merged_ds.latitude)} × {len(merged_ds.longitude)}")
        print(f"变量数量: {len(merged_ds.data_vars)}")
        
        # 保存
        merged_ds.to_netcdf(output_filename)
        print(f"输出文件: {output_filename}")
        
        return merged_ds
        
    except Exception as e:
        print(f"❌ 合并过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# 使用示例
if __name__ == "__main__":
    # 示例1: 合并3-5月
    print("示例1: 合并2024年3-5月")
    result1 = flexible_merge_months(
        ['2024-3.nc', '2024-4.nc', '2024-5.nc','2024-6.nc','2024-7.nc','2024-8.nc','2024-9.nc','2024-10.nc','2024-11.nc','2024-12.nc','2025-1.nc','2025-2.nc'],
        'era5_pressure_202403-202502.nc'
    )