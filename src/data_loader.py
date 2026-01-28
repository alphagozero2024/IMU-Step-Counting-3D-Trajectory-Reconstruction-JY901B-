"""
Data Loader for WitMotion IMU Format
加载WitMotion IMU传感器数据格式
"""

import pandas as pd
import numpy as np
import os


def load_witmotion_data(filepath, scenario_name=None, actual_steps=None):
    """
    加载WitMotion IMU传感器的数据文件
    
    Parameters:
    -----------
    filepath : str
        数据文件路径
    scenario_name : str, optional
        场景名称（如果文件中没有指定）
    actual_steps : int, optional
        实际步数（如果文件中没有指定）
        
    Returns:
    --------
    DataFrame : 标准化的IMU数据
    """
    # 读取文件，跳过第一行（StartTime）
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 检查是否有StartTime行
    start_idx = 0
    if lines[0].startswith('StartTime'):
        start_idx = 1
    
    # 读取数据（跳过StartTime和表头）
    df = pd.read_csv(filepath, sep='\t', skiprows=start_idx)
    
    print(f"✓ 已加载数据: {filepath}")
    print(f"  原始列数: {len(df.columns)}")
    print(f"  样本数: {len(df)}")
    
    # 提取需要的列
    # ax(g), ay(g), az(g) - 加速度（单位：g）
    # wx(deg/s), wy(deg/s), wz(deg/s) - 角速度（单位：度/秒）
    
    # 转换单位：g -> m/s², deg/s -> rad/s
    g_to_ms2 = 9.81
    deg_to_rad = np.pi / 180
    
    # 创建时间戳（相对时间）
    if 'Time(s)' in df.columns:
        # 解析时间字符串 HH:MM:SS.mmm
        time_str = df['Time(s)'].astype(str)
        timestamps = []
        first_time = None
        
        for t in time_str:
            # 解析 HH:MM:SS.mmm 格式
            try:
                parts = t.strip().split(':')
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                total_seconds = hours * 3600 + minutes * 60 + seconds
                
                if first_time is None:
                    first_time = total_seconds
                
                timestamps.append(total_seconds - first_time)
            except:
                timestamps.append(0)
        
        df_standard = pd.DataFrame({
            'timestamp': timestamps,
            'ax': df['ax(g)'].values * g_to_ms2,
            'ay': df['ay(g)'].values * g_to_ms2,
            'az': df['az(g)'].values * g_to_ms2,
            'gx': df['wx(deg/s)'].values * deg_to_rad,
            'gy': df['wy(deg/s)'].values * deg_to_rad,
            'gz': df['wz(deg/s)'].values * deg_to_rad
        })
    else:
        # 如果没有时间列，生成时间序列（假设100Hz采样率）
        sampling_rate = 100
        df_standard = pd.DataFrame({
            'timestamp': np.arange(len(df)) / sampling_rate,
            'ax': df['ax(g)'].values * g_to_ms2,
            'ay': df['ay(g)'].values * g_to_ms2,
            'az': df['az(g)'].values * g_to_ms2,
            'gx': df['wx(deg/s)'].values * deg_to_rad,
            'gy': df['wy(deg/s)'].values * deg_to_rad,
            'gz': df['wz(deg/s)'].values * deg_to_rad
        })
    
    # 添加元数据
    if scenario_name is None:
        # 尝试从文件名推断场景
        basename = os.path.basename(filepath)
        scenario_name = basename.replace('.txt', '').replace('.csv', '')
    
    df_standard['scenario'] = scenario_name
    
    if actual_steps is not None:
        df_standard['actual_steps'] = actual_steps
    else:
        df_standard['actual_steps'] = np.nan
    
    print(f"  转换后时长: {df_standard['timestamp'].max():.2f}秒")
    print(f"  实际采样率: {len(df_standard) / df_standard['timestamp'].max():.1f} Hz")
    
    return df_standard


def convert_witmotion_to_standard(input_file, output_file, scenario_name=None, actual_steps=None):
    """
    将WitMotion格式转换为标准CSV格式
    
    Parameters:
    -----------
    input_file : str
        输入文件路径（WitMotion格式）
    output_file : str
        输出文件路径（标准CSV格式）
    scenario_name : str, optional
        场景名称
    actual_steps : int, optional
        实际步数
    """
    df = load_witmotion_data(input_file, scenario_name, actual_steps)
    
    # 保存为标准CSV格式
    df.to_csv(output_file, index=False)
    print(f"✓ 已保存标准格式数据: {output_file}")
    
    return df


def batch_convert_directory(input_dir, output_dir):
    """
    批量转换目录中的所有WitMotion数据文件
    
    Parameters:
    -----------
    input_dir : str
        输入目录
    output_dir : str
        输出目录
    """
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有txt文件
    txt_files = glob.glob(os.path.join(input_dir, '*.txt'))
    
    if not txt_files:
        print("⚠️  未找到txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个数据文件\n")
    
    for txt_file in txt_files:
        basename = os.path.basename(txt_file)
        output_file = os.path.join(output_dir, basename.replace('.txt', '.csv'))
        
        print(f"处理: {basename}")
        
        # 尝试从文件名提取实际步数
        # 假设文件名格式类似: "walking_50steps.txt" 或 "scenario_name.txt"
        actual_steps = None
        if 'steps' in basename.lower():
            try:
                # 提取数字
                import re
                match = re.search(r'(\d+)\s*steps?', basename, re.IGNORECASE)
                if match:
                    actual_steps = int(match.group(1))
            except:
                pass
        
        # 如果没有自动提取到步数，询问用户
        if actual_steps is None:
            try:
                actual_steps_input = input(f"  请输入 {basename} 的实际步数（直接回车跳过）: ")
                if actual_steps_input.strip():
                    actual_steps = int(actual_steps_input)
            except:
                pass
        
        convert_witmotion_to_standard(txt_file, output_file, actual_steps=actual_steps)
        print()


def main():
    """
    主函数：演示数据加载和转换
    """
    print("=" * 70)
    print("WitMotion IMU数据加载器")
    print("=" * 70)
    
    # 检查sample_data.txt
    sample_file = "../sample_data.txt"
    if os.path.exists(sample_file):
        print("\n发现示例数据文件，加载中...")
        
        # 加载示例数据
        df = load_witmotion_data(sample_file, scenario_name="sample_demo")
        
        # 显示数据预览
        print("\n数据预览：")
        print(df.head(10))
        
        # 保存为标准格式
        output_file = "../data/sample_data_converted.csv"
        os.makedirs("../data", exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\n✓ 已保存为标准格式: {output_file}")
        
    else:
        print("\n⚠️  未找到 sample_data.txt")
    
    print("\n" + "=" * 70)
    print("使用说明：")
    print("=" * 70)
    print("1. 单个文件转换：")
    print("   df = load_witmotion_data('your_file.txt', scenario_name='walking', actual_steps=50)")
    print("   df.to_csv('output.csv', index=False)")
    print()
    print("2. 批量转换：")
    print("   batch_convert_directory('../raw_data', '../data')")
    print()
    print("转换后的文件可以直接用于 preprocessing.py, step_detection.py 等模块")
    print("=" * 70)


if __name__ == "__main__":
    main()
