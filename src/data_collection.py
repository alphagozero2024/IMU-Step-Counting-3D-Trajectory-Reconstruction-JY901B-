"""
Data Collection Module for IMU-based Step Counter
采集IMU数据的模块
"""

import numpy as np
import pandas as pd
from datetime import datetime
import time
import os

class IMUDataCollector:
    """
    IMU数据采集器
    用于从IMU传感器采集加速度和陀螺仪数据
    """
    
    def __init__(self, sampling_rate=100):
        """
        初始化数据采集器
        
        Parameters:
        -----------
        sampling_rate : int
            采样频率 (Hz)
        """
        self.sampling_rate = sampling_rate
        self.data = []
        self.is_collecting = False
        
    def collect_data(self, duration, scenario_name, actual_steps=0):
        """
        采集指定时长的IMU数据
        
        Parameters:
        -----------
        duration : float
            采集时长（秒）
        scenario_name : str
            场景名称（如 "walking_flat", "running", "pocket"等）
        actual_steps : int
            实际步数（人工计数）
            
        Returns:
        --------
        DataFrame : 包含时间戳和IMU数据的数据框
        """
        print(f"\n开始采集数据 - 场景: {scenario_name}")
        print(f"采集时长: {duration}秒")
        print(f"实际步数: {actual_steps}步")
        print("=" * 50)
        
        # 注意：这里需要连接实际的IMU硬件
        # 以下代码为模拟数据采集的框架
        print("\n⚠️  请确保IMU设备已连接并准备就绪")
        print("⚠️  需要根据您的IMU硬件修改此函数")
        print("\n示例：如果使用Arduino + MPU6050:")
        print("  1. 通过串口读取数据")
        print("  2. 解析加速度和陀螺仪值")
        print("  3. 存储到data列表中")
        
        # TODO: 替换为实际的IMU数据读取代码
        # 示例框架（需要根据实际硬件修改）：
        """
        import serial
        ser = serial.Serial('COM3', 115200)  # 根据实际端口修改
        
        start_time = time.time()
        while time.time() - start_time < duration:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8').strip()
                # 假设数据格式: "ax,ay,az,gx,gy,gz"
                values = line.split(',')
                if len(values) == 6:
                    timestamp = time.time()
                    self.data.append({
                        'timestamp': timestamp,
                        'ax': float(values[0]),
                        'ay': float(values[1]),
                        'az': float(values[2]),
                        'gx': float(values[3]),
                        'gy': float(values[4]),
                        'gz': float(values[5])
                    })
            time.sleep(1.0 / self.sampling_rate)
        """
        
        # 模拟数据（仅用于测试代码框架）
        print("\n⚠️  当前使用模拟数据进行测试")
        print("   请替换为实际IMU读取代码后再进行正式实验\n")
        
        n_samples = int(duration * self.sampling_rate)
        timestamps = np.linspace(0, duration, n_samples)
        
        # 模拟行走时的加速度模式
        step_freq = actual_steps / duration if actual_steps > 0 else 2.0
        
        for i, t in enumerate(timestamps):
            # 模拟垂直方向的周期性加速度变化（行走特征）
            ax = 0.1 * np.random.randn()
            ay = 0.1 * np.random.randn()
            az = 9.81 + 2.0 * np.sin(2 * np.pi * step_freq * t) + 0.3 * np.random.randn()
            
            gx = 0.05 * np.random.randn()
            gy = 0.05 * np.random.randn()
            gz = 0.05 * np.random.randn()
            
            self.data.append({
                'timestamp': t,
                'ax': ax,
                'ay': ay,
                'az': az,
                'gx': gx,
                'gy': gy,
                'gz': gz
            })
        
        df = pd.DataFrame(self.data)
        self.data = []  # 清空缓存
        
        print(f"✓ 数据采集完成！共采集 {len(df)} 个样本")
        return df, scenario_name, actual_steps
    
    def save_data(self, df, scenario_name, actual_steps, save_dir="../data"):
        """
        保存采集的数据到CSV文件
        
        Parameters:
        -----------
        df : DataFrame
            IMU数据
        scenario_name : str
            场景名称
        actual_steps : int
            实际步数
        save_dir : str
            保存目录
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scenario_name}_{actual_steps}steps_{timestamp_str}.csv"
        filepath = os.path.join(save_dir, filename)
        
        # 添加元数据
        df['scenario'] = scenario_name
        df['actual_steps'] = actual_steps
        
        df.to_csv(filepath, index=False)
        print(f"✓ 数据已保存到: {filepath}")
        
        return filepath


def main():
    """
    主函数：演示数据采集流程
    """
    print("=" * 60)
    print("IMU数据采集程序 - 计步器项目")
    print("=" * 60)
    
    collector = IMUDataCollector(sampling_rate=100)
    
    # 定义测试场景
    scenarios = [
        {"name": "walking_flat_hand", "duration": 30, "steps": 40, "desc": "平地行走-手持"},
        {"name": "walking_flat_pocket", "duration": 30, "steps": 40, "desc": "平地行走-口袋"},
        {"name": "walking_uphill", "duration": 30, "steps": 35, "desc": "上坡行走"},
        {"name": "walking_downhill", "duration": 30, "steps": 45, "desc": "下坡行走"},
        {"name": "running", "duration": 20, "steps": 60, "desc": "跑步"},
    ]
    
    print("\n建议的测试场景：")
    for i, scene in enumerate(scenarios, 1):
        print(f"{i}. {scene['desc']} - 时长{scene['duration']}秒，预计{scene['steps']}步")
    
    print("\n" + "=" * 60)
    print("使用说明：")
    print("=" * 60)
    print("1. 连接您的IMU设备（如Arduino + MPU6050）")
    print("2. 修改 collect_data() 函数中的IMU读取代码")
    print("3. 运行此脚本开始采集数据")
    print("4. 在每个场景中，手动计数实际步数作为参考")
    print("5. 每种场景至少采集3-5组不同步数的数据")
    print("\n建议的IMU传感器：")
    print("  - MPU6050 (常用，性价比高)")
    print("  - MPU9250 (9轴，更精确)")
    print("  - 手机内置IMU (可使用Sensor Logger等App)")
    print("=" * 60)
    
    # 示例：采集一组数据
    print("\n演示：采集一组模拟数据...")
    input("按Enter键开始采集（或Ctrl+C退出）...")
    
    df, scenario, steps = collector.collect_data(
        duration=10,
        scenario_name="demo_walking",
        actual_steps=15
    )
    
    collector.save_data(df, scenario, steps)
    
    print("\n✓ 演示完成！")
    print("  请根据您的硬件修改代码后进行正式数据采集。")


if __name__ == "__main__":
    main()
