"""
Trajectory Generation Module (Optional)
轨迹生成模块：基于IMU数据生成3D运动轨迹
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from preprocessing import DataPreprocessor
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class TrajectoryGenerator:
    """
    轨迹生成器
    基于IMU数据估算3D运动轨迹
    """
    
    def __init__(self, sampling_rate=100):
        """
        初始化轨迹生成器
        
        Parameters:
        -----------
        sampling_rate : int
            采样频率 (Hz)
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.preprocessor = DataPreprocessor(sampling_rate)
        
    def integrate_acceleration(self, acc_data, initial_velocity=None, initial_position=None):
        """
        对加速度进行积分得到速度和位置
        
        Parameters:
        -----------
        acc_data : array-like (N, 3)
            三轴加速度数据
        initial_velocity : array-like (3,), optional
            初始速度
        initial_position : array-like (3,), optional
            初始位置
            
        Returns:
        --------
        tuple : (velocity, position)
        """
        N = len(acc_data)
        
        if initial_velocity is None:
            initial_velocity = np.zeros(3)
        if initial_position is None:
            initial_position = np.zeros(3)
        
        velocity = np.zeros((N, 3))
        position = np.zeros((N, 3))
        
        velocity[0] = initial_velocity
        position[0] = initial_position
        
        # 梯形积分
        for i in range(1, N):
            velocity[i] = velocity[i-1] + (acc_data[i] + acc_data[i-1]) * self.dt / 2
            position[i] = position[i-1] + (velocity[i] + velocity[i-1]) * self.dt / 2
        
        return velocity, position
    
    def remove_gravity(self, acc_data, gravity_magnitude=9.81):
        """
        移除重力分量（简化方法：假设z轴主要受重力影响）
        
        Parameters:
        -----------
        acc_data : DataFrame or dict
            包含ax, ay, az的数据
        gravity_magnitude : float
            重力加速度大小
            
        Returns:
        --------
        np.array : 去除重力后的加速度 (N, 3)
        """
        if isinstance(acc_data, pd.DataFrame):
            acc = np.column_stack([acc_data['ax'], acc_data['ay'], acc_data['az']])
        else:
            acc = acc_data
        
        # 简化方法：假设静止时主要是z轴受重力
        # 更精确的方法需要使用陀螺仪数据进行姿态估计
        acc_corrected = acc.copy()
        acc_corrected[:, 2] -= gravity_magnitude  # 移除z轴重力
        
        return acc_corrected
    
    def apply_complementary_filter(self, acc_data, gyro_data, alpha=0.98):
        """
        互补滤波器用于姿态估计
        
        Parameters:
        -----------
        acc_data : array-like (N, 3)
            加速度数据
        gyro_data : array-like (N, 3)
            陀螺仪数据
        alpha : float
            互补滤波系数
            
        Returns:
        --------
        np.array : 估计的姿态角 (N, 3) [roll, pitch, yaw]
        """
        N = len(acc_data)
        angles = np.zeros((N, 3))
        
        for i in range(1, N):
            # 从加速度计算角度
            acc_roll = np.arctan2(acc_data[i, 1], acc_data[i, 2])
            acc_pitch = np.arctan2(-acc_data[i, 0], 
                                   np.sqrt(acc_data[i, 1]**2 + acc_data[i, 2]**2))
            
            # 陀螺仪角度增量
            gyro_angles = gyro_data[i] * self.dt
            
            # 互补滤波
            angles[i, 0] = alpha * (angles[i-1, 0] + gyro_angles[0]) + (1 - alpha) * acc_roll
            angles[i, 1] = alpha * (angles[i-1, 1] + gyro_angles[1]) + (1 - alpha) * acc_pitch
            angles[i, 2] = angles[i-1, 2] + gyro_angles[2]  # yaw只能从陀螺仪获得
        
        return angles
    
    def generate_trajectory(self, df, method='simple'):
        """
        生成运动轨迹
        
        Parameters:
        -----------
        df : DataFrame
            IMU数据
        method : str
            生成方法 ('simple', 'filtered', 'dead_reckoning')
            
        Returns:
        --------
        dict : 轨迹数据
        """
        # 提取加速度和陀螺仪数据
        acc_data = np.column_stack([df['ax'], df['ay'], df['az']])
        gyro_data = np.column_stack([df['gx'], df['gy'], df['gz']])
        
        if method == 'simple':
            # 简单方法：直接积分
            acc_corrected = self.remove_gravity(acc_data)
            velocity, position = self.integrate_acceleration(acc_corrected)
            
        elif method == 'filtered':
            # 滤波方法：先滤波再积分
            acc_corrected = self.remove_gravity(acc_data)
            
            # 对每个轴应用高通滤波去除漂移
            acc_filtered = np.zeros_like(acc_corrected)
            for axis in range(3):
                acc_filtered[:, axis] = self.preprocessor.apply_bandpass_filter(
                    acc_corrected[:, axis], low_freq=0.1, high_freq=5.0
                )
            
            velocity, position = self.integrate_acceleration(acc_filtered)
            
        elif method == 'dead_reckoning':
            # 航位推算：使用姿态估计
            angles = self.apply_complementary_filter(acc_data, gyro_data)
            
            # 将加速度转换到全局坐标系
            acc_global = np.zeros_like(acc_data)
            for i in range(len(acc_data)):
                r = Rotation.from_euler('xyz', angles[i])
                acc_global[i] = r.apply(acc_data[i])
            
            acc_global[:, 2] -= 9.81  # 移除重力
            velocity, position = self.integrate_acceleration(acc_global)
            
        else:
            raise ValueError(f"未知的生成方法: {method}")
        
        return {
            'position': position,
            'velocity': velocity,
            'time': df['timestamp'].values
        }
    
    def visualize_trajectory_3d(self, trajectory, title="3D运动轨迹", save_path=None):
        """
        可视化3D运动轨迹
        
        Parameters:
        -----------
        trajectory : dict
            轨迹数据
        title : str
            图表标题
        save_path : str, optional
            保存路径
        """
        position = trajectory['position']
        
        fig = plt.figure(figsize=(12, 10))
        
        # 3D轨迹图
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(position[:, 0], position[:, 1], position[:, 2], 
                linewidth=2, color='blue', alpha=0.7)
        ax1.scatter(position[0, 0], position[0, 1], position[0, 2], 
                   color='green', s=100, label='起点', marker='o')
        ax1.scatter(position[-1, 0], position[-1, 1], position[-1, 2], 
                   color='red', s=100, label='终点', marker='s')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D轨迹视图')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # XY平面投影
        ax2 = fig.add_subplot(222)
        ax2.plot(position[:, 0], position[:, 1], linewidth=2, color='blue', alpha=0.7)
        ax2.scatter(position[0, 0], position[0, 1], color='green', s=100, label='起点')
        ax2.scatter(position[-1, 0], position[-1, 1], color='red', s=100, label='终点')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY平面投影（俯视图）')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # XZ平面投影
        ax3 = fig.add_subplot(223)
        ax3.plot(position[:, 0], position[:, 2], linewidth=2, color='blue', alpha=0.7)
        ax3.scatter(position[0, 0], position[0, 2], color='green', s=100, label='起点')
        ax3.scatter(position[-1, 0], position[-1, 2], color='red', s=100, label='终点')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('XZ平面投影（侧视图）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # YZ平面投影
        ax4 = fig.add_subplot(224)
        ax4.plot(position[:, 1], position[:, 2], linewidth=2, color='blue', alpha=0.7)
        ax4.scatter(position[0, 1], position[0, 2], color='green', s=100, label='起点')
        ax4.scatter(position[-1, 1], position[-1, 2], color='red', s=100, label='终点')
        ax4.set_xlabel('Y (m)')
        ax4.set_ylabel('Z (m)')
        ax4.set_title('YZ平面投影（正视图）')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 轨迹可视化已保存: {save_path}")
        
        plt.show()
    
    def process_file(self, filepath, method='filtered', save_visualization=True):
        """
        处理单个数据文件并生成轨迹
        
        Parameters:
        -----------
        filepath : str
            数据文件路径
        method : str
            生成方法
        save_visualization : bool
            是否保存可视化结果
            
        Returns:
        --------
        dict : 轨迹数据
        """
        # 加载数据
        df = self.preprocessor.load_data(filepath)
        scenario = df['scenario'].iloc[0] if 'scenario' in df.columns else "unknown"
        
        print(f"\n生成轨迹 - 场景: {scenario}")
        print(f"方法: {method}")
        
        # 生成轨迹
        trajectory = self.generate_trajectory(df, method=method)
        
        # 计算轨迹统计
        position = trajectory['position']
        total_distance = np.sum(np.sqrt(np.sum(np.diff(position, axis=0)**2, axis=1)))
        displacement = np.linalg.norm(position[-1] - position[0])
        
        print(f"总行进距离: {total_distance:.2f} m")
        print(f"直线位移: {displacement:.2f} m")
        
        # 可视化
        if save_visualization:
            results_dir = "../results"
            os.makedirs(results_dir, exist_ok=True)
            
            filename = os.path.basename(filepath).replace('.csv', '')
            save_path = os.path.join(results_dir, f"{filename}_trajectory.png")
            
            self.visualize_trajectory_3d(
                trajectory,
                title=f"3D运动轨迹 - {scenario}",
                save_path=save_path
            )
        
        return trajectory


def main():
    """
    主函数：演示轨迹生成
    """
    print("=" * 60)
    print("轨迹生成演示（选做部分）")
    print("=" * 60)
    
    data_dir = "../data"
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print("\n⚠️  未找到数据文件！")
        print("   请先运行 data_collection.py 采集数据")
        return
    
    generator = TrajectoryGenerator(sampling_rate=100)
    
    # 处理第一个数据文件
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if data_files:
        filepath = os.path.join(data_dir, data_files[0])
        trajectory = generator.process_file(filepath, method='filtered')
        
        print("\n✓ 轨迹生成完成！")
        print("\n注意：")
        print("  - 轨迹估算基于IMU数据的双重积分")
        print("  - 由于传感器噪声和漂移，轨迹会有累积误差")
        print("  - 更精确的轨迹需要额外的传感器融合（如GPS）")
    else:
        print("\n⚠️  data目录中没有CSV文件")


if __name__ == "__main__":
    main()
