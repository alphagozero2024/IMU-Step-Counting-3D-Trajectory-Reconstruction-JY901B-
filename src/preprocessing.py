"""
Data Preprocessing Module
数据预处理模块：噪声分析、滤波、特征提取
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class DataPreprocessor:
    """
    数据预处理器
    负责噪声分析、滤波、特征提取等
    """
    
    def __init__(self, sampling_rate=100):
        """
        初始化预处理器
        
        Parameters:
        -----------
        sampling_rate : int
            采样频率 (Hz)
        """
        self.sampling_rate = sampling_rate
        
    def load_data(self, filepath):
        """
        加载IMU数据
        
        Parameters:
        -----------
        filepath : str
            数据文件路径（支持CSV格式）
            
        Returns:
        --------
        DataFrame : IMU数据
        """
        try:
            df = pd.read_csv(filepath)
            print(f"✓ 已加载数据: {filepath}")
            print(f"  样本数: {len(df)}, 时长: {df['timestamp'].max():.2f}秒")
            
            # 验证必要的列
            required_cols = ['timestamp', 'ax', 'ay', 'az']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️  缺少必要的列: {missing_cols}")
                print("  提示: 如果是WitMotion格式，请先使用 data_loader.py 转换")
            
            return df
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            print("  提示: 请确保数据格式正确或使用 data_loader.py 转换")
            raise
    
    def calculate_magnitude(self, df):
        """
        计算加速度向量的模（合成加速度）
        
        Parameters:
        -----------
        df : DataFrame
            包含ax, ay, az的数据框
            
        Returns:
        --------
        np.array : 加速度模值
        """
        acc_magnitude = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        return acc_magnitude
    
    def analyze_noise(self, signal_data, title="信号"):
        """
        分析信号的噪声特性
        
        Parameters:
        -----------
        signal_data : array-like
            信号数据
        title : str
            图表标题
        """
        print(f"\n{title} 噪声分析:")
        print(f"  均值: {np.mean(signal_data):.4f}")
        print(f"  标准差: {np.std(signal_data):.4f}")
        print(f"  最大值: {np.max(signal_data):.4f}")
        print(f"  最小值: {np.min(signal_data):.4f}")
        
        # 频域分析
        N = len(signal_data)
        yf = fft(signal_data)
        xf = fftfreq(N, 1/self.sampling_rate)[:N//2]
        power = 2.0/N * np.abs(yf[:N//2])
        
        # 找到主频率
        main_freq_idx = np.argmax(power[1:]) + 1  # 跳过DC分量
        main_freq = xf[main_freq_idx]
        
        print(f"  主频率: {main_freq:.2f} Hz")
        print(f"  功率: {power[main_freq_idx]:.4f}")
        
        return {
            'mean': np.mean(signal_data),
            'std': np.std(signal_data),
            'main_freq': main_freq,
            'power': power[main_freq_idx]
        }
    
    def apply_lowpass_filter(self, signal_data, cutoff_freq=5.0, order=4):
        """
        应用低通滤波器去除高频噪声
        
        Parameters:
        -----------
        signal_data : array-like
            原始信号
        cutoff_freq : float
            截止频率 (Hz)
        order : int
            滤波器阶数
            
        Returns:
        --------
        np.array : 滤波后的信号
        """
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = signal.filtfilt(b, a, signal_data)
        return filtered_signal
    
    def apply_bandpass_filter(self, signal_data, low_freq=0.5, high_freq=5.0, order=4):
        """
        应用带通滤波器（保留行走频率范围）
        
        Parameters:
        -----------
        signal_data : array-like
            原始信号
        low_freq : float
            低截止频率 (Hz)
        high_freq : float
            高截止频率 (Hz)
        order : int
            滤波器阶数
            
        Returns:
        --------
        np.array : 滤波后的信号
        """
        nyquist = 0.5 * self.sampling_rate
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = signal.butter(order, [low, high], btype='band', analog=False)
        filtered_signal = signal.filtfilt(b, a, signal_data)
        return filtered_signal
    
    def moving_average(self, signal_data, window_size=5):
        """
        移动平均滤波
        
        Parameters:
        -----------
        signal_data : array-like
            原始信号
        window_size : int
            窗口大小
            
        Returns:
        --------
        np.array : 滤波后的信号
        """
        return np.convolve(signal_data, np.ones(window_size)/window_size, mode='same')
    
    def visualize_preprocessing(self, df, save_path=None):
        """
        可视化预处理过程
        
        Parameters:
        -----------
        df : DataFrame
            IMU数据
        save_path : str, optional
            保存路径
        """
        # 计算加速度模值
        acc_mag = self.calculate_magnitude(df)
        
        # 应用不同的滤波方法
        filtered_lowpass = self.apply_lowpass_filter(acc_mag, cutoff_freq=5.0)
        filtered_bandpass = self.apply_bandpass_filter(acc_mag, low_freq=0.5, high_freq=5.0)
        filtered_ma = self.moving_average(acc_mag, window_size=5)
        
        # 创建子图
        fig, axes = plt.subplots(5, 1, figsize=(14, 12))
        
        time = df['timestamp'].values
        
        # 原始三轴加速度
        axes[0].plot(time, df['ax'], label='ax', alpha=0.7)
        axes[0].plot(time, df['ay'], label='ay', alpha=0.7)
        axes[0].plot(time, df['az'], label='az', alpha=0.7)
        axes[0].set_ylabel('加速度 (m/s²)')
        axes[0].set_title('原始三轴加速度')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 原始合成加速度
        axes[1].plot(time, acc_mag, label='原始信号', color='blue', alpha=0.7)
        axes[1].set_ylabel('加速度模值 (m/s²)')
        axes[1].set_title('原始合成加速度')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 低通滤波
        axes[2].plot(time, filtered_lowpass, label='低通滤波 (5Hz)', color='green', linewidth=2)
        axes[2].set_ylabel('加速度模值 (m/s²)')
        axes[2].set_title('低通滤波结果')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 带通滤波
        axes[3].plot(time, filtered_bandpass, label='带通滤波 (0.5-5Hz)', color='red', linewidth=2)
        axes[3].set_ylabel('加速度模值 (m/s²)')
        axes[3].set_title('带通滤波结果（推荐用于计步）')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # 移动平均
        axes[4].plot(time, filtered_ma, label='移动平均', color='orange', linewidth=2)
        axes[4].set_ylabel('加速度模值 (m/s²)')
        axes[4].set_xlabel('时间 (秒)')
        axes[4].set_title('移动平均滤波结果')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 预处理可视化已保存: {save_path}")
        
        plt.show()
        
        return {
            'original': acc_mag,
            'lowpass': filtered_lowpass,
            'bandpass': filtered_bandpass,
            'moving_average': filtered_ma
        }
    
    def extract_features(self, signal_data):
        """
        提取信号特征
        
        Parameters:
        -----------
        signal_data : array-like
            信号数据
            
        Returns:
        --------
        dict : 特征字典
        """
        features = {
            'mean': np.mean(signal_data),
            'std': np.std(signal_data),
            'max': np.max(signal_data),
            'min': np.min(signal_data),
            'range': np.max(signal_data) - np.min(signal_data),
            'median': np.median(signal_data),
            'rms': np.sqrt(np.mean(signal_data**2))
        }
        
        return features


def main():
    """
    主函数：演示预处理流程
    """
    import os
    
    print("=" * 60)
    print("数据预处理演示")
    print("=" * 60)
    
    # 检查是否有数据文件
    data_dir = "../data"
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print("\n⚠️  未找到数据文件！")
        print("   请先运行 data_collection.py 采集数据")
        return
    
    # 加载第一个数据文件
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if data_files:
        filepath = os.path.join(data_dir, data_files[0])
        
        preprocessor = DataPreprocessor(sampling_rate=100)
        df = preprocessor.load_data(filepath)
        
        # 噪声分析
        acc_mag = preprocessor.calculate_magnitude(df)
        preprocessor.analyze_noise(acc_mag, title="合成加速度")
        
        # 可视化预处理
        results_dir = "../results"
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "preprocessing_visualization.png")
        
        preprocessor.visualize_preprocessing(df, save_path=save_path)
        
        print("\n✓ 预处理演示完成！")
    else:
        print("\n⚠️  data目录中没有CSV文件")


if __name__ == "__main__":
    main()
