"""
Step Detection Algorithm
计步算法模块：基于峰值检测的计步算法
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import font_manager
from preprocessing import DataPreprocessor
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class StepDetector:
    """
    计步检测器
    使用峰值检测算法识别步数
    """
    
    def __init__(self, sampling_rate=100):
        """
        初始化计步器
        
        Parameters:
        -----------
        sampling_rate : int
            采样频率 (Hz)
        """
        self.sampling_rate = sampling_rate
        self.preprocessor = DataPreprocessor(sampling_rate)

    def _estimate_step_distance(self, signal_data, min_freq=0.5, max_freq=4.0,
                                height=None, prominence=None):
        """自适应估计最小峰距，优先用粗检测间距，无法估计时退回频域主频。"""
        min_spacing = max(1, int(self.sampling_rate / max_freq))
        max_spacing = max(min_spacing, int(self.sampling_rate / min_freq))

        # 粗检测峰值以估计实际间距，避免因低估步频导致距离过大漏检
        rough_peaks, _ = signal.find_peaks(signal_data, height=height, prominence=prominence)
        if len(rough_peaks) >= 2:
            spacing = np.diff(rough_peaks)
            median_spacing = int(np.median(spacing)) if len(spacing) > 0 else min_spacing
            return int(np.clip(median_spacing, min_spacing, max_spacing))

        # 若缺少足够峰值，用 Welch PSD 估计主导步频
        centered = signal_data - np.mean(signal_data)
        if len(centered) < 2:
            return min_spacing

        nperseg = min(256, len(centered))
        freqs, psd = signal.welch(centered, fs=self.sampling_rate, nperseg=nperseg)
        valid = (freqs >= min_freq) & (freqs <= max_freq)

        if not np.any(valid):
            return min_spacing

        dominant_freq = freqs[valid][np.argmax(psd[valid])]
        dominant_freq = max(min(dominant_freq, max_freq), min_freq)
        est_spacing = int(self.sampling_rate / dominant_freq)
        return int(np.clip(est_spacing, min_spacing, max_spacing))
        
    def detect_steps(self, signal_data, method='peak', **kwargs):
        """
        检测步数
        
        Parameters:
        -----------
        signal_data : array-like
            加速度信号（通常是滤波后的合成加速度）
        method : str
            检测方法 ('peak', 'threshold', 'adaptive', 'zero_crossing', 'autocorrelation')
        **kwargs : dict
            方法特定参数
            
        Returns:
        --------
        tuple : (步数, 峰值索引)
        """
        if method == 'peak':
            return self._detect_by_peaks(signal_data, **kwargs)
        elif method == 'threshold':
            return self._detect_by_threshold(signal_data, **kwargs)
        elif method == 'adaptive':
            return self._detect_adaptive(signal_data, **kwargs)
        elif method == 'zero_crossing':
            return self._detect_by_zero_crossing(signal_data, **kwargs)
        elif method == 'autocorrelation':
            return self._detect_by_autocorrelation(signal_data, **kwargs)
        else:
            raise ValueError(f"未知的检测方法: {method}")
    
    def _detect_by_peaks(self, signal_data, height=None, distance=None, prominence=None,
                        min_freq=0.5, max_freq=4.0, adaptive_distance=True):
        """
        基于峰值检测的计步算法
        
        Parameters:
        -----------
        signal_data : array-like
            信号数据
        height : float, optional
            最小峰值高度
        distance : int, optional
            峰值之间的最小距离（样本数）
        prominence : float, optional
            峰值突出度
        min_freq : float, optional
            预期最小步频(Hz)，用于自适应峰距估计
        max_freq : float, optional
            预期最大步频(Hz)，用于自适应峰距估计
        adaptive_distance : bool, optional
            是否根据主导步频动态调整最小峰距
            
        Returns:
        --------
        tuple : (步数, 峰值索引)
        """
        # 默认参数设置
        if height is None:
            height = np.mean(signal_data) + 0.5 * np.std(signal_data)

        if prominence is None:
            prominence = 0.3 * np.std(signal_data)
        
        if distance is None:
            if adaptive_distance:
                distance = self._estimate_step_distance(
                    signal_data,
                    min_freq=min_freq,
                    max_freq=max_freq,
                    height=height,
                    prominence=prominence
                )
            else:
                distance = int(self.sampling_rate / max_freq)
        
        # 使用scipy的find_peaks函数
        peaks, properties = signal.find_peaks(
            signal_data,
            height=height,
            distance=distance,
            prominence=prominence
        )
        
        step_count = len(peaks)
        
        return step_count, peaks, properties
    
    def _detect_by_threshold(self, signal_data, threshold=None, min_interval=0.3):
        """
        基于阈值的计步算法
        
        Parameters:
        -----------
        signal_data : array-like
            信号数据
        threshold : float, optional
            阈值
        min_interval : float
            最小步间隔（秒）
            
        Returns:
        --------
        tuple : (步数, 峰值索引)
        """
        if threshold is None:
            threshold = np.mean(signal_data) + 0.5 * np.std(signal_data)
        
        min_samples = int(min_interval * self.sampling_rate)
        
        peaks = []
        last_peak = -min_samples
        
        for i in range(1, len(signal_data) - 1):
            # 检测上升沿
            if (signal_data[i] > threshold and 
                signal_data[i] > signal_data[i-1] and 
                signal_data[i] >= signal_data[i+1] and
                i - last_peak >= min_samples):
                peaks.append(i)
                last_peak = i
        
        return len(peaks), np.array(peaks), {}
    
    def _detect_adaptive(self, signal_data, window_size=100):
        """
        自适应阈值计步算法
        
        Parameters:
        -----------
        signal_data : array-like
            信号数据
        window_size : int
            滑动窗口大小
            
        Returns:
        --------
        tuple : (步数, 峰值索引)
        """
        peaks = []
        min_samples = int(0.3 * self.sampling_rate)  # 最小步间隔
        last_peak = -min_samples
        
        for i in range(window_size, len(signal_data) - 1):
            # 计算局部窗口的统计量
            window = signal_data[i-window_size:i]
            local_mean = np.mean(window)
            local_std = np.std(window)
            threshold = local_mean + 0.5 * local_std
            
            # 检测峰值
            if (signal_data[i] > threshold and 
                signal_data[i] > signal_data[i-1] and 
                signal_data[i] >= signal_data[i+1] and
                i - last_peak >= min_samples):
                peaks.append(i)
                last_peak = i
        
        return len(peaks), np.array(peaks), {}
    
    def _detect_by_zero_crossing(self, signal_data, min_interval=0.3):
        """
        过零检测计步算法
        基于信号穿过均值线的次数来计步
        
        Parameters:
        -----------
        signal_data : array-like
            信号数据
        min_interval : float
            最小步间隔（秒）
            
        Returns:
        --------
        tuple : (步数, 过零点索引)
        """
        # 去除直流分量（减去均值）
        centered_signal = signal_data - np.mean(signal_data)
        
        min_samples = int(min_interval * self.sampling_rate)
        
        crossings = []
        last_crossing = -min_samples
        
        # 检测从负到正的过零点（上升沿）
        for i in range(1, len(centered_signal)):
            if (centered_signal[i-1] < 0 and centered_signal[i] >= 0 and
                i - last_crossing >= min_samples):
                crossings.append(i)
                last_crossing = i
        
        # 步数为过零次数的一半（一个完整步态周期包含两次过零）
        # 但我们只检测上升沿，所以直接返回过零次数
        step_count = len(crossings)
        
        return step_count, np.array(crossings), {'type': 'zero_crossing'}
    
    def _detect_by_autocorrelation(self, signal_data, min_freq=0.5, max_freq=3.0):
        """
        自相关函数计步算法
        通过分析信号的周期性来估计步数
        
        Parameters:
        -----------
        signal_data : array-like
            信号数据
        min_freq : float
            最小步频 (Hz)
        max_freq : float
            最大步频 (Hz)
            
        Returns:
        --------
        tuple : (步数, 估计的峰值位置)
        """
        # 去除直流分量
        centered_signal = signal_data - np.mean(signal_data)
        
        # 计算自相关函数
        n = len(centered_signal)
        autocorr = np.correlate(centered_signal, centered_signal, mode='full')
        autocorr = autocorr[n-1:]  # 取正半轴
        autocorr = autocorr / autocorr[0]  # 归一化
        
        # 步频范围对应的lag范围
        min_lag = int(self.sampling_rate / max_freq)  # 对应最大步频
        max_lag = int(self.sampling_rate / min_freq)  # 对应最小步频
        
        # 在有效范围内找第一个峰值（对应步态周期）
        search_region = autocorr[min_lag:min(max_lag, len(autocorr))]
        
        if len(search_region) == 0:
            return 0, np.array([]), {'period': 0, 'step_freq': 0}
        
        # 找到自相关函数的第一个主峰
        peaks, _ = signal.find_peaks(search_region, height=0.1)
        
        if len(peaks) == 0:
            # 如果没有找到峰值，使用最大值位置
            peak_idx = np.argmax(search_region)
        else:
            peak_idx = peaks[0]
        
        # 步态周期（样本数）
        step_period = min_lag + peak_idx
        
        # 步频 (Hz)
        step_freq = self.sampling_rate / step_period if step_period > 0 else 0
        
        # 总时长
        duration = len(signal_data) / self.sampling_rate
        
        # 估计步数
        step_count = int(duration * step_freq)
        
        # 生成估计的步态位置（用于可视化）
        estimated_peaks = np.arange(step_period // 2, len(signal_data), step_period).astype(int)
        estimated_peaks = estimated_peaks[estimated_peaks < len(signal_data)]
        
        return step_count, estimated_peaks, {
            'period': step_period / self.sampling_rate,
            'step_freq': step_freq,
            'autocorr': autocorr
        }
    
    def visualize_detection(self, time, signal_data, peaks, actual_steps=None, 
                          title="计步检测结果", save_path=None):
        """
        可视化计步检测结果
        
        Parameters:
        -----------
        time : array-like
            时间序列
        signal_data : array-like
            信号数据
        peaks : array-like
            检测到的峰值索引
        actual_steps : int, optional
            实际步数
        title : str
            图表标题
        save_path : str, optional
            保存路径
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 绘制信号
        ax.plot(time, signal_data, label='滤波后信号', linewidth=1.5, color='blue', alpha=0.7)
        
        # 标记检测到的峰值
        if len(peaks) > 0:
            ax.plot(time[peaks], signal_data[peaks], 'ro', 
                   label=f'检测到的步数: {len(peaks)}', markersize=8)
        
        # 绘制阈值线
        threshold = np.mean(signal_data) + 0.5 * np.std(signal_data)
        ax.axhline(y=threshold, color='green', linestyle='--', 
                  label=f'阈值: {threshold:.2f}', alpha=0.7)
        
        ax.set_xlabel('时间 (秒)', fontsize=12)
        ax.set_ylabel('加速度模值 (m/s²)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加准确度信息
        if actual_steps is not None and pd.notna(actual_steps):
            detected_steps = len(peaks)
            accuracy = (1 - abs(detected_steps - actual_steps) / actual_steps) * 100
            error = detected_steps - actual_steps
            
            textstr = f'实际步数: {actual_steps}\n检测步数: {detected_steps}\n误差: {int(error):+d}\n准确率: {accuracy:.1f}%'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 检测结果已保存: {save_path}")
        
        plt.show()
    
    def compare_methods(self, filepath, filter_type='bandpass', save_visualization=True):
        """
        比较三种计步方法的结果
        
        Parameters:
        -----------
        filepath : str
            数据文件路径（CSV格式）
        filter_type : str
            滤波类型 ('bandpass', 'lowpass', 'moving_average')
        save_visualization : bool
            是否保存可视化结果
            
        Returns:
        --------
        dict : 各方法的检测结果比较
        """
        # 加载数据
        try:
            df = self.preprocessor.load_data(filepath)
        except Exception as e:
            print(f"❌ 无法加载文件: {filepath}")
            return None
        
        # 计算合成加速度
        acc_mag = self.preprocessor.calculate_magnitude(df)
        
        # 应用滤波
        if filter_type == 'bandpass':
            filtered_signal = self.preprocessor.apply_bandpass_filter(acc_mag)
        elif filter_type == 'lowpass':
            filtered_signal = self.preprocessor.apply_lowpass_filter(acc_mag)
        elif filter_type == 'moving_average':
            filtered_signal = self.preprocessor.moving_average(acc_mag)
        else:
            filtered_signal = acc_mag
        
        # 获取实际步数
        actual_steps = df['actual_steps'].iloc[0] if 'actual_steps' in df.columns else None
        scenario = df['scenario'].iloc[0] if 'scenario' in df.columns else "unknown"
        
        # 定义三种方法
        methods = ['peak', 'zero_crossing', 'autocorrelation']
        method_names = {
            'peak': '峰值检测法',
            'zero_crossing': '过零检测法',
            'autocorrelation': '自相关函数法'
        }
        
        results = {}
        
        print(f"\n{'='*70}")
        print(f"三种计步方法比较 - 场景: {scenario}")
        print(f"{'='*70}")
        
        for method in methods:
            detected_steps, peaks, properties = self.detect_steps(filtered_signal, method=method)
            
            if actual_steps is not None and pd.notna(actual_steps):
                accuracy = (1 - abs(detected_steps - actual_steps) / actual_steps) * 100
                error = detected_steps - actual_steps
            else:
                accuracy = None
                error = None
            
            results[method] = {
                'name': method_names[method],
                'detected_steps': detected_steps,
                'error': error,
                'accuracy': accuracy,
                'peaks': peaks,
                'properties': properties
            }
            
            print(f"\n{method_names[method]}:")
            print(f"  检测步数: {detected_steps}")
            if accuracy is not None:
                print(f"  误差: {int(error):+d}")
                print(f"  准确率: {accuracy:.1f}%")
            if 'step_freq' in properties:
                print(f"  估计步频: {properties['step_freq']:.2f} Hz")
        
        # 比较结果可视化
        if save_visualization:
            self._visualize_comparison(
                df['timestamp'].values,
                filtered_signal,
                results,
                actual_steps,
                scenario,
                filepath
            )
        
        # 找出最佳方法
        if actual_steps is not None:
            best_method = max(results.keys(), key=lambda m: results[m]['accuracy'] if results[m]['accuracy'] else 0)
            print(f"\n{'='*70}")
            print(f"最佳方法: {method_names[best_method]} (准确率: {results[best_method]['accuracy']:.1f}%)")
            print(f"{'='*70}")
        
        return {
            'filepath': filepath,
            'scenario': scenario,
            'actual_steps': actual_steps,
            'results': results
        }
    
    def _visualize_comparison(self, time, signal_data, results, actual_steps, scenario, filepath):
        """
        可视化三种方法的比较结果
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        methods = ['peak', 'zero_crossing', 'autocorrelation']
        colors = ['red', 'green', 'purple']
        
        for idx, method in enumerate(methods):
            ax = axes[idx]
            result = results[method]
            peaks = result['peaks']
            
            # 绘制信号
            ax.plot(time, signal_data, label='滤波后信号', linewidth=1, color='blue', alpha=0.6)
            
            # 标记检测到的点
            if len(peaks) > 0:
                ax.plot(time[peaks], signal_data[peaks], 'o', color=colors[idx],
                       label=f'检测点: {result["detected_steps"]}步', markersize=6)
            
            # 阈值线（仅对峰值检测）
            if method == 'peak':
                threshold = np.mean(signal_data) + 0.5 * np.std(signal_data)
                ax.axhline(y=threshold, color='orange', linestyle='--', 
                          label=f'阈值: {threshold:.2f}', alpha=0.7)
            elif method == 'zero_crossing':
                ax.axhline(y=np.mean(signal_data), color='orange', linestyle='--',
                          label='均值线', alpha=0.7)
            
            ax.set_ylabel('加速度模值 (m/s²)', fontsize=10)
            ax.set_title(f'{result["name"]} - 检测: {result["detected_steps"]}步' + 
                        (f', 准确率: {result["accuracy"]:.1f}%' if result["accuracy"] else ''),
                        fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('时间 (秒)', fontsize=10)
        
        # 添加总体信息
        if actual_steps is not None and pd.notna(actual_steps):
            fig.suptitle(f'三种计步方法比较 - {scenario} (实际步数: {actual_steps})', 
                        fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # 保存图片
        results_dir = "../results"
        os.makedirs(results_dir, exist_ok=True)
        filename = os.path.basename(filepath).replace('.csv', '')
        save_path = os.path.join(results_dir, f"{filename}_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 比较结果已保存: {save_path}")
        
        plt.show()
    
    def process_file(self, filepath, method='peak', filter_type='bandpass', 
                    save_visualization=True):
        """
        处理单个数据文件并进行计步
        
        Parameters:
        -----------
        filepath : str
            数据文件路径（CSV格式）
        method : str
            检测方法 ('peak', 'threshold', 'adaptive')
        filter_type : str
            滤波类型 ('bandpass', 'lowpass', 'moving_average')
        save_visualization : bool
            是否保存可视化结果
            
        Returns:
        --------
        dict : 检测结果
        """
        # 加载数据
        try:
            df = self.preprocessor.load_data(filepath)
        except Exception as e:
            print(f"❌ 无法加载文件: {filepath}")
            return None
        
        # 计算合成加速度
        acc_mag = self.preprocessor.calculate_magnitude(df)
        
        # 应用滤波
        if filter_type == 'bandpass':
            filtered_signal = self.preprocessor.apply_bandpass_filter(acc_mag)
        elif filter_type == 'lowpass':
            filtered_signal = self.preprocessor.apply_lowpass_filter(acc_mag)
        elif filter_type == 'moving_average':
            filtered_signal = self.preprocessor.moving_average(acc_mag)
        else:
            filtered_signal = acc_mag
        
        # 检测步数
        detected_steps, peaks, properties = self.detect_steps(filtered_signal, method=method)
        
        # 获取实际步数
        actual_steps = df['actual_steps'].iloc[0] if 'actual_steps' in df.columns else None
        scenario = df['scenario'].iloc[0] if 'scenario' in df.columns else "unknown"
        
        # 计算准确度
        if actual_steps is not None and pd.notna(actual_steps):
            accuracy = (1 - abs(detected_steps - actual_steps) / actual_steps) * 100
            error = detected_steps - actual_steps
        else:
            accuracy = None
            error = None
        
        print(f"\n{'='*60}")
        print(f"场景: {scenario}")
        print(f"实际步数: {actual_steps}")
        print(f"检测步数: {detected_steps}")
        if accuracy is not None:
            print(f"误差: {int(error):+d}")
            print(f"准确率: {accuracy:.1f}%")
        print(f"{'='*60}")
        
        # 可视化
        if save_visualization:
            results_dir = "../results"
            os.makedirs(results_dir, exist_ok=True)
            
            filename = os.path.basename(filepath).replace('.csv', '')
            save_path = os.path.join(results_dir, f"{filename}_detection.png")
            
            self.visualize_detection(
                df['timestamp'].values,
                filtered_signal,
                peaks,
                actual_steps=actual_steps,
                title=f"计步检测结果 - {scenario}",
                save_path=save_path
            )
        
        return {
            'filepath': filepath,
            'scenario': scenario,
            'actual_steps': actual_steps,
            'detected_steps': detected_steps,
            'error': error,
            'accuracy': accuracy,
            'peaks': peaks
        }


def main():
    """
    主函数：演示计步算法并比较三种方法
    """
    print("=" * 70)
    print("计步检测算法演示 - 三种方法比较")
    print("=" * 70)
    print("\n支持的计步方法:")
    print("  1. 峰值检测法 (Peak Detection)")
    print("  2. 过零检测法 (Zero-Crossing Detection)")
    print("  3. 自相关函数法 (Autocorrelation)")
    
    data_dir = "../data"
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print("\n⚠️  未找到数据文件！")
        print("   请先运行 data_collection.py 采集数据")
        return
    
    detector = StepDetector(sampling_rate=100)
    
    # 处理所有数据文件
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    all_comparisons = []
    
    for data_file in data_files:
        filepath = os.path.join(data_dir, data_file)
        comparison = detector.compare_methods(filepath, filter_type='bandpass')
        if comparison:
            all_comparisons.append(comparison)
    
    # 总结所有文件的比较结果
    if all_comparisons:
        print("\n" + "=" * 70)
        print("所有数据文件 - 三种方法准确率汇总")
        print("=" * 70)
        
        summary_data = []
        for comp in all_comparisons:
            if comp['actual_steps'] is not None:
                row = {
                    '场景': comp['scenario'],
                    '实际步数': comp['actual_steps'],
                    '峰值检测': f"{comp['results']['peak']['detected_steps']} ({comp['results']['peak']['accuracy']:.1f}%)",
                    '过零检测': f"{comp['results']['zero_crossing']['detected_steps']} ({comp['results']['zero_crossing']['accuracy']:.1f}%)",
                    '自相关法': f"{comp['results']['autocorrelation']['detected_steps']} ({comp['results']['autocorrelation']['accuracy']:.1f}%)"
                }
                summary_data.append(row)
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            print(df_summary.to_string(index=False))
            
            # 计算各方法平均准确率
            print("\n" + "-" * 70)
            print("平均准确率:")
            for method in ['peak', 'zero_crossing', 'autocorrelation']:
                method_names = {'peak': '峰值检测法', 'zero_crossing': '过零检测法', 'autocorrelation': '自相关函数法'}
                accs = [c['results'][method]['accuracy'] for c in all_comparisons if c['results'][method]['accuracy']]
                if accs:
                    print(f"  {method_names[method]}: {np.mean(accs):.1f}%")
    
    print("\n✓ 计步检测比较完成！")


if __name__ == "__main__":
    main()
