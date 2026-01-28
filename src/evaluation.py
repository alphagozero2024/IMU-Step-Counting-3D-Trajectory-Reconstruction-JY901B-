"""
Evaluation and Testing Module
评估和测试模块：多场景测试和性能评估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from step_detection import StepDetector
import os
import glob

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class StepCounterEvaluator:
    """
    计步器评估器
    用于多场景测试和性能评估
    """
    
    def __init__(self, sampling_rate=100):
        """
        初始化评估器
        
        Parameters:
        -----------
        sampling_rate : int
            采样频率 (Hz)
        """
        self.detector = StepDetector(sampling_rate)
        self.results = []
        
    def evaluate_single_file(self, filepath, method='peak', filter_type='bandpass'):
        """
        评估单个数据文件
        
        Parameters:
        -----------
        filepath : str
            数据文件路径
        method : str
            检测方法
        filter_type : str
            滤波类型
            
        Returns:
        --------
        dict : 评估结果
        """
        result = self.detector.process_file(
            filepath, 
            method=method, 
            filter_type=filter_type,
            save_visualization=True
        )
        
        self.results.append(result)
        return result
    
    def evaluate_directory(self, data_dir, method='peak', filter_type='bandpass'):
        """
        评估目录中的所有数据文件
        
        Parameters:
        -----------
        data_dir : str
            数据目录路径
        method : str
            检测方法
        filter_type : str
            滤波类型
            
        Returns:
        --------
        DataFrame : 所有文件的评估结果
        """
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        if not csv_files:
            print("⚠️  未找到CSV数据文件")
            return None
        
        print(f"\n找到 {len(csv_files)} 个数据文件")
        print("=" * 60)
        
        self.results = []
        
        for i, filepath in enumerate(csv_files, 1):
            print(f"\n处理文件 {i}/{len(csv_files)}: {os.path.basename(filepath)}")
            self.evaluate_single_file(filepath, method, filter_type)
        
        return pd.DataFrame(self.results)
    
    def calculate_metrics(self, df_results):
        """
        计算评估指标
        
        Parameters:
        -----------
        df_results : DataFrame
            评估结果
            
        Returns:
        --------
        dict : 评估指标
        """
        if df_results is None or len(df_results) == 0:
            return None
        
        # 过滤掉没有实际步数的记录
        df_valid = df_results[df_results['actual_steps'].notna()].copy()
        
        if len(df_valid) == 0:
            print("⚠️  没有包含实际步数的数据")
            return None
        
        # 计算各项指标
        metrics = {
            'total_samples': len(df_valid),
            'mean_accuracy': df_valid['accuracy'].mean(),
            'std_accuracy': df_valid['accuracy'].std(),
            'min_accuracy': df_valid['accuracy'].min(),
            'max_accuracy': df_valid['accuracy'].max(),
            'mean_absolute_error': np.abs(df_valid['error']).mean(),
            'mean_error': df_valid['error'].mean(),
            'std_error': df_valid['error'].std()
        }
        
        # 按场景统计
        scenario_stats = df_valid.groupby('scenario').agg({
            'accuracy': ['mean', 'std', 'count'],
            'error': ['mean', 'std']
        }).round(2)
        
        metrics['scenario_stats'] = scenario_stats
        
        return metrics
    
    def visualize_results(self, df_results, save_path=None):
        """
        可视化评估结果
        
        Parameters:
        -----------
        df_results : DataFrame
            评估结果
        save_path : str, optional
            保存路径
        """
        if df_results is None or len(df_results) == 0:
            print("⚠️  没有结果可以可视化")
            return
        
        df_valid = df_results[df_results['actual_steps'].notna()].copy()
        
        if len(df_valid) == 0:
            print("⚠️  没有包含实际步数的数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 实际步数 vs 检测步数
        ax1 = axes[0, 0]
        ax1.scatter(df_valid['actual_steps'], df_valid['detected_steps'], 
                   s=100, alpha=0.6, edgecolors='black')
        
        # 添加理想线
        max_steps = max(df_valid['actual_steps'].max(), df_valid['detected_steps'].max())
        ax1.plot([0, max_steps], [0, max_steps], 'r--', linewidth=2, label='理想情况')
        
        ax1.set_xlabel('实际步数', fontsize=12)
        ax1.set_ylabel('检测步数', fontsize=12)
        ax1.set_title('实际步数 vs 检测步数', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率分布
        ax2 = axes[0, 1]
        ax2.hist(df_valid['accuracy'], bins=15, color='skyblue', 
                edgecolor='black', alpha=0.7)
        ax2.axvline(df_valid['accuracy'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'均值: {df_valid["accuracy"].mean():.1f}%')
        ax2.set_xlabel('准确率 (%)', fontsize=12)
        ax2.set_ylabel('频数', fontsize=12)
        ax2.set_title('准确率分布', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 误差分布
        ax3 = axes[1, 0]
        colors = ['green' if e >= 0 else 'red' for e in df_valid['error']]
        ax3.bar(range(len(df_valid)), df_valid['error'], color=colors, alpha=0.6, edgecolor='black')
        ax3.axhline(0, color='black', linewidth=1)
        ax3.set_xlabel('样本索引', fontsize=12)
        ax3.set_ylabel('误差 (步)', fontsize=12)
        ax3.set_title('检测误差分布', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 按场景的准确率
        ax4 = axes[1, 1]
        if 'scenario' in df_valid.columns:
            scenario_acc = df_valid.groupby('scenario')['accuracy'].mean().sort_values(ascending=False)
            colors_bar = plt.cm.viridis(np.linspace(0, 1, len(scenario_acc)))
            bars = ax4.barh(range(len(scenario_acc)), scenario_acc.values, color=colors_bar, 
                           edgecolor='black', alpha=0.8)
            ax4.set_yticks(range(len(scenario_acc)))
            ax4.set_yticklabels(scenario_acc.index, fontsize=10)
            ax4.set_xlabel('平均准确率 (%)', fontsize=12)
            ax4.set_title('各场景准确率对比', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, scenario_acc.values)):
                ax4.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 评估结果可视化已保存: {save_path}")
        
        plt.show()
    
    def generate_report(self, df_results, metrics, save_path=None):
        """
        生成评估报告
        
        Parameters:
        -----------
        df_results : DataFrame
            评估结果
        metrics : dict
            评估指标
        save_path : str, optional
            保存路径
        """
        if metrics is None:
            print("⚠️  没有评估指标可以生成报告")
            return
        
        report = []
        report.append("=" * 70)
        report.append("计步器性能评估报告")
        report.append("=" * 70)
        report.append("")
        
        # 总体性能
        report.append("【总体性能】")
        report.append(f"  测试样本数: {metrics['total_samples']}")
        report.append(f"  平均准确率: {metrics['mean_accuracy']:.2f}%")
        report.append(f"  准确率标准差: {metrics['std_accuracy']:.2f}%")
        report.append(f"  最高准确率: {metrics['max_accuracy']:.2f}%")
        report.append(f"  最低准确率: {metrics['min_accuracy']:.2f}%")
        report.append("")
        
        # 误差分析
        report.append("【误差分析】")
        report.append(f"  平均绝对误差: {metrics['mean_absolute_error']:.2f} 步")
        report.append(f"  平均误差: {metrics['mean_error']:.2f} 步")
        report.append(f"  误差标准差: {metrics['std_error']:.2f} 步")
        report.append("")
        
        # 场景统计
        if 'scenario_stats' in metrics:
            report.append("【分场景统计】")
            report.append(str(metrics['scenario_stats']))
            report.append("")
        
        # 详细结果
        report.append("【详细测试结果】")
        df_valid = df_results[df_results['actual_steps'].notna()].copy()
        for _, row in df_valid.iterrows():
            report.append(f"  场景: {row['scenario']}")
            report.append(f"    实际步数: {row['actual_steps']}")
            report.append(f"    检测步数: {row['detected_steps']}")
            report.append(f"    误差: {row['error']:+d}")
            report.append(f"    准确率: {row['accuracy']:.2f}%")
            report.append("")
        
        report.append("=" * 70)
        
        # 打印报告
        report_text = "\n".join(report)
        print(report_text)
        
        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n✓ 评估报告已保存: {save_path}")
        
        return report_text


def main():
    """
    主函数：执行完整的评估流程
    """
    print("=" * 70)
    print("计步器多场景测试与评估")
    print("=" * 70)
    
    data_dir = "../data"
    results_dir = "../results"
    
    if not os.path.exists(data_dir):
        print("\n⚠️  数据目录不存在！")
        print("   请先运行 data_collection.py 采集数据")
        return
    
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建评估器
    evaluator = StepCounterEvaluator(sampling_rate=100)
    
    # 评估所有数据文件
    print("\n开始评估...")
    df_results = evaluator.evaluate_directory(data_dir, method='peak', filter_type='bandpass')
    
    if df_results is not None and len(df_results) > 0:
        # 计算指标
        print("\n" + "=" * 70)
        print("计算评估指标...")
        print("=" * 70)
        metrics = evaluator.calculate_metrics(df_results)
        
        # 可视化结果
        print("\n生成可视化结果...")
        viz_path = os.path.join(results_dir, "evaluation_summary.png")
        evaluator.visualize_results(df_results, save_path=viz_path)
        
        # 生成报告
        print("\n生成评估报告...")
        report_path = os.path.join(results_dir, "evaluation_report.txt")
        evaluator.generate_report(df_results, metrics, save_path=report_path)
        
        # 保存结果数据
        csv_path = os.path.join(results_dir, "evaluation_results.csv")
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 结果数据已保存: {csv_path}")
        
        print("\n" + "=" * 70)
        print("✓ 评估完成！")
        print("=" * 70)
    else:
        print("\n⚠️  没有有效的评估结果")


if __name__ == "__main__":
    main()
