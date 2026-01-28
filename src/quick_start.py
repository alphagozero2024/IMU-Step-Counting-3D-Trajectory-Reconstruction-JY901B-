"""
Quick Start Demo - Complete Workflow
快速开始演示 - 完整工作流程
"""

import os
import sys

def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def get_user_choice(prompt, options):
    """获取用户选择"""
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    while True:
        try:
            choice = input("\n请输入选项编号: ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
            print("无效选项，请重新输入")
        except ValueError:
            print("请输入数字")

def main():
    """完整的演示流程"""
    print_section("IMU计步器项目 - 快速开始演示")
    
    # Step 1: 数据转换
    print_section("步骤 1/5: 转换WitMotion数据格式")
    print("如果您有WitMotion格式的数据文件，需要先转换...")
    
    # 检查sample_data.txt
    sample_file = "../sample_data.txt"
    if os.path.exists(sample_file):
        print(f"\n✓ 发现示例数据: {sample_file}")
        
        try:
            from data_loader import load_witmotion_data
            
            print("正在转换数据格式...")
            df = load_witmotion_data(
                sample_file, 
                scenario_name="sample_walking",
                actual_steps=None  # 如果您知道实际步数，请在这里填写
            )
            
            # 保存转换后的数据
            os.makedirs("../data", exist_ok=True)
            output_file = "../data/sample_data.csv"
            df.to_csv(output_file, index=False)
            print(f"\n✓ 数据已转换并保存: {output_file}")
            
        except Exception as e:
            print(f"✗ 数据转换失败: {e}")
            return
    else:
        print(f"\n⚠️  未找到 {sample_file}")
        print("   跳过数据转换步骤...")
    
    # Step 2: 检查数据目录
    print_section("步骤 2/5: 检查数据文件")
    
    data_dir = "../data"
    if not os.path.exists(data_dir):
        print(f"⚠️  数据目录不存在，创建中...")
        os.makedirs(data_dir)
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')] if os.path.exists(data_dir) else []
    
    if csv_files:
        print(f"\n✓ 找到 {len(csv_files)} 个CSV数据文件:")
        for f in csv_files:
            print(f"  - {f}")
    else:
        print("\n⚠️  data目录中没有CSV文件")
        print("   请使用 data_loader.py 转换您的IMU数据")
        print("   或者使用 data_collection.py 采集新数据")
        return
    
    # Step 3: 数据预处理
    print_section("步骤 3/5: 数据预处理和滤波")
    
    try:
        from preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor(sampling_rate=100)
        filepath = os.path.join(data_dir, csv_files[0])
        
        print(f"\n处理文件: {csv_files[0]}")
        df = preprocessor.load_data(filepath)
        
        # 分析噪声
        acc_mag = preprocessor.calculate_magnitude(df)
        print("\n正在分析信号特性...")
        preprocessor.analyze_noise(acc_mag, title="合成加速度")
        
        print("\n✓ 预处理分析完成")
        print("   提示: 运行 'python preprocessing.py' 可查看详细的滤波对比图")
        
    except Exception as e:
        print(f"✗ 预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: 计步检测 - 选择方法
    print_section("步骤 4/5: 计步检测")
    
    print("\n可用的计步方法:")
    methods = [
        ('peak', '峰值检测法 (Peak Detection) - 检测信号峰值'),
        ('zero_crossing', '过零检测法 (Zero-Crossing) - 检测信号过零点'),
        ('autocorrelation', '自相关函数法 (Autocorrelation) - 分析信号周期性'),
        ('compare', '比较所有三种方法')
    ]
    
    choice = get_user_choice("\n请选择计步方法 (输入 q 退出):", 
                             [m[1] for m in methods])
    
    if choice is None:
        print("\n已取消")
        return
    
    selected_method = methods[choice][0]
    
    try:
        from step_detection import StepDetector
        
        detector = StepDetector(sampling_rate=100)
        
        print(f"\n开始计步检测 - 使用方法: {methods[choice][1].split(' - ')[0]}")
        print("-" * 70)
        
        results = []
        
        for csv_file in csv_files:
            filepath = os.path.join(data_dir, csv_file)
            print(f"\n处理文件: {csv_file}")
            
            if selected_method == 'compare':
                # 比较所有三种方法
                comparison = detector.compare_methods(
                    filepath, 
                    filter_type='bandpass',
                    save_visualization=True
                )
                if comparison:
                    results.append(comparison)
            else:
                # 使用单一方法
                result = detector.process_file(
                    filepath, 
                    method=selected_method, 
                    filter_type='bandpass',
                    save_visualization=True
                )
                if result:
                    results.append(result)
        
        print("\n✓ 计步检测完成")
        print("   检测结果和可视化图已保存到 results/ 目录")
        
    except Exception as e:
        print(f"✗ 计步检测失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: 结果汇总
    print_section("步骤 5/5: 结果汇总")
    
    if results:
        if selected_method == 'compare':
            # 显示三种方法的比较结果
            print("\n三种方法检测结果比较:")
            print("-" * 90)
            print(f"{'场景':<15} {'实际':<8} {'峰值检测':<12} {'过零检测':<12} {'自相关法':<12}")
            print("-" * 90)
            
            for comp in results:
                scenario = (comp['scenario'][:13] if comp['scenario'] else 'unknown')
                actual = comp['actual_steps'] if comp['actual_steps'] else 'N/A'
                
                peak_res = comp['results']['peak']
                zc_res = comp['results']['zero_crossing']
                ac_res = comp['results']['autocorrelation']
                
                def fmt_result(r):
                    if r['accuracy'] is not None:
                        return f"{r['detected_steps']} ({r['accuracy']:.0f}%)"
                    return f"{r['detected_steps']}"
                
                print(f"{scenario:<15} {str(actual):<8} {fmt_result(peak_res):<12} {fmt_result(zc_res):<12} {fmt_result(ac_res):<12}")
            
            print("-" * 90)
        else:
            # 显示单一方法结果
            print("\n检测结果汇总:")
            print("-" * 70)
            print(f"{'场景':<20} {'实际步数':<10} {'检测步数':<10} {'误差':<10} {'准确率'}")
            print("-" * 70)
            
            for r in results:
                scenario = r['scenario'][:18] if r['scenario'] else 'unknown'
                actual = r['actual_steps'] if r['actual_steps'] else 'N/A'
                detected = r['detected_steps']
                error = r['error'] if r['error'] is not None else 'N/A'
                accuracy = f"{r['accuracy']:.1f}%" if r['accuracy'] is not None else 'N/A'
                
                actual_str = str(actual) if actual != 'N/A' else actual
                error_str = f"{error:+d}" if error != 'N/A' else error
                
                print(f"{scenario:<20} {actual_str:<10} {detected:<10} {error_str:<10} {accuracy}")
            
            print("-" * 70)
    
    # 完成提示
    print_section("演示完成！")
    
    print("\n✓ 快速开始演示已完成！")
    print("\n接下来您可以：")
    print("  1. 查看 results/ 目录中的可视化结果")
    print("  2. 运行 'python evaluation.py' 进行完整评估")
    print("  3. 运行 'python trajectory_generation.py' 生成3D轨迹（选做）")
    print("  4. 采集更多场景的数据进行测试")
    
    print("\n📚 详细使用说明请参考:")
    print("  - USAGE_GUIDE.md - 完整使用指南")
    print("  - README.md - 项目文档")
    print("  - HARDWARE_SETUP.md - 硬件连接指南")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
