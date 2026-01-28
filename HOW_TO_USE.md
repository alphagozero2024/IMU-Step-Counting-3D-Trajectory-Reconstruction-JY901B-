# 📖 如何使用本项目 - 完整指南

## 🎯 项目概述

本项目实现了基于IMU传感器的智能计步系统，您的数据格式是**WitMotion IMU传感器**格式。

## ⚡ 快速开始（3步完成）

### 1️⃣ 安装依赖

```powershell
cd C:\Users\Administrator\Desktop\Step_Recorder
pip install -r requirements.txt
```

### 2️⃣ 转换数据格式

```powershell
cd src
python data_loader.py
```

这会自动转换 `sample_data.txt` 为标准CSV格式。

### 3️⃣ 运行演示

```powershell
python quick_start.py
```

这将自动完成：数据转换 → 预处理分析 → 计步检测 → 结果展示

## 📋 详细使用流程

### 第一步：准备数据

您有两种方式准备数据：

**方式A：使用现有的sample_data.txt**
```powershell
cd src
python data_loader.py
```

**方式B：转换您自己的WitMotion数据**
```python
from data_loader import convert_witmotion_to_standard

convert_witmotion_to_standard(
    input_file='your_data.txt',
    output_file='../data/your_data.csv',
    scenario_name='walking_flat',  # 场景名称
    actual_steps=50  # 您手动计数的实际步数（重要！）
)
```

**方式C：批量转换多个文件**
```python
from data_loader import batch_convert_directory

# 将 raw_data 文件夹中的所有.txt文件转换为CSV
batch_convert_directory('../raw_data', '../data')
```

### 第二步：查看数据预处理效果

```powershell
python preprocessing.py
```

**输出内容：**
- 信号噪声分析统计
- 5种信号对比图：
  - 原始三轴加速度
  - 原始合成加速度
  - 低通滤波结果
  - 带通滤波结果（推荐）
  - 移动平均滤波结果
- 保存为：`results/preprocessing_visualization.png`

### 第三步：运行计步检测

```powershell
python step_detection.py
```

**输出内容：**
- 每个数据文件的检测结果
- 实际步数 vs 检测步数
- 误差和准确率统计
- 计步依据可视化图（峰值标记）
- 保存为：`results/[文件名]_detection.png`

### 第四步：完整性能评估

```powershell
python evaluation.py
```

**输出内容：**
- 详细评估报告：`results/evaluation_report.txt`
- 汇总可视化图：`results/evaluation_summary.png`
  - 实际vs检测步数散点图
  - 准确率分布直方图
  - 误差分布柱状图
  - 各场景准确率对比
- 结果数据：`results/evaluation_results.csv`

### 第五步：生成3D轨迹（选做）

```powershell
python trajectory_generation.py
```

**输出内容：**
- 3D运动轨迹可视化
- XY/XZ/YZ平面投影
- 行进距离统计
- 保存为：`results/[文件名]_trajectory.png`

## 🎓 为您的项目报告准备材料

### 必需的可视化图表

运行完整流程后，您将获得：

1. **预处理对比图** (`preprocessing_visualization.png`)
   - 展示信号处理过程
   - 说明滤波方法的选择

2. **计步检测图** (`*_detection.png`)
   - **最重要**：直观展示计步依据
   - 红点标记检测到的每一步
   - 绿线显示检测阈值

3. **评估汇总图** (`evaluation_summary.png`)
   - 展示多场景测试结果
   - 准确率统计
   - 误差分析

4. **评估报告** (`evaluation_report.txt`)
   - 详细的文字报告
   - 可直接用于项目文档

### 报告结构建议

```
1. 项目背景 (10%)
   - IMU传感器原理
   - 计步应用意义

2. 技术原理 (20%)
   - 信号采集和预处理
   - 峰值检测算法
   - 参数选择依据

3. 实验方案 (20%)
   - 数据采集方案
   - 测试场景设计
   - 设备佩戴方式

4. 算法实现 (20%)
   - 预处理流程（附preprocessing图）
   - 计步算法实现
   - 代码关键部分说明

5. 实验结果 (25%)
   - 计步检测可视化（附detection图，必须！）
   - 准确率统计表
   - 多场景对比分析（附evaluation图）
   - 误差分析

6. 总结与改进 (5%)
   - 项目总结
   - 不足和改进方向
```

## 📊 采集新数据的建议

如果您要采集新的实验数据：

### 推荐的测试场景

| 场景 | 描述 | 时长 | 目标步数 | 重复次数 |
|-----|------|------|---------|---------|
| 平地慢走-手持 | 匀速慢走，手持设备 | 30s | 35-45步 | 3-5次 |
| 平地快走-手持 | 匀速快走，手持设备 | 30s | 50-65步 | 3-5次 |
| 平地走-裤兜 | 手机放裤兜 | 30s | 40-50步 | 3-5次 |
| 上楼梯 | 匀速上楼 | 20s | 20-30步 | 3-5次 |
| 下楼梯 | 匀速下楼 | 20s | 20-30步 | 3-5次 |
| 跑步 | 匀速跑步 | 20s | 70-90步 | 3-5次 |

### 文件命名规范

使用描述性文件名，包含场景和步数信息：
- `walking_flat_hand_45steps.txt` ✓
- `walking_fast_pocket_58steps.txt` ✓
- `running_75steps.txt` ✓
- `upstairs_25steps.txt` ✓

这样程序可以自动识别实际步数！

### 采集注意事项

✅ **采集前：**
- 确保IMU设备正常工作
- 准备计数器或手机用于手动计步
- 固定设备佩戴位置和方向

✅ **采集中：**
- 开始记录数据
- **同时开始手动计数**（很重要！）
- 保持匀速运动
- 避免中途停顿

✅ **采集后：**
- 停止记录
- 立即保存数据
- 在文件名中标注实际步数
- 记录场景和佩戴方式

## 🔧 常见问题解答

### Q1: 如何处理sample_data.txt？

```powershell
cd src
python data_loader.py
```

这会自动将sample_data.txt转换为CSV格式，保存在 `data/` 目录。

### Q2: 如果没有实际步数怎么办？

运行转换后，手动编辑CSV文件：

```python
import pandas as pd

df = pd.read_csv('../data/sample_data.csv')
df['actual_steps'] = 45  # 填入您数的实际步数
df.to_csv('../data/sample_data.csv', index=False)
```

或者在检测时，程序仍会给出检测步数，只是无法计算准确率。

### Q3: 准确率低于85%怎么办？

**方法1：调整滤波参数**
```python
# 在 preprocessing.py 中
filtered_signal = preprocessor.apply_bandpass_filter(
    acc_mag, 
    low_freq=0.5,   # 尝试 0.3-0.7
    high_freq=5.0   # 尝试 4.0-6.0
)
```

**方法2：调整峰值检测参数**
```python
# 在 step_detection.py 中
detected_steps, peaks, _ = detector.detect_steps(
    filtered_signal,
    method='peak',
    height=10.0,      # 根据信号调整
    distance=30,      # 最小峰值间隔
    prominence=0.5    # 峰值突出度
)
```

### Q4: 如何只分析一个数据文件？

```python
from step_detection import StepDetector

detector = StepDetector(sampling_rate=100)
result = detector.process_file(
    '../data/your_file.csv',
    method='peak',
    filter_type='bandpass',
    save_visualization=True
)

print(f"检测步数: {result['detected_steps']}")
print(f"准确率: {result['accuracy']:.1f}%")
```

### Q5: 数据采样率不是100Hz怎么办？

修改所有脚本中的采样率参数：

```python
# 例如：50Hz
preprocessor = DataPreprocessor(sampling_rate=50)
detector = StepDetector(sampling_rate=50)
generator = TrajectoryGenerator(sampling_rate=50)
```

## 📈 结果解读

### 准确率评价标准

- **> 95%**: 优秀 ⭐⭐⭐⭐⭐
- **90-95%**: 良好 ⭐⭐⭐⭐
- **85-90%**: 可接受 ⭐⭐⭐
- **< 85%**: 需要改进 ⚠️

### 误差分析

- **平均绝对误差 < 2步**: 优秀
- **平均绝对误差 2-5步**: 良好
- **平均绝对误差 > 5步**: 需要调整参数

### 不同场景的预期准确率

| 场景 | 预期准确率 | 说明 |
|-----|----------|------|
| 平地走-手持 | 95%+ | 最容易检测 |
| 平地走-口袋 | 90-95% | 信号衰减 |
| 跑步 | 90-95% | 频率较高 |
| 上下楼梯 | 85-90% | 模式不规律 |

## 🎯 项目检查清单

完成项目前，确保：

- [ ] 安装了所有依赖包
- [ ] 转换了所有IMU数据为CSV格式
- [ ] 每个数据文件都有实际步数标注
- [ ] 至少有3种不同场景的数据
- [ ] 每种场景有3-5组测试数据
- [ ] 运行了预处理分析（有preprocessing图）
- [ ] 运行了计步检测（有detection图，必须！）
- [ ] 运行了性能评估（有evaluation报告和图）
- [ ] 平均准确率 > 85%
- [ ] 保存了所有结果到results目录
- [ ] （选做）生成了3D轨迹

## 📞 获取帮助

如果遇到问题：

1. **查看详细文档：**
   - `USAGE_GUIDE.md` - 完整使用指南
   - `README.md` - 项目文档
   - `HARDWARE_SETUP.md` - 硬件连接

2. **查看代码注释：**
   所有Python文件都有详细的中文注释

3. **运行示例：**
   ```powershell
   python quick_start.py  # 完整演示流程
   ```

## 🚀 进阶功能

### 实时在线检测

修改代码实现实时计步（需要实时数据流）

### 机器学习方法

使用scikit-learn训练分类模型区分走路和跑步

### 移动端部署

将算法移植到Android/iOS应用

---

**祝您项目顺利完成！** 🎉

有任何问题，请参考上述文档或查看代码注释。
