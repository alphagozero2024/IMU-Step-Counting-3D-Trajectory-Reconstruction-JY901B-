# 使用说明 - IMU计步器项目

## 🚀 快速开始

本项目已经为您准备好了完整的代码框架，您的IMU数据使用的是**WitMotion传感器格式**。

### 数据格式说明

您的数据格式（sample_data.txt）包含以下信息：
- **ax(g), ay(g), az(g)**: 三轴加速度（单位：g，1g = 9.81 m/s²）
- **wx(deg/s), wy(deg/s), wz(deg/s)**: 三轴角速度（单位：度/秒）
- 以及其他辅助信息（角度、温度、磁力计等）

## 📋 完整使用流程

### 步骤1: 安装依赖包

打开PowerShell，进入项目目录：

```powershell
cd C:\Users\Administrator\Desktop\Step_Recorder
pip install -r requirements.txt
```

### 步骤2: 转换数据格式

如果您有WitMotion格式的数据文件（.txt格式），需要先转换为标准CSV格式：

```powershell
cd src
python data_loader.py
```

**选项A - 转换sample_data.txt（示例）：**
```powershell
# 程序会自动检测并转换 sample_data.txt
python data_loader.py
```

**选项B - 批量转换多个文件：**
```python
# 在Python中运行
from data_loader import batch_convert_directory

# 假设您的原始数据在 raw_data 文件夹
batch_convert_directory('../raw_data', '../data')
```

**选项C - 转换单个文件（带实际步数）：**
```python
from data_loader import convert_witmotion_to_standard

convert_witmotion_to_standard(
    input_file='../raw_data/walking_flat.txt',
    output_file='../data/walking_flat_50steps.csv',
    scenario_name='walking_flat',
    actual_steps=50  # 您手动计数的实际步数
)
```

### 步骤3: 数据预处理和分析

查看数据质量和滤波效果：

```powershell
python preprocessing.py
```

**输出：**
- 噪声统计分析
- 多种滤波方法对比图
- 保存在 `results/preprocessing_visualization.png`

### 步骤4: 运行计步检测

执行计步算法：

```powershell
python step_detection.py
```

**输出：**
- 每个数据文件的检测步数
- 与实际步数的对比
- 准确率统计
- 计步可视化图（标记峰值）

### 步骤5: 多场景评估

全面评估算法性能：

```powershell
python evaluation.py
```

**输出：**
- 评估报告：`results/evaluation_report.txt`
- 汇总图表：`results/evaluation_summary.png`
- 结果数据：`results/evaluation_results.csv`

### 步骤6: 生成3D轨迹（选做）

```powershell
python trajectory_generation.py
```

**输出：**
- 3D运动轨迹可视化
- 多视角投影图

## 📊 数据采集指南

### 推荐的测试场景

| 场景名称 | 描述 | 建议时长 | 建议步数 | 文件命名示例 |
|---------|------|---------|---------|------------|
| walking_flat_hand | 平地行走-手持 | 30秒 | 40-50步 | walking_flat_hand_45steps.txt |
| walking_flat_pocket | 平地行走-口袋 | 30秒 | 40-50步 | walking_flat_pocket_42steps.txt |
| walking_uphill | 上坡行走 | 30秒 | 30-40步 | walking_uphill_35steps.txt |
| walking_downhill | 下坡行走 | 30秒 | 45-55步 | walking_downhill_50steps.txt |
| running | 跑步 | 20秒 | 60-80步 | running_70steps.txt |

### 采集注意事项

1. **准备工作：**
   - 确保IMU设备已连接并开始记录
   - 准备好计数器或手机用于手动计步
   - 固定好设备佩戴位置

2. **采集过程：**
   - 开始记录数据
   - 同时开始手动计数步数
   - 执行规定的场景动作
   - 停止记录并保存数据

3. **文件命名：**
   - 在文件名中包含场景和步数信息
   - 例如：`walking_flat_50steps.txt`
   - 这样程序可以自动识别实际步数

4. **重复测试：**
   - 每种场景至少采集3-5组不同步数的数据
   - 确保测试的全面性和鲁棒性

## 🔧 常见问题

### Q1: 如何使用自己的IMU数据？

**A:** 如果您的数据是WitMotion格式（与sample_data.txt相同）：
1. 将数据文件放入项目目录
2. 运行 `data_loader.py` 转换为CSV格式
3. 继续后续步骤

### Q2: 如果数据格式不同怎么办？

**A:** 修改 `data_loader.py` 中的列名映射，或者手动创建CSV文件，包含以下列：
- `timestamp`: 时间戳（秒）
- `ax, ay, az`: 三轴加速度（m/s²）
- `gx, gy, gz`: 三轴角速度（rad/s）
- `scenario`: 场景名称
- `actual_steps`: 实际步数

### Q3: 准确率较低怎么办？

**A:** 尝试调整参数：
```python
# 在 step_detection.py 中修改
detector = StepDetector(sampling_rate=100)

# 调整峰值检测参数
detected_steps, peaks, properties = detector.detect_steps(
    filtered_signal,
    method='peak',
    height=None,  # 改为具体数值，如 10.5
    distance=50,  # 调整最小峰值距离
    prominence=0.5  # 调整突出度
)
```

### Q4: 如何只处理sample_data.txt？

**A:** 运行以下命令：
```powershell
cd src

# 1. 转换数据
python data_loader.py

# 2. 查看预处理效果
python preprocessing.py

# 3. 运行计步检测
python step_detection.py

# 注意：如果sample_data.txt没有实际步数，检测结果不会显示准确率
```

### Q5: 如何为sample_data添加实际步数？

**A:** 有两种方法：

**方法1 - 在转换时指定：**
```python
from data_loader import convert_witmotion_to_standard

convert_witmotion_to_standard(
    input_file='../sample_data.txt',
    output_file='../data/sample_data.csv',
    scenario_name='sample_walking',
    actual_steps=45  # 填入您数的实际步数
)
```

**方法2 - 修改已转换的CSV文件：**
```python
import pandas as pd

df = pd.read_csv('../data/sample_data_converted.csv')
df['actual_steps'] = 45  # 填入实际步数
df.to_csv('../data/sample_data_converted.csv', index=False)
```

## 📈 结果解读

### 评估指标说明

- **准确率**: `(1 - |检测步数 - 实际步数| / 实际步数) × 100%`
  - > 95%: 优秀
  - 90-95%: 良好
  - 85-90%: 可接受
  - < 85%: 需要改进

- **平均绝对误差**: 平均每次检测的步数偏差
  - < 2步: 优秀
  - 2-5步: 良好
  - > 5步: 需要改进

### 可视化图表说明

1. **preprocessing_visualization.png**
   - 展示原始信号和各种滤波效果
   - 帮助选择最佳滤波方法

2. **[filename]_detection.png**
   - 显示计步检测过程
   - 红点标记检测到的步数
   - 绿色虚线是检测阈值

3. **evaluation_summary.png**
   - 包含4个子图：
     - 实际vs检测步数散点图
     - 准确率分布直方图
     - 误差柱状图
     - 各场景准确率对比

## 📝 项目报告建议

### 报告必要内容

1. **技术原理** （20%）
   - IMU传感器工作原理
   - 计步算法原理（峰值检测）
   - 信号预处理方法

2. **实验设计** （20%）
   - 数据采集方案
   - 测试场景设计
   - 设备佩戴方式

3. **算法实现** （30%）
   - 预处理流程
   - 计步算法实现
   - 参数选择依据

4. **结果分析** （30%）
   - 各场景准确率统计
   - 误差分析
   - 可视化展示（必须包含计步依据图）
   - 不同场景对比分析

### 关键可视化图表

**必须包含：**
- ✅ 计步检测结果图（显示峰值标记）
- ✅ 准确率统计表
- ✅ 误差分析图

**推荐包含：**
- 信号预处理对比图
- 多场景性能对比图
- 3D运动轨迹（选做）

## 🎯 项目完成检查清单

- [ ] 安装了所有依赖包
- [ ] 成功转换了IMU数据格式
- [ ] 采集了至少3种不同场景的数据
- [ ] 每种场景有3-5组不同步数的测试
- [ ] 运行了预处理分析
- [ ] 运行了计步检测
- [ ] 生成了评估报告
- [ ] 准确率 > 85%
- [ ] 保存了所有可视化结果
- [ ] （选做）生成了3D轨迹

## 💡 进阶优化建议

1. **提高准确率：**
   - 尝试不同的滤波参数
   - 调整峰值检测阈值
   - 考虑自适应阈值方法

2. **增强鲁棒性：**
   - 添加步态识别（区分走路和跑步）
   - 实现在线实时检测
   - 处理异常数据

3. **扩展功能：**
   - 添加步频分析
   - 实现轨迹跟踪
   - 结合机器学习方法

---

如有任何问题，请参考：
- **README.md** - 完整项目文档
- **HARDWARE_SETUP.md** - 硬件连接指南
- 代码中的注释和文档字符串

**祝您项目顺利完成！** 🎉
