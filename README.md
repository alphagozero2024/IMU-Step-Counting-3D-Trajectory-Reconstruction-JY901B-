# 智能检测系统 - 基于惯性测量单元的计步器

## 项目简介

本项目实现了一个基于IMU（惯性测量单元）传感器的计步器系统，能够准确检测行走和跑步时的步数，并可选地生成3D运动轨迹。

## 功能特点

- ✅ **数据采集**：从IMU传感器采集三轴加速度和陀螺仪数据
- ✅ **数据预处理**：噪声分析、多种滤波方法（低通、带通、移动平均）
- ✅ **计步算法**：基于峰值检测的高精度计步算法
- ✅ **多场景测试**：支持多种佩戴方式和运动模式的测试
- ✅ **可视化展示**：直观展示计步依据和检测结果
- ✅ **性能评估**：全面的准确率和误差分析
- 🎯 **轨迹生成**（选做）：基于IMU数据生成3D运动轨迹

## 项目结构

```
Step_Recorder/
├── data/                    # 数据存储目录
│   └── *.csv               # IMU采集数据
├── results/                # 结果输出目录
│   ├── *.png              # 可视化图表
│   ├── evaluation_report.txt  # 评估报告
│   └── evaluation_results.csv # 评估结果数据
├── src/                    # 源代码目录
│   ├── data_collection.py      # 数据采集模块
│   ├── preprocessing.py        # 数据预处理模块
│   ├── step_detection.py       # 计步算法模块
│   ├── trajectory_generation.py # 轨迹生成模块（选做）
│   └── evaluation.py           # 评估测试模块
├── instructions.txt        # 项目需求说明
├── requirements.txt        # Python依赖包
└── README.md              # 本文件
```

## 环境配置

### 1. Python环境

建议使用Python 3.8或更高版本。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- numpy: 数值计算
- pandas: 数据处理
- scipy: 信号处理和科学计算
- matplotlib: 数据可视化

## 使用指南

### 步骤1: 硬件准备

**推荐的IMU传感器：**
- **MPU6050**（常用，性价比高）
- **MPU9250**（9轴，更精确）
- 手机内置IMU（可使用Sensor Logger等App导出数据）

**连接方式：**
- Arduino + MPU6050：通过串口连接
- 手机：使用Sensor Logger等应用导出CSV数据

### 步骤2: 数据采集

1. 修改 `src/data_collection.py` 中的IMU读取代码，适配您的硬件
2. 运行数据采集：

```bash
cd src
python data_collection.py
```

**建议的测试场景：**

| 场景 | 描述 | 建议时长 | 建议步数 |
|------|------|---------|---------|
| walking_flat_hand | 平地行走-手持 | 30秒 | 40步 |
| walking_flat_pocket | 平地行走-口袋 | 30秒 | 40步 |
| walking_uphill | 上坡行走 | 30秒 | 35步 |
| walking_downhill | 下坡行走 | 30秒 | 45步 |
| running | 跑步 | 20秒 | 60步 |

**重要提示：**
- 每种场景至少采集3-5组不同步数的数据
- 采集时需人工计数实际步数作为参考
- 数据会自动保存到 `data/` 目录

### 步骤3: 数据预处理分析

查看数据质量和预处理效果：

```bash
python preprocessing.py
```

输出：
- 噪声分析报告
- 多种滤波方法对比图
- 预处理可视化结果

### 步骤4: 计步检测

运行计步算法：

```bash
python step_detection.py
```

输出：
- 每个数据文件的检测结果
- 计步可视化图（标记峰值位置）
- 准确率统计

### 步骤5: 多场景评估

执行完整的性能评估：

```bash
python evaluation.py
```

输出：
- 所有场景的汇总评估
- 准确率分布图
- 误差分析图
- 详细评估报告

### 步骤6: 轨迹生成（选做）

生成3D运动轨迹：

```bash
python trajectory_generation.py
```

输出：
- 3D轨迹可视化
- 多视角投影图
- 行进距离统计

## 算法说明

### 计步算法原理

1. **信号采集**：获取三轴加速度数据
2. **合成加速度**：计算加速度向量模值
3. **带通滤波**：保留0.5-5Hz的行走频率范围
4. **峰值检测**：识别周期性的加速度峰值
5. **步数统计**：峰值计数即为步数

### 关键参数

- **采样频率**：100 Hz（推荐）
- **滤波频率**：0.5-5 Hz（带通滤波）
- **峰值高度**：均值 + 0.5×标准差
- **最小峰值距离**：0.33秒（对应最大3步/秒）

## 评估指标

- **准确率**：`(1 - |检测步数 - 实际步数| / 实际步数) × 100%`
- **平均绝对误差**：`mean(|检测步数 - 实际步数|)`
- **平均误差**：`mean(检测步数 - 实际步数)`

## 硬件连接示例

### Arduino + MPU6050

```cpp
// Arduino代码示例
#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  mpu.initialize();
}

void loop() {
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  
  // 转换为m/s²和rad/s
  float ax_ms2 = ax / 16384.0 * 9.81;
  float ay_ms2 = ay / 16384.0 * 9.81;
  float az_ms2 = az / 16384.0 * 9.81;
  float gx_rads = gx / 131.0 * PI / 180;
  float gy_rads = gy / 131.0 * PI / 180;
  float gz_rads = gz / 131.0 * PI / 180;
  
  // 输出CSV格式
  Serial.print(ax_ms2); Serial.print(",");
  Serial.print(ay_ms2); Serial.print(",");
  Serial.print(az_ms2); Serial.print(",");
  Serial.print(gx_rads); Serial.print(",");
  Serial.print(gy_rads); Serial.print(",");
  Serial.println(gz_rads);
  
  delay(10);  // 100Hz采样率
}
```

### Python串口读取

在 `data_collection.py` 中修改：

```python
import serial

ser = serial.Serial('COM3', 115200)  # 根据实际端口修改

while time.time() - start_time < duration:
    if ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()
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
```

## 注意事项

1. **数据采集**：
   - 确保IMU设备稳定连接
   - 采集时保持设备佩戴位置一致
   - 手动计数时要准确记录实际步数

2. **参数调整**：
   - 不同场景可能需要调整滤波参数
   - 可以在 `step_detection.py` 中修改峰值检测参数

3. **轨迹生成**：
   - 轨迹估算基于双重积分，会有累积误差
   - 建议配合其他传感器（如GPS）提高精度

## 项目报告建议

### 报告结构

1. **项目背景**：IMU原理和计步应用
2. **技术原理**：算法设计和实现方法
3. **实验步骤**：
   - 数据采集方案
   - 预处理流程
   - 算法实现
   - 多场景测试
4. **结果分析**：
   - 准确率统计
   - 误差分析
   - 不同场景对比
   - 可视化展示
5. **总结与展望**

### 关键内容

- 展示计步依据的可视化图（峰值标记）
- 分析不同场景的准确率差异
- 讨论误差来源和改进方法
- （选做）展示3D轨迹可视化

## 故障排除

**问题1：准确率较低**
- 检查滤波参数是否合适
- 调整峰值检测阈值
- 确保采样率稳定

**问题2：检测步数偏多**
- 增大最小峰值距离
- 提高峰值突出度要求

**问题3：检测步数偏少**
- 降低峰值高度阈值
- 检查信号是否被过度滤波

## 联系与支持

如有问题或建议，请查看课程资料或联系助教。

---

**祝项目顺利完成！** 🎉
