# IMU硬件连接指南

## 硬件选择

### 推荐方案1: Arduino + MPU6050
- **成本**: 低（约￥20-30）
- **难度**: 中等
- **精度**: 中等

### 推荐方案2: 手机内置IMU
- **成本**: 免费
- **难度**: 简单
- **精度**: 高

### 推荐方案3: MPU9250/BMI160
- **成本**: 中等（约￥50-100）
- **难度**: 中等
- **精度**: 高

## 方案1: Arduino + MPU6050 连接

### 硬件连接

```
MPU6050 <-> Arduino
VCC     <-> 5V
GND     <-> GND
SCL     <-> A5 (SCL)
SDA     <-> A4 (SDA)
```

### Arduino代码

将以下代码上传到Arduino：

```cpp
#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  
  mpu.initialize();
  
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed");
    while(1);
  }
  
  Serial.println("MPU6050 ready");
  delay(1000);
}

void loop() {
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  
  // 转换为标准单位
  float ax_ms2 = ax / 16384.0 * 9.81;
  float ay_ms2 = ay / 16384.0 * 9.81;
  float az_ms2 = az / 16384.0 * 9.81;
  float gx_rads = gx / 131.0 * PI / 180;
  float gy_rads = gy / 131.0 * PI / 180;
  float gz_rads = gz / 131.0 * PI / 180;
  
  // 输出CSV格式
  Serial.print(ax_ms2, 4); Serial.print(",");
  Serial.print(ay_ms2, 4); Serial.print(",");
  Serial.print(az_ms2, 4); Serial.print(",");
  Serial.print(gx_rads, 4); Serial.print(",");
  Serial.print(gy_rads, 4); Serial.print(",");
  Serial.println(gz_rads, 4);
  
  delay(10);  // 100Hz采样率
}
```

### 修改data_collection.py

在 `collect_data()` 函数中替换为：

```python
import serial
import time

# 打开串口（根据实际端口修改）
ser = serial.Serial('COM3', 115200, timeout=1)  # Windows
# ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # Linux

time.sleep(2)  # 等待Arduino初始化

start_time = time.time()
while time.time() - start_time < duration:
    if ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8').strip()
            values = line.split(',')
            
            if len(values) == 6:
                timestamp = time.time() - start_time
                self.data.append({
                    'timestamp': timestamp,
                    'ax': float(values[0]),
                    'ay': float(values[1]),
                    'az': float(values[2]),
                    'gx': float(values[3]),
                    'gy': float(values[4]),
                    'gz': float(values[5])
                })
        except:
            pass

ser.close()
```

## 方案2: 使用手机IMU

### 步骤1: 安装App

**Android:**
- Sensor Logger (推荐)
- Physics Toolbox Sensor Suite
- HyperIMU

**iOS:**
- Sensor Logger
- sensorLog

### 步骤2: 配置采集参数

在App中设置：
- 采样频率: 100 Hz
- 传感器: 加速度计 + 陀螺仪
- 输出格式: CSV

### 步骤3: 采集数据

1. 打开App并开始记录
2. 执行行走/跑步测试（手动计数步数）
3. 停止记录并导出CSV文件
4. 将CSV文件复制到 `data/` 目录

### 步骤4: 数据格式转换

如果App导出的CSV格式不同，需要转换为项目要求的格式：

```python
# 在src目录创建 convert_phone_data.py
import pandas as pd

# 读取手机导出的数据
df = pd.read_csv('../data/phone_data.csv')

# 根据实际列名调整
# 假设手机数据列名为: time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
df_new = pd.DataFrame({
    'timestamp': df['time'] - df['time'].iloc[0],  # 归零时间戳
    'ax': df['accel_x'],
    'ay': df['accel_y'],
    'az': df['accel_z'],
    'gx': df['gyro_x'],
    'gy': df['gyro_y'],
    'gz': df['gyro_z'],
    'scenario': 'phone_walking',
    'actual_steps': 50  # 手动填入实际步数
})

df_new.to_csv('../data/converted_data.csv', index=False)
print("转换完成！")
```

## 方案3: 树莓派 + IMU模块

### 硬件连接（I2C）

```
MPU6050/MPU9250 <-> Raspberry Pi
VCC             <-> 3.3V
GND             <-> GND
SCL             <-> GPIO 3 (SCL)
SDA             <-> GPIO 2 (SDA)
```

### Python代码（使用smbus）

```python
import smbus
import time
import math

class IMU:
    def __init__(self, address=0x68):
        self.bus = smbus.SMBus(1)
        self.address = address
        
        # 唤醒MPU6050
        self.bus.write_byte_data(self.address, 0x6B, 0)
        
    def read_raw_data(self, addr):
        high = self.bus.read_byte_data(self.address, addr)
        low = self.bus.read_byte_data(self.address, addr+1)
        value = (high << 8) | low
        if value > 32768:
            value = value - 65536
        return value
    
    def get_data(self):
        # 读取加速度
        ax = self.read_raw_data(0x3B) / 16384.0 * 9.81
        ay = self.read_raw_data(0x3D) / 16384.0 * 9.81
        az = self.read_raw_data(0x3F) / 16384.0 * 9.81
        
        # 读取陀螺仪
        gx = self.read_raw_data(0x43) / 131.0 * math.pi / 180
        gy = self.read_raw_data(0x45) / 131.0 * math.pi / 180
        gz = self.read_raw_data(0x47) / 131.0 * math.pi / 180
        
        return ax, ay, az, gx, gy, gz

# 在data_collection.py中使用
imu = IMU()
start_time = time.time()

while time.time() - start_time < duration:
    ax, ay, az, gx, gy, gz = imu.get_data()
    timestamp = time.time() - start_time
    
    self.data.append({
        'timestamp': timestamp,
        'ax': ax, 'ay': ay, 'az': az,
        'gx': gx, 'gy': gy, 'gz': gz
    })
    
    time.sleep(0.01)  # 100Hz
```

## 数据采集注意事项

### 1. 传感器校准

在采集前进行校准：

```python
# 静止状态下采集100个样本
calibration_samples = []
for i in range(100):
    # 读取数据
    ax, ay, az, gx, gy, gz = get_imu_data()
    calibration_samples.append([gx, gy, gz])
    time.sleep(0.01)

# 计算偏移量
gyro_offset = np.mean(calibration_samples, axis=0)

# 在采集时减去偏移
gx -= gyro_offset[0]
gy -= gyro_offset[1]
gz -= gyro_offset[2]
```

### 2. 佩戴位置

**推荐位置：**
- 手持（握住设备）
- 裤兜（前口袋）
- 上臂绑带
- 腰部绑带

**注意方向：**
- 确保每次采集时设备方向一致
- 记录设备的坐标系方向

### 3. 测试场景建议

| 场景 | 描述 | 步数范围 | 重复次数 |
|------|------|---------|---------|
| 慢走-平地 | 平地慢速行走 | 30-50步 | 3-5次 |
| 快走-平地 | 平地快速行走 | 50-80步 | 3-5次 |
| 跑步 | 平地跑步 | 60-100步 | 3-5次 |
| 上楼梯 | 爬楼梯 | 20-40步 | 3-5次 |
| 下楼梯 | 下楼梯 | 20-40步 | 3-5次 |

### 4. 采集检查清单

- [ ] IMU设备连接正常
- [ ] 采样频率设置为100Hz
- [ ] 准备好计数器（手动计步）
- [ ] 确定测试场景和路线
- [ ] 设备佩戴位置固定
- [ ] 数据保存路径正确

## 故障排除

### 问题1: 串口连接失败
- 检查端口号是否正确（Windows: COM3, Linux: /dev/ttyUSB0）
- 确认Arduino驱动已安装
- 检查波特率是否匹配（115200）

### 问题2: 数据格式错误
- 检查串口输出格式是否为CSV
- 确认数据包含6个值（ax,ay,az,gx,gy,gz）
- 检查单位转换是否正确

### 问题3: 采样率不稳定
- 减少串口通信的延迟
- 使用定时器中断而不是delay()
- 检查缓冲区是否溢出

### 问题4: IMU无法初始化
- 检查I2C连接是否正常
- 确认电源供电稳定
- 尝试更换I2C地址（0x68或0x69）

## 测试验证

采集完成后，运行以下代码验证数据质量：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('../data/test_data.csv')

# 检查数据
print(f"样本数: {len(df)}")
print(f"时长: {df['timestamp'].max():.2f}秒")
print(f"采样率: {len(df) / df['timestamp'].max():.1f} Hz")

# 可视化
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 加速度
axes[0].plot(df['timestamp'], df['ax'], label='ax')
axes[0].plot(df['timestamp'], df['ay'], label='ay')
axes[0].plot(df['timestamp'], df['az'], label='az')
axes[0].set_ylabel('Acceleration (m/s²)')
axes[0].legend()
axes[0].grid()

# 陀螺仪
axes[1].plot(df['timestamp'], df['gx'], label='gx')
axes[1].plot(df['timestamp'], df['gy'], label='gy')
axes[1].plot(df['timestamp'], df['gz'], label='gz')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Gyroscope (rad/s)')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()
```

---

选择适合您的方案开始数据采集吧！如有问题，请参考主README.md文档。
