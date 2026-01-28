# 📋 项目使用说明 - 快速参考卡

## 🎯 核心信息

**项目类型：** IMU计步器（智能检测系统）  
**数据格式：** WitMotion IMU传感器格式（.txt文件）  
**编程语言：** Python 3.8+  
**目标：** 实现准确的步数检测并生成详细报告

## ⚡ 3分钟快速开始

### 第1步：安装依赖（1分钟）
```powershell
cd C:\Users\Administrator\Desktop\Step_Recorder
pip install -r requirements.txt
```

### 第2步：转换数据（30秒）
```powershell
cd src
python data_loader.py
```

### 第3步：一键运行（1分钟）
```powershell
python quick_start.py
```

完成！结果保存在 `results/` 文件夹。

---

## 📊 完整工作流程图

```
[WitMotion数据.txt]
        ↓
   data_loader.py (转换格式)
        ↓
[标准CSV数据] → data/
        ↓
   preprocessing.py (信号分析)
        ↓
[滤波后的信号]
        ↓
   step_detection.py (峰值检测)
        ↓
[检测步数 + 可视化图]
        ↓
   evaluation.py (性能评估)
        ↓
[完整报告 + 汇总图表] → results/
```

---

## 📁 目录结构说明

```
Step_Recorder/
├── data/                      # 转换后的CSV数据
│   └── *.csv
├── results/                   # 所有输出结果
│   ├── *_detection.png       # 计步检测图（最重要！）
│   ├── preprocessing_visualization.png
│   ├── evaluation_summary.png
│   └── evaluation_report.txt
├── src/                       # 源代码
│   ├── data_loader.py        # 数据格式转换
│   ├── preprocessing.py      # 预处理和滤波
│   ├── step_detection.py     # 计步算法
│   ├── evaluation.py         # 性能评估
│   ├── trajectory_generation.py  # 3D轨迹（选做）
│   └── quick_start.py        # 一键运行
├── sample_data.txt            # 示例IMU数据
├── HOW_TO_USE.md             # 详细使用说明
└── README.md                  # 项目文档
```

---

## 🔑 关键文件说明

| 文件 | 用途 | 何时使用 |
|-----|------|---------|
| `data_loader.py` | 转换WitMotion格式 | 首次使用，有新数据时 |
| `preprocessing.py` | 查看滤波效果 | 想优化信号处理时 |
| `step_detection.py` | 计步检测 | **核心功能**，必须运行 |
| `evaluation.py` | 完整评估 | **生成报告**，必须运行 |
| `trajectory_generation.py` | 3D轨迹 | 选做部分 |
| `quick_start.py` | 一键演示 | 快速测试整个流程 |

---

## 📈 输出文件说明

### 必须包含在报告中的图表

1. **`*_detection.png`** ⭐⭐⭐⭐⭐
   - **最重要的图！**
   - 展示计步依据（峰值标记）
   - 红点 = 检测到的每一步
   - 必须在报告中详细说明

2. **`evaluation_summary.png`** ⭐⭐⭐⭐
   - 4个子图展示完整性能
   - 准确率统计
   - 误差分析

3. **`evaluation_report.txt`** ⭐⭐⭐⭐
   - 文字版详细报告
   - 可直接复制到文档

### 可选的辅助图表

4. **`preprocessing_visualization.png`** ⭐⭐⭐
   - 说明预处理过程
   - 展示滤波效果

5. **`*_trajectory.png`** ⭐⭐
   - 选做部分
   - 3D运动轨迹

---

## 🎓 常用命令速查

### 数据转换
```powershell
# 转换单个文件
python -c "from data_loader import convert_witmotion_to_standard; convert_witmotion_to_standard('../raw_data/walk.txt', '../data/walk.csv', 'walking', 50)"

# 批量转换
python -c "from data_loader import batch_convert_directory; batch_convert_directory('../raw_data', '../data')"
```

### 运行分析
```powershell
# 数据预处理
python preprocessing.py

# 计步检测
python step_detection.py

# 完整评估
python evaluation.py

# 3D轨迹（选做）
python trajectory_generation.py
```

### 一键运行
```powershell
# 完整演示流程
python quick_start.py
```

---

## ⚙️ 参数调优速查

### 如果检测步数偏多
```python
# 在 step_detection.py 中调整
distance=50,        # 增大（默认33）
prominence=0.8,     # 增大（默认0.3）
height=11.0        # 增大（默认自动）
```

### 如果检测步数偏少
```python
# 在 step_detection.py 中调整
distance=25,        # 减小
prominence=0.2,     # 减小
height=9.5         # 减小
```

### 调整滤波参数
```python
# 在 preprocessing.py 中调整
low_freq=0.5,      # 低频截止（0.3-0.7）
high_freq=5.0,     # 高频截止（4.0-6.0）
```

---

## ✅ 项目提交前检查清单

**数据准备：**
- [ ] 已转换所有IMU数据为CSV格式
- [ ] 每个文件都标注了实际步数
- [ ] 至少3种场景，每种3-5组数据

**代码运行：**
- [ ] 成功运行 `data_loader.py`
- [ ] 成功运行 `step_detection.py`
- [ ] 成功运行 `evaluation.py`
- [ ] 平均准确率 > 85%

**结果文件：**
- [ ] `results/` 文件夹有所有图表
- [ ] 至少有3张 `*_detection.png` 图
- [ ] 有 `evaluation_summary.png`
- [ ] 有 `evaluation_report.txt`

**报告内容：**
- [ ] 包含计步依据可视化图
- [ ] 包含准确率统计表
- [ ] 包含多场景对比分析
- [ ] 包含误差分析

---

## 🆘 故障排除

| 问题 | 解决方案 |
|-----|---------|
| 找不到模块 | `pip install -r requirements.txt` |
| 数据加载失败 | 先运行 `data_loader.py` 转换格式 |
| 准确率低于85% | 参考"参数调优速查"章节 |
| 图表不显示中文 | 正常，已配置中文字体 |
| 缺少实际步数 | 手动编辑CSV文件添加 |

---

## 📞 帮助资源

1. **详细教程：** `HOW_TO_USE.md`
2. **项目文档：** `README.md`
3. **使用指南：** `USAGE_GUIDE.md`
4. **硬件连接：** `HARDWARE_SETUP.md`

---

## 💡 报告撰写提示

### 关键内容
1. **技术原理** - 峰值检测算法原理
2. **实验设计** - 测试场景和数据采集方案
3. **算法实现** - 预处理流程和计步算法
4. **结果分析** - ⭐ 必须包含detection图展示计步依据
5. **性能评估** - 准确率统计和误差分析

### 图表使用建议
- `*_detection.png` → 算法实现章节（重点讲解）
- `evaluation_summary.png` → 结果分析章节
- `preprocessing_visualization.png` → 技术原理章节
- `evaluation_report.txt` → 附录或性能评估章节

---

**打印这张卡片，随时查阅！** 📄

更多详情请查看 `HOW_TO_USE.md`
