# UDP 上位机 PyQt6 版本

PyQt6 重构版本，模块化设计，功能完整。

## 快速开始

```powershell
# 1. 创建环境
conda env create -f environment.yml

# 2. 激活环境
conda activate smartcar-udp-pyqt6

# 3. 运行程序
.\scripts\run_gui.bat
# 或
python src\udp_gui_qt.py
```

**首次启动**：窗口默认最大化显示，深色主题

## 核心功能

| 功能 | 说明 |
|-----|------|
| **运行** | UDP 监听、PNG 保存、日志记录 |
| **视频** | PNG 序列合成 MP4、**实时录制 MP4** ⭐ |
| **对齐** | 图像帧与日志按时间戳对齐 |
| **示波器** | 实时波形、FFT 分析、位提取 |
| **发送** | UDP 数据发送、定时发送 |
| **自定义帧** | 图像帧/日志帧/日志变量配置、**彩色索引图** ⭐ |

### 🆕 新增功能

#### 1. 二值图（自定义8位）编码格式
在"自定义帧 → 图像帧配置"中，新增图像编码格式选项：**二值图(自定义8位)**

**特性：**
- `0` → 显示为黑色
- `255` → 显示为白色  
- `1-254` → 根据数值映射到彩色（HSV 色轮）

**应用场景：**
- 语义分割结果可视化（不同类别用不同颜色）
- 目标检测标注显示
- 路径规划结果展示
- 传感器数据热力图

**配置示例：**
```
帧头: A0FFFFA0
帧尾: B0B00A0D
编码格式: 二值图(自定义8位)  ← 选择此项
固定尺寸: 120x188
```

#### 2. 实时 MP4 录制（无需保存 PNG）
在"实时视频"显示区域，统计信息右侧新增 **⏺ 开始录制** 按钮。

**特性：**
- 直接从实时视频流录制 MP4
- 无需保存中间 PNG 文件
- 支持灰度和彩色视频
- 自动检测分辨率和格式
- 录制完成后可直接打开文件位置

**使用流程：**
1. 启动 UDP 监听并接收视频
2. 点击 **⏺ 开始录制**，选择保存路径
3. 录制过程中按钮变为 **⏹ 停止录制**（红色）
4. 再次点击停止，MP4 文件自动保存

**技术参数：**
- 默认帧率：30 FPS（与 UI 刷新同步）
- 编码器：MP4V (OpenCV VideoWriter)
- 自动适配：灰度/RGB/BGR 格式

## 模块结构

```
src/
├── udp_gui_qt.py          # 主窗口
├── tab_run.py             # 运行标签页
├── tab_video.py           # 视频标签页
├── tab_align.py           # 对齐标签页
├── tab_scope.py           # 示波器标签页
├── tab_send.py            # 发送标签页
├── tab_custom_frame.py    # 自定义帧标签页
├── udp_receiver_qt.py     # UDP 接收器
├── udp_sender_qt.py       # UDP 发送器
├── video_processor.py     # 视频处理
├── config.py              # 配置管理
└── utils.py               # 工具函数
```

## 技术栈

- **Python** 3.10
- **PyQt6** 6.6+ (现代化 Qt6 + 深色主题)
- **OpenCV** 图像处理
- **NumPy** 数值计算
- **Matplotlib** 示波器可视化

## 特性对比

| 项 | 原版 (Tkinter) | PyQt6 版 |
|---|----------------|----------|
| UI 框架 | ttkbootstrap | PyQt6 + 深色主题 |
| 代码组织 | 单文件 4000+ 行 | 11 模块 ~300 行/模块 |
| 线程安全 | Lock + callback | Qt Signals/Slots |
| 中文显示 | 部分乱码 | 完美支持 SimHei 字体 |
| 可维护性 | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 环境管理

```powershell
# 更新环境
conda env update -f environment.yml

# 删除环境
conda env remove -n smartcar-udp-pyqt6

# 测试环境
python test_environment.py
```

## 数据兼容性

✅ 完全兼容原版数据格式  
✅ CSV 文件格式相同  
✅ 两版本可共存使用
