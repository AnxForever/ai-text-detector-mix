# Matplotlib中文显示规则

## ⚠️ 重要规则：所有生成可视化的代码必须配置中文字体！

### 问题
matplotlib默认不支持中文显示，会显示为方框（□□□）。

### 解决方案

#### 1. Windows系统（推荐）

```python
import matplotlib.pyplot as plt

# 在任何绘图代码之前添加以下配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
```

**可用的Windows中文字体：**
- `Microsoft YaHei` - 微软雅黑（推荐）
- `SimHei` - 黑体
- `SimSun` - 宋体
- `KaiTi` - 楷体

#### 2. Linux/WSL系统

需要先安装中文字体：
```bash
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei
```

然后配置：
```python
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

#### 3. macOS系统

```python
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False
```

### 完整示例代码

```python
import matplotlib.pyplot as plt
import numpy as np

# ====== 中文字体配置（必须在绘图前） ======
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# =========================================

# 示例数据
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# 绘图（现在中文可以正常显示）
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'o-', label='数据曲线')
plt.xlabel('横轴标签', fontsize=12)
plt.ylabel('纵轴标签', fontsize=12)
plt.title('中文标题示例', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('test_chinese.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 验证字体是否生效

```python
from matplotlib import font_manager

# 列出所有可用字体
for font in font_manager.fontManager.ttflist:
    if 'Microsoft' in font.name or 'Sim' in font.name or 'WenQuanYi' in font.name:
        print(f"可用中文字体: {font.name} - {font.fname}")
```

### 项目中的应用

本项目中的所有可视化脚本必须包含中文字体配置：

- ✅ `scripts/evaluation/complete_evaluation_windows.py` - 已配置
- ✅ `scripts/evaluation/plot_training_curves.py` - 需检查
- ✅ 所有future的可视化脚本 - 必须配置

### 检查清单

每次编写matplotlib可视化代码时：

- [ ] 在导入matplotlib后立即配置中文字体
- [ ] 设置 `plt.rcParams['font.sans-serif']`
- [ ] 设置 `plt.rcParams['axes.unicode_minus'] = False`
- [ ] 测试生成的图片中文是否正常显示
- [ ] 如果是跨平台代码，配置多个备选字体

## 运行指南

### 在Windows PowerShell中运行

```powershell
# 激活虚拟环境（如果使用）
.\.venv\Scripts\Activate.ps1

# 运行Windows版评估脚本（自动配置中文字体）
python scripts/evaluation/complete_evaluation_windows.py
```

### 在WSL/Linux中运行

```bash
# 需要先安装中文字体
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei

# 激活虚拟环境
source .venv/bin/activate

# 运行评估脚本
python scripts/evaluation/complete_evaluation_fixed.py
```

## 常见问题

### Q: 为什么我的图表中文显示为方框？
A: 没有配置中文字体，或者配置的字体系统中不存在。检查上面的配置代码是否添加。

### Q: 如何知道我的系统有哪些中文字体？
A: 运行上面的"验证字体是否生效"代码。

### Q: WSL环境没有sudo权限怎么办？
A: 使用Windows PowerShell运行Python脚本，它会自动使用Windows系统字体。

---

**记住：任何使用matplotlib的代码都必须配置中文字体！** 🎨
