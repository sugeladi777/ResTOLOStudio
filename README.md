# ReSTOLO Studio

这是一个用于扫描采集、图像标注、模型训练和结果推理的一体化桌面应用项目。

## 目录说明

- `app/`：桌面应用主代码
- `app/windows/`：主窗口、采集面板、结果面板与窗口操作逻辑
- `app/legacy/`：历史工作台代码的隔离层
- `app/services/`：运行时服务封装
- `app/ui/`：可复用界面组件
- `app/utils/`：工具类、训练/推理管理器、SXM 解析等辅助代码
- `app/core/`：核心路径与基础对象
- `assets/`：本地资源目录
- `assets/models/`：本地模型文件与类别文件
- `assets/config/`：配置资源，如错误模式匹配规则
- `ml/`：YOLO / ResNet 训练与推理相关代码
- `ml/models/`：模型结构定义与配置
- `ml/utils/`：训练与推理底层工具
- `ml/data/`：训练超参数等数据配置
- `nanonis/`：Nanonis TCP 控制相关代码
- `sessions/`：运行过程中生成的扫描、训练和推理结果目录
- `main.py`：应用启动入口
- `environment.yml`：Conda 环境定义
- `requirements.txt`：Python 依赖列表
- `.gitignore`：版本控制忽略规则
