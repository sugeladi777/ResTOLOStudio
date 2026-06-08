# ReSTOLO Studio

这是一个桌面应用，用于整合以下工作流：

- Nanonis STM 控制与扫描采集
- ReSTOLO 图像推理、标注与训练
- 基于会话的结果管理

## 运行方式

```powershell
pip install -r requirements.txt
python main.py
```

## Conda 环境

当前工作站推荐环境：

- Python `3.10`
- PyTorch `2.5.1`
- CUDA `12.1`
- 已验证 GPU：`NVIDIA GeForce RTX 4060`

创建并使用环境：

```powershell
conda env create -f environment.yml
conda activate restolo-py310
python main.py
```

如果环境已经存在，可以更新：

```powershell
conda env update -f environment.yml --prune
```

## 项目结构

- `main.py`：精简启动入口
- `app/bootstrap.py`：应用启动与 Qt 装配
- `app/runtime.py`：运行时与服务装配
- `app/windows/`：顶层窗口层
- `app/windows/studio_window.py`：主窗口入口
- `app/windows/studio_panels.py`：采集与结果面板
- `app/windows/studio_actions.py`：窗口级操作逻辑，包括 Nanonis、会话与推理
- `app/windows/__init__.py`：窗口层对外导出入口
- `app/legacy/`：历史 UI 与旧流程边界
- `app/legacy/workbench_impl.py`：精简后的 legacy 壳类
- `app/legacy/workbench_ui.py`：legacy 主题与通用控件辅助
- `app/legacy/workbench_layout.py`：legacy 主窗口布局构建
- `app/legacy/workbench_data.py`：数据加载、SXM 转换、标注与模型加载相关逻辑
- `app/legacy/workbench_training.py`：YOLO 与 ResNet 训练流程辅助
- `app/legacy/workbench_runtime.py`：训练/推理运行时回调与界面状态处理
- `app/legacy/workbench_state.py`：标签页切换与按钮状态逻辑
- `app/legacy/workbench_bindings.py`：集中管理 legacy 方法绑定
- `app/services/`：运行时服务层
- `app/ui/`：可复用界面组件
- `app/utils/`：工具与管理器
- `app/core/`：核心路径与基础对象
- `app/core/paths.py`：统一管理资源目录、会话目录、配置文件与 YOLO 相关资源路径
- `assets/`：与源码分离的本地资源目录
- `assets/models/`：本地模型目录，用于存放检测/分类权重和 `classes.yaml`
- `assets/config/error_patterns.yaml`：错误模式匹配配置
- `nanonis/`：Nanonis TCP 后端
- `ml/`：YOLO / ResNet 训练与推理逻辑
- `ml/models/yolov5m_molecule.yaml`：当前桌面训练流程使用的 YOLO 结构配置
- `sessions/`：扫描、推理和训练输出目录

## 仓库约定

- `__pycache__/` 等运行缓存不纳入版本控制
- `.vs/`、`.suo` 等 IDE 元数据不纳入版本控制
- 训练产物不保留在源码目录下
- 不属于桌面应用工作流的上游部署辅助文件会被移除
- 当前桌面应用不会使用的可选实验跟踪能力会降级为本地兼容空实现
- `sessions/` 视为运行输出目录，本地生成，不纳入仓库
- 本地大模型权重默认不纳入仓库，按需自行放入 `assets/models/`
