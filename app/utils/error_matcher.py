"""
错误匹配框架：根据日志输出正则匹配错误类型，返回自定义的友好提示信息。

使用方式：
1. 在 error_patterns.yaml 中定义错误模式
2. 调用 ErrorMatcher.match(log_text) 获取匹配结果
"""
import re
import os
import yaml


class ErrorRule:
    """单条错误匹配规则"""
    
    def __init__(self, name, pattern, message, category="通用", suggestion=""):
        """
        name: 规则名称
        pattern: 正则表达式（匹配日志中的错误信息）
        message: 匹配成功后展示给用户的友好提示
        category: 错误分类（如"显存"、"模型"、"数据"等）
        suggestion: 建议的解决方法
        """
        self.name = name
        self.pattern = pattern
        self.message = message
        self.category = category
        self.suggestion = suggestion
        self._compiled = re.compile(pattern, re.IGNORECASE)
    
    def match(self, text):
        """尝试匹配文本，返回 Match 对象或 None"""
        return self._compiled.search(text)


class ErrorMatcher:
    """错误匹配器，从配置文件加载规则并匹配日志"""
    
    def __init__(self, config_path=None):
        self.rules = []
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self._load_default_rules()
    
    def load_config(self, config_path):
        """从YAML配置文件加载错误规则"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config and 'rules' in config:
                for rule_data in config['rules']:
                    rule = ErrorRule(
                        name=rule_data.get('name', ''),
                        pattern=rule_data['pattern'],
                        message=rule_data['message'],
                        category=rule_data.get('category', '通用'),
                        suggestion=rule_data.get('suggestion', '')
                    )
                    self.rules.append(rule)
        except Exception:
            self._load_default_rules()
    
    def _load_default_rules(self):
        """加载默认的错误匹配规则"""
        defaults = [
            # 显存相关
            ErrorRule(
                name="cuda_oom",
                pattern=r"CUDA out of memory|RuntimeError.*out of memory",
                message="显存不足，无法完成操作",
                category="显存",
                suggestion="尝试减小 batch_size 或降低 img_size"
            ),
            ErrorRule(
                name="cuda_not_available",
                pattern=r"CUDA not available|No CUDA GPUs are available",
                message="未检测到可用的GPU",
                category="显存",
                suggestion="请确认已正确安装NVIDIA驱动和CUDA"
            ),
            # 模型相关
            ErrorRule(
                name="model_not_found",
                pattern=r"No such file.*\.pt|FileNotFoundError.*\.pt|No such file.*\.pth|FileNotFoundError.*\.pth",
                message="模型文件不存在",
                category="模型",
                suggestion="请检查模型文件路径是否正确"
            ),
            ErrorRule(
                name="model_load_error",
                pattern=r"Error.*loading.*model|load_state_dict.*error|size mismatch",
                message="模型加载失败",
                category="模型",
                suggestion="请确认模型文件与当前配置匹配，或重新选择模型"
            ),
            ErrorRule(
                name="pretrained_model_load_failed",
                pattern=r"加载预训练模型失败|Failed to load pretrained model",
                message="预训练模型加载失败",
                category="模型",
                suggestion="请确认预训练模型文件完整且格式正确"
            ),
            # 数据相关
            ErrorRule(
                name="no_images",
                pattern=r"没有.*图片|No images|0 images|No such file.*image",
                message="未找到训练图片",
                category="数据",
                suggestion="请确认图片路径正确，且图片文件存在"
            ),
            ErrorRule(
                name="no_annotations",
                pattern=r"没有.*标注|No annotations|0 labels|No labels found",
                message="未找到标注数据",
                category="数据",
                suggestion="请先加载或创建标注文件"
            ),
            ErrorRule(
                name="data_dir_not_found",
                pattern=r"数据.*目录.*不存在|Data directory not found|No such file.*data",
                message="数据目录不存在",
                category="数据",
                suggestion="请确认数据目录路径正确"
            ),
            # 训练相关
            ErrorRule(
                name="train_failed",
                pattern=r"训练失败|Training failed|train.*error",
                message="训练过程出错",
                category="训练",
                suggestion="请查看日志了解详细错误信息"
            ),
            ErrorRule(
                name="nan_loss",
                pattern=r"Loss is NaN|nan.*loss|loss.*nan",
                message="训练出现NaN损失值",
                category="训练",
                suggestion="尝试降低学习率或检查数据是否存在异常值"
            ),
            # 推理相关
            ErrorRule(
                name="inference_failed",
                pattern=r"推理失败|Inference failed|detect.*error",
                message="推理过程出错",
                category="推理",
                suggestion="请确认模型和图片是否正确加载"
            ),
            # 通用
            ErrorRule(
                name="permission_denied",
                pattern=r"Permission denied|权限不足|Access denied",
                message="文件访问权限不足",
                category="通用",
                suggestion="请确认文件/目录的读写权限"
            ),
            ErrorRule(
                name="disk_full",
                pattern=r"No space left|磁盘空间不足|disk full",
                message="磁盘空间不足",
                category="通用",
                suggestion="请清理磁盘空间后重试"
            ),
        ]
        self.rules = defaults
    
    def match(self, text):
        """
        匹配日志文本，返回第一个匹配的 ErrorRule，或 None。
        按规则顺序匹配，先定义的优先级更高。
        """
        for rule in self.rules:
            if rule.match(text):
                return rule
        return None
    
    def match_all(self, text):
        """匹配日志文本，返回所有匹配的 ErrorRule 列表"""
        results = []
        for rule in self.rules:
            if rule.match(text):
                results.append(rule)
        return results
    
    def format_error(self, text):
        """
        匹配日志并格式化错误信息。
        返回 (friendly_message, category, suggestion) 或 None。
        """
        rule = self.match(text)
        if rule:
            return rule.message, rule.category, rule.suggestion
        return None
