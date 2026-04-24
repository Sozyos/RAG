"""
配置模块
使用 Python 标准库的 dataclass 实现类型安全的配置管理
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class DataConfig:
    """数据配置类"""
    kg_file: str  # 知识图谱数据文件路径
    embedding_model: str = "all-MiniLM-L6-v2"  # 默认使用的 embedding 模型
    batch_size: int = 32  # 批量处理大小


@dataclass
class RetrieverConfig:
    """检索器配置类"""
    top_k: int = 2  # 返回最相似的 top_k 个三元组
    similarity_threshold: float = 0.5  # 相似度阈值，低于此值的三元组将被过滤
    use_gpu: bool = False  # 是否使用 GPU 加速


@dataclass
class SystemConfig:
    """系统配置类"""
    data: DataConfig
    retriever: RetrieverConfig
    debug: bool = False  # 调试模式
    log_level: str = "INFO"  # 日志级别


class ConfigManager:
    """配置管理器"""

    @staticmethod
    def from_file(filepath: str) -> SystemConfig:
        """
        从 JSON 文件加载配置

        Args:
            filepath: JSON 配置文件路径

        Returns:
            SystemConfig: 系统配置对象

        Raises:
            FileNotFoundError: 配置文件不存在
            json.JSONDecodeError: JSON 格式错误
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # 构建嵌套的配置对象
        data_config = DataConfig(**config_dict.get('data', {}))
        retriever_config = RetrieverConfig(**config_dict.get('retriever', {}))

        system_config = SystemConfig(
            data=data_config,
            retriever=retriever_config,
            debug=config_dict.get('debug', False),
            log_level=config_dict.get('log_level', 'INFO')
        )

        return system_config

    @staticmethod
    def to_dict(config: SystemConfig) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        return {
            'data': {
                'kg_file': config.data.kg_file,
                'embedding_model': config.data.embedding_model,
                'batch_size': config.data.batch_size
            },
            'retriever': {
                'top_k': config.retriever.top_k,
                'similarity_threshold': config.retriever.similarity_threshold,
                'use_gpu': config.retriever.use_gpu
            },
            'debug': config.debug,
            'log_level': config.log_level
        }

    @staticmethod
    def save_to_file(config: SystemConfig, filepath: str):
        """将配置保存到 JSON 文件"""
        config_dict = ConfigManager.to_dict(config)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)


# 默认配置
DEFAULT_CONFIG = SystemConfig(
    data=DataConfig(kg_file="data/kg.json"),
    retriever=RetrieverConfig()
)