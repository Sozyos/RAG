"""
知识图谱模块
实现简单的三元组存储和倒排索引
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


@dataclass
class Triplet:
    """三元组数据类"""
    head: str  # 头实体
    relation: str  # 关系
    tail: str  # 尾实体
    metadata: Dict[str, Any] = None  # 元数据，如置信度、来源等

    def __post_init__(self):
        """初始化后处理"""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'head': self.head,
            'relation': self.relation,
            'tail': self.tail,
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        """字符串表示"""
        return f"({self.head}, {self.relation}, {self.tail})"

    @property
    def text(self) -> str:
        """获取三元组的文本表示，用于 embedding（规范化关系表达，提升语义匹配）"""
        rel = self.relation
        # 规范化：将文言/书面语关系转为更自然的口语表达
        if rel.endswith('为'):
            rel = rel[:-1] + '是'  # "挚爱为" → "挚爱是"
        elif rel.endswith('于'):
            rel = rel[:-1] + '在'  # "被镇压于" → "被镇压在", "统一于" → "统一在"
        return f"{self.head} {rel} {self.tail}"


class SimpleKG:
    """简单知识图谱类"""

    def __init__(self):
        """初始化空的知识图谱"""
        self.triplets: List[Triplet] = []
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)  # 词到三元组索引的映射
        self.entity_index: Dict[str, List[int]] = defaultdict(list)  # 实体到三元组索引的映射

    def add_triplet(self, triplet: Triplet):
        """
        添加三元组到知识图谱

        Args:
            triplet: 要添加的三元组
        """
        idx = len(self.triplets)
        self.triplets.append(triplet)

        # 更新倒排索引（基于词）
        words = self._extract_words(triplet.text)
        for word in words:
            self.inverted_index[word].append(idx)

        # 更新实体索引
        self.entity_index[triplet.head].append(idx)
        self.entity_index[triplet.tail].append(idx)

    def load_from_json(self, filepath: str):
        """
        从 JSON 文件加载知识图谱数据

        Args:
            filepath: JSON 文件路径

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON 格式错误
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON 数据应该是一个三元组列表")

        for item in data:
            # 支持两种格式：直接三元组或包含 metadata 的对象
            if isinstance(item, list) and len(item) >= 3:
                head, relation, tail = item[:3]
                metadata = item[3] if len(item) > 3 else {}
                triplet = Triplet(head, relation, tail, metadata)
            elif isinstance(item, dict):
                triplet = Triplet(
                    head=item.get('head', ''),
                    relation=item.get('relation', ''),
                    tail=item.get('tail', ''),
                    metadata=item.get('metadata', {})
                )
            else:
                raise ValueError(f"无法解析的三元组格式: {item}")

            self.add_triplet(triplet)

    def search_by_keyword(self, keyword: str) -> List[Triplet]:
        """
        根据关键词搜索三元组

        Args:
            keyword: 搜索关键词

        Returns:
            包含关键词的三元组列表
        """
        indices = self.inverted_index.get(keyword.lower(), [])
        return [self.triplets[i] for i in indices]

    def search_by_entity(self, entity: str) -> List[Triplet]:
        """
        根据实体搜索三元组

        Args:
            entity: 实体名称

        Returns:
            包含该实体的三元组列表（作为头实体或尾实体）
        """
        indices = self.entity_index.get(entity, [])
        return [self.triplets[i] for i in indices]

    def get_all_triplets(self) -> List[Triplet]:
        """获取所有三元组"""
        return self.triplets.copy()

    def get_triplet_texts(self) -> List[str]:
        """获取所有三元组的文本表示"""
        return [triplet.text for triplet in self.triplets]

    def size(self) -> int:
        """获取三元组数量"""
        return len(self.triplets)

    def _extract_words(self, text: str) -> List[str]:
        """
        从文本中提取单词（简单实现）

        Args:
            text: 输入文本

        Returns:
            单词列表
        """
        # 简单的分词：按空格分割，转换为小写，过滤空字符串
        words = text.lower().split()
        return [word.strip() for word in words if word.strip()]

    def save_to_json(self, filepath: str):
        """
        将知识图谱保存到 JSON 文件

        Args:
            filepath: 保存路径
        """
        data = [triplet.to_dict() for triplet in self.triplets]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def __str__(self) -> str:
        """字符串表示"""
        return f"SimpleKG with {self.size()} triplets"

    def __len__(self) -> int:
        """长度"""
        return self.size()