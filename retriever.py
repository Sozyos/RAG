"""
检索器模块
实现基于向量相似度的批量三元组检索
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# 导入第三方库（在运行时检查）
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"警告: 缺少依赖库 - {e}")
    print("请运行: pip install sentence-transformers scikit-learn")
    IMPORT_SUCCESS = False

from kg import Triplet, SimpleKG


@dataclass
class RetrievalResult:
    """检索结果数据类"""
    query: str  # 查询文本
    triplet: Triplet  # 检索到的三元组
    similarity: float  # 相似度分数
    rank: int  # 排名（从1开始）

    def __str__(self) -> str:
        return f"查询: '{self.query}' -> {self.triplet} (相似度: {self.similarity:.4f}, 排名: {self.rank})"


class BatchTripletRetriever:
    """批量三元组检索器（支持混合检索：语义相似度 + 实体匹配 + 关系约束）"""

    def __init__(self, kg: SimpleKG, model_name: str = "all-MiniLM-L6-v2",
                 device: str = "cpu", top_k: int = 5, threshold: float = 0.5):
        """
        初始化检索器

        Args:
            kg: 知识图谱对象
            model_name: sentence-transformers 模型名称
            device: 计算设备 ('cpu' 或 'cuda')
            top_k: 返回最相似的 top_k 个结果
            threshold: 相似度阈值，低于此值的结果将被过滤
        """
        if not IMPORT_SUCCESS:
            raise ImportError("缺少必要的依赖库，请安装 sentence-transformers 和 scikit-learn")

        self.kg = kg
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self.threshold = threshold

        # 加载 embedding 模型
        print(f"正在加载模型: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)

        # 预计算所有三元组的 embedding
        print("正在预计算知识图谱 embedding...")
        self.triplet_texts = kg.get_triplet_texts()
        self.triplet_embeddings = self._encode_batch(self.triplet_texts)
        print(f"预计算完成，共 {len(self.triplet_texts)} 个三元组")

        # 构建实体字典（去重），用于实体识别
        all_entities = set()
        for triplet in kg.triplets:
            all_entities.add(triplet.head)
            all_entities.add(triplet.tail)
        # 按长度降序排序，确保长实体优先匹配（如"《红楼梦》"优先于"红楼梦"）
        self.known_entities = sorted(all_entities, key=lambda x: len(x), reverse=True)

        # 构建关系词到关系名的映射（用于关系约束）
        self._build_relation_keywords()

    def _build_relation_keywords(self):
        """构建关系关键词映射表"""
        # 关键词权重映射（具体关键词权重高，泛化疑问词权重低）
        self.keyword_weights: Dict[str, float] = {
            # 泛化疑问词（权重低，避免淹没具体关键词）
            "是谁": 0.15, "是什么": 0.15, "是哪": 0.15, "是哪个": 0.15,
            "哪里": 0.25, "住在": 0.30, "都有谁": 0.20,
            "谁": 0.15,
            # 具体关键词（权重大）
            "事迹": 0.40, "经典": 0.35, "包括": 0.35, "包含": 0.30,
            "属于": 0.35, "出自": 0.35, "统一": 0.35,
            "武器是什么": 0.50, "事迹有哪些": 0.50, "首领是谁": 0.50,
            "武器": 0.40, "法宝": 0.40, "作者": 0.40,
            "徒弟都是": 0.45, "徒弟": 0.35, "师傅": 0.35,
            "父亲": 0.35, "母亲": 0.35, "哥哥": 0.35,
            "挚爱": 0.50, "妻子": 0.35, "挚爱是谁": 0.55,
            "身份": 0.30, "称号": 0.30, "性格": 0.30,
            "结局": 0.35, "故事": 0.30, "本领": 0.40, "技能": 0.35,
            "能力": 0.35, "压": 0.30, "哪个": 0.25, "哪些": 0.35,
            "多少": 0.30, "首领": 0.35, "人物": 0.35,
        }

        # 常见问题词对应的关系关键词集合
        self.relation_keywords: Dict[str, List[str]] = {
            "作者": ["作者是", "作者为"],
            "武器": ["武器是", "兵器是"],
            "法宝": ["法宝是", "武器是"],
            "兵器": ["兵器是", "武器是"],
            "哪里": ["居所是", "地点是", "被镇压于", "被压在", "根据地是"],
            "住在": ["居所是", "地点是"],
            "谁": ["身份是", "称号是", "身份为", "是"],
            "是什么": ["身份是", "称号是", "类型是", "别名是", "武器是", "法宝是"],
            "是哪": ["身份是", "称号是", "类型是"],
            "包括": ["包含", "包括"],
            "包含": ["包含", "包括"],
            "事迹": ["事迹是", "经典事迹", "事迹"],
            "事件": ["事迹是", "经典事迹", "事迹"],
            "经典": ["事迹是", "经典事迹"],
            "故事": ["事迹是", "经典事迹", "主线是"],
            "结局": ["结局是"],
            "性格": ["性格是"],
            "身份": ["身份是", "职位是"],
            "称号": ["称号是"],
            "本领": ["神通是", "技能是", "绝招是", "武器是"],
            "技能": ["神通是", "技能是", "绝招是"],
            "能力": ["神通是", "技能是", "绝招是"],
            "压": ["被压在", "被镇压于"],
            "属于": ["出自", "包含", "属于"],
            "出自": ["出自", "包含"],
            "统一": ["统一于", "统一者是", "最终统一者是"],
            "首领": ["地位是", "身份是", "称号是"],
            "首领是谁": ["地位是", "身份是", "首领是"],
            "徒弟": ["徒弟是", "大徒弟是", "二徒弟是", "三徒弟是", "徒弟"],
            "徒弟都是": ["徒弟是", "大徒弟是", "二徒弟是", "三徒弟是"],
            "师傅": ["师傅是"],
            "父亲": ["父亲是"],
            "母亲": ["母亲是"],
            "哥哥": ["哥哥是"],
            "哪个": ["出自", "作者是", "类型是"],
            "哪些": ["事迹是", "经典事迹", "包含", "包括", "神通是"],
            "事迹有哪些": ["事迹是", "经典事迹"],
            "多少": ["排名是"],
            "武器是什么": ["武器是", "兵器是"],
            "人物": ["人物是", "主人公是", "核心人物是"],
            "挚爱是谁": ["挚爱为", "挚爱是"],
            "都有谁": ["人物是", "身份是", "地位是"],
        }

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量编码文本为向量

        Args:
            texts: 文本列表

        Returns:
            embedding 矩阵，形状为 (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        # 使用 sentence-transformers 进行批量编码
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # 归一化以便使用余弦相似度
        )
        return embeddings

    def _extract_entities(self, query: str) -> List[str]:
        """
        从查询中提取已知实体（长实体优先，互斥匹配避免短实体侵蚀）

        Args:
            query: 查询文本

        Returns:
            查询中包含的已知实体列表
        """
        matched = []
        remaining = query
        for entity in self.known_entities:
            if entity in remaining:
                matched.append(entity)
                # 将已匹配的实体从剩余文本中移除，避免短实体重复匹配
                remaining = remaining.replace(entity, "◆" * len(entity), 1)
        return matched

    def _compute_entity_boost(self, query: str) -> np.ndarray:
        """
        计算实体匹配增强向量（精确实体匹配 + 关系约束）

        Args:
            query: 查询文本

        Returns:
            boost 向量，形状为 (n_triplets,)，匹配的三元组获得额外加分
        """
        n_triplets = len(self.kg.triplets)
        boost = np.zeros(n_triplets, dtype=np.float32)

        # 1) 实体匹配增强（精确实体匹配，而非子串匹配）
        matched_entities = self._extract_entities(query)
        if matched_entities:
            for entity in matched_entities:
                for idx, triplet in enumerate(self.kg.triplets):
                    # 精确实体匹配：head/tail 必须完全等于实体名，避免"贾宝玉"误配"贾宝玉首席丫鬟"
                    if entity == triplet.head or entity == triplet.tail:
                        boost[idx] += 0.45  # 实体匹配加分

        # 2) 关系关键词约束：取最具体的关键词匹配（越长越具体），避免泛词覆盖
        for idx, triplet in enumerate(self.kg.triplets):
            max_weight = 0.0
            for q_word, relations in self.relation_keywords.items():
                if q_word in query:
                    for rel in relations:
                        if rel == triplet.relation:
                            # 使用显式权重表（具体关键词权重大，泛化疑问词权重低）
                            weight = self.keyword_weights.get(q_word, 0.25)
                            max_weight = max(max_weight, weight)
            boost[idx] += max_weight

        return boost

    def retrieve_batch(self, queries: List[str]) -> List[List[RetrievalResult]]:
        """
        批量检索：为每个查询检索最相关的三元组
        使用混合评分 = cosine_similarity + entity_boost + relation_matching

        Args:
            queries: 查询文本列表

        Returns:
            检索结果列表，每个元素是一个查询的结果列表
        """
        if not queries:
            return []

        # 批量编码查询
        print(f"正在编码 {len(queries)} 个查询...")
        query_embeddings = self._encode_batch(queries)

        # 计算余弦相似度矩阵
        print("正在计算相似度...")
        similarity_matrix = cosine_similarity(query_embeddings, self.triplet_embeddings)

        # 为每个查询获取 top-k 结果（混合评分）
        all_results = []
        for i, query in enumerate(queries):
            # 语义相似度分数
            query_similarities = similarity_matrix[i].copy()

            # 实体匹配 + 关系约束增强
            entity_boost = self._compute_entity_boost(query)
            hybrid_scores = query_similarities + entity_boost

            # 获取混合评分最高的索引
            top_indices = np.argsort(hybrid_scores)[::-1]

            # 过滤和收集结果
            query_results = []
            seen_pairs = set()  # 去重：避免同一个 (head, relation, tail) 重复
            rank = 1
            for idx in top_indices:
                hybrid_score = hybrid_scores[idx]
                similarity = query_similarities[idx]

                # 应用阈值过滤
                if hybrid_score < self.threshold:
                    continue

                # 获取对应的三元组
                triplet = self.kg.triplets[idx]

                # 去重检查
                triplet_key = (triplet.head, triplet.relation, triplet.tail)
                if triplet_key in seen_pairs:
                    continue
                seen_pairs.add(triplet_key)

                # 创建结果对象（记录原始相似度和混合评分）
                result = RetrievalResult(
                    query=query,
                    triplet=triplet,
                    similarity=round(hybrid_score, 4),
                    rank=rank
                )
                query_results.append(result)

                # 达到 top_k 限制时停止
                if len(query_results) >= self.top_k:
                    break

                rank += 1

            all_results.append(query_results)

        return all_results

    def retrieve_single(self, query: str) -> List[RetrievalResult]:
        """
        单个查询检索（批量检索的包装方法）

        Args:
            query: 查询文本

        Returns:
            检索结果列表
        """
        results = self.retrieve_batch([query])
        return results[0] if results else []

    def get_triplet_by_index(self, index: int) -> Optional[Triplet]:
        """
        根据索引获取三元组

        Args:
            index: 三元组索引

        Returns:
            三元组对象，如果索引无效则返回 None
        """
        if 0 <= index < len(self.kg.triplets):
            return self.kg.triplets[index]
        return None

    def get_embedding_dim(self) -> int:
        """获取 embedding 维度"""
        if len(self.triplet_embeddings) > 0:
            return self.triplet_embeddings.shape[1]
        return 0

    def __str__(self) -> str:
        """字符串表示"""
        return (f"BatchTripletRetriever(model={self.model_name}, "
                f"triplets={len(self.triplet_texts)}, "
                f"top_k={self.top_k}, threshold={self.threshold})")