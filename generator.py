"""
生成器模块
实现模拟 LLM 和基于知识图谱的文本生成
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from kg import Triplet
from retriever import RetrievalResult


@dataclass
class GenerationConfig:
    """生成配置类"""
    max_tokens: int = 200  # 生成的最大 token 数
    temperature: float = 0.7  # 温度参数
    use_template: bool = True  # 是否使用模板
    include_metadata: bool = False  # 是否包含元数据


class MockLLM:
    """模拟大型语言模型"""

    # 关系关键词权重映射
    KEYWORD_WEIGHTS = {
        "是谁": 0.15, "是什么": 0.15, "是哪": 0.15, "是哪个": 0.15,
        "哪里": 0.25, "住在": 0.30, "都有谁": 0.20,
        "谁": 0.15,
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

    # 关系关键词映射表
    RELATION_KEYWORDS = {
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

    def __init__(self, name: str = "MockGPT-3.5"):
        """
        初始化模拟 LLM

        Args:
            name: 模型名称
        """
        self.name = name
        self.call_count = 0

        # 预定义的响应模板（模拟不同主题的响应）
        self.response_templates = [
            "{answer}",
            "检索增强生成结果：{answer}",
            "基于知识图谱的回答：{answer}",
        ]

        # 预定义的答案片段（用于生成模拟答案）
        self.answer_fragments = [
            "根据知识图谱中的三元组信息，可以得出这一结论。",
            "检索到的知识支持这一观点。",
            "基于语义相似度计算，该推断具有较高的可信度。",
            "这一分析结果与知识图谱中的信息一致。",
            "从知识表示的角度看，这一理解是合理的。",
            "该结论得到了相关三元组的支持。",
            "基于检索增强生成的机制，这一回答是可靠的。",
            "知识图谱的语义信息验证了这一观点。"
        ]

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """
        模拟文本生成

        Args:
            prompt: 输入提示
            config: 生成配置

        Returns:
            生成的文本
        """
        self.call_count += 1

        if config is None:
            config = GenerationConfig()

        # 解析 prompt 中的上下文和查询（简单实现）
        context = self._extract_context(prompt)
        query = self._extract_query(prompt)

        # 生成模拟答案
        answer = self._generate_answer(context, query, config)

        # 选择响应模板
        template = random.choice(self.response_templates)

        # 填充模板
        response = template.format(
            context=context[:100] + "..." if len(context) > 100 else context,
            query=query[:50] + "..." if len(query) > 50 else query,
            answer=answer
        )

        # 限制响应长度
        if len(response) > config.max_tokens * 4:  # 粗略估计：1 token ≈ 4字符
            response = response[:config.max_tokens * 4] + "..."

        return response

    def _extract_context(self, prompt: str) -> str:
        """从 prompt 中提取上下文"""
        # 查找 ### 上下文 ### 标记
        if '### 上下文 ###' in prompt:
            parts = prompt.split('### 上下文 ###')
            if len(parts) > 1:
                # 获取上下文部分，直到下一个 ### 标记或结尾
                context_part = parts[1]
                # 查找下一个 ### 标记
                next_marker = context_part.find('###')
                if next_marker != -1:
                    context = context_part[:next_marker].strip()
                else:
                    context = context_part.strip()
                return context
        # 回退到旧逻辑
        lines = prompt.split('\n')
        context_lines = [line for line in lines if '上下文' in line or 'Context' in line]
        if context_lines:
            return context_lines[0].replace('上下文:', '').replace('Context:', '').strip()
        return prompt[:200]  # 返回前200个字符作为上下文

    def _extract_query(self, prompt: str) -> str:
        """从 prompt 中提取查询（只取第一行有效查询，排除后续指令文本）"""
        if '### 查询 ###' in prompt:
            parts = prompt.split('### 查询 ###')
            if len(parts) > 1:
                query_part = parts[1]
                # 只取第一个 `###` 标记之前的内容
                next_marker = query_part.find('###')
                if next_marker != -1:
                    query = query_part[:next_marker].strip()
                else:
                    # 没有 ###，则只取第一段（空行前的内容）
                    query = query_part.strip().split('\n\n')[0].strip()
                return query
        # 回退到旧逻辑
        lines = prompt.split('\n')
        query_lines = [line for line in lines if '查询' in line or 'Query' in line or '问题' in line]
        if query_lines:
            return query_lines[0].replace('查询:', '').replace('Query:', '').replace('问题:', '').strip()
        return "未知查询"

    def _generate_answer(self, context: str, query: str, config: GenerationConfig) -> str:
        """生成模拟答案（关系匹配优先，实体匹配辅助）"""
        triplets_info = self._parse_context_triplets(context)

        if not triplets_info:
            return "知识库中暂无与该问题直接相关的知识。请尝试其他查询。"

        # 对每个三元组独立打分：关系匹配（越具体权重越高）> 实体匹配 > 语义相似度
        scored = []
        for head, rel, tail, sim in triplets_info:
            score = sim  # 基础分 = 语义相似度

            # 关系匹配：使用显式权重表（具体关键词权重大，泛化疑问词权重低）
            has_rel_match = False
            max_rel_weight = 0.0
            for kw, rels in self.RELATION_KEYWORDS.items():
                if kw in query and rel in rels:
                    weight = self.KEYWORD_WEIGHTS.get(kw, 0.25)
                    if weight > max_rel_weight:
                        max_rel_weight = weight
                        has_rel_match = True
            score += max_rel_weight

            # 实体匹配：整个实体名出现在查询中
            if head in query or tail in query:
                score += 0.3

            scored.append((head, rel, tail, sim, score, has_rel_match))

        # 按最终得分降序排列
        scored.sort(key=lambda x: x[4], reverse=True)

        # 取得分最高的三元组
        head, rel, tail, sim, final_score, has_rel_match = scored[0]

        # 当查询是列举型（都有谁/都是谁/哪些/数量词），优先选择包含多值的三元组
        listing_patterns = ["都有谁", "都是谁", "有哪些", "哪些", "四个", "三个"]
        if any(p in query for p in listing_patterns):
            multi_valued = [(h, r, t, s, sc, hr) for h, r, t, s, sc, hr in scored
                           if "、" in t or "、" in r]
            if multi_valued:
                head, rel, tail, sim, final_score, has_rel_match = multi_valued[0]

        # 实体匹配到的直接返回
        if head in query or tail in query:
            return f"根据知识图谱数据，{head}{rel}{tail}。"
        # 关系匹配（非泛词）且得分合理，也接受
        if has_rel_match:
            return f"根据知识图谱数据，{head}{rel}{tail}。"
        return "知识库中暂无与该问题直接相关的知识。请尝试其他查询。"

    def _parse_context_triplets(self, context: str) -> List[Tuple[str, str, str, float]]:
        """
        解析上下文中的三元组信息

        Args:
            context: 上下文字符串

        Returns:
            三元组信息列表 [(head, relation, tail, similarity), ...]
        """
        if not context or context == "未检索到相关三元组。":
            return []

        triplets_info = []
        lines = context.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 尝试解析格式: "head relation tail (相似度: x.xxx)" 或 "head relation tail"
            sim = 0.0
            # 检查是否有相似度信息
            if ' (相似度:' in line:
                parts = line.split(' (相似度:')
                content_part = parts[0].strip()
                try:
                    # 先按 ')' 分割提取数值部分，避免元数据干扰
                    sim_str = parts[1].split(')')[0].strip()
                    sim = float(sim_str)
                except (ValueError, IndexError):
                    sim = 0.0
            else:
                content_part = line
                # 检查是否有元数据
                if ' [' in content_part:
                    content_part = content_part.split(' [')[0]

            # 解析三元组内容
            # 格式: "head relation tail"
            content_parts = content_part.split()
            if len(content_parts) >= 3:
                head = content_parts[0]
                relation = content_parts[1]
                tail = ' '.join(content_parts[2:])
                triplets_info.append((head, relation, tail, sim))

        return triplets_info

    def reset_counter(self):
        """重置调用计数器"""
        self.call_count = 0

    def __str__(self) -> str:
        return f"MockLLM(name={self.name}, calls={self.call_count})"


class KGTextGenerator:
    """基于知识图谱的文本生成器"""

    def __init__(self, llm: MockLLM, config: Optional[GenerationConfig] = None):
        """
        初始化生成器

        Args:
            llm: 语言模型实例
            config: 生成配置
        """
        self.llm = llm
        self.config = config or GenerationConfig()

    def generate_from_triplets(self, query: str, triplets: List[Triplet],
                               results: Optional[List[RetrievalResult]] = None) -> str:
        """
        基于检索到的三元组生成文本

        Args:
            query: 用户查询
            triplets: 检索到的三元组列表
            results: 检索结果（包含相似度等信息）

        Returns:
            生成的文本
        """
        # 构建上下文
        context = self._build_context(triplets, results)

        # 构建 prompt
        prompt = self._build_prompt(query, context)

        # 调用 LLM 生成
        response = self.llm.generate(prompt, self.config)

        return response

    def _build_context(self, triplets: List[Triplet],
                       results: Optional[List[RetrievalResult]] = None) -> str:
        """
        构建上下文字符串

        Args:
            triplets: 三元组列表
            results: 检索结果

        Returns:
            上下文字符串
        """
        if not triplets:
            return "未检索到相关三元组。"

        context_parts = []
        for i, triplet in enumerate(triplets):
            # 如果有检索结果，包含相似度信息
            if results and i < len(results):
                similarity = results[i].similarity
                context_parts.append(f"{triplet.text} (相似度: {similarity:.3f})")
            else:
                context_parts.append(triplet.text)

            # 如果配置中包含元数据，添加元数据
            if self.config.include_metadata and triplet.metadata:
                metadata_str = ", ".join(f"{k}: {v}" for k, v in triplet.metadata.items())
                context_parts[-1] += f" [{metadata_str}]"

        return "\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """
        构建 prompt

        Args:
            query: 用户查询
            context: 上下文

        Returns:
            prompt 字符串
        """
        if self.config.use_template:
            prompt = f"""### 上下文 ###
{context}

### 查询 ###
{query}

请严格基于以上上下文回答问题。注意：
1. 只使用与查询实体直接相关的信息，忽略无关的三元组
2. 如果上下文中没有与查询相关的知识，请回答"知识库暂无该问题答案"
3. 回答应当简洁准确"""
        else:
            prompt = f"上下文：{context}\n\n问题：{query}\n\n回答："

        return prompt

    def batch_generate(self, queries: List[str],
                       triplets_list: List[List[Triplet]]) -> List[str]:
        """
        批量生成文本

        Args:
            queries: 查询列表
            triplets_list: 每个查询对应的三元组列表

        Returns:
            生成的文本列表
        """
        if len(queries) != len(triplets_list):
            raise ValueError("queries 和 triplets_list 的长度必须相同")

        responses = []
        for query, triplets in zip(queries, triplets_list):
            response = self.generate_from_triplets(query, triplets)
            responses.append(response)

        return responses

    def __str__(self) -> str:
        return f"KGTextGenerator(llm={self.llm.name})"