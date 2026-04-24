"""
主程序
演示基于半结构化知识图谱的 RAG 系统
"""

import os
import sys
from typing import List
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


from config import ConfigManager, SystemConfig
from kg import SimpleKG, Triplet
from retriever import BatchTripletRetriever, RetrievalResult
from generator import MockLLM, KGTextGenerator, GenerationConfig


def create_sample_kg_data(filepath: str = "data/kg.json"):
    """
    创建示例知识图谱数据

    Args:
        filepath: 数据文件路径
    """
    import json
    import os

    # 创建数据目录
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 示例三元组数据（半结构化格式）
    sample_data = [
        # 格式1: [head, relation, tail]
        ["Python", "是一种", "编程语言"],
        ["编程语言", "用于", "软件开发"],
        ["机器学习", "是", "人工智能的子领域"],
        ["深度学习", "基于", "神经网络"],
        ["神经网络", "受启发于", "人脑结构"],

        # 格式2: 包含 metadata 的字典
        {
            "head": "Python",
            "relation": "由",
            "tail": "Guido van Rossum 创建",
            "metadata": {"年份": 1991, "置信度": 0.95}
        },
        {
            "head": "TensorFlow",
            "relation": "是",
            "tail": "机器学习框架",
            "metadata": {"开发者": "Google", "开源": True}
        },
        {
            "head": "PyTorch",
            "relation": "是",
            "tail": "深度学习框架",
            "metadata": {"开发者": "Facebook", "动态计算图": True}
        },
        {
            "head": "RAG",
            "relation": "代表",
            "tail": "检索增强生成",
            "metadata": {"应用领域": "自然语言处理"}
        },
        {
            "head": "知识图谱",
            "relation": "用于",
            "tail": "知识表示",
            "metadata": {"类型": "语义网络"}
        }
    ]

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print(f"示例数据已创建: {filepath}")


def create_sample_config(filepath: str = "config.json"):
    """
    创建示例配置文件

    Args:
        filepath: 配置文件路径
    """
    import json

    config_data = {
        "data": {
            "kg_file": "data/kg.json",
            "embedding_model": "all-MiniLM-L6-v2",
            "batch_size": 32
        },
        "retriever": {
            "top_k": 3,
            "similarity_threshold": 0.4,
            "use_gpu": False
        },
        "debug": True,
        "log_level": "INFO"
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    print(f"示例配置已创建: {filepath}")


def print_results(query: str, results: List[RetrievalResult]):
    """
    打印检索结果

    Args:
        query: 查询文本
        results: 检索结果列表
    """
    print(f"\n查询: '{query}'")
    print("-" * 60)

    if not results:
        print("未找到相关三元组")
        return

    for result in results:
        print(f"排名 {result.rank}: {result.triplet}")
        print(f"  相似度: {result.similarity:.4f}")
        if result.triplet.metadata:
            print(f"  元数据: {result.triplet.metadata}")
        print()


def main():
    """主函数"""
    print("=" * 70)
    print("基于半结构化知识图谱的 RAG 系统")
    print("=" * 70)

    # 1. 创建示例数据文件（如果不存在）
    kg_file = "data/kg.json"
    config_file = "config.json"

    if not os.path.exists(kg_file):
        print("创建示例知识图谱数据...")
        create_sample_kg_data(kg_file)

    if not os.path.exists(config_file):
        print("创建示例配置文件...")
        create_sample_config(config_file)

    # 2. 加载配置
    print("\n1. 加载配置...")
    try:
        config = ConfigManager.from_file(config_file)
        print(f"   配置加载成功: {config_file}")
        print(f"   知识图谱文件: {config.data.kg_file}")
        print(f"   Embedding 模型: {config.data.embedding_model}")
        print(f"   检索 top_k: {config.retriever.top_k}")
        print(f"   相似度阈值: {config.retriever.similarity_threshold}")
    except Exception as e:
        print(f"   配置加载失败: {e}")
        return

    # 3. 初始化知识图谱
    print("\n2. 初始化知识图谱...")
    try:
        kg = SimpleKG()
        kg.load_from_json(config.data.kg_file)
        print(f"   知识图谱加载成功，包含 {kg.size()} 个三元组")

        # 打印前几个三元组作为示例
        print("   示例三元组:")
        for i, triplet in enumerate(kg.triplets[:3]):
            print(f"     {i+1}. {triplet}")
    except Exception as e:
        print(f"   知识图谱加载失败: {e}")
        return

    # 4. 初始化检索器
    print("\n3. 初始化检索器...")
    try:
        retriever = BatchTripletRetriever(
            kg=kg,
            model_name=config.data.embedding_model,
            device="cuda" if config.retriever.use_gpu else "cpu",
            top_k=config.retriever.top_k,
            threshold=config.retriever.similarity_threshold
        )
        print(f"   检索器初始化成功")
        print(f"   Embedding 维度: {retriever.get_embedding_dim()}")
    except Exception as e:
        print(f"   检索器初始化失败: {e}")
        print("   请确保已安装依赖: pip install sentence-transformers scikit-learn")
        return

    # 5. 定义测试查询
    test_queries = [
      
    "中国四大名著分别是哪四本？",
    "《红楼梦》的作者是谁？",
    "林黛玉住在大观园的哪里？",
    "贾宝玉的挚爱是谁？",
    "《西游记》孙悟空的武器是什么？",
    "唐僧的四个徒弟都是谁？",
    "《三国演义》桃园三结义都有谁？",
    "诸葛亮最有名的事迹有哪些？",
    "关羽的武器是什么？",
    "《水浒传》武松有哪些经典事件？",
    "梁山一百单八将的首领是谁？",
    "鲁智深最有名的事迹是什么？",
    "三国最终统一于哪个朝代？",
    "金陵十二钗都属于哪本书？",
    "孙悟空被如来佛祖压在哪里？"

    ]

    print(f"\n4. 执行批量检索 ({len(test_queries)} 个查询)...")

    # 6. 执行批量检索
    try:
        all_results = retriever.retrieve_batch(test_queries)

        # 打印每个查询的结果
        for query, results in zip(test_queries, all_results):
            print_results(query, results)
    except Exception as e:
        print(f"   检索失败: {e}")
        return

    # 7. 初始化生成器并进行文本生成
    print("\n5. 文本生成演示...")
    try:
        # 初始化模拟 LLM
        llm = MockLLM(name="MockGPT-3.5")
        print(f"   初始化 LLM: {llm.name}")

        # 初始化生成器
        gen_config = GenerationConfig(
            max_tokens=150,
            temperature=0.7,
            use_template=True,
            include_metadata=True
        )
        generator = KGTextGenerator(llm, gen_config)
        print(f"   初始化生成器: {generator}")

        # 为每个查询生成文本
        for i, query in enumerate(test_queries):
            if i < len(all_results) and all_results[i]:
                # 获取检索到的三元组（传递更多候选给生成器自主评分）
                triplets = [result.triplet for result in all_results[i]]
                results = all_results[i]

                # 生成文本
                print(f"\n   生成回答 for: '{query}'")
                response = generator.generate_from_triplets(query, triplets, results)
                print(f"   生成结果: {response[:100]}...")  # 只打印前100字符
                print(f"   LLM 调用次数: {llm.call_count}")
    except Exception as e:
        print(f"   文本生成失败: {e}")

    # 8. 系统统计信息
    print("\n" + "=" * 70)
    print("系统统计信息:")
    print(f"   知识图谱大小: {kg.size()} 个三元组")
    print(f"   测试查询数量: {len(test_queries)}")
    print(f"   检索器配置: top_k={config.retriever.top_k}, threshold={config.retriever.similarity_threshold}")
    if 'llm' in locals():
        print(f"   LLM 总调用次数: {llm.call_count}")
    print("=" * 70)
    print("演示完成！")


if __name__ == "__main__":
    main()