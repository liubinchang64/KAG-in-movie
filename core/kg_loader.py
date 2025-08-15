import json
import os
import yaml
import logging
import networkx as nx
from core.llm.models import BGEEmbeddingModel
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeGraphLoader:
    """
    知识图谱加载器类
    用于加载已构建并保存的知识图谱，支持后续直接调用，无需重复构建
    """
    def __init__(self, config):
        """
        初始化知识图谱加载器
        
        Args:
            config: 配置字典，包含知识图谱文件路径等信息
        """
        self.config = config
        self.G = None  # 知识图谱
        self.vector_store = None  # 向量存储
        self.embed_model = BGEEmbeddingModel(config)  # 嵌入模型
        
        # 验证并设置文件路径
        required_paths = ["kg_gexf_path", "kg_json_path", "kg_vector_store_path"]
        for path_key in required_paths:
            if path_key not in config:
                raise ValueError(f"配置中缺少必要的路径: {path_key}")
                
        self.gexf_path = config["kg_gexf_path"]
        self.json_path = config["kg_json_path"]
        self.vector_store_path = config["kg_vector_store_path"]
        
    def load_graph(self, format="gexf"):
        """
        加载知识图谱
        
        Args:
            format: 加载格式，支持'gexf'或'json'
        
        Returns:
            bool: 加载是否成功
        """
        if format == "gexf" and os.path.exists(self.gexf_path):
            try:
                self.G = nx.read_gexf(self.gexf_path)
                logger.info(f"成功从GEXF文件加载知识图谱: {self.gexf_path}")
                logger.info(f"知识图谱统计: {len(self.G.nodes)} 个节点, {len(self.G.edges)} 条边")
                return True
            except Exception as e:
                logger.error(f"加载GEXF格式知识图谱失败: {e}")
                return False
        elif format == "json" and os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                
                self.G = nx.DiGraph()
                
                # 添加节点
                for node_data in graph_data['nodes']:
                    node_id = node_data.pop('id')
                    self.G.add_node(node_id, **node_data)
                
                # 添加边
                for edge_data in graph_data['edges']:
                    self.G.add_edge(edge_data['source'], edge_data['target'], relation=edge_data['relation'])
                
                logger.info(f"成功从JSON文件加载知识图谱: {self.json_path}")
                logger.info(f"知识图谱统计: {len(self.G.nodes)} 个节点, {len(self.G.edges)} 条边")
                return True
            except Exception as e:
                logger.error(f"加载JSON格式知识图谱失败: {e}")
                return False
        else:
            logger.error(f"指定格式 {format} 的知识图谱文件不存在")
            return False
    
    def load_vector_store(self):
        """
        加载节点嵌入向量存储
        
        Returns:
            bool: 加载是否成功
        """
        if os.path.exists(self.vector_store_path):
            try:
                with open(self.vector_store_path, 'r', encoding='utf-8') as f:
                    vector_store_data = json.load(f)
                
                self.vector_store = SimpleVectorStore.from_dict(vector_store_data)
                logger.info(f"成功加载节点嵌入向量存储: {self.vector_store_path}")
                # 直接使用vector_store_data的长度，因为它本身就是embedding_dict
                logger.info(f"向量存储包含 {len(vector_store_data)} 个节点嵌入")
                return True
            except Exception as e:
                logger.error(f"加载节点嵌入向量存储失败: {e}")
                return False
        else:
            logger.error(f"节点嵌入向量存储文件不存在: {self.vector_store_path}")
            return False
    
    def get_node_info(self, node_id):
        """
        获取节点信息
        
        Args:
            node_id: 节点ID
        
        Returns:
            dict: 节点属性字典，如果节点不存在则返回None
        """
        if self.G is None or node_id not in self.G.nodes:
            logger.warning(f"节点 {node_id} 不存在或知识图谱未加载")
            return None
        
        return dict(self.G.nodes[node_id])
    
    def get_node_relations(self, node_id, relation_type=None):
        """
        获取节点的关系
        
        Args:
            node_id: 节点ID
            relation_type: 关系类型，可选，指定则只返回该类型的关系
        
        Returns:
            list: 关系列表，每个元素为(source, target, relation)
        """
        if self.G is None or node_id not in self.G.nodes:
            logger.warning(f"节点 {node_id} 不存在或知识图谱未加载")
            return []
        
        relations = []
        # outgoing edges (node -> others)
        for u, v, d in self.G.out_edges(node_id, data=True):
            if relation_type is None or d['relation'] == relation_type:
                relations.append((u, v, d['relation']))
        
        # incoming edges (others -> node)
        for u, v, d in self.G.in_edges(node_id, data=True):
            if relation_type is None or d['relation'] == relation_type:
                relations.append((u, v, d['relation']))
        
        return relations
    
    def _create_query_embedding(self, query_text):
        """
        创建查询嵌入向量
        
        Args:
            query_text: 查询文本
        
        Returns:
            list: 嵌入向量，如果失败则返回None
        """
        try:
            query_embedding = self.embed_model._get_text_embedding(query_text)
            if not query_embedding:
                logger.error("未能生成查询嵌入")
                return None
            return query_embedding
        except Exception as e:
            logger.error(f"生成查询嵌入失败: {e}")
            return None
    
    def search_similar_nodes(self, query_text, top_k=5):
        """
        基于文本查询搜索相似节点
        
        Args:
            query_text: 查询文本
            top_k: 返回的最相似节点数量
        
        Returns:
            list: 相似节点列表，每个元素为(node_id, similarity_score)
        """
        if self.vector_store is None:
            logger.error("向量存储未加载，无法进行相似性搜索")
            return []
        
        try:
            # 生成查询嵌入
            query_embedding = self._create_query_embedding(query_text)
            if not query_embedding:
                return []
            
            # 创建查询对象
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                filters=None
            )
            
            # 执行查询
            results = self.vector_store.query(query)
            
            # 处理查询结果
            if hasattr(results, 'nodes') and results.nodes is not None:
                similar_nodes = [(node.node_id, node.score) for node in results.nodes]
                logger.info(f"找到 {len(similar_nodes)} 个相似节点")
                return similar_nodes
            elif hasattr(results, 'similarities') and hasattr(results, 'ids'):
                similar_nodes = list(zip(results.ids, results.similarities))
                logger.info(f"找到 {len(similar_nodes)} 个相似节点")
                return similar_nodes
            else:
                logger.warning(f"查询结果格式不符合预期: {results}")
                return []
        except Exception as e:
            logger.error(f"相似性搜索失败: {e}")
            return []
    
    def get_related_nodes(self, node_id, relation_type=None, node_type=None):
        """
        获取与指定节点相关的节点
        
        Args:
            node_id: 节点ID
            relation_type: 关系类型，可选
            node_type: 节点类型，可选
        
        Returns:
            list: 相关节点列表，每个元素为(node_id, node_info)
        """
        if self.G is None or node_id not in self.G.nodes:
            logger.warning(f"节点 {node_id} 不存在或知识图谱未加载")
            return []
        
        related_nodes = []
        # 检查出边
        for u, v, d in self.G.out_edges(node_id, data=True):
            if (relation_type is None or d['relation'] == relation_type) and (node_type is None or self.G.nodes[v].get('type') == node_type):
                related_nodes.append((v, self.G.nodes[v]))
        
        return related_nodes

# 示例使用代码
if __name__ == '__main__':
    try:
        # 加载配置
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 创建知识图谱加载器
        kg_loader = KnowledgeGraphLoader(config)
        
        # 加载知识图谱
        kg_loader.load_graph(format="gexf")
        
        # 加载向量存储
        kg_loader.load_vector_store()
        
        # 示例1: 获取节点信息
        movie_info = kg_loader.get_node_info("肖申克的救赎")
        print(f"\n电影信息: {movie_info}")
        
        # 示例2: 获取节点关系
        relations = kg_loader.get_node_relations("肖申克的救赎")
        print(f"\n关系信息: {relations}")
        
        # 示例3: 搜索相似节点
        similar_nodes = kg_loader.search_similar_nodes("感人的剧情片")
        print(f"\n相似节点: {similar_nodes}")
        
        # 示例4: 获取导演的电影
        director_movies = kg_loader.get_related_nodes("弗兰克·德拉邦特", relation_type="由...执导", node_type="movie")
        print(f"\n弗兰克·德拉邦特导演的电影: {[movie[0] for movie in director_movies]}")
        
        # 示例5: 获取电影的演员
        movie_actors = kg_loader.get_related_nodes("肖申克的救赎", relation_type="由...主演", node_type="person")
        print(f"\n肖申克的救赎的演员: {[actor[0] for actor in movie_actors]}")
    except Exception as e:
        logger.error(f"示例代码运行失败: {e}")
        import traceback
        traceback.print_exc()