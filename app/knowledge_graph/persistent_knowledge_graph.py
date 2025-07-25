# /app/knowledge_graph/persistent_knowledge_graph.py
# title: 永続的知識グラフ管理
# role: 知識グラフをファイルに保存し、ロードし、マージする機能を提供する。

import json
import logging
import os
from typing import Set, Dict
from datetime import datetime

from .models import KnowledgeGraph, Node, Edge

logger = logging.getLogger(__name__)

class PersistentKnowledgeGraph:
    """
    ファイルベースで知識グラフを永続化し、更新を管理するクラス。
    """
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.graph = self._load()

    def _load(self) -> KnowledgeGraph:
        """ストレージから知識グラフをロードする。"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return KnowledgeGraph.model_validate(data)
            except (IOError, json.JSONDecodeError) as e:
                logger.error(f"永続的知識グラフのロードに失敗しました: {e}. 新しいグラフを作成します。")
        return KnowledgeGraph()

    def save(self) -> None:
        """現在の知識グラフをストレージに保存する。"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                f.write(self.graph.model_dump_json(indent=4))
            logger.info(f"知識グラフが {self.storage_path} に保存されました。")
        except IOError as e:
            logger.error(f"知識グラフの保存に失敗しました: {e}")

    def merge(self, new_graph: KnowledgeGraph) -> None:
        """
        新しいグラフを既存のグラフにマージする。
        """
        if not new_graph or not hasattr(new_graph, 'nodes'):
            logger.warning("マージ対象の知識グラフが無効です。")
            return

        existing_node_ids: Set[str] = {node.id for node in self.graph.nodes}
        for new_node in new_graph.nodes:
            if new_node.id not in existing_node_ids:
                self.graph.nodes.append(new_node)
                existing_node_ids.add(new_node.id)

        edge_map: Dict[str, Edge] = {
            f"{edge.source}-{edge.label}-{edge.target}": edge for edge in self.graph.edges
        }

        for new_edge in new_graph.edges:
            edge_key = f"{new_edge.source}-{new_edge.label}-{new_edge.target}"
            if edge_key in edge_map:
                existing_edge = edge_map[edge_key]
                existing_edge.weight += new_edge.weight
                logger.info(f"Edge weight updated (LTP): {edge_key}, new weight: {existing_edge.weight}")
            else:
                self.graph.edges.append(new_edge)
                edge_map[edge_key] = new_edge
        
        logger.info(f"知識グラフをマージしました。現在のノード数: {len(self.graph.nodes)}, エッジ数: {len(self.graph.edges)}")

    def get_graph(self) -> KnowledgeGraph:
        """現在のグラフオブジェクトを返す。"""
        return self.graph

    def get_summary(self) -> str:
        """知識グラフの概要を返す。"""
        if not self.graph.nodes and not self.graph.edges:
            return "知識グラフは空です。"
        
        num_nodes = len(self.graph.nodes)
        num_edges = len(self.graph.edges)
        
        sample_labels = list(set(node.label for node in self.graph.nodes[:5]))
        
        return (f"知識グラフには {num_nodes}個のノードと {num_edges}個のエッジが含まれています。"
                f"主なエンティティカテゴリ: {sample_labels}")

    def access_node(self, node_id: str) -> None:
        """ノードへのアクセスを記録し、最終アクセス日時を更新する。"""
        for node in self.graph.nodes:
            if node.id == node_id:
                if "last_accessed" in node.metadata:
                    node.metadata["last_accessed"] = datetime.utcnow().isoformat()
                break