# /app/knowledge_graph/models.py
# title: 知識グラフデータモデル
# role: 知識グラフを構成するNode, Edge, KnowledgeGraphのデータ構造を定義する。

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class Node(BaseModel):
    """
    知識グラフのノード（エンティティ）を表すクラス。
    """
    id: str = Field(..., description="エンティティの一意なID（例：'地球'）")
    label: str = Field(..., description="エンティティのカテゴリ（例：'Planet'）")
    properties: Dict[str, Any] = Field(default_factory=dict, description="エンティティの属性")
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {"created_at": datetime.utcnow().isoformat(), "last_accessed": datetime.utcnow().isoformat()},
        description="ノードのメタデータ"
    )

class Edge(BaseModel):
    """
    知識グラフのエッジ（関係）を表すクラス。
    """
    source: str = Field(..., description="関係の始点となるノードのID")
    target: str = Field(..., description="関係の終点となるノードのID")
    label: str = Field(..., description="関係のタイプ")
    properties: Dict[str, Any] = Field(default_factory=dict, description="関係の属性")
    weight: float = Field(default=1.0, description="関係の強度や確信度。")

class KnowledgeGraph(BaseModel):
    """
    ノードとエッジのコレクションとして知識グラフを表現するクラス。
    """
    nodes: List[Node] = Field(default_factory=list, description="グラフ内のノードのリスト")
    edges: List[Edge] = Field(default_factory=list, description="グラフ内のエッジのリスト")

    def to_string(self) -> str:
        """
        知識グラフの内容を人間が読める文字列形式に変換する。
        """
        if not self.nodes and not self.edges:
            return "知識グラフは空です。"

        node_str = "\n".join([f"- ノード: {n.id} (ラベル: {n.label}, プロパティ: {n.properties})" for n in self.nodes])
        edge_str = "\n".join([f"- 関係: ({e.source})-[{e.label} (信頼度: {e.weight:.2f})]->({e.target})" for e in self.edges])

        return f"--- 知識グラフ ---\n[ノード]\n{node_str}\n\n[関係]\n{edge_str}\n----------------"