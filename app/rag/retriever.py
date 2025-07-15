# /app/rag/retriever.py
# title: 情報検索（レトリーバー）
# role: ナレッジベースと知識グラフから、与えられたクエリに関連する情報を検索する。

from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from app.rag.knowledge_base import KnowledgeBase
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from app.knowledge_graph.persistent_knowledge_graph import PersistentKnowledgeGraph
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

class Retriever:
    """
    ナレッジベースから関連情報を検索するクラス。
    """
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def __init__(self, knowledge_base: KnowledgeBase, persistent_knowledge_graph: PersistentKnowledgeGraph):
        """
        コンストラクタ。
        """
        if not knowledge_base.vector_store:
            raise ValueError("ナレッジベースがロードされていません。")
        
        self.langchain_retriever: Runnable = knowledge_base.vector_store.as_retriever()
        self.knowledge_graph = persistent_knowledge_graph
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def invoke(self, query: str) -> List[Document]:
        """
        指定されたクエリに最も関連性の高いドキュメントを検索します。
        ベクトルストアと知識グラフの両方から情報を取得します。
        """
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 1. ベクトルストアから情報を検索
        vector_docs = self.langchain_retriever.invoke(query)
        
        # 2. 知識グラフから関連情報を検索（簡易的なキーワード検索）
        graph_summary = self.knowledge_graph.get_summary()
        graph_docs = []
        if query.lower() in graph_summary.lower():
             graph_content = self.knowledge_graph.get_graph().to_string()
             graph_docs.append(Document(page_content=graph_content, metadata={"source": "knowledge_graph"}))
        
        # 3. 両方の結果を統合して返す
        return vector_docs + graph_docs
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️