# /app/rag/retriever.py
# title: 情報検索（レトリーバー）
# role: ナレッジベースから、与えられたクエリに関連する情報を検索する。

from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from app.rag.knowledge_base import KnowledgeBase

class Retriever:
    """
    ナレッジベースから関連情報を検索するクラス。
    """
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        コンストラクタ。
        """
        if not knowledge_base.vector_store:
            raise ValueError("ナレッジベースがロードされていません。")
        
        self.langchain_retriever: Runnable = knowledge_base.vector_store.as_retriever()

    def invoke(self, query: str) -> List[Document]:
        """
        指定されたクエリに最も関連性の高いドキュメントを検索します。
        """
        return self.langchain_retriever.invoke(query)