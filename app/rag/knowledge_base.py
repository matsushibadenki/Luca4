# /app/rag/knowledge_base.py
# title: ナレッジベース管理
# role: ドキュメントの読み込み、追加、ベクトルストアの構築と管理を行う。

from __future__ import annotations
import os
import logging
from typing import Optional, List
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    ドキュメントを管理し、ベクトルストアを構築・更新するクラス。
    """
    def __init__(self, embedding_model_name: str):
        self.vector_store: Optional[FAISS] = None
        self.embeddings = OllamaEmbeddings(model=embedding_model_name)
        self.text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def _load_and_build_store(self, source_file_path: str):
        """
        指定されたソースからドキュメントを読み込み、ベクトルストアを構築する内部メソッド。
        """
        if not os.path.exists(source_file_path):
            logger.warning(f"ナレッジベースのソースファイルが見つかりません: {source_file_path}。空のナレッジベースで起動します。")
            self.vector_store = FAISS.from_texts([""], self.embeddings)
            return

        try:
            with open(source_file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            texts = self.text_splitter.split_text(raw_text)
            documents = [Document(page_content=t) for t in texts]
            
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"ナレッジベースが {source_file_path} から正常に読み込まれ、インデックス化されました。")

        except Exception as e:
            logger.error(f"ナレッジベースの読み込み中に問題が発生しました: {e}", exc_info=True)
            self.vector_store = FAISS.from_texts([""], self.embeddings)

    @classmethod
    def create_and_load(cls, source_file_path: str) -> KnowledgeBase:
        """
        インスタンスを生成し、ドキュメントをロードするクラスメソッド。
        """
        kb = cls(embedding_model_name=settings.EMBEDDING_MODEL_NAME)
        kb._load_and_build_store(source_file_path)
        return kb

    def add_documents(self, documents: List[Document]):
        """
        既存のベクトルストアに新しいドキュメントを追加する。
        """
        if not self.vector_store:
            logger.error("知識ベースが初期化されていないため、ドキュメントを追加できません。")
            return

        logger.info(f"{len(documents)}個の新しいドキュメントを知識ベースに追加します。")
        try:
            chunks = self.text_splitter.split_documents(documents)
            self.vector_store.add_documents(chunks)
            logger.info("知識ベースの更新が完了しました。")
        except Exception as e:
            logger.error(f"ドキュメントの追加中にエラーが発生しました: {e}", exc_info=True)