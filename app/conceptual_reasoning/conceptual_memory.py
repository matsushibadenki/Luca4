# /app/conceptual_reasoning/conceptual_memory.py
# title: 概念記憶
# role: 概念ベクトルを保存・検索するための専門の記憶領域（ベクトルデータベース）。

import logging
import numpy as np
import faiss
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ConceptualMemory:
    """
    FAISSを利用して概念ベクトルを効率的に保存・検索するクラス。
    """
    def __init__(self, dimension: int):
        # 確実にint型に変換してfaissの型エラーを回避する
        self.dimension = int(dimension)
        # FAISSインデックスの初期化
        self.index = faiss.IndexFlatL2(self.dimension)
        # ベクトルとそれに対応するメタデータ（例：元のテキスト）を保存するリスト
        self.stored_vectors: list[np.ndarray] = []
        self.metadata: list[dict] = []
        logger.info(f"概念記憶が次元数 {self.dimension} で初期化されました。")

    def add_concepts(self, vectors: np.ndarray, metadata_list: list[dict]):
        """
        複数の新しい概念ベクトルとメタデータを記憶に追加する。
        """
        if vectors.shape[1] != self.dimension:
            logger.error(f"追加しようとしたベクトルの次元 ({vectors.shape[1]}) が、メモリの次元 ({self.dimension}) と一致しません。")
            return
        
        self.index.add(vectors.astype('float32'))
        self.stored_vectors.extend(list(vectors))
        self.metadata.extend(metadata_list)
        logger.info(f"{len(vectors)}個の新しい概念が記憶に追加されました。現在の総数: {self.index.ntotal}")

    def search_similar_concepts(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        与えられたベクトルに最も類似する概念をk個検索し、そのメタデータを返す。
        """
        if self.index.ntotal == 0:
            return []
            
        distances, indices = self.index.search(np.array([query_vector]).astype('float32'), k)
        
        results: List[Dict[str, Any]] = []
        for idx, i in enumerate(indices[0]):
            if i != -1:
                results.append({
                    "metadata": self.metadata[i],
                    "vector": self.stored_vectors[i].tolist(),
                    "distance": float(distances[0][idx])
                })
        return results