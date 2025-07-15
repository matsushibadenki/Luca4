# /app/conceptual_reasoning/imagination_engine.py
# title: 想像エンジン
# role: 潜在空間で概念ベクトルを操作し、新しい概念を「想像」し、創発させる。

import logging
import numpy as np

logger = logging.getLogger(__name__)

class ImaginationEngine:
    """
    潜在空間（ベクトル空間）で概念を操作するためのエンジン。
    """
    def combine_concepts(self, vectors: list[np.ndarray], weights: list[float]) -> np.ndarray:
        """
        複数の概念ベクトルを重み付きで合成し、新しい概念を生成する。
        例：「ライオン」のベクトルと「猫」のベクトルを合成して「ライオンのような猫」を表現する。
        """
        if len(vectors) != len(weights) or not vectors:
            logger.error("ベクトルと重みの数が一致しないか、入力が空です。")
            return np.array([])
        
        weighted_vectors = [vec * w for vec, w in zip(vectors, weights)]
        combined_vector = np.sum(weighted_vectors, axis=0)
        
        # 結果を正規化して返す
        norm = np.linalg.norm(combined_vector)
        if norm == 0:
            return combined_vector
        return combined_vector / norm

    def find_analogy(self, start_vec_a: np.ndarray, end_vec_a: np.ndarray, start_vec_b: np.ndarray) -> np.ndarray:
        """
        アナロジー（類推）によって新しい概念ベクトルを見つける。
        「AにとってのBは、Cにとっての何か？」という問いに答える。
        例: King - Man + Woman = Queen
        """
        analogy_vector = start_vec_b + (end_vec_a - start_vec_a)
        
        # 結果を正規化して返す
        norm = np.linalg.norm(analogy_vector)
        if norm == 0:
            return analogy_vector
        return analogy_vector / norm