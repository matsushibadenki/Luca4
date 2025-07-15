# /app/conceptual_reasoning/sensory_processing_unit.py
# title: 感覚処理ユニット
# role: テキスト、画像、音声などの多様なモダリティの情報を、共通の潜在空間上の「概念ベクトル」に変換する。

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SensoryProcessingUnit:
    """
    多様なモダリティの情報を共通の概念ベクトルに変換するユニット。
    sentence-transformersライブラリの事前学習済みCLIPモデルを利用する。
    """
    def __init__(self, model_name: str = 'clip-ViT-B-32'):
        try:
            self.model = SentenceTransformer(model_name)
            self._embedding_dimension: Optional[int] = None # キャッシュ用
            logger.info(f"感覚処理ユニットがモデル '{model_name}' で初期化されました。")
        except Exception as e:
            logger.error(f"SentenceTransformerモデル '{model_name}' のロードに失敗しました: {e}", exc_info=True)
            raise

    def get_embedding_dimension(self) -> int:
        """
        エンベディングの次元数を取得する。初回呼び出し時に計算し、キャッシュする。
        """
        if self._embedding_dimension is None:
            # sentence-transformersの組み込みメソッドを試す
            dimension = self.model.get_sentence_embedding_dimension()
            if dimension is None:
                # 取得できない場合は、ダミーエンコーディングで次元を決定する
                logger.warning("get_sentence_embedding_dimension()がNoneを返しました。ダミーエンコーディングで次元を特定します。")
                dummy_embedding = self.model.encode("test")
                dimension = dummy_embedding.shape[0]
            self._embedding_dimension = int(dimension) # 確実にint型にする
            logger.info(f"埋め込み次元が {self._embedding_dimension} に設定されました。")
        return self._embedding_dimension

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        テキストのリストを概念ベクトルのnumpy配列に変換する。
        """
        logger.info(f"エンコード対象テキスト: {texts}")
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"テキストのエンコード中にエラーが発生しました: {e}", exc_info=True)
            return np.array([])