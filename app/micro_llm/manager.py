# /app/micro_llm/manager.py
# title: マイクロLLMマネージャー
# role: マイクロLLMのライフサイクル（作成、一覧取得など）を管理する。

import logging
from typing import List, Dict, Any
from app.llm_providers.base import LLMProvider
from app.micro_llm.creator import MicroLLMCreator

logger = logging.getLogger(__name__)

class MicroLLMManager:
    """
    マイクロLLMの学習と管理を行うエージェント。
    """
    def __init__(self, llm_provider: LLMProvider, creator: MicroLLMCreator):
        self.llm_provider = llm_provider
        self.creator = creator

    def get_specialized_models(self) -> List[Dict[str, str]]:
        """
        利用可能なマイクロLLM（専門家モデル）のリストを取得する。

        Returns:
            List[Dict[str, str]]: モデル名とトピックを含む辞書のリスト。
        """
        logger.info("利用可能な専門家モデルをスキャンしています...")
        try:
            response = self.llm_provider.list_models()
            all_models = response.get("models", []) if "models" in response else []

            specialized_models = []
            for model_info in all_models:
                model_name = model_info.get("name") if "name" in model_info else None
                if model_name and model_name.startswith("luca4-micro-"):
                    topic = model_name.replace("luca4-micro-", "").replace("-", " ").title()
                    specialized_models.append({
                        "name": model_name,
                        "topic": topic
                    })
            logger.info(f"{len(specialized_models)}個の専門家モデルが見つかりました。")
            return specialized_models
        except Exception as e:
            logger.error(f"専門家モデルのスキャン中にエラーが発生しました: {e}", exc_info=True)
            return []

    def run_creation_cycle(self, topic: str):
        """
        指定されたトピックに関する知識を抽出し、マイクロLLMを作成する。
        """
        logger.info(f"マイクロLLM作成サイクル開始: トピック='{topic}'")
        self.creator.create_model_from_topic(topic)
        logger.info(f"マイクロLLM作成サイクル完了: トピック='{topic}'")