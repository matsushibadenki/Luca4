# /app/micro_llm/creator.py
# title: マイクロLLMクリエーター
# role: 特定のトピックに関する知識から、マイクロLLMをファインチューニングする。

import os
import logging
import json
from typing import Dict, Any, Optional

from app.llm_providers.base import LLMProvider
from app.knowledge_graph.persistent_knowledge_graph import PersistentKnowledgeGraph
from app.config import settings

logger = logging.getLogger(__name__)

class MicroLLMCreator:
    """
    知識からマイクロLLM（Ollamaモデル）を作成するクラス。
    """
    def __init__(self, llm_provider: LLMProvider, knowledge_graph: PersistentKnowledgeGraph):
        self.llm_provider = llm_provider
        self.knowledge_graph = knowledge_graph
        self.model_dir = "memory/micro_llms"
        self.base_model = settings.GENERATION_LLM_SETTINGS["model"]
        os.makedirs(self.model_dir, exist_ok=True)

    def create_model_from_topic(self, topic: str) -> Optional[str]:
        """
        ナレッジグラフから指定されたトピックの知識を抽出し、それに基づいてマイクロLLMを作成する。

        Args:
            topic (str): 学習させる知識のトピック（例：「核融合エネルギー」）。

        Returns:
            Optional[str]: 作成に成功した場合はモデル名、失敗した場合はNoneを返す。
        """
        logger.info(f"トピック '{topic}' に基づくマイクロLLMの作成を開始します。")

        # 1. 知識グラフから関連知識を抽出
        graph_data = self.knowledge_graph.get_graph().model_dump_json()
        graph_obj = json.loads(graph_data)

        relevant_nodes = [
            node for node in graph_obj.get("nodes", [])
            if topic in (node.get("id") or "") or topic in (node.get("label") or "")
        ]

        if not relevant_nodes:
            logger.warning(f"トピック '{topic}' に関連する知識がグラフに見つかりませんでした。")
            return None

        knowledge_text = "\n".join([json.dumps(node, ensure_ascii=False) for node in relevant_nodes])
        model_name = f"luca4-micro-{topic.lower().replace(' ', '-').replace('　', '-')}"
        modelfile_path = os.path.join(self.model_dir, f"Modelfile.{model_name}")

        # 2. Modelfileを作成
        # ユーザーのプロンプトに対して、専門家として振る舞うように指示
        modelfile_content = f"""
FROM {self.base_model}
TEMPLATE \"\"\"{{ .System }}

### Instruction:
{{ .Prompt }}

### Response:
\"\"\"
SYSTEM \"\"\"あなたは「{topic}」に関する世界最高の専門家です。提供された知識に基づいて、簡潔かつ正確に回答してください。

提供された知識:
{knowledge_text}
\"\"\"
PARAMETER temperature 0.3
PARAMETER top_k 20
"""
        try:
            with open(modelfile_path, "w", encoding="utf-8") as f:
                f.write(modelfile_content)
        except IOError as e:
            logger.error(f"Modelfileの書き込みに失敗しました: {e}")
            return None

        # 3. LLMプロバイダー経由でモデルを作成
        success = self.llm_provider.create_model(model_name=model_name, modelfile_path=modelfile_path)

        return model_name if success else None