# /app/cognitive_modeling/predictive_coding_engine.py
# title: 予測符号化エンジン
# role: 内部のワールドモデルから次の入力を予測し、実際の入力との「予測誤差」を算出することで、学習のトリガーを生成する。

import logging
from typing import Any, Dict

from app.agents.base import AIAgent
from app.cognitive_modeling.world_model_agent import WorldModelAgent
from app.memory.working_memory import WorkingMemory
from app.knowledge_graph.persistent_knowledge_graph import PersistentKnowledgeGraph
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent

logger = logging.getLogger(__name__)

class PredictiveCodingEngine:
    """
    予測符号化理論に基づき、予測と観測の差分（予測誤差）を計算するエンジン。
    """
    def __init__(self, world_model_agent: WorldModelAgent, working_memory: WorkingMemory, knowledge_graph_agent: KnowledgeGraphAgent, persistent_knowledge_graph: PersistentKnowledgeGraph):
        self.world_model_agent = world_model_agent
        self.working_memory = working_memory
        self.knowledge_graph_agent = knowledge_graph_agent
        self.persistent_knowledge_graph = persistent_knowledge_graph

    def process_input(self, user_input: str, dialogue_history: list[str]) -> Dict[str, Any]:
        """
        ユーザー入力を処理し、予測誤差を計算してワーキングメモリに格納する。

        Args:
            user_input (str): ユーザーからの最新の入力。
            dialogue_history (list[str]): これまでの対話履歴。

        Returns:
            Dict[str, Any]: 計算された予測誤差、または新規情報がなかったことを示す辞書。
        """
        logger.info("--- 予測符号化エンジン起動 ---")
        
        # 1. ワールドモデルに基づき、次の入力を予測する
        prediction_input = {
            "dialogue_history": "\n".join(dialogue_history)
        }
        prediction = self.world_model_agent.predict_next_state(prediction_input)
        logger.info(f"予測された次の状態: {prediction}")

        # 2. 予測と実際の入力を比較し、予測誤差を計算する
        error_calculation_input = {
            "prediction": prediction,
            "actual_input": user_input
        }
        prediction_error = self.world_model_agent.calculate_prediction_error(error_calculation_input)
        logger.info(f"計算された予測誤差: {prediction_error}")
        
        # 3. 予測誤差（新規情報）をワーキングメモリに格納
        if "error_type" in prediction_error and prediction_error["error_type"] != "新規情報なし" and "key_info" in prediction_error and prediction_error["key_info"]:
            self.working_memory.add_prediction_error(prediction_error)
            logger.info(f"予測誤差をワーキングメモリに追加しました: {prediction_error['summary']}")
            
            # ワールドモデルの更新をトリガーし、知識グラフに統合
            self.world_model_agent.update_model({
                "dialogue_history": "\n".join(dialogue_history),
                "prediction_error": prediction_error.get("summary", "")
            })
        else:
            logger.info("予測誤差は検出されませんでした（学習の必要なし）。")
            
        logger.info("--- 予測符号化エンジン終了 ---")
        return prediction_error