# /app/cognitive_modeling/world_model_agent.py
# title: ワールドモデルAIエージェント
# role: 対話の文脈から世界の次の状態を予測し、予測誤差を計算・分析し、内部の世界モデルを更新する。

import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing import Any, Dict, Optional

from app.agents.base import AIAgent
from app.knowledge_graph.persistent_knowledge_graph import PersistentKnowledgeGraph
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent
from app.knowledge_graph.models import KnowledgeGraph

logger = logging.getLogger(__name__)

class WorldModelAgent(AIAgent):
    """
    世界の内部モデルを維持し、予測と学習を行うエージェント。
    """
    def __init__(self, llm: Any, knowledge_graph_agent: KnowledgeGraphAgent, persistent_knowledge_graph: PersistentKnowledgeGraph):
        self.llm = llm
        self.knowledge_graph_agent = knowledge_graph_agent
        self.persistent_knowledge_graph = persistent_knowledge_graph
        super().__init__()

    def build_chain(self) -> Optional[Runnable]:
        """
        このエージェントは特定の内部チェーンを使用するため、
        単一のメインチェーンは構築しません。
        """
        return None

    def predict_next_state(self, input_data: Dict[str, Any]) -> str:
        """対話履歴から次のユーザーの意図や発言を予測する。"""
        prompt = ChatPromptTemplate.from_template(
            """あなたは対話の文脈を読む専門家です。以下の対話履歴に基づき、ユーザーが次にどのような発言をするか、その意図や内容を予測してください。
            
            対話履歴:
            {dialogue_history}
            ---
            予測される次のユーザーの発言/意図:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(input_data)

    def calculate_prediction_error(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """予測と実際の入力の差分（予測誤差）を分析し、構造化して返す。"""
        prompt = ChatPromptTemplate.from_template(
            """あなたは、予測と現実のズレを分析する認知科学者です。AIの「予測」と実際の「ユーザー入力」を比較し、その間の「予測誤差」（＝新規性、驚き）を分析してください。
            出力は、誤差のカテゴリ、要約、キーワードを含む厳密なJSON形式でなければなりません。
            
            AIの予測:
            {prediction}

            実際のユーザー入力:
            {actual_input}
            ---
            予測誤差の分析結果 (JSON):
            {{
                "error_type": "予測誤差のカテゴリ（例: トピックの急な変更, 予期せぬ詳細情報, 矛盾した情報, 新規情報なし）",
                "summary": "誤差の簡単な要約",
                "key_info": ["関連するキーワード1", "関連するキーワード2"]
            }}
            """
        )
        chain = prompt | self.llm | JsonOutputParser()
        return chain.invoke(input_data)

    def update_model(self, input_data: Dict[str, Any]) -> str:
        """予測誤差に基づき、ワールドモデル（この場合はLLMの内部状態）の解釈を更新するための要約を生成し、知識グラフに統合する。"""
        logger.info("ワールドモデルの更新を開始します。")
        prompt = ChatPromptTemplate.from_template(
            """あなたは学習するAIです。これまでの文脈と、新たに発生した「予測誤差」を踏まえ、世界の理解をどのように更新すべきか、内省的なメモを記述してください。
            このメモは、知識グラフに追加するのに適した客観的な事実や関係性を含んでください。

            これまでの文脈:
            {dialogue_history}
            
            発生した予測誤差:
            {prediction_error}
            ---
            ワールドモデル更新のための内省メモ（知識グラフ形式で解釈可能な事実の箇条書きなど）:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        update_summary = chain.invoke(input_data)
        logger.info(f"ワールドモデル更新メモ: {update_summary}")

        if update_summary.strip():
            logger.info("ワールドモデル更新メモを知識グラフに統合しています...")
            kg_input = {"text_chunk": update_summary}
            try:
                new_knowledge_graph: KnowledgeGraph = self.knowledge_graph_agent.invoke(kg_input)
                self.persistent_knowledge_graph.merge(new_knowledge_graph)
                self.persistent_knowledge_graph.save()
                logger.info("ワールドモデル更新が知識グラフに永続化されました。")
            except Exception as e:
                logger.error(f"ワールドモデル更新の知識グラフへの統合に失敗しました: {e}", exc_info=True)
        else:
            logger.info("ワールドモデル更新のための内省メモが空のため、知識グラフへの統合をスキップします。")

        return update_summary