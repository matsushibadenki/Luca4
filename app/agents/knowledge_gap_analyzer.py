# /app/agents/knowledge_gap_analyzer.py
# title: 知識ギャップ分析AIエージェント
# role: 対話履歴と知識グラフを比較し、知識が不足しているトピックを発見する。

import logging
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser

from app.agents.base import AIAgent
from app.memory.memory_consolidator import MemoryConsolidator
from app.knowledge_graph.persistent_knowledge_graph import PersistentKnowledgeGraph

logger = logging.getLogger(__name__)

class KnowledgeGapAnalyzerAgent(AIAgent):
    """
    対話履歴とナレッジグラフを分析し、知識が不足している領域を特定するエージェント。
    """
    def __init__(
        self,
        llm: Any,
        output_parser: JsonOutputParser,
        prompt_template: ChatPromptTemplate,
        memory_consolidator: MemoryConsolidator,
        knowledge_graph: PersistentKnowledgeGraph
    ):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        self.memory_consolidator = memory_consolidator
        self.knowledge_graph = knowledge_graph
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        知識ギャップ分析のためのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def analyze_for_gaps(self) -> Optional[str]:
        """
        知識のギャップを分析し、強化すべきトピックを一つ返す。

        Returns:
            Optional[str]: 強化すべきトピック名。見つからなければNone。
        """
        logger.info("知識ギャップの分析を開始します...")

        # 1. 最近の対話履歴からクエリを取得
        recent_interactions = self.memory_consolidator.get_recent_events(limit=20)
        recent_queries = [
            event["query"] for event in recent_interactions
            if "type" in event and event["type"] == "interaction" and "query" in event
        ]

        if not recent_queries:
            logger.info("分析対象の対話履歴がありません。")
            return None

        # 2. 現在の知識グラフの概要を取得
        graph_summary = self.knowledge_graph.get_summary()

        # 3. LLMに分析を依頼
        analysis_input = {
            "recent_queries": "\n- ".join(recent_queries),
            "knowledge_graph_summary": graph_summary
        }

        try:
            result: Dict[str, Any] = self.invoke(analysis_input)
            
            if "topic" in result and result["topic"] and result["topic"] != "なし":
                topic = result["topic"]
                logger.info(f"知識ギャップが発見されました。強化推奨トピック: '{topic}'")
                return topic
            else:
                logger.info("顕著な知識ギャップは見つかりませんでした。")
                return None
        except Exception as e:
            logger.error(f"知識ギャップ分析中にエラーが発生しました: {e}", exc_info=True)
            return None

