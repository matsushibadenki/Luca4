# /app/digital_homeostasis/integrity_monitor.py
# title: 整合性モニター
# role: AIの知識ベース（知識グラフ）の論理的整合性や情報鮮度を継続的に監視し、知的健全性の状態を評価する。

import logging
import time
from typing import Dict, Any, List, TYPE_CHECKING

from app.knowledge_graph.persistent_knowledge_graph import PersistentKnowledgeGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

if TYPE_CHECKING:
    from app.analytics import AnalyticsCollector

logger = logging.getLogger(__name__)

class IntegrityMonitor:
    """
    AIの知識ベースの健全性を監視するクラス。
    """
    def __init__(self, llm: Any, knowledge_graph: PersistentKnowledgeGraph, analytics_collector: "AnalyticsCollector"):
        self.llm = llm
        self.knowledge_graph = knowledge_graph
        self.analytics_collector = analytics_collector
        self.consistency_check_prompt = ChatPromptTemplate.from_template(
            """あなたは論理分析の専門家です。以下の知識グラフの断片に、論理的な矛盾や不整合がないかを確認してください。
            矛盾を発見した場合は、その内容を具体的に指摘してください。問題がなければ「問題なし」と回答してください。

            知識グラフの断片:
            {graph_snippet}
            ---
            分析結果:
            """
        )

    async def check_logical_consistency(self) -> List[str]:
        """
        知識グラフ全体の論理的整合性を非同期でチェックする。
        """
        logger.info("知識グラフの論理的整合性チェックを開始します...")
        graph_string = self.knowledge_graph.get_graph().to_string()
        
        graph_snippet = graph_string[:4000] if len(graph_string) > 4000 else graph_string

        if "知識グラフは空です" in graph_snippet:
             logger.info("知識グラフが空のため、整合性チェックをスキップします。")
             return []

        chain = self.consistency_check_prompt | self.llm | StrOutputParser()
        # 修正: LLM呼び出しを非同期に
        result = await chain.ainvoke({"graph_snippet": graph_snippet})

        if "問題なし" in result:
            logger.info("論理的整合性に問題は見つかりませんでした。")
            return []
        else:
            logger.warning(f"論理的な不整合の可能性が検出されました: {result}")
            return [result]

    async def get_health_status(self) -> Dict[str, Any]:
        """
        現在の知的健全性の全体的なステータスを非同期で返し、アナリティクスに送信する。
        """
        # 修正: 非同期メソッドの呼び出し
        inconsistencies = await self.check_logical_consistency()
        
        status = {
            "is_healthy": not inconsistencies,
            "inconsistencies": inconsistencies,
            "last_checked": time.time()
        }
        logger.info(f"現在の知的健全性ステータス: {'健全' if status['is_healthy'] else '要注意'}")
        
        # 修正: アナリティクスへの送信を非同期に
        await self.analytics_collector.log_event("integrity_status", status)
        
        return status