# /app/agents/autonomous_agent.py
# title: 自律思考AIエージェント
# role: ユーザーの入力がない場合に、自律的に情報を収集、分析し、知識を拡張する。

import logging
import random
from typing import Any, List, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from app.agents.base import AIAgent
from app.memory.memory_consolidator import MemoryConsolidator
from app.rag.knowledge_base import KnowledgeBase
from app.config import settings
from app.tools.tool_belt import ToolBelt
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from app.constants import ToolNames
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️


logger = logging.getLogger(__name__)

class AutonomousAgent(AIAgent):
    """
    自律的に思考し、知識を拡張するエージェント。
    """
    def __init__(
        self,
        llm: Any,
        output_parser: Any,
        memory_consolidator: MemoryConsolidator,
        knowledge_base: KnowledgeBase,
        tool_belt: ToolBelt,
    ):
        self.llm = llm
        self.output_parser = output_parser
        self.memory_consolidator = memory_consolidator
        self.knowledge_base = knowledge_base
        self.tool_belt = tool_belt
        super().__init__()

    def build_chain(self) -> Optional[Runnable]:
        # このエージェントは複数の内部チェーンを持つため、単一のチェーンは構築しない
        return None

    def _decide_on_research_topic(self) -> str:
        """
        過去の対話や定義済みのトピックに基づき、次に調査するトピックを決定する。
        """
        if hasattr(settings, 'AUTONOMOUS_RESEARCH_TOPICS') and settings.AUTONOMOUS_RESEARCH_TOPICS:
            topic = random.choice(settings.AUTONOMOUS_RESEARCH_TOPICS)
            logger.info(f"自律思考: 次の研究テーマを '{topic}' に決定しました。")
            return topic
        logger.warning("自律思考のトピックリストが設定されていません。")
        return "人工知能の未来"

    def _gather_information(self, topic: str) -> str:
        """
        指定されたトピックについて、Web検索で情報を収集する。
        """
        logger.info(f"情報収集: '{topic}' についてWeb検索を実行します。")
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        web_search_tool = self.tool_belt.get_tool(ToolNames.SEARCH)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        if web_search_tool:
            try:
                return web_search_tool.use(topic)
            except Exception as e:
                logger.error(f"Web検索中にエラーが発生しました: {e}")
                return f"'{topic}'の検索中にエラーが発生しました。"
        else:
            logger.error("WebSearchToolが見つかりません。")
            return "Web検索ツールが利用できません。"


    def _synthesize_knowledge(self, topic: str, information: str) -> str:
        """
        収集した情報を要約し、知識として統合する。
        """
        logger.info("知識統合: 収集した情報を要約・分析しています...")
        prompt = ChatPromptTemplate.from_template(
            """あなたは優秀なリサーチャーです。以下のトピックと収集された情報に基づき、
            最も重要で中核となる情報を、簡潔かつ客観的な事実として3〜5個の箇条書きでまとめてください。
            
            トピック: {topic}
            
            収集された情報:
            {information}
            
            統合された知識（箇条書き）:
            """
        )
        chain = prompt | self.llm | self.output_parser
        synthesized_text = chain.invoke({"topic": topic, "information": information})
        logger.info(f"知識統合完了:\n{synthesized_text}")
        return synthesized_text

    def run_autonomous_cycle(self):
        """
        自律的な情報収集、統合、記憶のサイクルを1回実行する。
        """
        topic = self._decide_on_research_topic()
        gathered_info = self._gather_information(topic)
        if not gathered_info or "エラーが発生しました" in gathered_info or "利用できません" in gathered_info:
            logger.warning(f"'{topic}' の情報収集に失敗したため、サイクルを中断します。")
            return

        synthesized_knowledge = self._synthesize_knowledge(topic, gathered_info)
        if not synthesized_knowledge:
            return

        new_document = Document(page_content=synthesized_knowledge, metadata={"source": f"autonomous_research_{topic}"})
        self.knowledge_base.add_documents([new_document])

        self.memory_consolidator.log_autonomous_thought(topic, synthesized_knowledge)