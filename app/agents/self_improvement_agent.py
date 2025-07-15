# /app/agents/self_improvement_agent.py
# title: 自己改善AIエージェント
# role: AI自身のパフォーマンスに対する自己批判を分析し、具体的な改善提案を生成する。

import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser # For structured output

from app.agents.base import AIAgent

logger = logging.getLogger(__name__)

class SelfImprovementAgent(AIAgent):
    """
    自己批判に基づいて、具体的な改善提案を生成するAIエージェント。
    """
    def __init__(self, llm: Any, output_parser: JsonOutputParser, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = output_parser # Expecting JsonOutputParser
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        自己改善提案を生成するためのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> List[Dict[str, Any]]:
        """
        自己批判とコンテキストを分析し、改善提案のリストを生成します。
        """
        if not isinstance(input_data, dict):
            raise TypeError("SelfImprovementAgent expects a dictionary as input.")
        
        if self._chain is None:
            raise RuntimeError("SelfImprovementAgent's chain is not initialized.")
        
        result: List[Dict[str, Any]] = self._chain.invoke(input_data)
        return result