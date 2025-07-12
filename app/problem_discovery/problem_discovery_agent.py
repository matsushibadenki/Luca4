# /app/problem_discovery/problem_discovery_agent.py
# title: 問題発見AIエージェント
# role: ユーザーのクエリや対話コンテキストから、まだ明示されていない潜在的な問題や関連する疑問を発見する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from typing import Any, Dict, List

from app.agents.base import AIAgent

class ProblemDiscoveryAgent(AIAgent):
    """
    ユーザーの潜在的な問題や関連する疑問を発見するAIエージェント。
    """
    def __init__(self, llm: Any, output_parser: Any, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = JsonOutputParser()
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> List[str]:
        if not isinstance(input_data, dict):
            raise TypeError("ProblemDiscoveryAgent expects a dictionary as input.")
        
        if self._chain is None:
            raise RuntimeError("ProblemDiscoveryAgent's chain is not initialized.")
        result: List[str] = self._chain.invoke(input_data)
        return result