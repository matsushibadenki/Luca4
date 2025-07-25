# /app/agents/query_refinement_agent.py
# title: 検索クエリ改善AIエージェント
# role: 検索品質の評価に基づき、より良い検索クエリを生成する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Any

from app.agents.base import AIAgent

class QueryRefinementAgent(AIAgent):
    """
    検索クエリを改善するためのAIエージェント。
    """
    def __init__(self, llm: Any, output_parser: Any, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        クエリ改善エージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser