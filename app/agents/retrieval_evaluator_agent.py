# /app/agents/retrieval_evaluator_agent.py
# title: 検索品質評価AIエージェント
# role: 検索結果の品質を多角的に評価し、改善提案を行う。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from typing import Any, Dict

from app.agents.base import AIAgent

class RetrievalEvaluatorAgent(AIAgent):
    """
    RAGによって検索された情報の品質を評価するAIエージェント。
    """
    def __init__(self, llm: Any, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.prompt_template = prompt_template
        self.output_parser = JsonOutputParser()
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        検索品質評価エージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> Dict[str, Any]:
        """
        検索結果を評価し、評価結果を辞書として返します。
        """
        if not isinstance(input_data, dict):
            raise TypeError("RetrievalEvaluatorAgent expects a dictionary as input.")
        
        if self._chain is None:
            raise RuntimeError("RetrievalEvaluatorAgent's chain is not initialized.")
        
        result: Dict[str, Any] = self._chain.invoke(input_data)
        return result