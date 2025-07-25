# /app/agents/thought_evaluator_agent.py
# title: 思考評価AIエージェント
# role: Tree of Thoughtsの各思考ステップの有望性を評価し、探索をガイドするためのスコアを生成する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from typing import Any, Dict

from app.agents.base import AIAgent

class ThoughtEvaluatorAgent(AIAgent):
    """
    思考の有望性を評価するAIエージェント。
    """
    def __init__(self, llm: Any, output_parser: JsonOutputParser, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        思考評価エージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> Dict[str, Any]:
        """
        思考の経路を評価し、スコアと理由を含む辞書を返します。
        """
        if not isinstance(input_data, dict):
            raise TypeError("ThoughtEvaluatorAgent expects a dictionary as input.")
        
        if self._chain is None:
            raise RuntimeError("ThoughtEvaluatorAgent's chain is not initialized.")
        
        result: Dict[str, Any] = self._chain.invoke(input_data)
        return result