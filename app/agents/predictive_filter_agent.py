# /app/agents/predictive_filter_agent.py
# title: 予測フィルターAIエージェント
# role: 脳の予測符号化を模倣し、入力から予測可能で冗長な情報をフィルタリングし、新規性の高い「予測誤差」のみを抽出する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from typing import Any, Dict

from app.agents.base import AIAgent

class PredictiveFilterAgent(AIAgent):
    """
    入力情報から予測誤差（新規で驚きのある情報）を抽出するAIエージェント。
    """
    def __init__(self, llm: Any, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = JsonOutputParser()
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        予測フィルターエージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> Dict[str, Any]:
        """
        入力テキストを分析し、予測誤差をJSON形式で返します。
        """
        if not isinstance(input_data, dict):
            raise TypeError("PredictiveFilterAgent expects a dictionary as input.")
        
        if self._chain is None:
            raise RuntimeError("PredictiveFilterAgent's chain is not initialized.")
        
        result: Dict[str, Any] = self._chain.invoke(input_data)
        return result