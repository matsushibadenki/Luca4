# /app/agents/fact_checking_agent.py
# title: 情報検証AIエージェント
# role: AIによって生成された最終回答案が、参照情報と矛盾していないか、または参照情報にない情報を主張していないかを検証する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Any, Dict

from app.agents.base import AIAgent

class FactCheckingAgent(AIAgent):
    """
    回答の事実性を検証するAIエージェント。
    """
    def __init__(self, llm: Any, output_parser: Any, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> str:
        if not isinstance(input_data, dict):
            raise TypeError("FactCheckingAgent expects a dictionary as input.")
        
        if self._chain is None:
            raise RuntimeError("FactCheckingAgent's chain is not initialized.")
        result: str = self._chain.invoke(input_data)
        return result