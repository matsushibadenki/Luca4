# /app/meta_cognition/self_critic_agent.py
# title: 自己批判AIエージェント
# role: AI自身の応答を客観的に評価し、改善点を提案する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Dict
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

from app.agents.base import AIAgent

class SelfCriticAgent(AIAgent):
    """
    自己批判とフィードバックを行うAIエージェント。
    """
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def __init__(self, llm: Any, output_parser: StrOutputParser, prompt_template: ChatPromptTemplate):
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> str:
        if not isinstance(input_data, dict):
            raise TypeError("SelfCriticAgent expects a dictionary as input.")

        if self._chain is None:
            raise RuntimeError("SelfCriticAgent's chain is not initialized.")
        result: str = self._chain.invoke(input_data)
        return result
