# /app/agents/information_agent.py
# title: 情報収集AIエージェント
# role: ユーザーの要求に関連する具体的な情報や選択肢を収集し提供する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Any

from app.agents.base import AIAgent

class InformationAgent(AIAgent):
    """
    情報収集に特化したAIエージェント。
    """
    def __init__(self, llm: Any, output_parser: Any, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        情報収集エージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser