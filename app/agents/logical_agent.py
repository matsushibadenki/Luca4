# /app/agents/logical_agent.py
# title: 論理的思考AIエージェント
# role: 提供された情報とユーザーの要求に基づき、論理的な観点から最適な選択肢や解決策を分析し提案する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Any

from app.agents.base import AIAgent

class LogicalAgent(AIAgent):
    """
    論理的思考に特化したAIエージェント。
    """
    def __init__(self, llm: Any, output_parser: Any, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        論理的思考エージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser