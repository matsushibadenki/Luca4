# /app/agents/knowledge_assimilation_agent.py
# title: 知識生成AIエージェント
# role: 与えられたキーワードに基づき、ナレッジベースに追加するための説明文を生成する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Any

from app.agents.base import AIAgent

class KnowledgeAssimilationAgent(AIAgent):
    """
    キーワードから知識を生成し、システムに統合するAIエージェント。
    """
    def __init__(self, llm: Any, output_parser: Any, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        知識生成エージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser