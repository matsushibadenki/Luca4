# /app/agents/tool_using_agent.py
# title: ツール使用判断AIエージェント
# role: 与えられたタスクに基づき、利用可能なツールの中から最適なものを選択する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Any

from app.agents.base import AIAgent

class ToolUsingAgent(AIAgent):
    """
    タスクに最適なツールを選択し、使用方法を決定するAIエージェント。
    """
    def __init__(self, llm: Any, output_parser: Any, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        ツール使用判断エージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser