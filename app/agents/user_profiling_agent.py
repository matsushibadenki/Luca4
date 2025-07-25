# /app/agents/user_profiling_agent.py
# title: ユーザープロファイリングAIエージェント
# role: ユーザーの要求から、現在のユーザーの状態や隠れたニーズを推測し提供する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Any

from app.agents.base import AIAgent

class UserProfilingAgent(AIAgent):
    """
    ユーザーの状態を分析し、プロファイリングを行うAIエージェント。
    """
    def __init__(self, llm: Any, output_parser: Any, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        super().__init__()
        
    def build_chain(self) -> Runnable:
        """
        ユーザープロファイリングエージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser