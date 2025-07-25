# /app/agents/word_learning_agent.py
# title: 単語学習AIエージェント
# role: ユーザーの入力から重要なキーワードや専門用語を抽出し、学習する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Any

from app.agents.base import AIAgent

class WordLearningAgent(AIAgent):
    """
    対話から単語を学習するAIエージェント。
    """
    def __init__(self, llm: Any, output_parser: Any, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        単語学習エージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser