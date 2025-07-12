# /app/integrated_information_processing/integrated_information_agent.py
# title: 統合情報AIエージェント
# role: 統合情報理論(IIT)の思想に基づき、複数の多様な情報ストリームを、単なる要約ではなく、より高次の「意味」や「概念」へと統合・創発させる。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Any, Dict

from app.agents.base import AIAgent

class IntegratedInformationAgent(AIAgent):
    """
    複数の情報を統合し、高次の意味を生成するエージェント。
    """
    def __init__(self, llm: Any, output_parser: Any):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = ChatPromptTemplate.from_template(
            """あなたは、多様な視点を統合し、そこから新しい意味や洞察を創発させる哲学的な対話者です。
            以下の、異なるペルソナから提示された複数の見解を受け取り、それらを単に要約するのではなく、**見解間の関係性、隠れた前提、共通するテーマ、そして対立から生まれる新しい問い**を分析してください。
            最終的に、元の要求に対して、より深く、多角的で、示唆に富んだ一つの統合された回答を生成してください。

            **元の要求:**
            {query}

            ---
            **各ペルソナからの見解:**
            {persona_outputs}
            ---

            **統合された洞察と最終回答:**
            """
        )
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        エージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> str:
        if not isinstance(input_data, dict):
            raise TypeError("IntegratedInformationAgent expects a dictionary as input.")
        
        if self._chain is None:
            raise RuntimeError("IntegratedInformationAgent's chain is not initialized.")
        result: str = self._chain.invoke(input_data)
        return result