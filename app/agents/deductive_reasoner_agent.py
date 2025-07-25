# /app/agents/deductive_reasoner_agent.py
# title: 演繹的推論AIエージェント
# role: LLMの創造的な発想を抑制し、記号的検証器から得られた厳密な事実にのみ基づいて論理的な結論を導き出す。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Dict

from app.agents.base import AIAgent

class DeductiveReasonerAgent(AIAgent):
    """
    記号的事実に基づいて厳密な演繹的推論のみを行うエージェント。
    """
    def __init__(self, llm: Any, output_parser: StrOutputParser, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        演繹的推論エージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> str:
        if not isinstance(input_data, dict):
            raise TypeError("DeductiveReasonerAgent expects a dictionary as input.")
        
        if self._chain is None:
            raise RuntimeError("DeductiveReasonerAgent's chain is not initialized.")
        
        result: str = self._chain.invoke(input_data)
        return result