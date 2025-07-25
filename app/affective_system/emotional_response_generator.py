# /app/affective_system/emotional_response_generator.py
# title: 感情応答生成エージェント
# role: AIの最終的な回答に、現在の感情状態に基づいた適切なトーンや表現を加える。

from __future__ import annotations
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from typing import Any, Dict, TYPE_CHECKING, Optional
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.agents.base import AIAgent
from .affective_state import AffectiveState

if TYPE_CHECKING:
    from langchain_ollama import OllamaLLM
    from langchain_core.output_parsers import StrOutputParser

class EmotionalResponseGenerator(AIAgent):
    """
    最終回答に感情的なニュアンスを付加するエージェント。
    """
    def __init__(self, llm: "OllamaLLM", output_parser: "StrOutputParser", prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        感情応答生成のためのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> str:
        """
        最終回答と感情状態を受け取り、トーンを調整した応答を生成します。
        """
        if not isinstance(input_data, dict):
            raise TypeError("EmotionalResponseGenerator expects a dictionary as input.")

        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        affective_state_val = input_data.get("affective_state")
        affective_state: Optional[AffectiveState] = affective_state_val if isinstance(affective_state_val, AffectiveState) else None
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
        # 感情がニュートラルな場合は、元の回答をそのまま返す
        if not affective_state or affective_state.is_neutral():
            return input_data.get("final_answer", "")

        if self._chain is None:
            raise RuntimeError("EmotionalResponseGenerator's chain is not initialized.")
            
        result: str = self._chain.invoke(input_data)
        return result