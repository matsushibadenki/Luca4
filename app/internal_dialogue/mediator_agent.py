# /app/internal_dialogue/mediator_agent.py
# title: 調停者AIエージェント
# role: 内的家族システム(IFS)のセラピストのように、多様な思考エージェント間の対話を促進し、対立を解消し、統合的な結論へと導く。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Dict

from app.agents.base import AIAgent

class MediatorAgent(AIAgent):
    """
    思考エージェント間の対話を司り、統合を促す調停者。
    """
    def __init__(self, llm: Any):
        self.llm = llm
        self.output_parser = StrOutputParser()
        self.prompt_template = ChatPromptTemplate.from_template(
            """あなたは、多様な意見を持つ会議をまとめる、卓越したファシリテーター兼調停者です。
            以下の対話履歴と元の要求を踏まえ、次に行うべきことを指示してください。

            指示の選択肢:
            1. **質問を投げる**: 特定のエージェントを指名し、他の意見に対する反論や深掘りを促す質問を投げかける。（例: 「@楽観主義者さん、その計画のリスクについて@現実主義者さんが指摘していますが、どうお考えですか？」）
            2. **要約と論点整理**: 議論が発散した場合、現在の論点を整理し、共通点や対立点を明確にする。
            3. **結論の統合**: 全員の意見が出揃ったと判断した場合、すべての意見を統合し、元の要求に対する最終的な結論を導き出すよう指示する。

            元の要求:
            {query}

            これまでの対話履歴:
            {dialogue_history}
            ---
            次のアクション（質問、要約、または結論の指示）:
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
            raise TypeError("MediatorAgent expects a dictionary as input.")

        if self._chain is None:
            raise RuntimeError("MediatorAgent's chain is not initialized.")
        return self._chain.invoke(input_data)