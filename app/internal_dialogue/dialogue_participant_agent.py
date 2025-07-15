# /app/internal_dialogue/dialogue_participant_agent.py
# title: 対話参加者生成エージェント
# role: マーヴィン・ミンスキーの「心の社会」理論に基づき、与えられたテーマに対して多様な視点を持つ思考エージェント群を生成する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from typing import Any, Dict, List

from app.agents.base import AIAgent

class DialogueParticipantAgent(AIAgent):
    """
    与えられた要求に応じて、多様な視点を持つ対話参加者（思考エージェント）を生成する。
    """
    def __init__(self, llm: Any):
        self.llm = llm
        self.output_parser = JsonOutputParser()
        self.prompt_template = ChatPromptTemplate.from_template(
            """あなたは、複雑な問題に対して多様な視点を生み出す「アイデアの生成者」です。
            以下のユーザー要求を分析し、この問題について議論すべき、独立した視点を持つ思考エージェント（ペルソナ）を5人、JSON形式で生成してください。
            各エージェントには、その視点を象徴する「名前」と、思考の方向性を定める「ペルソナ説明」を与えてください。

            ユーザー要求:
            {query}
            ---
            思考エージェントのリスト (JSON):
            {{
                "participants": [
                    {{"name": "エージェント1の名前", "persona": "エージェント1のペルソナ説明"}},
                    {{"name": "エージェント2の名前", "persona": "エージェント2のペルソナ説明"}},
                    {{"name": "エージェント3の名前", "persona": "エージェント3のペルソナ説明"}},
                    {{"name": "エージェント4の名前", "persona": "エージェント4のペルソナ説明"}},
                    {{"name": "エージェント5の名前", "persona": "エージェント5のペルソナ説明"}}
                ]
            }}
            """
        )
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        エージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> List[Dict[str, str]]:
        if not isinstance(input_data, dict):
            raise TypeError("DialogueParticipantAgent expects a dictionary as input.")
        
        if self._chain is None:
            raise RuntimeError("DialogueParticipantAgent's chain is not initialized.")
        
        result: Dict[str, List[Dict[str, str]]] = self._chain.invoke(input_data)
        return result.get("participants", [])