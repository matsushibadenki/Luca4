# /app/internal_dialogue/consciousness_staging_area.py
# title: 意識のステージングエリア
# role: 内的対話が行われる「場」を提供し、調停者の指示に従って対話の進行を管理する。

import logging
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

from app.internal_dialogue.mediator_agent import MediatorAgent

logger = logging.getLogger(__name__)

class ConsciousnessStagingArea:
    """
    多様な思考エージェントが対話を行う仮想的なステージ。
    """
    def __init__(self, llm: Any, mediator_agent: MediatorAgent):
        self.llm = llm
        self.mediator_agent = mediator_agent
        self.output_parser = StrOutputParser()
        self.dialogue_history: List[str] = []

    def _run_single_turn(self, query: str, participant: Dict[str, str], current_history: str) -> str:
        """個々の思考エージェントの意見を生成する。"""
        prompt = ChatPromptTemplate.from_template(
            """あなたは {persona}
            以下の元の要求とこれまでの議論を踏まえ、あなたの視点から意見を述べてください。

            元の要求: {query}
            
            これまでの議論:
            {history}
            ---
            あなたの意見 (@{name}):
            """
        )
        chain = prompt | self.llm | self.output_parser
        response = chain.invoke({
            "name": participant["name"],
            "persona": participant["persona"],
            "query": query,
            "history": current_history
        })
        return f"@{participant['name']}: {response}"

    def run_dialogue(self, query: str, participants: List[Dict[str, str]], max_turns: int = 5) -> str:
        """
        内省的な対話の全プロセスを実行する。
        """
        self.dialogue_history = []
        logger.info(f"--- 内的対話開始 --- 要求: '{query}'")
        logger.info(f"参加エージェント: {[p['name'] for p in participants]}")

        for turn in range(max_turns):
            logger.info(f"--- 対話ターン {turn + 1}/{max_turns} ---")
            
            # 全員に一度ずつ発言させる
            if turn == 0:
                for p in participants:
                    statement = self._run_single_turn(query, p, "\n".join(self.dialogue_history))
                    self.dialogue_history.append(statement)
                    logger.info(statement)
            
            # 調停者が介入
            mediator_input = {
                "query": query,
                "dialogue_history": "\n".join(self.dialogue_history)
            }
            mediator_action = self.mediator_agent.invoke(mediator_input)
            self.dialogue_history.append(f"@調停者: {mediator_action}")
            logger.info(f"@調停者: {mediator_action}")

            # 結論を出すように指示されたら終了
            if "結論" in mediator_action or "統合" in mediator_action or "まとめ" in mediator_action:
                logger.info("調停者が結論を促したため、対話を終了します。")
                break
                
            # 特定のエージェントが指名された場合、そのエージェントに発言させる
            mentioned_agents = [p for p in participants if f"@{p['name']}" in mediator_action]
            if mentioned_agents:
                for p in mentioned_agents:
                     statement = self._run_single_turn(query, p, "\n".join(self.dialogue_history))
                     self.dialogue_history.append(statement)
                     logger.info(statement)
            else: # 指名がない場合は全員に再度発言させる
                 for p in participants:
                    statement = self._run_single_turn(query, p, "\n".join(self.dialogue_history))
                    self.dialogue_history.append(statement)
                    logger.info(statement)

        final_summary = "\n".join(self.dialogue_history)
        logger.info("--- 内的対話終了 ---")
        return final_summary