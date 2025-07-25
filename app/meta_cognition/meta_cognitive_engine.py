# /app/meta_cognition/meta_cognitive_engine.py
# title: メタ認知エンジン
# role: 自己批判エージェントを利用して、AIの応答と思考プロセスに対するメタ認知的な評価（自己批判）を実行する。

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.meta_cognition.self_critic_agent import SelfCriticAgent

class MetaCognitiveEngine:
    """
    AIの応答と思考プロセスを自己批判するメタ認知エンジン。
    """
    def __init__(self, self_critic_agent: 'SelfCriticAgent'):
        self.self_critic_agent = self_critic_agent

    def critique_process_and_response(
        self, query: str, plan: str, cognitive_loop_output: str, final_answer: str
    ) -> str:
        """
        与えられた思考プロセス全体と最終応答を自己批判エージェントに評価させます。
        """
        input_data = {
            "query": query,
            "plan": plan,
            "cognitive_loop_output": cognitive_loop_output,
            "final_answer": final_answer,
        }
        criticism = self.self_critic_agent.invoke(input_data)
        return criticism