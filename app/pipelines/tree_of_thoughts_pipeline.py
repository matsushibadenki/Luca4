# /app/pipelines/tree_of_thoughts_pipeline.py
# title: Tree of Thoughts (ToT) パイプライン
# role: 思考の木探索プロセス全体を管理し、最終的な結論を導き出す。

import logging
import time
from typing import Any, Dict

from app.pipelines.base import BasePipeline
from app.models import MasterAgentResponse, OrchestrationDecision
from app.agents.tree_of_thoughts_agent import TreeOfThoughtsAgent

logger = logging.getLogger(__name__)

class TreeOfThoughtsPipeline(BasePipeline):
    """
    Tree of Thoughts探索を実行するためのパイプライン。
    """
    def __init__(self, tree_of_thoughts_agent: TreeOfThoughtsAgent):
        self.tree_of_thoughts_agent = tree_of_thoughts_agent
        # ToTパイプラインは非同期処理を多用するため、arunを実装する
        # runはarunのラッパーとして機能させる

    def run(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        import asyncio
        return asyncio.run(self.arun(query, orchestration_decision))

    async def arun(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        """
        パイプラインを非同期で実行する。
        """
        start_time = time.time()
        logger.info("--- Tree of Thoughts Pipeline START ---")
        
        # ToT探索のパラメータ
        k = 3  # 初期思考の幅
        T = 3  # 思考の深さ
        b = 2  # 各ステップでのビーム幅

        # 思考の木を探索
        best_thought = self.tree_of_thoughts_agent.search(query, k, T, b)

        if best_thought:
            final_answer = best_thought.state
            retrieved_info = f"Tree of Thoughts探索により、{T}ステップの思考を経て結論に達しました。\n最良の思考経路の最終スコア: {best_thought.evaluation_score:.2f}"
        else:
            final_answer = "複雑な思考の末、明確な結論には至りませんでした。"
            retrieved_info = "Tree of Thoughts探索を行いましたが、有効な解決策を見つけられませんでした。"

        logger.info(f"--- Tree of Thoughts Pipeline END ({(time.time() - start_time):.2f} s) ---")
        
        return {
            "final_answer": final_answer,
            "self_criticism": "Tree of Thoughtsパイプラインは、複数の思考経路を評価・探索し、最も有望な結論を導き出しました。",
            "potential_problems": "探索の幅(k, b)や深さ(T)が不適切な場合、計算コストが増大するか、最適解を見逃す可能性があります。",
            "retrieved_info": retrieved_info
        }