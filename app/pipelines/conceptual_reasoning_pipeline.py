# /app/pipelines/conceptual_reasoning_pipeline.py
# title: 概念推論パイプライン
# role: 抽象的な問いに対し、概念の合成や類推といった操作を通じて答えを導き出す。

from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING, Dict, Any

from app.pipelines.base import BasePipeline
from app.models import MasterAgentResponse, OrchestrationDecision

if TYPE_CHECKING:
    from app.agents.planning_agent import PlanningAgent
    from app.agents.cognitive_loop_agent import CognitiveLoopAgent
    from app.agents.master_agent import MasterAgent

logger = logging.getLogger(__name__)

class ConceptualReasoningPipeline(BasePipeline):
    """
    概念操作と思考ループを組み合わせる、高度な推論パイプライン。
    """
    def __init__(
        self,
        planning_agent: PlanningAgent,
        cognitive_loop_agent: CognitiveLoopAgent,
        master_agent: MasterAgent,
    ):
        self.planning_agent = planning_agent
        self.cognitive_loop_agent = cognitive_loop_agent
        self.master_agent = master_agent

    async def arun(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        start_time = time.time()
        logger.info("--- Conceptual Reasoning Pipeline START ---")

        # 1. 計画立案: 概念操作を含む計画を立てる
        # このパイプラインが選択された時点で、クエリは概念操作を必要とすると判断されている
        planning_input = {
            "query": query,
            "reasoning_instruction": "このタスクは抽象的な概念操作を必要とします。思考のステップには「概念のベクトル化」「概念の合成」「概念の分析」などを含めてください。"
        }
        plan = self.planning_agent.invoke(planning_input)
        logger.info(f"Generated Plan for Conceptual Reasoning:\n{plan}")

        # 2. 認知ループの実行: 計画に基づき、概念操作と分析を行う
        # CognitiveLoopAgentは、内部でImaginationEngineやConceptualMemoryを使うように拡張されている
        cognitive_loop_input = {
            "query": query,
            "plan": plan,
            "reasoning_instruction": orchestration_decision.get("reasoning_instruction", "")
        }
        cognitive_loop_output = await self.cognitive_loop_agent.ainvoke(cognitive_loop_input)
        logger.info(f"Cognitive Loop Output with Conceptual Analysis:\n{cognitive_loop_output}")

        # 3. 最終回答の生成
        master_agent_input = {
            "query": query,
            "plan": plan,
            "cognitive_loop_output": cognitive_loop_output
        }
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # ainvokeはMasterAgentに存在しないため、意図されたgenerate_final_answer_asyncを呼び出す
        final_answer = await self.master_agent.generate_final_answer_async(master_agent_input, orchestration_decision)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
        logger.info(f"--- Conceptual Reasoning Pipeline END ({(time.time() - start_time):.2f} s) ---")

        return {
            "final_answer": final_answer,
            "self_criticism": "概念推論パイプラインは、潜在空間での概念操作を通じて、より深いレベルでの回答を試みました。",
            "potential_problems": "概念のベクトル表現が不正確な場合や、ベクトル演算の結果が解釈不能な場合に、推論が失敗する可能性があります。",
            "retrieved_info": f"Plan:\n{plan}\n\nCognitive Loop Output:\n{cognitive_loop_output}"
        }

    def run(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        import asyncio
        return asyncio.run(self.arun(query, orchestration_decision))