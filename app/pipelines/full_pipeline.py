# /app/pipelines/full_pipeline.py
# title: 完全思考パイプライン
# role: 複雑な要求に対し、計画、情報収集、自己評価、自己改善を含む包括的な思考プロセスを実行する。

from __future__ import annotations
import time
import logging
from typing import Dict, Any, TYPE_CHECKING
import asyncio

from app.pipelines.base import BasePipeline
from app.models import MasterAgentResponse, OrchestrationDecision

if TYPE_CHECKING:
    from app.agents.master_agent import MasterAgent
    from app.agents.planning_agent import PlanningAgent
    from app.agents.cognitive_loop_agent import CognitiveLoopAgent
    from app.meta_cognition.meta_cognitive_engine import MetaCognitiveEngine
    from app.problem_discovery.problem_discovery_agent import ProblemDiscoveryAgent
    from app.memory.memory_consolidator import MemoryConsolidator
    from app.meta_intelligence.self_improvement.evolution import SelfEvolvingSystem
    from app.analytics import AnalyticsCollector


logger = logging.getLogger(__name__)

class FullPipeline(BasePipeline):
    """
    計画、実行、評価、改善のサイクルを含む、完全な思考パイプライン。
    """
    def __init__(
        self,
        master_agent: 'MasterAgent',
        planning_agent: 'PlanningAgent',
        cognitive_loop_agent: 'CognitiveLoopAgent',
        meta_cognitive_engine: 'MetaCognitiveEngine',
        problem_discovery_agent: 'ProblemDiscoveryAgent',
        memory_consolidator: 'MemoryConsolidator',
        self_evolving_system: 'SelfEvolvingSystem',
        analytics_collector: 'AnalyticsCollector',
    ):
        self.master_agent = master_agent
        self.planning_agent = planning_agent
        self.cognitive_loop_agent = cognitive_loop_agent
        self.meta_cognitive_engine = meta_cognitive_engine
        self.problem_discovery_agent = problem_discovery_agent
        self.memory_consolidator = memory_consolidator
        self.self_evolving_system = self_evolving_system
        self.analytics_collector = analytics_collector

    def run(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        """同期版は非同期版を呼び出すラッパーとする。"""
        return asyncio.run(self.arun(query, orchestration_decision))

    async def arun(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        """
        完全な思考パイプラインを非同期で実行する。
        """
        if self.master_agent is None:
            raise RuntimeError("MasterAgent has not been set for the FullPipeline.")

        start_time = time.time()
        logger.info(f"--- Full Pipeline started for query: '{query}' ---")

        # 思考の軌跡を記録
        reasoning_trace: Dict[str, Any] = {}

        # オーケストレーション決定から思考の強調点を取得
        reasoning_emphasis = orchestration_decision.get("reasoning_emphasis")
        reasoning_instruction = ""
        if reasoning_emphasis == "bird's_eye_view":
            reasoning_instruction = "回答は、概念間の関係性、全体像、長期的な影響、または抽象的な原則を強調してください。"
        elif reasoning_emphasis == "detail_oriented":
            reasoning_instruction = "回答は、具体的な事実、詳細な手順、明確なデータ、または精密な論理構造を強調してください。"

        # 1. 計画立案
        planning_input = {
            "query": query,
            "reasoning_instruction": reasoning_instruction
        }
        plan = self.planning_agent.invoke(planning_input)
        reasoning_trace["step_1_plan"] = plan
        logger.info(f"Generated Plan:\n{plan}")

        # 2. 認知ループ（情報収集・分析）
        cognitive_loop_input = {
            "query": query,
            "plan": plan,
            "reasoning_instruction": reasoning_instruction
        }
        cognitive_loop_output = await self.cognitive_loop_agent.ainvoke(cognitive_loop_input)
        reasoning_trace["step_2_cognitive_loop_output"] = cognitive_loop_output
        logger.info(f"Cognitive Loop Output:\n{cognitive_loop_output}")

        # 3. 最終回答の生成（ユーザーへの応答）
        if "https?://" in query:
            final_answer = cognitive_loop_output
            reasoning_trace["step_3_final_answer_generation"] = "Cognitive loop output was directly used as final answer due to URL in query."
            logger.info("URLクエリのため、Cognitive Loopの出力を最終回答として採用します。")
        else:
            max_length = 8000
            truncated_output = cognitive_loop_output[:max_length] if len(cognitive_loop_output) > max_length else cognitive_loop_output
            master_agent_input = {
                "query": query,
                "plan": plan,
                "cognitive_loop_output": truncated_output
            }
            final_answer = await self.master_agent.generate_final_answer_async(master_agent_input, orchestration_decision)
            reasoning_trace["step_3_final_answer_generation"] = final_answer
        
        # 4. 自己評価と潜在的問題の発見
        self_criticism = self.meta_cognitive_engine.critique_process_and_response(
            query=query,
            plan=plan,
            cognitive_loop_output=cognitive_loop_output,
            final_answer=final_answer
        )
        reasoning_trace["step_4_self_criticism"] = self_criticism
        logger.info(f"Self-Criticism:\n{self_criticism}")
        await self.analytics_collector.log_event("self_criticism", self_criticism)

        potential_problems_list = self.problem_discovery_agent.invoke({
            "query": query,
            "plan": plan,
            "cognitive_loop_output": cognitive_loop_output,
        })
        potential_problems = "\n".join(potential_problems_list) if potential_problems_list else "特になし"
        reasoning_trace["step_5_potential_problems"] = potential_problems
        logger.info(f"Discovered Potential Problems: {potential_problems}")
        await self.analytics_collector.log_event("potential_problems", potential_problems)

        # 5. AIの内部メンテナンスと自己進化プロセス（バックグラウンドで実行）
        async def background_tasks():
            await self.master_agent.run_internal_maintenance_async(query, final_answer)
            trace_data = {
                "query": query,
                "reasoning_trace": reasoning_trace,
                "final_answer": final_answer,
                "self_criticism": self_criticism,
            }
            await self.self_evolving_system.collect_execution_trace(trace_data)
            logger.info("Execution trace collected for potential self-evolution.")

        asyncio.create_task(background_tasks())
        
        logger.info(f"--- Full Pipeline END ({(time.time() - start_time):.2f} s) ---")

        return {
            "final_answer": final_answer,
            "self_criticism": self_criticism,
            "potential_problems": potential_problems,
            "retrieved_info": cognitive_loop_output,
        }