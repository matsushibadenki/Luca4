# /app/meta_intelligence/self_improvement/evolution.py
# title: Self-Evolving System
# role: Analyzes and improves its own intelligence.

from typing import Dict, Any, List, TYPE_CHECKING
import logging

from app.agents.self_improvement_agent import SelfImprovementAgent
from app.agents.self_correction_agent import SelfCorrectionAgent
from app.meta_cognition.meta_cognitive_engine import MetaCognitiveEngine

if TYPE_CHECKING:
    from app.analytics import AnalyticsCollector
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    from app.agents.process_reward_agent import ProcessRewardAgent
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

logger = logging.getLogger(__name__)

class SelfEvolvingSystem:
    """
    自分自身を分析し、改善する知能。
    思考プロセスを客観視し、弱点を発見し、改善戦略を立案・実装（検討）する。
    """
    def __init__(
        self,
        meta_cognitive_engine: MetaCognitiveEngine,
        self_improvement_agent: SelfImprovementAgent,
        self_correction_agent: SelfCorrectionAgent,
        analytics_collector: "AnalyticsCollector",
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        process_reward_agent: "ProcessRewardAgent",
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    ):
        """
        自己進化システムを初期化します。
        """
        self.meta_cognitive_engine = meta_cognitive_engine
        self.self_improvement_agent = self_improvement_agent
        self.self_correction_agent = self_correction_agent
        self.analytics_collector = analytics_collector
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.process_reward_agent = process_reward_agent
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.performance_traces: List[Dict[str, Any]] = []

    async def collect_execution_trace(self, trace_data: Dict[str, Any]):
        """
        AIの実行トレース（思考の記録）を収集し、アナリティクスに送信します。
        """
        self.performance_traces.append(trace_data)
        logger.info("Execution trace collected for self-analysis.")
        await self.analytics_collector.log_event("execution_trace", trace_data)

    async def analyze_own_performance(self) -> None:
        """
        収集された実行トレースを基に、自己のパフォーマンスを分析し、改善サイクルを実行します。
        """
        if not self.performance_traces:
            logger.warning("No performance traces to analyze. Skipping self-evolution cycle.")
            return

        logger.info("--- Starting Self-Evolution Cycle ---")

        latest_trace = self.performance_traces[-1]
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # Step 1: Process Reward Modelによる各ステップの評価
        process_feedback = []
        reasoning_trace = latest_trace.get("reasoning_trace", {})
        query = latest_trace.get("query", "")
        for step_name, step_content in reasoning_trace.items():
            reward_input = {
                "query": query,
                "step_name": step_name,
                "step_content": str(step_content),
            }
            reward_result = self.process_reward_agent.invoke(reward_input)
            process_feedback.append({
                "step": step_name,
                "reward": reward_result.get("reward_score", 0.0),
                "justification": reward_result.get("justification", "")
            })
        logger.info(f"Process Reward Model Feedback: {process_feedback}")
        await self.analytics_collector.log_event("process_feedback", process_feedback)

        # Step 2: Meta-cognitive analysis on the final output
        logger.info("Step 2: Performing meta-cognitive analysis on the final trace.")
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self_criticism = self.meta_cognitive_engine.critique_process_and_response(
            query=latest_trace.get("query", ""),
            plan=latest_trace.get("plan", ""),
            cognitive_loop_output=latest_trace.get("cognitive_loop_output", ""),
            final_answer=latest_trace.get("final_answer", "")
        )
        logger.info(f"Meta-cognitive Analysis (Self-Criticism): {self_criticism}")
        
        await self.analytics_collector.log_event("self_criticism", self_criticism)

        if not self_criticism or "問題なし" in self_criticism:
            logger.info("No significant weaknesses found. Concluding self-evolution cycle.")
            self.performance_traces.clear()
            return

        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # Step 3: Designing self-improvement plan based on weaknesses and process rewards
        logger.info("Step 3: Designing self-improvement plan based on weaknesses and process rewards.")
        improvement_input = {
            "trace_data": latest_trace,
            "process_feedback": str(process_feedback),
            "self_criticism": self_criticism
        }
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        improvement_suggestions = self.self_improvement_agent.invoke(improvement_input)
        
        if not improvement_suggestions:
            logger.warning("Could not design any improvement suggestions.")
            self.performance_traces.clear()
            return
            
        logger.info(f"Generated Improvement Suggestions: {improvement_suggestions}")
        await self.analytics_collector.log_event("improvement_suggestions", improvement_suggestions)

        logger.info("Step 4: Implementing (considering) improvements.")
        self.self_correction_agent.consider_and_log_application(improvement_suggestions)
        
        self.performance_traces.clear()
        logger.info("--- Self-Evolution Cycle Completed ---")