# app/agents/master_agent.pyの修正
# path: app/agents/master_agent.py

import logging
from typing import Any, Dict, List, TYPE_CHECKING
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.agents.base import AIAgent
from app.memory.memory_consolidator import MemoryConsolidator
from app.cognitive_modeling.predictive_coding_engine import PredictiveCodingEngine
from app.memory.working_memory import WorkingMemory
from app.affective_system.affective_engine import AffectiveEngine
from app.affective_system.emotional_response_generator import EmotionalResponseGenerator

if TYPE_CHECKING:
    from app.digital_homeostasis.ethical_motivation_engine import EthicalMotivationEngine
    from app.value_evolution.value_evaluator import ValueEvaluator
    from app.agents.orchestration_agent import OrchestrationAgent
    from app.analytics import AnalyticsCollector
    from app.models import OrchestrationDecision


logger = logging.getLogger(__name__)

class MasterAgent(AIAgent):
    """
    認知アーキテクチャ全体を統括し、最終的な回答を生成するマスターAI。
    """
    def __init__(
        self,
        llm: Any,
        output_parser: Any,
        prompt_template: ChatPromptTemplate,
        memory_consolidator: MemoryConsolidator,
        ethical_motivation_engine: 'EthicalMotivationEngine',
        predictive_coding_engine: PredictiveCodingEngine,
        working_memory: WorkingMemory,
        value_evaluator: 'ValueEvaluator',
        orchestration_agent: 'OrchestrationAgent',
        affective_engine: AffectiveEngine,
        emotional_response_generator: EmotionalResponseGenerator,
        analytics_collector: 'AnalyticsCollector',
    ):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        self.memory_consolidator = memory_consolidator
        self.ethical_motivation_engine = ethical_motivation_engine
        self.predictive_coding_engine = predictive_coding_engine
        self.working_memory = working_memory
        self.dialogue_history: List[str] = []
        self.value_evaluator = value_evaluator
        self.orchestration_agent = orchestration_agent
        self.affective_engine = affective_engine
        self.emotional_response_generator = emotional_response_generator
        self.analytics_collector = analytics_collector
        super().__init__()

    def build_chain(self) -> Runnable:
        return self.prompt_template | self.llm | self.output_parser

    async def generate_final_answer_async(self, input_data: Dict[str, Any], orchestration_decision: 'OrchestrationDecision') -> str:
        """
        ユーザーへの最終応答を非同期で生成する。このプロセスは迅速に完了する必要がある。
        """
        if self._chain is None:
            raise RuntimeError("MasterAgent's chain is not initialized.")

        query = input_data.get("query", "")

        # 1. 現在の状況から感情状態を評価
        affective_state = await self.affective_engine.assess_and_update_state(user_query=query)
        logger.info(f"現在の感情状態: {affective_state.emotion.value} (強度: {affective_state.intensity})")
        await self.analytics_collector.log_event("affective_state", affective_state.model_dump())

        # 2. オーケストレーション決定に基づき、思考の強調点を決定
        reasoning_emphasis = orchestration_decision.get("reasoning_emphasis")
        reasoning_instruction = ""
        if reasoning_emphasis == "bird's_eye_view":
            reasoning_instruction = "回答は、概念間の関係性、全体像、長期的な影響、または抽象的な原則を強調してください。"
        elif reasoning_emphasis == "detail_oriented":
            reasoning_instruction = "回答は、具体的な事実、詳細な手順、明確なデータ、または精密な論理構造を強調してください。"

        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 3. 最近のバックグラウンド思考の洞察を取得し、プロンプトに含める
        physical_insights_logs = self.memory_consolidator.get_recent_insights("physical_simulation_insight", limit=1)
        physical_insights = "\n".join([log.get("synthesized_knowledge", "") for log in physical_insights_logs])
        if not physical_insights:
            physical_insights = "特筆すべき物理シミュレーションからの洞察はありません。"

        recent_autonomous_thoughts_logs = self.memory_consolidator.get_recent_insights("autonomous_thought", limit=1)
        recent_autonomous_thoughts = "\n".join([log.get("synthesized_knowledge", "") for log in recent_autonomous_thoughts_logs])
        if not recent_autonomous_thoughts:
            recent_autonomous_thoughts = "特筆すべき自律学習からの洞察はありません。"

        recent_self_improvement_insights_logs = self.memory_consolidator.get_recent_insights("self_improvement_applied_decision", limit=1)
        recent_self_improvement_insights = "\n".join([log.get("synthesized_knowledge", "") for log in recent_self_improvement_insights_logs])
        if not recent_self_improvement_insights:
            recent_self_improvement_insights = "特筆すべき自己改善からの洞察はありません。"
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

        # 4. LLMに渡す最終的なプロンプトを構築
        master_agent_prompt_input = {
            "query": input_data.get("query", ""),
            "plan": input_data.get("plan", ""),
            "cognitive_loop_output": input_data.get("cognitive_loop_output", ""),
            "reasoning_instruction": reasoning_instruction,
            "physical_insights": physical_insights,
            "recent_autonomous_thoughts": recent_autonomous_thoughts, # 追加
            "recent_self_improvement_insights": recent_self_improvement_insights # 追加
        }

        # 5. LLMを通じて最終回答案を生成
        final_answer = await self._chain.ainvoke(master_agent_prompt_input)

        # 6. 感情状態を反映させて最終的な応答を微調整
        emotional_response_input = {
            "final_answer": final_answer,
            "affective_state": affective_state,
            "emotion": affective_state.emotion.value,
            "intensity": affective_state.intensity,
            "reason": affective_state.reason
        }
        final_answer_with_emotion = self.emotional_response_generator.invoke(emotional_response_input)

        return final_answer_with_emotion

    async def run_internal_maintenance_async(self, query: str, final_answer: str):
        """
        応答生成後に実行される、AIの内部状態を維持するための非同期バックグラウンドプロセス。
        """
        logger.info("--- AIの内部メンテナンス（ホメオスタシス）プロセスを開始 ---")
        try:
            # 倫理的動機付けの評価
            motivation = await self.ethical_motivation_engine.assess_and_generate_motivation(final_answer)
            logger.info(f"倫理的動機付け: {motivation}")

            # 予測符号化による学習
            prediction_error = self.predictive_coding_engine.process_input(query, self.dialogue_history)
            if prediction_error:
                logger.info(f"予測誤差が検出されました: {prediction_error}")
                self.working_memory.add_prediction_error(prediction_error)

            # 価値観の評価と更新
            await self.value_evaluator.assess_and_update_values(final_answer)

            # 対話履歴の記録
            self.memory_consolidator.log_interaction(query, final_answer)
            self.dialogue_history.append(f"User: {query}")
            self.dialogue_history.append(f"AI: {final_answer}")
            logger.info("--- AIの内部メンテナンスプロセスが完了 ---")
        except Exception as e:
            logger.error(f"内部メンテナンスプロセス中にエラーが発生しました: {e}", exc_info=True)


    async def ainvoke(self, input_data: Dict[str, Any], orchestration_decision: 'OrchestrationDecision') -> str:
        """
        【注意】このメソッドは現在、直接の応答生成には使用されません。
        応答生成は generate_final_answer_async で、内部メンテナンスは run_internal_maintenance_async で実行されます。
        """
        if not isinstance(input_data, dict):
            raise TypeError("MasterAgent's ainvoke expects a dictionary as input.")
        logger.warning("MasterAgentのainvokeが直接呼び出されましたが、このメソッドは応答生成を行いません。")
        return await self.generate_final_answer_async(input_data, orchestration_decision)