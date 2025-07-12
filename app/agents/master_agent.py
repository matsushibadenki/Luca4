# /app/agents/master_agent.py
# title: マスターAIエージェント
# role: 全ての思考プロセスを統括し、最終的な応答を生成する司令塔。

from __future__ import annotations
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
    from app.models import OrchestrationDecision # Import OrchestrationDecision


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
        # self.engine の依存関係を削除
        self.dialogue_history: List[str] = []
        self.value_evaluator = value_evaluator
        self.orchestration_agent = orchestration_agent
        self.affective_engine = affective_engine
        self.emotional_response_generator = emotional_response_generator
        self.analytics_collector = analytics_collector
        super().__init__()

    def build_chain(self) -> Runnable:
        return self.prompt_template | self.llm | self.output_parser
    
    async def ainvoke(self, input_data: Dict[str, Any] | str, orchestration_decision: 'OrchestrationDecision') -> str:
        """非同期でMasterAgentのロジックを実行する。"""
        if not isinstance(input_data, dict):
            raise TypeError("MasterAgent expects a dictionary as input.")
        
        if self._chain is None:
            raise RuntimeError("MasterAgent's chain is not initialized.")
        
        query = input_data.get("query", "")
        
        affective_state = await self.affective_engine.assess_and_update_state(user_query=query)
        
        logger.info(f"現在の感情状態: {affective_state.emotion.value} (強度: {affective_state.intensity})")
        await self.analytics_collector.log_event("affective_state", affective_state.model_dump())

        # Determine reasoning instruction based on orchestration decision
        reasoning_emphasis = orchestration_decision.get("reasoning_emphasis")
        reasoning_instruction = ""
        if reasoning_emphasis == "bird's_eye_view":
            reasoning_instruction = "回答は、概念間の関係性、全体像、長期的な影響、または抽象的な原則を強調してください。"
        elif reasoning_emphasis == "detail_oriented":
            reasoning_instruction = "回答は、具体的な事実、詳細な手順、明確なデータ、または精密な論理構造を強調してください。"

        # Add reasoning_instruction to input_data for the prompt
        master_agent_prompt_input = {
            "query": input_data.get("query", ""),
            "plan": input_data.get("plan", ""),
            "cognitive_loop_output": input_data.get("cognitive_loop_output", ""),
            "reasoning_instruction": reasoning_instruction
        }

        final_answer = await self._chain.ainvoke(master_agent_prompt_input)

        emotional_response_input = {
            "final_answer": final_answer,
            "affective_state": affective_state,
            "emotion": affective_state.emotion.value,
            "intensity": affective_state.intensity,
            "reason": affective_state.reason
        }
        final_answer_with_emotion = self.emotional_response_generator.invoke(emotional_response_input)

        motivation = await self.ethical_motivation_engine.assess_and_generate_motivation(final_answer)
        logger.info(f"倫理的動機付け: {motivation}")
        
        prediction_error = self.predictive_coding_engine.process_input(query, self.dialogue_history)
        if prediction_error:
            logger.info(f"予測誤差が検出されました: {prediction_error}")
            self.working_memory.add_prediction_error(prediction_error)
        
        await self.value_evaluator.assess_and_update_values(final_answer)

        self.memory_consolidator.log_interaction(query, final_answer_with_emotion)

        self.dialogue_history.append(f"User: {query}")
        self.dialogue_history.append(f"AI: {final_answer_with_emotion}")
        
        return final_answer_with_emotion

    # 既存のinvokeメソッドは削除し、aitaskを呼び出すメソッドのみに集約
    # (AIAgentのinvokeメソッドを継承し、直接は利用しない想定)