# tests/test_master_agent.pyの修正
# path: tests/test_master_agent.py

import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from langchain_core.runnables.base import RunnableConfig # RunnableConfigをインポートし直す
from langchain_core.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
# ChatPromptValue や HumanMessage はテストのモックで対応するため、ここではインポートしない
# from langchain_core.prompts.chat import ChatPromptValue # 削除
# from langchain_core.messages import HumanMessage # 削除
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from typing import Any, Callable, Awaitable
import asyncio

# テスト対象のモジュールをインポート
from app.agents.master_agent import MasterAgent
from app.memory.memory_consolidator import MemoryConsolidator
from app.digital_homeostasis.ethical_motivation_engine import EthicalMotivationEngine
from app.cognitive_modeling.predictive_coding_engine import PredictiveCodingEngine
from app.memory.working_memory import WorkingMemory
from app.value_evolution.value_evaluator import ValueEvaluator
from app.affective_system.affective_engine import AffectiveEngine
from app.affective_system.affective_state import AffectiveState, Emotion
from app.affective_system.emotional_response_generator import EmotionalResponseGenerator
from app.analytics.collector import AnalyticsCollector
from app.agents.orchestration_agent import OrchestrationAgent

class MockLLM(Runnable):
    """LangChain LLMを模倣するモッククラス"""
    def __init__(self, response_content: str):
        self.response_content = response_content

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        def dummy_sync_func(inner_input):
            return self.response_content
        return self._call_with_config(dummy_sync_func, input, config)

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        async def dummy_async_func(inner_input):
            return self.response_content
        return await self._acall_with_config(dummy_async_func, input, config)

    def _call_with_config(
        self,
        func: Callable[[Any], Any] | Callable[[Any, CallbackManagerForChainRun], Any] | Callable[[Any, CallbackManagerForChainRun, RunnableConfig], Any],
        input_: Any,
        config: RunnableConfig | None,
        run_type: str | None = None,
        serialized: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.response_content

    async def _acall_with_config(
        self,
        func: Callable[[Any], Awaitable[Any]] | Callable[[Any, AsyncCallbackManagerForChainRun], Awaitable[Any]] | Callable[[Any, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Any]],
        input_: Any,
        config: RunnableConfig | None,
        run_type: str | None = None,
        serialized: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.response_content


class TestMasterAgent(unittest.IsolatedAsyncioTestCase):
    """MasterAgentのテストスイート"""

    async def asyncSetUp(self):
        # MasterAgentの__init__に渡す全ての依存関係をモック
        self.mock_llm = MagicMock(spec=Runnable)
        self.mock_llm.ainvoke.return_value = "Final answer from MasterAgent."
        
        self.mock_output_parser = StrOutputParser()
        # ChatPromptTemplateをモックし、ainvokeの戻り値をAsyncMockとして設定
        self.mock_prompt_template = MagicMock(spec=ChatPromptTemplate)
        
        # HumanMessage や ChatPromptValue を直接インポートしないため、構造をモックする
        mock_human_message = MagicMock()
        mock_human_message.content = "Mocked formatted prompt content"
        
        mock_chat_prompt_value = MagicMock() # ChatPromptValueの代わりにMagicMockを使用
        mock_chat_prompt_value.messages = [mock_human_message] # messages属性を設定
        
        self.mock_prompt_template.ainvoke = AsyncMock(return_value=mock_chat_prompt_value)
        self.expected_chat_prompt_value_mock = mock_chat_prompt_value # アサーションのためにモックを保持
        
        self.mock_memory_consolidator = MagicMock(spec=MemoryConsolidator)
        if asyncio.iscoroutinefunction(getattr(MemoryConsolidator, 'log_interaction', None)):
            self.mock_memory_consolidator.log_interaction = AsyncMock()
        else:
            self.mock_memory_consolidator.log_interaction = MagicMock()
        self.mock_memory_consolidator.get_recent_insights.side_effect = lambda event_type, limit: []
        
        self.mock_ethical_motivation_engine = MagicMock(spec=EthicalMotivationEngine)
        # assess_and_generate_motivationのサイドエフェクトとしてlog_eventを呼び出すモック関数を定義
        async def mock_assess_and_generate_motivation(final_answer):
            await self.mock_analytics_collector.log_event("integrity_status", {'homeostatic_state': 'stable', 'drive_summary': 'Stable.'})
            return {"homeostatic_state": "stable", "drive_summary": "Stable."}
        self.mock_ethical_motivation_engine.assess_and_generate_motivation = AsyncMock(side_effect=mock_assess_and_generate_motivation)
        
        self.mock_predictive_coding_engine = MagicMock(spec=PredictiveCodingEngine)
        self.mock_predictive_coding_engine.process_input = MagicMock(return_value=None)
        
        self.mock_working_memory = MagicMock(spec=WorkingMemory)
        self.mock_working_memory.add_prediction_error = MagicMock()
        
        self.mock_value_evaluator = MagicMock(spec=ValueEvaluator)
        self.mock_value_evaluator.assess_and_update_values = AsyncMock()
        
        self.mock_affective_engine = MagicMock(spec=AffectiveEngine)
        mock_affective_state = MagicMock(spec=AffectiveState)
        mock_affective_state.emotion = MagicMock(spec=Emotion)
        mock_affective_state.emotion.value = "neutral"
        mock_affective_state.intensity = 0.5
        mock_affective_state.reason = "initial state"
        mock_affective_state.model_dump.return_value = {"emotion": "neutral", "intensity": 0.5}
        
        self.mock_affective_engine.assess_and_update_state = AsyncMock(return_value=mock_affective_state)
        
        self.mock_emotional_response_generator = MagicMock(spec=EmotionalResponseGenerator)
        self.mock_emotional_response_generator.invoke.return_value = "Final answer with emotional tone."
        
        self.mock_analytics_collector = MagicMock(spec=AnalyticsCollector)
        self.mock_analytics_collector.log_event = AsyncMock()

        self.mock_orchestration_agent = MagicMock(spec=OrchestrationAgent)
        
        # MasterAgentのインスタンスを生成
        self.master_agent = MasterAgent(
            llm=self.mock_llm,
            output_parser=self.mock_output_parser,
            prompt_template=self.mock_prompt_template,
            memory_consolidator=self.mock_memory_consolidator,
            ethical_motivation_engine=self.mock_ethical_motivation_engine,
            predictive_coding_engine=self.mock_predictive_coding_engine,
            working_memory=self.mock_working_memory,
            value_evaluator=self.mock_value_evaluator,
            orchestration_agent=self.mock_orchestration_agent,
            affective_engine=self.mock_affective_engine,
            emotional_response_generator=self.mock_emotional_response_generator,
            analytics_collector=self.mock_analytics_collector,
        )
        self.master_agent._chain = AsyncMock(spec=Runnable)
        self.master_agent._chain.ainvoke.return_value = "Final answer from MasterAgent's mocked chain."

    async def test_ainvoke_calls_generate_final_answer_async(self):
        query = "テストクエリ"
        orchestration_decision = {
            "chosen_mode": "full",
            "reason": "test",
            "agent_configs": {},
            "reasoning_emphasis": "none"
        }
        input_data = {
            "query": query,
            "plan": "テスト計画",
            "cognitive_loop_output": "認知ループ出力",
        }

        with patch.object(self.master_agent, 'generate_final_answer_async', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "Mocked final answer."
            
            result = await self.master_agent.ainvoke(input_data, orchestration_decision)
            
            mock_generate.assert_called_once_with(input_data, orchestration_decision)
            self.assertEqual(result, "Mocked final answer.")

    async def test_ainvoke_raises_type_error_for_non_dict_input(self):
        with self.assertRaises(TypeError):
            await self.master_agent.ainvoke("無効な入力", {})

    async def test_generate_final_answer_async_default_emphasis(self):
        query = "簡単な質問"
        plan = "単純な回答"
        cognitive_loop_output = "情報なし"
        orchestration_decision = {
            "chosen_mode": "simple",
            "reason": "test",
            "agent_configs": {},
            "reasoning_emphasis": None
        }
        input_data = {
            "query": query,
            "plan": plan,
            "cognitive_loop_output": cognitive_loop_output,
        }

        self.master_agent._chain.ainvoke.return_value = "Default emphasis final answer from mocked chain."

        result = await self.master_agent.generate_final_answer_async(input_data, orchestration_decision)
        
        self.assertEqual(result, "Final answer with emotional tone.")

        expected_prompt_input = {
            "query": query,
            "plan": plan,
            "cognitive_loop_output": cognitive_loop_output,
            "reasoning_instruction": "",
            "physical_insights": "特筆すべき物理シミュレーションからの洞察はありません。",
            "recent_autonomous_thoughts": "特筆すべき自律学習からの洞察はありません。",
            "recent_self_improvement_insights": "特筆すべき自己改善からの洞察はありません。",
        }
        self.master_agent._chain.ainvoke.assert_called_once_with(expected_prompt_input)
        self.mock_affective_engine.assess_and_update_state.assert_called_once()
        self.mock_emotional_response_generator.invoke.assert_called_once_with({
            "final_answer": self.master_agent._chain.ainvoke.return_value,
            "affective_state": self.mock_affective_engine.assess_and_update_state.return_value,
            "emotion": self.mock_affective_engine.assess_and_update_state.return_value.emotion.value,
            "intensity": self.mock_affective_engine.assess_and_update_state.return_value.intensity,
            "reason": self.mock_affective_engine.assess_and_update_state.return_value.reason
        })
        self.mock_analytics_collector.log_event.assert_awaited_with("affective_state", {"emotion": "neutral", "intensity": 0.5})

    async def test_generate_final_answer_async_bird_eye_view(self):
        query = "AIの未来の全体像について"
        plan = "全体像の分析"
        cognitive_loop_output = "未来のAIに関する広範な情報"
        orchestration_decision = {
            "chosen_mode": "full",
            "reason": "test",
            "agent_configs": {},
            "reasoning_emphasis": "bird's_eye_view"
        }
        input_data = {
            "query": query,
            "plan": plan,
            "cognitive_loop_output": cognitive_loop_output,
        }

        self.master_agent._chain.ainvoke.return_value = "Bird's eye view final answer from mocked chain."

        result = await self.master_agent.generate_final_answer_async(input_data, orchestration_decision)
        
        self.assertEqual(result, "Final answer with emotional tone.") 

        expected_prompt_input = {
            "query": query,
            "plan": plan,
            "cognitive_loop_output": cognitive_loop_output,
            "reasoning_instruction": "回答は、概念間の関係性、全体像、長期的な影響、または抽象的な原則を強調してください。",
            "physical_insights": "特筆すべき物理シミュレーションからの洞察はありません。",
            "recent_autonomous_thoughts": "特筆すべき自律学習からの洞察はありません。",
            "recent_self_improvement_insights": "特筆すべき自己改善からの洞察はありません。",
        }
        self.master_agent._chain.ainvoke.assert_called_once_with(expected_prompt_input)

    async def test_generate_final_answer_async_detail_oriented(self):
        query = "AIの実装手順の詳細について"
        plan = "詳細な実装計画"
        cognitive_loop_output = "AIの実装ステップに関する具体的な情報"
        orchestration_decision = {
            "chosen_mode": "full",
            "reason": "test",
            "agent_configs": {},
            "reasoning_emphasis": "detail_oriented"
        }
        input_data = {
            "query": query,
            "plan": plan,
            "cognitive_loop_output": cognitive_loop_output,
        }

        self.master_agent._chain.ainvoke.return_value = "Detail oriented final answer from mocked chain."

        result = await self.master_agent.generate_final_answer_async(input_data, orchestration_decision)
        
        self.assertEqual(result, "Final answer with emotional tone.") 

        expected_prompt_input = {
            "query": query,
            "plan": plan,
            "cognitive_loop_output": cognitive_loop_output,
            "reasoning_instruction": "回答は、具体的な事実、詳細な手順、明確なデータ、または精密な論理構造を強調してください。",
            "physical_insights": "特筆すべき物理シミュレーションからの洞察はありません。",
            "recent_autonomous_thoughts": "特筆すべき自律学習からの洞察はありません。",
            "recent_self_improvement_insights": "特筆すべき自己改善からの洞察はありません。",
        }
        self.master_agent._chain.ainvoke.assert_called_once_with(expected_prompt_input)

    async def test_run_internal_maintenance_async_no_prediction_error(self):
        query = "テストメンテナンス"
        final_answer = "テスト回答"
        self.mock_predictive_coding_engine.process_input.return_value = {}
        
        await self.master_agent.run_internal_maintenance_async(query, final_answer)
        
        self.mock_ethical_motivation_engine.assess_and_generate_motivation.assert_called_once_with(final_answer)
        self.mock_predictive_coding_engine.process_input.assert_called_once()
        self.mock_working_memory.add_prediction_error.assert_not_called()
        self.mock_value_evaluator.assess_and_update_values.assert_called_once_with(final_answer)
        self.mock_memory_consolidator.log_interaction.assert_called_once_with(query, final_answer)
        self.mock_analytics_collector.log_event.assert_awaited_with("integrity_status", {'homeostatic_state': 'stable', 'drive_summary': 'Stable.'})
        self.assertEqual(len(self.master_agent.dialogue_history), 2)
        self.assertEqual(self.master_agent.dialogue_history[0], f"User: {query}")
        self.assertEqual(self.master_agent.dialogue_history[1], f"AI: {final_answer}")

    async def test_run_internal_maintenance_async_with_prediction_error(self):
        query = "テストメンテナンス"
        final_answer = "テスト回答"
        prediction_error_data = {"error_type": "新規情報", "summary": "新しい概念", "key_info": ["概念A"]}
        self.mock_predictive_coding_engine.process_input.return_value = prediction_error_data
        
        await self.master_agent.run_internal_maintenance_async(query, final_answer)
        
        self.mock_predictive_coding_engine.process_input.assert_called_once()
        self.mock_working_memory.add_prediction_error.assert_called_once_with(prediction_error_data)
        self.mock_memory_consolidator.log_interaction.assert_called_once_with(query, final_answer)

    async def test_generate_final_answer_async_with_recent_insights(self):
        query = "AIの能力について"
        plan = "能力分析"
        cognitive_loop_output = "能力に関する詳細なデータ"
        orchestration_decision = {
            "chosen_mode": "full",
            "reason": "test",
            "agent_configs": {},
            "reasoning_emphasis": "none"
        }
        input_data = {
            "query": query,
            "plan": plan,
            "cognitive_loop_output": cognitive_loop_output,
        }

        self.mock_memory_consolidator.get_recent_insights.side_effect = [
            [{"synthesized_knowledge": "最近の物理シミュレーション洞察。"}],
            [{"synthesized_knowledge": "最近の自律学習洞察。"}],
            [{"synthesized_knowledge": "最近の自己改善洞察。"}],
        ]

        self.master_agent._chain.ainvoke.return_value = "Final answer with all insights from mocked chain."

        result = await self.master_agent.generate_final_answer_async(input_data, orchestration_decision)

        self.assertEqual(result, "Final answer with emotional tone.")

        expected_prompt_input = {
            "query": query,
            "plan": plan,
            "cognitive_loop_output": cognitive_loop_output,
            "reasoning_instruction": "",
            "physical_insights": "最近の物理シミュレーション洞察。",
            "recent_autonomous_thoughts": "最近の自律学習洞察。",
            "recent_self_improvement_insights": "最近の自己改善洞察。",
        }
        self.master_agent._chain.ainvoke.assert_called_once_with(expected_prompt_input)
        self.mock_memory_consolidator.get_recent_insights.assert_any_call("physical_simulation_insight", limit=1)
        self.mock_memory_consolidator.get_recent_insights.assert_any_call("autonomous_thought", limit=1)
        self.mock_memory_consolidator.get_recent_insights.assert_any_call("self_improvement_applied_decision", limit=1)

if __name__ == '__main__':
    unittest.main()