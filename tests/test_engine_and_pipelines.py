# tests/test_engine_and_pipelines.pyの修正
# path: tests/test_engine_and_pipelines.py

import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from langchain_core.runnables.base import RunnableConfig
from langchain_core.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from typing import Any, Dict, Callable, Awaitable
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

# テスト対象のモジュールと依存関係のモック
from app.engine.engine import MetaIntelligenceEngine
from app.pipelines.simple_pipeline import SimplePipeline
from app.pipelines.base import BasePipeline
from app.models import MasterAgentResponse, OrchestrationDecision

# --- 共通のモッククラス ---
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


class MockLLMProvider:
    def get_llm_instance(self, model: str, **kwargs) -> MockLLM:
        return MockLLM(response_content="dummy llm instance response")

class MockResourceArbiter:
    def arbitrate(self, decision: OrchestrationDecision) -> OrchestrationDecision:
        """仲裁ロジックをシミュレートし、無効なモードの場合はフォールバックする"""
        if decision.get("chosen_mode") not in ["simple", "full"]:
            new_decision = decision.copy()
            new_decision["chosen_mode"] = "simple"
            new_decision["reason"] = (
                f"FALLBACK: Insufficient energy for original choice "
                f"'{decision.get('chosen_mode', 'N/A')}'. Original reason: {decision.get('reason', 'N/A')}"
            )
            return new_decision
        return decision

class MockRetriever:
    def __init__(self, docs: list[Document]):
        self.docs = docs

    def invoke(self, query: str) -> list[Document]:
        return self.docs

class MockPromptManager:
    def get_prompt(self, name: str) -> ChatPromptTemplate:
        if name == "ROUTING_PROMPT":
            return ChatPromptTemplate.from_template("Route query: {query}")
        elif name == "SIMPLE_MASTER_AGENT_PROMPT":
            return ChatPromptTemplate.from_template("Answer based on: {query} and {retrieved_info}")
        elif name == "DIRECT_RESPONSE_PROMPT":
            return ChatPromptTemplate.from_template("Direct answer: {query}")
        return ChatPromptTemplate.from_template(f"Default prompt for {name}: {{query}}")


class TestMetaIntelligenceEngine(unittest.IsolatedAsyncioTestCase):
    """MetaIntelligenceEngineのユニットテスト"""

    async def asyncSetUp(self):
        self.patcher_settings = patch('app.config.settings')
        self.mock_settings = self.patcher_settings.start()
        self.mock_settings.PIPELINE_SETTINGS = {
            "simple": {"num_drafts": 1},
            "full": {"max_iterations": 1},
        }
        self.mock_settings.LLM_BACKEND = "ollama"
        self.mock_settings.GENERATION_LLM_SETTINGS = {"model": "mock", "temperature": 0.7}
        self.mock_settings.EMBEDDING_MODEL_NAME = "mock-embed"

        self.mock_resource_arbiter = MockResourceArbiter()
        
        self.mock_simple_pipeline = MagicMock(spec=BasePipeline)
        self.mock_simple_pipeline.arun = AsyncMock(return_value={
            "final_answer": "Mocked Simple Pipeline Response for Engine Test",
            "self_criticism": "", "potential_problems": "", "retrieved_info": ""
        })

        self.mock_full_pipeline = MagicMock(spec=BasePipeline)
        self.mock_full_pipeline.arun = AsyncMock(return_value={
            "final_answer": "Mocked Full Pipeline Response for Engine Test",
            "self_criticism": "", "potential_problems": "", "retrieved_info": ""
        })

        self.pipelines = {
            "simple": self.mock_simple_pipeline,
            "full": self.mock_full_pipeline,
        }
        self.engine = MetaIntelligenceEngine(
            pipelines=self.pipelines,
            resource_arbiter=self.mock_resource_arbiter
        )

    async def asyncTearDown(self):
        self.patcher_settings.stop()

    async def test_meta_intelligence_engine_simple_mode_execution(self):
        query = "今日の天気は？"
        orchestration_decision: OrchestrationDecision = {
            "chosen_mode": "simple",
            "reason": "weather_query",
            "agent_configs": {},
            "reasoning_emphasis": "current_info"
        }
        
        response = await self.engine.arun(query, orchestration_decision)

        self.assertEqual(response["final_answer"], "Mocked Simple Pipeline Response for Engine Test")
        self.mock_simple_pipeline.arun.assert_called_once_with(query, orchestration_decision)
        self.mock_full_pipeline.arun.assert_not_called()

    async def test_meta_intelligence_engine_full_mode_execution(self):
        query = "AIの意識とは何か、哲学的に論じなさい。"
        orchestration_decision: OrchestrationDecision = {
            "chosen_mode": "full",
            "reason": "philosophical_query",
            "agent_configs": {},
            "reasoning_emphasis": "conceptual"
        }
        
        response = await self.engine.arun(query, orchestration_decision)

        self.assertEqual(response["final_answer"], "Mocked Full Pipeline Response for Engine Test")
        self.mock_full_pipeline.arun.assert_called_once_with(query, orchestration_decision)
        self.mock_simple_pipeline.arun.assert_not_called()

    async def test_meta_intelligence_engine_invalid_mode_fallback(self):
        query = "何でもいいよ"
        orchestration_decision: OrchestrationDecision = {
            "chosen_mode": "invalid_mode",
            "reason": "testing_fallback",
            "agent_configs": {},
            "reasoning_emphasis": "none"
        }
        
        response = await self.engine.arun(query, orchestration_decision)

        self.assertEqual(response["final_answer"], "Mocked Simple Pipeline Response for Engine Test")
        
        expected_fallback_decision: OrchestrationDecision = {
            "chosen_mode": "simple",
            "reason": "FALLBACK: Insufficient energy for original choice 'invalid_mode'. Original reason: testing_fallback",
            "agent_configs": {},
            "reasoning_emphasis": "none"
        }
        self.mock_simple_pipeline.arun.assert_called_once_with(query, expected_fallback_decision)
        self.mock_full_pipeline.arun.assert_not_called()


class TestSimplePipeline(unittest.IsolatedAsyncioTestCase):
    """SimplePipelineのユニットテスト (内部ロジックに焦点を当てる)"""

    async def asyncSetUp(self):
        self.patcher_settings = patch('app.config.settings')
        self.mock_settings = self.patcher_settings.start()
        self.mock_settings.PIPELINE_SETTINGS = {
            "simple": {"num_drafts": 1},
            "full": {"max_iterations": 1},
        }
        self.mock_settings.LLM_BACKEND = "ollama"
        self.mock_settings.GENERATION_LLM_SETTINGS = {"model": "mock", "temperature": 0.7}
        self.mock_settings.EMBEDDING_MODEL_NAME = "mock-embed"

        self.mock_prompt_manager = MockPromptManager()
        
        self.mock_retriever = MagicMock(spec=MockRetriever)
        self.mock_retriever.invoke.return_value = [Document(page_content="retrieved info for rag")]

        self.mock_llm_router = MagicMock(spec=Runnable)
        self.mock_llm_direct = MagicMock(spec=Runnable)
        self.mock_llm_rag = MagicMock(spec=Runnable)

        self.simple_pipeline = SimplePipeline(
            llm=self.mock_llm_direct,
            output_parser=StrOutputParser(),
            retriever=self.mock_retriever,
            prompt_manager=self.mock_prompt_manager
        )
        
        self.simple_pipeline.router_chain = self.mock_prompt_manager.get_prompt("ROUTING_PROMPT") | self.mock_llm_router | JsonOutputParser()
        self.simple_pipeline.direct_chain = self.mock_prompt_manager.get_prompt("DIRECT_RESPONSE_PROMPT") | self.mock_llm_direct | StrOutputParser()
        self.simple_pipeline.rag_chain = self.mock_prompt_manager.get_prompt("SIMPLE_MASTER_AGENT_PROMPT") | self.mock_llm_rag | StrOutputParser()


    async def asyncTearDown(self):
        self.patcher_settings.stop()

    async def test_simple_pipeline_direct_route(self):
        self.mock_llm_router.ainvoke.return_value = '{"route": "DIRECT"}'
        self.mock_llm_direct.ainvoke.return_value = "Mocked Direct response for query: こんにちは"
        
        query = "こんにちは"
        orchestration_decision: OrchestrationDecision = {
            "chosen_mode": "simple", "reason": "greeting", "agent_configs": {}, "reasoning_emphasis": "concise"
        }
        
        response = await self.simple_pipeline.arun(query, orchestration_decision)

        self.assertEqual(response["final_answer"], "Mocked Direct response for query: こんにちは")
        self.assertEqual(response["retrieved_info"], "")
        self.mock_retriever.invoke.assert_not_called()
        self.mock_llm_router.ainvoke.assert_called_once()
        self.mock_llm_direct.ainvoke.assert_called_once()
        self.mock_llm_rag.ainvoke.assert_not_called()


    async def test_simple_pipeline_rag_route_success(self):
        self.mock_llm_router.ainvoke.return_value = '{"route": "RAG"}'
        self.mock_llm_rag.ainvoke.return_value = "Mocked RAG combined response for query: さんまについて教えてください with retrieved info for rag"
        
        query = "さんまについて教えてください"
        orchestration_decision: OrchestrationDecision = {
            "chosen_mode": "simple", "reason": "info_query", "agent_configs": {}, "reasoning_emphasis": "factual"
        }
        
        response = await self.simple_pipeline.arun(query, orchestration_decision)

        self.assertEqual(response["final_answer"], "Mocked RAG combined response for query: さんまについて教えてください with retrieved info for rag")
        self.assertEqual(response["retrieved_info"], "retrieved info for rag")
        self.mock_retriever.invoke.assert_called_once_with(query)
        self.mock_llm_router.ainvoke.assert_called_once()
        self.mock_llm_rag.ainvoke.assert_called_once()
        self.mock_llm_direct.ainvoke.assert_not_called()


    async def test_simple_pipeline_rag_route_no_retrieval_fallback(self):
        self.mock_llm_router.ainvoke.return_value = '{"route": "RAG"}'
        self.mock_retriever.invoke.return_value = [] 
        self.mock_llm_direct.ainvoke.return_value = "Mocked Direct response for query: 存在しないトピックについて教えてください (fallback)"
        
        query = "存在しないトピックについて教えてください"
        orchestration_decision: OrchestrationDecision = {
            "chosen_mode": "simple", "reason": "info_query_no_data", "agent_configs": {}, "reasoning_emphasis": "factual"
        }
        
        response = await self.simple_pipeline.arun(query, orchestration_decision)

        self.assertEqual(response["final_answer"], "Mocked Direct response for query: 存在しないトピックについて教えてください (fallback)")
        self.assertEqual(response["retrieved_info"], "")
        self.mock_retriever.invoke.assert_called_once_with(query)
        self.mock_llm_router.ainvoke.assert_called_once()
        self.mock_llm_direct.ainvoke.assert_called_once()
        self.mock_llm_rag.ainvoke.assert_not_called()


if __name__ == '__main__':
    unittest.main()