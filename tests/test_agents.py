# tests/test_agents.pyの修正
# path: tests/test_agents.py

import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from langchain_core.runnables.base import RunnableConfig
from langchain_core.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from typing import Any, Callable, Awaitable
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
import asyncio # asyncioをインポート

# テスト対象のモジュールをインポート
from app.agents.planning_agent import PlanningAgent

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


class TestPlanningAgent(unittest.IsolatedAsyncioTestCase):
    """PlanningAgentのテストスイート"""

    async def asyncSetUp(self):
        """各テストケースの前に実行されるセットアップ"""
        self.mock_llm = MockLLM("DECOMPOSE, RAG_SEARCH, SYNTHESIZE")
        self.output_parser = StrOutputParser()
        self.prompt_template = ChatPromptTemplate.from_template("Test prompt: {query}")
        
        self.planning_agent = PlanningAgent(
            llm=self.mock_llm,
            output_parser=self.output_parser,
            prompt_template=self.prompt_template
        )
        self.planning_agent.build_chain()

    async def test_select_thinking_modules(self):
        """
        select_thinking_modulesメソッドが正しい思考モジュールシーケンスを返すかテストする。
        """
        query = "複雑な問題を解決する方法を教えてください。"
        expected_modules = "DECOMPOSE, RAG_SEARCH, SYNTHESIZE"
        
        result = self.planning_agent.select_thinking_modules(query)
        
        self.assertEqual(result, expected_modules)

if __name__ == '__main__':
    unittest.main()