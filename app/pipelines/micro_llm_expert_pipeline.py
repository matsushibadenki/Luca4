# /app/pipelines/micro_llm_expert_pipeline.py
# title: マイクロLLM専門家パイプライン
# role: 専門的なクエリに対し、対応するマイクロLLMツールを選択・実行して回答を生成する。

from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING, Dict, Any

from app.pipelines.base import BasePipeline
from app.models import MasterAgentResponse, OrchestrationDecision
from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from app.agents.tool_using_agent import ToolUsingAgent
    from app.tools.tool_belt import ToolBelt
    from app.llm_providers.base import LLMProvider

logger = logging.getLogger(__name__)

class MicroLLMExpertPipeline(BasePipeline):
    """
    専門的なクエリに対して、対応するマイクロLLMツールを活用するパイプライン。
    """
    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_using_agent: ToolUsingAgent,
        tool_belt: ToolBelt,
    ):
        self.llm_provider = llm_provider
        self.tool_using_agent = tool_using_agent
        self.tool_belt = tool_belt
        # 回答を整形するための汎用LLMインスタンス
        self.formatter_llm = self.llm_provider.get_llm_instance(
            model="gemma3:latest", temperature=0.7
        )

    async def arun(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        """
        パイプラインを非同期で実行する。
        """
        start_time = time.time()
        logger.info(f"--- MicroLLM Expert Pipeline START for query: '{query}' ---")

        # 1. 適切な専門家ツール（マイクロLLM）を選択
        tool_descriptions = self.tool_belt.get_tool_descriptions()
        tool_selection_input = {"tools": tool_descriptions, "task": query}
        tool_decision_str: str = self.tool_using_agent.invoke(tool_selection_input)

        tool_name, tool_query = (
            [s.strip() for s in tool_decision_str.split(":", 1)]
            if ":" in tool_decision_str
            else (None, None)
        )

        # ツール名とクエリの両方が有効かチェック
        if not tool_name or not tool_query or not tool_name.startswith("Specialist_"):
            logger.warning("適切な専門家ツールまたはクエリが見つかりませんでした。Fullパイプラインにフォールバックすべき状況です。")
            return {
                "final_answer": "申し訳ありません、この質問に答えられる専門家が見つかりませんでした",
                "self_criticism": "専門家ツール選択またはクエリ生成に失敗しました",
                "potential_problems": "対応するマイクロLLMがまだ作成されていないか、LLMがクエリを生成できませんでした",
                "retrieved_info": f"ツール選択結果: {tool_decision_str}"
            }

        # 2. 選択された専門家ツールを実行
        expert_tool = self.tool_belt.get_tool(tool_name)
        if not expert_tool:
            logger.error(f"選択されたツール '{tool_name}' がToolBelt内に見つかりません。")
            return {"final_answer": "エラーが発生しました", "self_criticism": "", "potential_problems": "", "retrieved_info": ""}


        logger.info(f"専門家ツール '{tool_name}' をクエリ '{tool_query}' で実行します。")
        # tool_queryがNoneでないことは上でチェック済みのため、mypyエラーは発生しない
        expert_answer = expert_tool.use(tool_query)

        # 3. 専門家の回答を整形して最終的な応答を生成
        formatter_prompt = ChatPromptTemplate.from_template(
            """あなたは優秀なアシスタントです。以下の専門家からの回答を、ユーザーにとってより自然で分かりやすい言葉遣いに整形し、最終的な回答を作成してください。

            ユーザーの元の質問:
            {user_query}

            専門家からの回答:
            {expert_answer}
            ---
            最終的な回答:
            """
        )
        formatter_chain = formatter_prompt | self.formatter_llm
        final_answer = formatter_chain.invoke({
            "user_query": query,
            "expert_answer": expert_answer
        })

        retrieved_info = f"専門家ツール '{tool_name}' を使用しました。\n専門家の回答:\n{expert_answer}"
        logger.info(f"--- MicroLLM Expert Pipeline END ({(time.time() - start_time):.2f} s) ---")

        return {
            "final_answer": final_answer,
            "self_criticism": f"専門家ツール '{tool_name}' を活用して回答を生成しました",
            "potential_problems": "専門家の回答が限定的すぎる場合、整形後の回答も情報が不足する可能性があります",
            "retrieved_info": retrieved_info
        }

    def run(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        import asyncio
        return asyncio.run(self.arun(query, orchestration_decision))
