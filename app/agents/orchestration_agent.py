# /app/agents/orchestration_agent.py
# title: オーケストレーションエージェント
# role: ユーザーの要求の複雑さやAIの感情状態に応じて、最適な思考パイプライン（実行モード）を選択する。

import logging
import re
from typing import Dict, Any, TYPE_CHECKING, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser

from app.agents.base import AIAgent
from app.models import OrchestrationDecision
from app.reasoning.complexity_analyzer import ComplexityAnalyzer
from app.llm_providers.base import LLMProvider
from app.affective_system.affective_state import AffectiveState

if TYPE_CHECKING:
    from app.tools.tool_belt import ToolBelt

logger = logging.getLogger(__name__)

class OrchestrationAgent(AIAgent):
    """
    ユーザーの要求を分析し、最適な実行モードを決定するエージェント。
    """
    _chain: Runnable

    def __init__(
        self,
        llm_provider: LLMProvider,
        output_parser: JsonOutputParser,
        prompt_template: ChatPromptTemplate,
        complexity_analyzer: ComplexityAnalyzer,
        tool_belt: "ToolBelt",
    ):
        # llmインスタンスはプロバイダー経由で取得
        self.llm = llm_provider.get_llm_instance(model="gemma3:latest")
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        self.complexity_analyzer = complexity_analyzer
        self.tool_belt = tool_belt
        super().__init__()

    def build_chain(self) -> Runnable:
        """このエージェント専用のチェーンを構築する。"""
        return self.prompt_template | self.llm | self.output_parser

    def _determine_reasoning_emphasis(self, query: str) -> Optional[str]:
        """クエリに基づいて推論の強調を決定する。"""
        query_lower = query.lower()
        bird_keywords = ["全体像", "戦略", "将来", "哲学", "概要", "大局", "ビジョン", "抽象"]
        detail_keywords = ["具体例", "詳細", "手順", "データ", "正確な", "特定", "実装", "技術"]

        bird_score = sum(1 for kw in bird_keywords if kw in query_lower)
        detail_score = sum(1 for kw in detail_keywords if kw in query_lower)

        if bird_score > detail_score and bird_score > 0:
            return "bird's_eye_view"
        elif detail_score > bird_score and detail_score > 0:
            return "detail_oriented"
        else:
            return None

    def invoke(self, input_data: Dict[str, Any] | str) -> OrchestrationDecision:
        """
        要求を処理し、オーケストレーションの決定を返す。
        """
        if not isinstance(input_data, dict) or "query" not in input_data:
            raise TypeError("OrchestrationAgent expects a dictionary with a 'query' key.")

        query = input_data["query"]
        affective_state_val = input_data.get("affective_state")
        affective_state: Optional[AffectiveState] = affective_state_val if isinstance(affective_state_val, AffectiveState) else None
        affective_state_summary = f"{affective_state.emotion.value} (強度: {affective_state.intensity})" if affective_state else "不明"

        # 推論の強調を決定
        reasoning_emphasis = self._determine_reasoning_emphasis(query)
        logger.info(f"推論の強調: {reasoning_emphasis}")

        # 1. URLが含まれているかチェック (優先度高)
        url_pattern = re.compile(r'https?://\S+')
        if url_pattern.search(query):
            logger.info("URLが検出されたため、強制的に 'full' モードを選択します。")
            return {
                "chosen_mode": "full",
                "reason": "URLが含まれているため、Webブラウジング機能を持つfullパイプラインが選択されました",
                "agent_configs": {},
                "reasoning_emphasis": reasoning_emphasis # ここにも含める
            }

        # 2. 専門家ツール（マイクロLLM）が利用可能かチェック
        tool_descriptions = self.tool_belt.get_tool_descriptions()
        # 専門家ツールが存在する場合のみ、LLMに選択を問い合わせる
        if "Specialist_" in tool_descriptions:
            expert_check_prompt = ChatPromptTemplate.from_template(
                """あなたはタスクを専門家に割り振るのが得意なマネージャーです。
                以下の「ユーザーの要求」が、提供されている「専門家ツールリスト」のいずれかの専門分野に合致するかどうかを判断してください。
                合致する場合はそのツール名を、合致しない場合は「none」とだけ答えてください。

                専門家ツールリスト:
                {tools}

                ユーザーの要求: {query}
                ---
                判断結果（ツール名またはnone）:
                """
            )
            expert_check_chain = expert_check_prompt | self.llm
            tool_decision = expert_check_chain.invoke({
                "tools": tool_descriptions,
                "query": query
            }).strip()

            if tool_decision != "none" and tool_decision.startswith("Specialist_"):
                 logger.info(f"専門家ツール '{tool_decision}' が適していると判断されました。'micro_llm_expert' モードを選択します。")
                 return {
                    "chosen_mode": "micro_llm_expert",
                    "reason": f"要求が専門分野に合致し、対応するツール '{tool_decision}' が存在するため",
                    "agent_configs": {},
                    "reasoning_emphasis": reasoning_emphasis # ここにも含める
                 }

        # 3. 従来の複雑度分析に基づくモード選択
        complexity_result = self.complexity_analyzer.analyze(query)
        complexity_level = complexity_result.get("complexity_level", "Level 2")

        agent_input = {
            "query": query,
            "complexity_level": complexity_level,
            "affective_state": affective_state_summary
        }

        try:
            logger.info(f"Invoking orchestration chain with input: {agent_input}")
            if self._chain is None:
                 raise RuntimeError("OrchestrationAgent's chain is not initialized.")
            decision: OrchestrationDecision = self._chain.invoke(agent_input)
            # 推論の強調を決定に追加
            decision["reasoning_emphasis"] = reasoning_emphasis
            logger.info(f"Orchestration decision: {decision}")
            return decision
        except Exception as e:
            logger.error(f"Error invoking orchestration chain: {e}", exc_info=True)
            return {
                "chosen_mode": "full",
                "reason": "An error occurred during orchestration, falling back to full mode.",
                "agent_configs": {},
                "reasoning_emphasis": reasoning_emphasis # ここにも含める
            }