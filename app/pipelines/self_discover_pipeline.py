# /app/pipelines/self_discover_pipeline.py
# title: 自己発見パイプライン
# role: 問題の性質に応じて思考モジュールを動的に組み合わせ、解決戦略を自律的に構築する。

import logging
import time
from typing import Any, Dict, List

from app.pipelines.base import BasePipeline
from app.agents.planning_agent import PlanningAgent
from app.agents.thinking_modules import DecomposeAgent, CritiqueAgent, SynthesizeAgent
from app.agents.cognitive_loop_agent import CognitiveLoopAgent
from app.models import MasterAgentResponse
from app.models import OrchestrationDecision # ADDED

logger = logging.getLogger(__name__)

class SelfDiscoverPipeline(BasePipeline):
    """
    思考モジュールを動的に組み合わせて問題解決を行うパイプライン。
    """
    def __init__(
        self,
        planning_agent: PlanningAgent,
        decompose_agent: DecomposeAgent,
        critique_agent: CritiqueAgent,
        synthesize_agent: SynthesizeAgent,
        cognitive_loop_agent: CognitiveLoopAgent,
    ):
        self.planning_agent = planning_agent
        self.thinking_modules = {
            "DECOMPOSE": decompose_agent,
            "CRITIQUE": critique_agent,
            "SYNTHESIZE": synthesize_agent,
            "RAG_SEARCH": cognitive_loop_agent, # RAG検索はCognitiveLoopAgentが担当
        }

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def run(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        """
        パイプラインを実行する。
        """
        start_time = time.time()
        logger.info("--- Self-Discover Pipeline START ---")

        # 1. 思考戦略の選択
        strategy_sequence_str = self.planning_agent.select_thinking_modules(query)
        strategy_sequence = [s.strip() for s in strategy_sequence_str.split(',')]
        logger.info(f"選択された思考戦略シーケンス: {strategy_sequence}")

        # 2. 戦略の実行
        execution_context: Dict[str, Any] = {"query": query}
        execution_trace: List[str] = []

        for module_name in strategy_sequence:
            if module_name not in self.thinking_modules:
                logger.warning(f"未知の思考モジュール '{module_name}' はスキップされました。")
                continue

            agent = self.thinking_modules[module_name]
            
            # 入力を準備
            input_data: Dict[str, Any] | str
            if module_name == "DECOMPOSE":
                input_data = {"query": execution_context["query"]}
            elif module_name == "CRITIQUE":
                input_data = {"draft": execution_context.get("last_output", "")}
            elif module_name == "SYNTHESIZE":
                # これまでの出力をリストとして渡す
                info_list = "\n---\n".join(execution_trace)
                input_data = {"information_list": info_list}
            elif module_name == "RAG_SEARCH":
                 input_data = {"query": execution_context["query"], "plan": "関連情報の検索"}
            else:
                input_data = execution_context["query"]

            logger.info(f"実行中モジュール: {module_name}, 入力: {input_data}")
            output = agent.invoke(input_data)
            
            execution_context["last_output"] = output
            trace_entry = f"【{module_name}の出力】\n{output}"
            execution_trace.append(trace_entry)
            logger.info(trace_entry)

        final_answer = execution_context.get("last_output", "処理が完了しましたが、明確な最終出力はありません。")
        retrieved_info = "\n\n".join(execution_trace)

        logger.info(f"--- Self-Discover Pipeline END ({(time.time() - start_time):.2f} s) ---")
        
        return {
            "final_answer": final_answer,
            "self_criticism": f"自己発見パイプラインは、[{', '.join(strategy_sequence)}]という戦略で回答を導出しました。",
            "potential_problems": "選択された戦略が最適でない場合、非効率な思考プロセスになる可能性があります。",
            "retrieved_info": retrieved_info
        }