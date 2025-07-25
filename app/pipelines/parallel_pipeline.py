# /app/pipelines/parallel_pipeline.py
# title: 並列推論パイプライン
# role: 複数の思考プロセスを並列実行し、最も優れた回答を選択する。

from __future__ import annotations
import logging
import time
from typing import Any, List, Dict, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from app.pipelines.base import BasePipeline
from app.models import MasterAgentResponse, OrchestrationDecision
from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from app.agents.cognitive_loop_agent import CognitiveLoopAgent
    from langchain_ollama import OllamaLLM
    from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class ParallelPipeline(BasePipeline):
    """
    複数の複雑性レジームでCognitiveLoopAgentを並列実行し、最良の結果を選択するパイプライン。
    """
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def __init__(
        self,
        llm: 'OllamaLLM',
        output_parser: 'StrOutputParser',
        cognitive_loop_agent_factory: Any
    ):
        self.llm = llm
        self.output_parser = output_parser
        self.cognitive_loop_agent_factory = cognitive_loop_agent_factory
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def _run_single_loop(self, query: str, complexity: str) -> Dict[str, Any]:
        """単一の認知ループを実行する"""
        agent: 'CognitiveLoopAgent' = self.cognitive_loop_agent_factory()
        output = agent.invoke({"query": f"({complexity}の複雑度で分析) {query}", "plan": "並列分析"})
        return {"complexity": complexity, "output": output}

    def run(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        """
        パイプラインを実行する。
        """
        start_time = time.time()
        logger.info("--- Parallel Pipeline START ---")

        complexities = ["low", "medium", "high"]
        results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=len(complexities)) as executor:
            futures = [executor.submit(self._run_single_loop, query, comp) for comp in complexities]
            for future in futures:
                results.append(future.result())

        formatted_results = "\n\n---\n\n".join(
            [f"【{res['complexity']}複雑度での分析結果】\n{res['output']}" for res in results]
        )
        
        selection_prompt = ChatPromptTemplate.from_template(
            """あなたは複数の分析結果を統合し、最も優れた回答を選択する編集長です。
            以下の異なる視点からの分析結果を読み、ユーザーの元の要求に対して最も包括的で質の高い最終回答を1つだけ生成してください。

            元の要求: {query}

            分析結果リスト:
            {results}
            ---
            統合・選択された最終回答:
            """
        )
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        selection_chain = selection_prompt | self.llm | self.output_parser
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        final_answer = selection_chain.invoke({"query": query, "results": formatted_results})
        
        logger.info(f"--- Parallel Pipeline END ({(time.time() - start_time):.2f} s) ---")
        
        return {
            "final_answer": final_answer,
            "self_criticism": "並列パイプラインは複数の視点から回答を生成しました。",
            "potential_problems": "各分析の視点が異なるため、統合時にニュアンスが失われる可能性があります。",
            "retrieved_info": formatted_results
        }