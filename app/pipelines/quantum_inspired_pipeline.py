# /app/pipelines/quantum_inspired_pipeline.py
# title: 量子インスパイアード推論パイプライン
# role: 複数のペルソナの視点から並列で仮説を生成し、一つの包括的な回答に統合する。

from __future__ import annotations
import logging
import time
from typing import Any, List, Dict, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from app.pipelines.base import BasePipeline
from app.models import MasterAgentResponse, OrchestrationDecision
from app.config import settings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

if TYPE_CHECKING:
    from app.integrated_information_processing.integrated_information_agent import IntegratedInformationAgent
    from langchain_ollama import OllamaLLM
    from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class QuantumInspiredPipeline(BasePipeline):
    """
    多様なペルソナの視点から並列で思考し、結果を統合するパイプライン。
    """
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def __init__(
        self,
        llm: 'OllamaLLM',
        output_parser: 'StrOutputParser',
        integrated_information_agent: 'IntegratedInformationAgent'
    ):
        self.llm = llm
        self.output_parser = output_parser
        self.integrated_information_agent = integrated_information_agent
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def _run_persona_thought(self, query: str, persona_data: Dict[str, str]) -> Dict[str, Any]:
        """単一のペルソナで思考を実行する"""
        persona_prompt = ChatPromptTemplate.from_template(
            """{persona}
            あなたは上記のペルソナになりきり、以下の要求に対して回答を生成してください。
            
            要求: {query}
            ---
            ペルソナとしての回答:
            """
        )
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        chain: Runnable = persona_prompt | self.llm | self.output_parser
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        output = chain.invoke({"query": query, "persona": persona_data["persona"]})
        return {"name": persona_data["name"], "output": output}

    def run(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        """
        パイプラインを実行する。
        """
        start_time = time.time()
        logger.info("--- Quantum-Inspired Pipeline START ---")

        personas = settings.QUANTUM_PERSONAS if hasattr(settings, 'QUANTUM_PERSONAS') else []
        results: List[Dict[str, Any]] = []

        if not personas:
            logger.warning("量子インスパイアードパイプライン用のペルソナが設定されていません。")
            return {
                "final_answer": "多様な視点での検討ができませんでした。ペルソナが設定されていません。",
                "self_criticism": "ペルソナが設定されていなかったため、パイプラインを実行できませんでした。",
                "potential_problems": "設定ファイル(config.py)のQUANTUM_PERSONASが空または存在しない可能性があります。",
                "retrieved_info": ""
            }

        with ThreadPoolExecutor(max_workers=len(personas)) as executor:
            futures = [executor.submit(self._run_persona_thought, query, p) for p in personas]
            for future in futures:
                results.append(future.result())

        formatted_results = "\n\n---\n\n".join(
            [f"【{res['name']}の視点】\n{res['output']}" for res in results]
        )
        
        synthesis_input = {
            "query": query,
            "persona_outputs": formatted_results
        }
        final_answer = self.integrated_information_agent.invoke(synthesis_input)
        
        logger.info(f"--- Quantum-Inspired Pipeline END ({(time.time() - start_time):.2f} s) ---")
        
        return {
            "final_answer": final_answer,
            "self_criticism": "量子インスパイアードパイプラインは、多様なペルソナの視点を統合して回答を生成しました。",
            "potential_problems": "ペルソナ間の意見の対立が激しい場合、最終的な回答が中立的になりすぎる可能性があります。",
            "retrieved_info": formatted_results
        }