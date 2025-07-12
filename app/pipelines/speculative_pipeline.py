# /app/pipelines/speculative_pipeline.py
# title: 投機的思考パイプライン
# role: 高速なローカルモデルで思考ドラフトを生成し、高性能モデルで検証・統合する。

import logging
import time
from typing import Any, List, Dict
from concurrent.futures import ThreadPoolExecutor

from app.pipelines.base import BasePipeline
from app.models import MasterAgentResponse, OrchestrationDecision
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from app.config import settings

logger = logging.getLogger(__name__)

class SpeculativePipeline(BasePipeline):
    """
    軽量モデルで複数のドラフトを生成し、高性能モデルで検証・統合するパイプライン。
    """
    def __init__(self, drafter_llm: OllamaLLM, verifier_llm: OllamaLLM, output_parser: Any):
        self.drafter_llm = drafter_llm
        self.verifier_llm = verifier_llm
        self.output_parser = output_parser

    def _generate_draft(self, query: str, draft_number: int) -> str:
        """単一の思考ドラフトを生成する"""
        logger.info(f"思考ドラフト {draft_number} を生成中...")
        draft_prompt = ChatPromptTemplate.from_template(
            """あなたは高速にアイデアを出すブレーンストーミングAIです。以下の要求に対して、完璧でなくて良いので、とにかく思考のドラフト（下書き）を生成してください。
            
            要求: {query}
            ---
            思考ドラフト:"""
        )
        chain = draft_prompt | self.drafter_llm | self.output_parser
        return chain.invoke({"query": query})

    def run(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        """
        パイプラインを実行する。
        """
        start_time = time.time()
        logger.info("--- Speculative Pipeline START ---")

        num_drafts = settings.PIPELINE_SETTINGS["speculative"]["num_drafts"]
        drafts: List[str] = []

        with ThreadPoolExecutor(max_workers=num_drafts) as executor:
            futures = [executor.submit(self._generate_draft, query, i + 1) for i in range(num_drafts)]
            for future in futures:
                drafts.append(future.result())

        formatted_drafts = "\n\n---\n\n".join(
            [f"【ドラフト {i+1}】\n{draft}" for i, draft in enumerate(drafts)]
        )
        logger.info("全ての思考ドラフトが生成されました。")
        
        verification_prompt = ChatPromptTemplate.from_template(
            """あなたは優秀な編集者兼ファクトチェッカーです。
            以下は、複数のアシスタントが生成した思考ドラフトです。これらのドラフトをレビューし、
            最も正確で質の高い情報を抽出し、矛盾点を解消し、一つの洗練された最終回答にまとめてください。
            
            元の要求: {query}
            
            思考ドラフト集:
            {drafts}
            ---
            検証・統合された最終回答:
            """
        )
        
        verification_chain = verification_prompt | self.verifier_llm | self.output_parser
        final_answer = verification_chain.invoke({"query": query, "drafts": formatted_drafts})
        
        logger.info(f"--- Speculative Pipeline END ({(time.time() - start_time):.2f} s) ---")
        
        return {
            "final_answer": final_answer,
            "self_criticism": "投機的思考パイプラインは、高速なドラフト生成と高品質な検証を組み合わせて回答しました。",
            "potential_problems": "ドラフトの質が低い場合、最終的な回答の質も影響を受ける可能性があります。",
            "retrieved_info": formatted_drafts
        }