# /app/pipelines/simple_pipeline.py
# title: 可変式シンプル推論パイプライン
# role: 質問の性質を判断し、単純な直接応答とRAGベースの応答を動的に切り替える。

from __future__ import annotations
import time
import logging
from typing import Dict, Any, TYPE_CHECKING
import asyncio

from app.pipelines.base import BasePipeline
from app.models import MasterAgentResponse, OrchestrationDecision
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable

if TYPE_CHECKING:
    from app.rag.retriever import Retriever
    from langchain_ollama import OllamaLLM
    from langchain_core.output_parsers import StrOutputParser
    from app.prompts.manager import PromptManager

logger = logging.getLogger(__name__)

class SimplePipeline(BasePipeline):
    """
    質問の性質に応じて、直接応答とRAG（Retrieval-Augmented Generation）を動的に切り替える、
    より洗練されたシンプルな推論パイプライン。
    """
    def __init__(self, llm: 'OllamaLLM', output_parser: 'StrOutputParser', retriever: 'Retriever', prompt_manager: 'PromptManager'):
        self.llm = llm
        self.output_parser = output_parser
        self.retriever = retriever

        # ルーティング用のチェーン
        routing_prompt = prompt_manager.get_prompt("ROUTING_PROMPT")
        self.router_chain = routing_prompt | self.llm | JsonOutputParser()

        # RAGを使用する応答用のチェーン
        rag_prompt = prompt_manager.get_prompt("SIMPLE_MASTER_AGENT_PROMPT")
        self.rag_chain = rag_prompt | self.llm | self.output_parser

        # RAGを使用しない直接応答用のチェーン
        direct_prompt = prompt_manager.get_prompt("DIRECT_RESPONSE_PROMPT")
        self.direct_chain = direct_prompt | self.llm | self.output_parser

    async def arun(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        """
        パイプラインを非同期で実行する。
        """
        start_time = time.time()
        logger.info("--- Simple Pipeline START ---")
        
        retrieved_info = ""
        final_answer = ""

        try:
            # 1. ルーティング判断
            logger.info(f"クエリのルーティングを判断中: '{query}'")
            routing_result = await self.router_chain.ainvoke({"query": query})
            route = routing_result.get("route", "DIRECT") # .get()で安全にアクセス
            logger.info(f"ルーティング結果: '{route}'")

            # 2. ルートに応じて処理を分岐
            if route == "RAG":
                logger.info("RAGルートが選択されました。内部知識ベースを検索します。")
                docs = self.retriever.invoke(query)
                retrieved_info = "\n\n".join([doc.page_content for doc in docs])
                if not retrieved_info.strip():
                     logger.warning("RAG検索を実行しましたが、関連情報が見つかりませんでした。DIRECTルートにフォールバックします。")
                     final_answer = await self.direct_chain.ainvoke({"query": query})
                else:
                    rag_input = {"query": query, "retrieved_info": retrieved_info}
                    final_answer = await self.rag_chain.ainvoke(rag_input)
            else: # DIRECTルート
                logger.info("DIRECTルートが選択されました。LLMが直接応答します。")
                final_answer = await self.direct_chain.ainvoke({"query": query})

        except Exception as e:
            logger.error(f"SimplePipelineの実行中にエラーが発生しました: {e}", exc_info=True)
            logger.info("エラーのため、DIRECTルートにフォールバックして応答を試みます。")
            try:
                final_answer = await self.direct_chain.ainvoke({"query": query})
            except Exception as final_e:
                 logger.error(f"フォールバック処理中にもエラーが発生しました: {final_e}", exc_info=True)
                 final_answer = "申し訳ありません、ご質問の処理中にエラーが発生しました。"


        end_time = time.time()
        logger.info(f"--- Simple Pipeline END ({(end_time - start_time):.2f} s) ---")

        return {
            "final_answer": final_answer,
            "self_criticism": "シンプルモードでは自己評価は実行されません。",
            "potential_problems": "シンプルモードでは潜在的な問題の発見は実行されません。",
            "retrieved_info": retrieved_info
        }

    def run(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        """同期版のrunメソッド"""
        return asyncio.run(self.arun(query, orchestration_decision))