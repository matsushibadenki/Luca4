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
from app.agents.prompts import SIMPLE_MASTER_AGENT_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable

if TYPE_CHECKING:
    from app.rag.retriever import Retriever
    from langchain_ollama import OllamaLLM
    from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# 質問の性質を判断するためのプロンプト
ROUTING_PROMPT = ChatPromptTemplate.from_template(
    """あなたはユーザーの質問を分析し、最適な回答方法を判断するルーティングエージェントです。
以下の質問が、一般的な知識や挨拶、創造的な応答で答えられるものか、それとも特定の文書や知識ベースを参照する必要がある専門的なものかを判断してください。

- 一般的な知識や挨拶、創造的な応答で答えられる場合 -> "DIRECT"
- 魚の名前、動物、植物、あるいは特定の文書に記載されていそうな詳細情報を尋ねている場合 -> "RAG"

質問: {query}
---
判断結果をJSON形式で出力してください:
{{
    "route": "DIRECT or RAG"
}}
"""
)

# RAGを使用しない場合の直接応答プロンプト
DIRECT_RESPONSE_PROMPT = ChatPromptTemplate.from_template(
    """あなたは親切で知識豊富なAIアシスタントです。以下の質問に対して、あなたの知識を使って直接回答してください。

質問: {query}
---
回答:
"""
)

class SimplePipeline(BasePipeline):
    """
    質問の性質に応じて、直接応答とRAG（Retrieval-Augmented Generation）を動的に切り替える、
    より洗練されたシンプルな推論パイプライン。
    """
    def __init__(self, llm: 'OllamaLLM', output_parser: 'StrOutputParser', retriever: 'Retriever'):
        self.llm = llm
        self.output_parser = output_parser
        self.retriever = retriever

        # ルーティング用のチェーン
        self.router_chain = ROUTING_PROMPT | self.llm | JsonOutputParser()

        # RAGを使用する応答用のチェーン
        self.rag_chain = SIMPLE_MASTER_AGENT_PROMPT | self.llm | self.output_parser

        # RAGを使用しない直接応答用のチェーン
        self.direct_chain = DIRECT_RESPONSE_PROMPT | self.llm | self.output_parser

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
            route = routing_result["route"] if "route" in routing_result else "DIRECT"
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