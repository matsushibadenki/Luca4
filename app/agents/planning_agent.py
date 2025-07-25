# /app/agents/planning_agent.py
# title: プランニングAIエージェント
# role: ユーザーの要求を分析し、実行可能な行動計画や思考モジュールの組み合わせを立案する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Any

from app.agents.base import AIAgent

class PlanningAgent(AIAgent):
    """
    ユーザーの要求から行動計画や思考戦略を生成するAIエージェント。
    """
    def __init__(self, llm: Any, output_parser: Any, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        プランニングエージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser
        
    def select_thinking_modules(self, query: str) -> str:
        """Self-Discover Pipelineのために、使用する思考モジュールのシーケンスを決定する"""
        module_selection_prompt = ChatPromptTemplate.from_template(
            """あなたは思考戦略家です。与えられた要求を解決するために、以下の思考モジュールの中から最も効果的なものを、適切な順番でカンマ区切りでリストアップしてください。
            
            利用可能なモジュール:
            - DECOMPOSE: 複雑な問題を単純なサブタスクに分解する。
            - CRITIQUE: 提案の弱点や欠点を指摘する。
            - SYNTHESIZE: 複数の情報を統合して要約や結論を出す。
            - RAG_SEARCH: 知識ベースから関連情報を検索する。
            
            要求: {query}
            ---
            思考モジュールシーケンス (例: DECOMPOSE, RAG_SEARCH, SYNTHESIZE):
            """
        )
        chain = module_selection_prompt | self.llm | self.output_parser
        return chain.invoke({"query": query})