# /app/agents/thinking_modules.py
# title: 原子レベル思考モジュール群
# role: Self-Discover Pipelineが動的に組み合わせるための、基本的な思考スキルを個別のエージェントとして提供する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Any
from .base import AIAgent

# --- プロンプトの定義 ---

DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """あなたは問題を小さなステップに分解する専門家です。以下の複雑な要求を、より単純なサブタスクのリストに分解してください。
    
    複雑な要求: {query}
    ---
    分解されたサブタスクリスト:"""
)

CRITIQUE_PROMPT = ChatPromptTemplate.from_template(
    """あなたは鋭い批評家です。以下の提案やアイデアに含まれる、弱点、欠点、または見落とされている点を指摘してください。
    
    評価対象の提案: {draft}
    ---
    批判的な評価:"""
)

SYNTHESIZE_PROMPT = ChatPromptTemplate.from_template(
    """あなたは多様な情報を統合する専門家です。以下の複数の情報や視点を組み合わせ、一貫性のある包括的な要約または結論を生成してください。
    
    統合対象の情報:
    {information_list}
    ---
    統合された要約/結論:"""
)


# --- エージェントクラスの定義 ---

class DecomposeAgent(AIAgent):
    def __init__(self, llm: Any, output_parser: Any):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = DECOMPOSE_PROMPT
        super().__init__()
    def build_chain(self) -> Runnable:
        return self.prompt_template | self.llm | self.output_parser

class CritiqueAgent(AIAgent):
    def __init__(self, llm: Any, output_parser: Any):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = CRITIQUE_PROMPT
        super().__init__()
    def build_chain(self) -> Runnable:
        return self.prompt_template | self.llm | self.output_parser

class SynthesizeAgent(AIAgent):
    def __init__(self, llm: Any, output_parser: Any):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = SYNTHESIZE_PROMPT
        super().__init__()
    def build_chain(self) -> Runnable:
        return self.prompt_template | self.llm | self.output_parser