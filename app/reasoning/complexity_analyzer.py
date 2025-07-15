# /app/reasoning/complexity_analyzer.py
# title: クエリ複雑度分析器
# role: ユーザーのクエリの複雑さを分析し、適切な思考パイプラインを選択するための指標を提供する。

import logging
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaLLM
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

logger = logging.getLogger(__name__)

COMPLEXITY_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """ユーザーの要求を分析し、その複雑度を評価してください。
以下のカテゴリから最も適切なものを一つ選択し、その理由とともにJSON形式で出力してください。

カテゴリ:
- **Level 1 (Simple)**: 簡単な挨拶、自己紹介、単純な事実確認など、外部情報なしで即答できる質問。
- **Level 2 (Moderate)**: Web検索やデータベース検索など、単一のツールを使用して回答できる質問。長所・短所の比較など、ある程度の推論が必要。
- **Level 3 (Complex)**: 複数の情報源からの情報を組み合わせ、深い分析や複数ステップの推論を必要とする質問。創造的な提案や詳細な計画立案など。
- **Level 4 (Highly Complex)**: 複数の専門分野にまたがる知識を統合し、内省的な対話や自己発見的なプロセスを通じて、新しい洞察を生み出す必要がある哲学的・抽象的な問い。

ユーザーの要求:
{query}
---
評価結果 (JSON):
{{
    "complexity_level": "Level X",
    "reason": "このレベルと判断した理由"
}}
"""
)

class ComplexityAnalyzer:
    """
    LLMを使用してユーザーのクエリの複雑さを分析するクラス。
    """
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def __init__(self, llm: OllamaLLM):
        """
        コンストラクタ。依存性は外部から注入される。
        Args:
            llm: 使用するLLMインスタンス。
        """
        self.llm = llm
        self.parser = JsonOutputParser()
        self.chain = COMPLEXITY_ANALYSIS_PROMPT | self.llm | self.parser
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def analyze(self, query: str) -> dict:
        """
        クエリの複雑さを分析し、レベルと理由を返す。

        Args:
            query: ユーザーからのクエリ文字列。

        Returns:
            複雑度レベルと理由を含む辞書。
            例: {"complexity_level": "Level 2", "reason": "..."}
        """
        try:
            logger.info(f"Analyzing complexity for query: '{query}'")
            response = self.chain.invoke({"query": query})
            logger.info(f"Complexity analysis result: {response}")
            if "complexity_level" not in response:
                logger.warning("Complexity analysis did not return 'complexity_level'. Defaulting to Level 2.")
                return {"complexity_level": "Level 2", "reason": "Default due to parsing error."}
            return response
        except Exception as e:
            logger.error(f"Error during complexity analysis: {e}", exc_info=True)
            return {"complexity_level": "Level 2", "reason": "Default due to an exception."}