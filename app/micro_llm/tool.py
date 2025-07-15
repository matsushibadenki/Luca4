# /app/micro_llm/tool.py
# title: マイクロLLMツール
# role: 動的に作成されたマイクロLLMをLangChainのツールとしてラップする。

import logging
from typing import Any
from langchain_core.runnables import Runnable

from app.tools.base import Tool
from app.llm_providers.base import LLMProvider

logger = logging.getLogger(__name__)

class MicroLLMTool(Tool):
    """
    ファインチューニングされたマイクロLLMをツールとして扱うためのクラス。
    """
    def __init__(self, model_name: str, description: str, llm_provider: LLMProvider):
        """
        Args:
            model_name (str): このツールが使用するマイクロLLMのモデル名。
            description (str): ツールの説明文。ToolUsingAgentがツール選択に利用する。
            llm_provider (LLMProvider): LLMの処理を実行するプロバイダー。
        """
        self.name = f"Specialist_{model_name.replace(':', '_').replace('/', '_')}" # ツール名として無効な文字を置換
        self.description = description
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.llm_instance = self.llm_provider.get_llm_instance(model=self.model_name)

    def use(self, query: str) -> str:
        """
        マイクロLLMツールを実行（推論）する。
        """
        logger.info(f"マイクロLLMツール '{self.name}' をクエリ '{query}' で実行します。")
        try:
            # LLMプロバイダーを通じて推論を実行
            return self.llm_provider.invoke(self.llm_instance, query)
        except Exception as e:
            logger.error(f"マイクロLLMツール '{self.name}' の実行中にエラーが発生しました: {e}", exc_info=True)
            return f"専門家ツール '{self.name}' の呼び出しに失敗しました。"