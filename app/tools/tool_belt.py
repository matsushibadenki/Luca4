# /app/tools/tool_belt.py
# title: ツールベルト
# role: システムで利用可能なすべてのツールを保持し、名前で呼び出す機能を提供する。

import os
import logging
from typing import List, Dict, Optional

from app.tools.base import Tool
from .tavily_search_tool import TavilySearchTool
from .wikipedia_search_tool import WikipediaSearchTool
from .playwright_browser_tool import PlaywrightBrowserTool
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from app.micro_llm.manager import MicroLLMManager
from app.micro_llm.tool import MicroLLMTool
from app.llm_providers.base import LLMProvider
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

logger = logging.getLogger(__name__)

class ToolBelt:
    """
    利用可能なツールのコレクションを管理するクラス。
    マイクロLLMツールを動的にロードする機能を持つ。
    """
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def __init__(self, llm_provider: LLMProvider, micro_llm_manager: MicroLLMManager):
        """
        Args:
            llm_provider (LLMProvider): LLM処理を実行するためのプロバイダー。
            micro_llm_manager (MicroLLMManager): マイクロLLMを管理するマネージャー。
        """
        self._tools: List[Tool] = [
            WikipediaSearchTool(),
            PlaywrightBrowserTool(),
        ]
        if 'TAVILY_API_KEY' in os.environ and os.environ.get('TAVILY_API_KEY'):
            self._tools.append(TavilySearchTool())

        self._tool_map: Dict[str, Tool] = {tool.name: tool for tool in self._tools}

        # マイクロLLMツールを動的にロード
        self._load_micro_llm_tools(llm_provider, micro_llm_manager)

    def _load_micro_llm_tools(self, llm_provider: LLMProvider, micro_llm_manager: MicroLLMManager):
        """利用可能なマイクロLLMをスキャンし、ツールとして登録する。"""
        logger.info("専門家マイクロLLMツールをロードしています...")
        specialized_models = micro_llm_manager.get_specialized_models()
        for model_info in specialized_models:
            model_name = model_info["name"]
            topic = model_info["topic"]
            description = f"「{topic}」に関する非常に詳細な質問に回答するための専門家ツール。"
            tool_instance = MicroLLMTool(
                model_name=model_name,
                description=description,
                llm_provider=llm_provider
            )
            if tool_instance.name not in self._tool_map:
                self._tools.append(tool_instance)
                self._tool_map[tool_instance.name] = tool_instance
                logger.info(f"専門家ツール '{tool_instance.name}' が正常にロードされました。")
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        指定された名前のツールを取得する。
        """
        return self._tool_map.get(tool_name)

    def get_tool_descriptions(self) -> str:
        """
        すべてのツールの名前と説明をフォーマットされた文字列として取得する。
        """
        return "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in self._tools]
        )