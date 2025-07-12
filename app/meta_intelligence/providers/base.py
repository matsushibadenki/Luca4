# /app/meta_intelligence/providers/base.py
# title: LLMプロバイダー基底クラス
# role: すべてのLLMプロバイダーが従うべき基本的なインターフェースを定義する。

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Type

class ProviderCapability(Enum):
    """LLMプロバイダーの能力を定義する列挙型。"""
    STANDARD_CALL = "standard_call"
    STREAMING = "streaming"
    TOOLS = "tools"
    FUNCTION_CALLING = "function_calling"
    ENHANCED_REASONING = "enhanced_reasoning"

class LLMProvider(ABC):
    """すべてのLLMプロバイダーの抽象基底クラス。"""

    @abstractmethod
    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        """
        プロバイダーの能力を辞書形式で返す。
        """
        pass

    async def call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        能力とロジックに基づいて enhanced_call または standard_call を呼び出すメインのエントリメソッド。
        """
        if self.should_use_enhancement(prompt, **kwargs):
            # EnhancedLLMProviderで実装されることを期待
            if hasattr(self, 'enhanced_call') and callable(self.enhanced_call):
                return await self.enhanced_call(prompt, system_prompt, **kwargs)
        return await self.standard_call(prompt, system_prompt, **kwargs)

    @abstractmethod
    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        標準的な（非拡張）LLM呼び出しを実行する。
        """
        pass

    @abstractmethod
    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        """
        与えられたプロンプトとkwargsに対して拡張機能を使用すべきかどうかを判断する。
        """
        pass

class EnhancedLLMProvider(LLMProvider):
    """MetaIntelligenceの能力を提供するための標準プロバイダーのラッパークラス。"""

    def __init__(self, standard_provider: LLMProvider):
        """
        拡張プロバイダーを初期化する。
        """
        if not isinstance(standard_provider, LLMProvider):
            raise TypeError("standard_providerはLLMProviderのインスタンスである必要があります。")
        self.standard_provider = standard_provider

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        """ラッパーされたプロバイダーの能力を返し、拡張能力を追加する。"""
        capabilities = self.standard_provider.get_capabilities()
        capabilities[ProviderCapability.ENHANCED_REASONING] = True
        return capabilities
    
    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """標準呼び出しをラップされたプロバイダーに委任する。"""
        return await self.standard_provider.standard_call(prompt, system_prompt, **kwargs)

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        """拡張機能を使用すべきかどうかのロジックをラップされたプロバイダーに委任する。"""
        return self.standard_provider.should_use_enhancement(prompt, **kwargs)

    @abstractmethod
    async def enhanced_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        拡張呼び出しのための共通ロジック。通常、MetaIntelligenceの推論コアを編成する。
        """
        pass