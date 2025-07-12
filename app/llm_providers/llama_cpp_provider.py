# /app/llm_providers/llama_cpp_provider.py
# title: Llama.cpp LLMプロバイダー
# role: LLMProviderインターフェースに基づき、llama.cpp特有の処理を実装する。

import logging
from typing import Any, Dict, Optional, List

from langchain_community.llms import LlamaCpp
from app.llm_providers.base import LLMProvider
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from app.meta_intelligence.providers.base import ProviderCapability
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

logger = logging.getLogger(__name__)

class LlamaCppProvider(LLMProvider):
    """
    llama.cppをLLM実行環境として利用するためのプロバイダークラス。
    """
    def __init__(self, model_path: str, n_ctx: int = 2048, n_batch: int = 512, **kwargs):
        """
        Args:
            model_path (str): ロードするGGUFモデルのフルパス。
            n_ctx (int): コンテキストの最大長。
            n_batch (int): バッチサイズ。
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.client_kwargs = kwargs
        logger.info(f"LlamaCppProvider initialized with model: {self.model_path}")

    def get_llm_instance(self, model: str, **kwargs) -> Any:
        """
        指定されたモデル名のLLMインスタンスを取得または生成する。
        llama.cppの場合、model引数は使用せず、初期化時のmodel_pathを使用する。
        """
        if not self.model_path:
            raise ValueError("LlamaCppProviderにはモデルのパスが指定されている必要があります。")
        
        # kwargsをself.client_kwargsにマージし、個別の呼び出しでオーバーライドできるようにする
        instance_kwargs = {**self.client_kwargs, **kwargs}
        
        return LlamaCpp(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            verbose=False, # LangChainのLlamaCppはデフォルトで詳細ログを出すため、通常はFalseに設定
            **instance_kwargs
        )

    def invoke(self, model_instance: Any, prompt: str, **kwargs) -> str:
        """
        指定されたLLMインスタンスを使用して推論を実行する。
        """
        if not isinstance(model_instance, LlamaCpp):
            raise TypeError("model_instance must be an instance of LlamaCpp")
        
        return model_instance.invoke(prompt, **kwargs)

    def create_model(self, model_name: str, modelfile_path: str, **kwargs) -> bool:
        """
        llama.cppはモデルの作成（ファインチューニング）機能を直接提供しません。
        したがって、このメソッドは常にFalseを返します。
        """
        logger.warning(f"LlamaCppProviderはモデルの作成を直接サポートしていません。'{model_name}'の作成リクエストは無視されます。")
        return False

    def list_models(self) -> Dict[str, Any]:
        """
        llama.cppはインストールされているモデルを一覧表示するAPIを持ちません。
        したがって、このメソッドは空のリストを返します。
        """
        logger.warning("LlamaCppProviderは利用可能なモデルの一覧表示を直接サポートしていません。")
        return {"models": []}

    def get_capabilities(self) -> Dict[Any, bool]:
        """LLMプロバイダーの能力を定義する。"""
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        return {
            ProviderCapability.STANDARD_CALL: True,
            ProviderCapability.STREAMING: False, # langchain-communityのLlamaCppはStreamingに対応していない場合がある
            ProviderCapability.TOOLS: False,
            ProviderCapability.FUNCTION_CALLING: False,
            ProviderCapability.ENHANCED_REASONING: False,
        }
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️