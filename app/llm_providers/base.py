# /app/llm_providers/base.py
# title: LLMプロバイダー 抽象基底クラス
# role: OllamaやLlama.cppなど、すべてのLLM実行環境が従うべき共通のインターフェースを定義する。

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class LLMProvider(ABC):
    """
    LLM実行環境の抽象基底クラス。
    """

    @abstractmethod
    def get_llm_instance(self, model: str, **kwargs) -> Any:
        """
        指定されたモデル名のLLMインスタンスを取得または生成する。
        """
        pass

    @abstractmethod
    def invoke(self, model_instance: Any, prompt: str, **kwargs) -> str:
        """
        指定されたLLMインスタンスを使用して推論を実行する。
        """
        pass

    @abstractmethod
    def create_model(self, model_name: str, modelfile_path: str, **kwargs) -> bool:
        """
        指定されたModelfileから新しいモデルを作成（ファインチューニング）する。
        """
        pass

    @abstractmethod
    def list_models(self) -> Dict[str, Any]:
        """
        利用可能なローカルモデルのリストを取得する。
        """
        pass