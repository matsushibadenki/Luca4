# /app/llm_providers/ollama_provider.py
# title: Ollama LLMプロバイダー
# role: LLMProviderインターフェースに基づき、Ollama特有の処理を実装する。

import logging
import subprocess
import json
from typing import Any, Dict, Optional
from langchain_ollama import OllamaLLM
from ollama import Client

from app.llm_providers.base import LLMProvider

logger = logging.getLogger(__name__)

class OllamaProvider(LLMProvider):
    """
    OllamaをLLM実行環境として利用するためのプロバイダークラス。
    """
    def __init__(self, host: Optional[str] = None):
        self.client = Client(host=host) if host else Client()

    def get_llm_instance(self, model: str, **kwargs) -> Any:
        """
        Ollama用のLangChain LLMインスタンスを返す。
        """
        return OllamaLLM(model=model, **kwargs)

    def invoke(self, model_instance: Any, prompt: str, **kwargs) -> str:
        """
        LangChainのOllamaLLMインスタンスを使って推論を実行する。
        """
        if not isinstance(model_instance, OllamaLLM):
            raise TypeError("model_instance must be an instance of OllamaLLM")
        return model_instance.invoke(prompt, **kwargs)

    def create_model(self, model_name: str, modelfile_path: str, **kwargs) -> bool:
        """
        'ollama create'コマンドを実行してモデルを作成する。
        """
        logger.info(f"Ollamaモデル '{model_name}' の作成を開始します (Modelfile: {modelfile_path})...")
        try:
            # ollama createコマンドを実行
            process = subprocess.run(
                ["ollama", "create", model_name, "-f", modelfile_path],
                capture_output=True, text=True, check=True, encoding='utf-8'
            )
            logger.info(f"Ollamaモデル '{model_name}' の作成に成功しました。\n{process.stdout}")
            return True
        except FileNotFoundError:
            logger.error("`ollama` コマンドが見つかりません。Ollamaがインストールされ、PATHが通っていることを確認してください。")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Ollamaモデル '{model_name}' の作成に失敗しました。\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Ollamaモデル作成中に予期せぬエラーが発生しました: {e}")
            return False

    def list_models(self) -> Dict[str, Any]:
        """
        'ollama list'コマンドの結果を返す。
        """
        try:
            # self.client.list()はListResponse (TypedDict)を返す。
            # mypyの型チェックエラーを解決するため、明示的にdictに変換する。
            response = self.client.list()
            return dict(response)
        except Exception as e:
            logger.error(f"Ollamaモデルリストの取得中にエラーが発生しました: {e}")
            return {"models": []}
