# /app/utils/ollama_utils.py
# title: Ollama Utilities
# role: Provides utility functions for interacting with the Ollama service.

import sys
import logging
from typing import List, Any, Sequence
import ollama

logger = logging.getLogger(__name__)

def check_ollama_models_availability(required_models: List[str]) -> bool:
    """
    ローカルのOllama環境に必要なモデルが存在するかを確認する。
    """
    missing_models: List[str] = []
    
    try:
        response_obj = ollama.list()
        
        models_list: Sequence[Any] = []
        if hasattr(response_obj, 'models'):
            models_list = response_obj['models']
        elif isinstance(response_obj, dict) and 'models' in response_obj:
            models_list = response_obj['models']

        local_models: List[str] = []
        for model_data in models_list:
            model_name = None
            if hasattr(model_data, 'model'):
                model_name = model_data.model
            elif hasattr(model_data, 'name'):
                model_name = model_data.name
            elif isinstance(model_data, dict):
                model_name = model_data.get('name') or model_data.get('model')

            if model_name:
                local_models.append(model_name)

        for model_name in required_models:
            required_base_name = model_name.split(':')[0]
            is_available = any(
                local_model.split(':')[0] == required_base_name for local_model in local_models
            )
            
            if not is_available:
                logger.error(f"モデル '{model_name}' がローカル環境に見つかりません。")
                missing_models.append(model_name)
            else:
                logger.info(f"モデル '{model_name}' は利用可能です。")

    except ollama.ResponseError as e:
        logger.error(f"Ollama APIからエラーが返されました。Ollamaが起動しているか確認してください。")
        logger.error(f"詳細: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Ollamaサーバーへの接続または処理中に予期せぬエラーが発生しました。")
        logger.error(f"詳細: {e}", exc_info=True)
        return False
            
    if missing_models:
        print("\n[エラー] アプリケーションの実行に必要なOllamaモデルが不足しています。")
        for model_name in missing_models:
            print("-" * 50)
            print(f"モデル '{model_name}' が見つかりません。")
            print("以下のコマンドをターミナルで実行して、モデルをダウンロードしてください:")
            print(f"  ollama pull {model_name}")
            print("-" * 50)
        return False

    return True