# /app/utils/api_key_checker.py
# title: API Key Checker
# role: Checks for the presence of necessary API keys.

import os
import logging

logger = logging.getLogger(__name__)

def check_search_api_key() -> bool:
    """
    Web検索機能に必要な環境変数が設定されているかを確認する。
    """
    # このプロジェクトではTavily Searchを使用するため、TAVILY_API_KEYをチェックします。
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("-" * 60)
        logger.warning("警告: 環境変数 'TAVILY_API_KEY' が設定されていません。")
        logger.warning("Web検索機能は無効化された状態で起動します。")
        logger.warning("Web検索を利用したい場合は、READMEに従ってAPIキーを設定してください。")
        logger.warning("-" * 60)
        return False
    logger.info("Web検索用のAPIキーが設定されています。")
    return True