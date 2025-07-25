# /app/prompts/manager.py
# title: プロンプトマネージャー
# role: JSONファイルからプロンプトを動的に読み込み、管理、更新する。

import json
import logging
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
import threading

logger = logging.getLogger(__name__)

class PromptManager:
    """
    JSONファイルからプロンプトを管理するシングルトンクラス。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, file_path: str = "data/prompts/prompts.json"):
        if not hasattr(self, '_initialized'):
            self.file_path = file_path
            self._prompts: Dict[str, str] = self._load_prompts()
            self._initialized = True
            logger.info(f"PromptManager initialized. Loaded {len(self._prompts)} prompts from {self.file_path}")

    def _load_prompts(self) -> Dict[str, str]:
        """JSONファイルからプロンプトのテンプレート文字列を読み込む。"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load prompts from {self.file_path}: {e}")
            return {}

    def _save_prompts(self):
        """現在のプロンプトの状態をJSONファイルに保存する。"""
        with self._lock:
            try:
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    json.dump(self._prompts, f, indent=4, ensure_ascii=False)
                logger.info(f"Prompts successfully saved to {self.file_path}")
            except IOError as e:
                logger.error(f"Failed to save prompts to {self.file_path}: {e}")

    def get_prompt(self, name: str) -> ChatPromptTemplate:
        """指定された名前のプロンプトをChatPromptTemplateオブジェクトとして取得する。"""
        template_str = self._prompts.get(name)
        if template_str is None:
            logger.error(f"Prompt '{name}' not found.")
            # 存在しない場合は、エラーを示すダミーのプロンプトを返す
            return ChatPromptTemplate.from_template(f"ERROR: Prompt '{name}' not found.")
        return ChatPromptTemplate.from_template(template_str)

    def update_prompt(self, name: str, new_template_string: str) -> bool:
        """指定されたプロンプトを更新し、ファイルに保存する。"""
        if name in self._prompts:
            logger.info(f"Updating prompt '{name}'...")
            self._prompts[name] = new_template_string
            self._save_prompts()
            return True
        else:
            logger.error(f"Attempted to update non-existent prompt '{name}'.")
            return False