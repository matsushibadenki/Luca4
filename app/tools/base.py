# /app/tools/base.py
# title: ツール基底クラス
# role: アプリケーション内で使用されるすべてのツールの基本的なインターフェースを定義する。

from abc import ABC, abstractmethod
from typing import Any

class Tool(ABC):
    """
    すべてのツールが継承する抽象基底クラス。
    """
    name: str
    description: str

    @abstractmethod
    def use(self, query: str) -> Any:
        """
        ツールを実行するメソッド。
        """
        pass