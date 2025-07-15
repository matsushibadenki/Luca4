# /app/agents/base.py
# title: AIエージェント 抽象基底クラス
# role: すべてのAIエージェントの基本的な構造とインターフェースを定義する。

from langchain_core.runnables import Runnable
from typing import Any, Dict, Optional


class AIAgent:
    """
    AIエージェントの抽象基底クラス。
    このクラスの__init__は、サブクラスの属性がすべて設定された後に呼び出されることを想定しています。
    """
    _chain: Optional[Runnable]

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def __init__(self, *args, **kwargs) -> None:
        """
        コンストラクタ。
        サブクラスの__init__の最後に呼び出され、チェーンを構築する。
        *args, **kwargs を受け取ることで、サブクラスのコンストラクタシグネチャを柔軟にする。
        """
        self._chain = self.build_chain()
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def build_chain(self) -> Optional[Runnable]:
        """
        LangChain互換のチェーンを構築する。
        このメソッドは、必ずサブクラスでオーバーライド（上書き）されなければなりません。
        エージェントが単一のメインチェーンを持たない場合はNoneを返すことができます。
        """
        raise NotImplementedError("build_chain() must be implemented by all agent subclasses.")

    def invoke(self, input_data: Dict[str, Any] | str) -> Any:
        """
        構築されたチェーンを実行（invoke）します。
        このメソッドは、単一のチェーンを持つエージェントでのみ使用されるべきです。

        Args:
            input_data: チェーンへの入力データ。

        Returns:
            チェーンの実行結果。
        """
        if not hasattr(self, '_chain') or self._chain is None:
            raise RuntimeError(
                f"{self.__class__.__name__} is not designed to be invoked directly. "
                "It may use multiple internal chains. Call a specific method instead."
            )
        return self._chain.invoke(input_data)