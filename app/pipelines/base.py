# /app/pipelines/base.py
# title: パイプライン基底クラス
# role: すべての推論パイプラインが従うべき基本的なインターフェースを定義する。

from abc import ABC, abstractmethod
from typing import Dict, Any
from app.models import MasterAgentResponse
from app.models import OrchestrationDecision

class BasePipeline(ABC):
    """
    すべての推論パイプラインの抽象基底クラス。
    """
    @abstractmethod
    def run(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        """
        パイプラインを実行するメソッド。（同期版）
        """
        pass

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    async def arun(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        """
        パイプラインを非同期で実行するメソッド。
        デフォルトでは同期版を呼び出すが、非同期処理が必要なパイプラインはこれをオーバーライドする。
        """
        return self.run(query, orchestration_decision)
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️