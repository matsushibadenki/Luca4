# /app/memory/working_memory.py
# title: 海馬的ワーキングメモリ
# role: 現在のセッションにおける新規性の高い情報（予測誤差）を保持する短期記憶領域。

from typing import List, Dict, Any
import uuid

class WorkingMemory:
    """
    海馬のように、現在の対話の文脈で新規性が高い情報を一時的に保持するクラス。
    セッションごとに一意のIDが割り振られます。
    """
    def __init__(self) -> None:
        self.session_id: str = str(uuid.uuid4())
        self.prediction_errors: List[Dict[str, Any]] = []
        self.context_summary: str = ""

    def add_prediction_error(self, error_data: Dict[str, Any]) -> None:
        """
        予測フィルターによって検出された予測誤差（新規情報）を追加します。
        """
        self.prediction_errors.append(error_data)

    def get_contents(self) -> Dict[str, Any]:
        """
        現在のワーキングメモリの内容を返します。
        """
        return {
            "session_id": self.session_id,
            "prediction_errors": self.prediction_errors,
            "context_summary": self.context_summary,
        }

    def clear(self) -> None:
        """
        ワーキングメモリの内容をリセットし、新しいセッションを開始します。
        """
        self.session_id = str(uuid.uuid4())
        self.prediction_errors = []
        self.context_summary = ""