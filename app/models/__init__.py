# /app/models/__init__.py
# title: アプリケーションデータモデル
# role: アプリケーション全体で使用されるデータ構造（TypedDictなど）を定義する。

from typing import TypedDict, Dict, Any

class MasterAgentResponse(TypedDict):
    """
    マスターエージェントの最終的な応答形式を定義する型。
    """
    final_answer: str
    self_criticism: str
    potential_problems: str
    retrieved_info: str

class OrchestrationDecision(TypedDict):
    """
    OrchestrationAgentによって決定される、実行モードと関連する設定。
    """
    chosen_mode: str
    reason: str
    agent_configs: Dict[str, Dict[str, Any]]
    reasoning_emphasis: str | None # 追加

class OrchestrationInput(TypedDict):
    """
    OrchestrationAgentのinvokeメソッドに渡される入力の型。
    """
    query: str