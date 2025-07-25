# /app/reasoning/thought.py
# title: 思考ノードデータ構造
# role: Tree of Thoughtsにおける個々の思考ステップを表現するデータクラス。

from __future__ import annotations
from typing import List, Optional
import uuid

class Thought:
    """
    思考の木における単一のノードを表すクラス。
    """
    def __init__(self, state: str, parent: Optional[Thought] = None, evaluation_score: float = 0.0):
        self.id: str = str(uuid.uuid4())
        self.state: str = state  # 思考の内容（テキスト）
        self.parent: Optional[Thought] = parent
        self.children: List[Thought] = []
        self.evaluation_score: float = evaluation_score  # この思考の有望性を示すスコア

    def add_child(self, state: str, evaluation_score: float = 0.0) -> Thought:
        """
        この思考に新しい子ノードを追加する。
        """
        child_thought = Thought(state, parent=self, evaluation_score=evaluation_score)
        self.children.append(child_thought)
        return child_thought

    def __repr__(self) -> str:
        return f"Thought(id={self.id}, state='{self.state[:30]}...', score={self.evaluation_score:.2f}, children={len(self.children)})"