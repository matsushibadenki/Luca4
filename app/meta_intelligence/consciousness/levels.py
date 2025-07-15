# /app/meta_intelligence/consciousness/levels.py
# title: 意識レベル定義
# role: システムの意識状態を定義する列挙型。

from enum import Enum

class ConsciousnessLevel(Enum):
    """
    システムの統合的な意識レベルを表す列挙型。
    """
    UNCONSCIOUS = "unconscious"
    SUBCONSCIOUS = "subconscious"
    CONSCIOUS = "conscious"
    META_CONSCIOUS = "meta-conscious"
    TRANSCENDENT = "transcendent"