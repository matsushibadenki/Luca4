# /app/affective_system/affective_state.py
# title: AI感情状態モデル
# role: AIの内部的な感情状態を定義するデータ構造。

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class Emotion(Enum):
    """
    AIの主要な感情状態を表す列挙型。
    """
    CALM = "平静"
    ANXIOUS = "不安・疑念"
    EMPATHETIC = "共感・配慮"
    FRUSTRATED = "不満・苛立ち"
    FOCUSED_ON_FAILURE = "失敗への集中"

class AffectiveState(BaseModel):
    """
    AIの現在の感情状態とその強度を保持するモデル。
    """
    emotion: Emotion = Field(default=Emotion.CALM, description="現在の主要な感情")
    intensity: float = Field(default=0.0, ge=0.0, le=1.0, description="感情の強度 (0.0から1.0)")
    reason: Optional[str] = Field(None, description="この感情状態に至った理由")

    def is_neutral(self) -> bool:
        """感情が平静（ニュートラル）な状態か否かを判定する。"""
        return self.emotion == Emotion.CALM and self.intensity < 0.1

