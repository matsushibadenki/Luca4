# /app/affective_system/__init__.py
# title: 感情システムパッケージ
# role: 感情関連の主要クラスをインポートし、パッケージとして利用可能にする。

from .affective_state import AffectiveState, Emotion
from .affective_engine import AffectiveEngine
from .emotional_response_generator import EmotionalResponseGenerator
