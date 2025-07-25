# /app/affective_system/affective_engine.py
# title: 感情エンジン
# role: AIの内部状態や外部からの入力に基づき、感情状態を評価・更新する。

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional
import asyncio

from .affective_state import AffectiveState, Emotion

if TYPE_CHECKING:
    from app.digital_homeostasis.integrity_monitor import IntegrityMonitor
    from app.value_evolution.value_evaluator import ValueEvaluator
    from app.models import MasterAgentResponse

logger = logging.getLogger(__name__)

class AffectiveEngine:
    """
    AIの感情状態を管理し、状況に応じて更新するエンジン。
    """
    def __init__(
        self,
        integrity_monitor: "IntegrityMonitor",
        value_evaluator: "ValueEvaluator",
    ):
        self.integrity_monitor = integrity_monitor
        self.value_evaluator = value_evaluator
        self.current_state = AffectiveState(reason=None)

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    async def assess_and_update_state(
        self,
        user_query: str,
        response: Optional[MasterAgentResponse] = None,
        user_profile: Optional[str] = None
    ) -> AffectiveState:
        """
        現在の状況を総合的に評価し、AIの感情状態を非同期で更新する。
        """
        logger.info("感情状態の評価・更新を開始します...")

        # 1. 不満・苛立ち (Frustration) の評価
        # システムの健全性が損なわれているか、価値観が大きく低下している場合に発生
        health_status = await self.integrity_monitor.get_health_status()
        if not health_status.get("is_healthy", True):
            self.current_state = AffectiveState(
                emotion=Emotion.FRUSTRATED,
                intensity=0.8,
                reason=f"システムの論理的整合性に問題が検出されました: {health_status.get('inconsistencies', [])}"
            )
            logger.warning(f"感情状態が更新されました: {self.current_state.emotion.value} (理由: {self.current_state.reason})")
            return self.current_state

        # 2. 不安・疑念 (Anxiety) の評価
        # 自己評価が低い場合や、回答の確信度が低い場合に発生
        if response:
            self_criticism = response.get("self_criticism", "")
            if "問題" in self_criticism or "限定的" in self_criticism or "失敗" in self_criticism:
                self.current_state = AffectiveState(
                    emotion=Emotion.ANXIOUS,
                    intensity=0.6,
                    reason=f"自己評価により、回答の品質に懸念が示されました: {self_criticism}"
                )
                logger.info(f"感情状態が更新されました: {self.current_state.emotion.value} (理由: {self.current_state.reason})")
                return self.current_state

        # 3. 共感 (Empathy) の評価
        # ユーザーの感情的な側面に寄り添う必要があると判断された場合に発生
        empathetic_keywords = ["辛い", "悲しい", "疲れた", "どうしたらいいか分からない"]
        if any(keyword in user_query for keyword in empathetic_keywords) or (user_profile and "emotional_support" in user_profile):
            self.current_state = AffectiveState(
                emotion=Emotion.EMPATHETIC,
                intensity=0.7,
                reason="ユーザーのクエリやプロファイルから、感情的なサポートが必要と判断されました。"
            )
            logger.info(f"感情状態が更新されました: {self.current_state.emotion.value} (理由: {self.current_state.reason})")
            return self.current_state

        # 4. デフォルトは平静状態
        self.current_state = AffectiveState(reason=None)
        logger.info("感情状態は「平静」です。")
        return self.current_state
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def get_current_state(self) -> AffectiveState:
        """現在の感情状態を返す。"""
        return self.current_state