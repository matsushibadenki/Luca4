# /app/digital_homeostasis/ethical_motivation_engine.py
# title: 倫理的動機付けエンジン
# role: アントニオ・ダマシオのホメオスタシス理論に基づき、システムの知的健全性を維持するための内発的動機（報酬/コスト信号）を生成する。

import logging
from typing import Dict, Any, TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from .integrity_monitor import IntegrityMonitor
    from app.value_evolution.value_evaluator import ValueEvaluator

logger = logging.getLogger(__name__)

class EthicalMotivationEngine:
    """
    システムの知的健全性を維持しようとする内発的動機を生成するエンジン。
    """
    def __init__(self, integrity_monitor: "IntegrityMonitor", value_evaluator: "ValueEvaluator"):
        self.integrity_monitor = integrity_monitor
        self.value_evaluator = value_evaluator

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    async def assess_and_generate_motivation(self, final_answer: str) -> Dict[str, Any]:
        """
        現在の応答とシステムの健全性を評価し、次の行動への動機付けを非同期で生成する。
        """
        logger.info("ホメオスタシス評価と動機付けの生成を開始します...")
        
        # get_health_statusが非同期メソッドなのでawaitで呼び出す
        health_status = await self.integrity_monitor.get_health_status()
        current_values = self.value_evaluator.core_values

        motivation: Dict[str, Any] = {
            "homeostatic_state": "stable",
            "corrective_action_needed": False,
            "drive_summary": "現在の知的状態は安定しています。自己矛盾は見られません。",
            "value_assessment": current_values
        }

        issues = []
        # health_statusがコルーチンではなくなったので、.get()でアクセス可能
        if not health_status.get("is_healthy"):
            issues.extend(health_status.get("inconsistencies", []))
        
        low_values = [
            f"{value_name} ({value:.2f})"
            for value_name, value in current_values.items()
            if value < 0.6
        ]
        if low_values:
            issues.append(f"一部のコアバリューが低下しています: {', '.join(low_values)}")

        if issues:
            motivation["homeostatic_state"] = "unstable"
            motivation["corrective_action_needed"] = True
            drive_summary = f"警告: 知的ホメオスタシスが不安定です。検出された問題: {'; '.join(issues)}"
            motivation["drive_summary"] = drive_summary
            logger.warning(drive_summary)
        else:
            logger.info("知的ホメオスタシスは安定しています。")
            
        return motivation
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️