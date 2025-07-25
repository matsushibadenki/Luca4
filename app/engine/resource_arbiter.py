# /app/engine/resource_arbiter.py
# title: リソースアービター
# role: パイプライン実行をエネルギー残量に基づいて仲裁し、システムの過負荷を防ぐ。

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from app.models import OrchestrationDecision
    from app.meta_intelligence.cognitive_energy.manager import CognitiveEnergyManager

logger = logging.getLogger(__name__)

# パイプラインごとのエネルギーコスト定義
PIPELINE_ENERGY_COSTS: Dict[str, float] = {
    "simple": 5.0,
    "full": 20.0,
    "parallel": 25.0,
    "quantum": 30.0,
    "speculative": 15.0,
    "self_discover": 22.0,
    "internal_dialogue": 18.0,
    "conceptual_reasoning": 28.0,
    "micro_llm_expert": 10.0,
    "default": 10.0 # フォールバック用
}

class ResourceArbiter:
    """
    OrchestrationAgentの決定を、現在の認知エネルギー状態に基づいて「仲裁」するクラス。
    """
    def __init__(self, energy_manager: "CognitiveEnergyManager"):
        self.energy_manager = energy_manager

    def arbitrate(self, decision: "OrchestrationDecision") -> "OrchestrationDecision":
        """
        パイプライン実行の決定を仲裁する。

        Args:
            decision (OrchestrationDecision): OrchestrationAgentからの決定。

        Returns:
            OrchestrationDecision: 実行が許可された最終的な決定。
        """
        chosen_pipeline = decision.get("chosen_mode", "simple")
        cost = PIPELINE_ENERGY_COSTS.get(chosen_pipeline, PIPELINE_ENERGY_COSTS["default"])

        logger.info(f"Arbitrating decision for pipeline '{chosen_pipeline}' with energy cost {cost}.")
        current_energy = self.energy_manager.get_current_energy_level()
        logger.info(f"Current energy level: {current_energy:.2f}")

        if self.energy_manager.consume_energy(cost):
            logger.info(f"Sufficient energy. Permitting execution of '{chosen_pipeline}'.")
            return decision
        else:
            logger.warning(
                f"Insufficient energy for pipeline '{chosen_pipeline}'. "
                f"Required: {cost}, Available: {current_energy:.2f}. "
                "Falling back to 'simple' pipeline."
            )
            
            fallback_decision = decision.copy()
            fallback_decision["chosen_mode"] = "simple"
            
            original_reason = fallback_decision.get("reason", "")
            fallback_reason = (
                f"FALLBACK: Insufficient energy for original choice '{chosen_pipeline}'. "
                f"Original reason: {original_reason}"
            )
            fallback_decision["reason"] = fallback_reason
            
            # フォールバックパイプラインのエネルギーを消費しようと試みる
            fallback_cost = PIPELINE_ENERGY_COSTS.get("simple", 5.0)
            self.energy_manager.consume_energy(fallback_cost)
            
            return fallback_decision