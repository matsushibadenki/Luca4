# /app/engine.py
# title: メタインテリジェンスエンジン
# role: 実行モードに応じて適切な推論パイプラインを選択し、処理を実行する。

from __future__ import annotations
import logging
import asyncio
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipelines.base import BasePipeline
    from app.models import MasterAgentResponse, OrchestrationDecision

logger = logging.getLogger(__name__)

class MetaIntelligenceEngine:
    """
    推論パイプラインを管理し、実行するコアエンジン。
    """
    def __init__(self, pipelines: Dict[str, 'BasePipeline']):
        self.pipelines = pipelines

    def run(self, query: str, orchestration_decision: 'OrchestrationDecision') -> 'MasterAgentResponse':
        """
        同期的なコンテキストからエンジンを実行するためのラッパーメソッド。
        """
        return asyncio.run(self.arun(query, orchestration_decision))

    async def arun(self, query: str, orchestration_decision: 'OrchestrationDecision') -> 'MasterAgentResponse':
        """
        指定されたモードで適切なパイプラインを非同期で実行する。
        """
        chosen_mode = orchestration_decision.get("chosen_mode", "simple")
        current_pipeline = self.pipelines.get(chosen_mode)

        if not current_pipeline:
            logger.warning(f"無効な実行モード '{chosen_mode}' が指定されました。'simple' モードにフォールバックします。")
            current_pipeline = self.pipelines["simple"]
        
        try:
            logger.info(f"メインパイプライン '{chosen_mode}' で実行中...")
            response = await current_pipeline.arun(query, orchestration_decision)
            return response
        except Exception as e:
            logger.critical(f"パイプライン '{chosen_mode}' の実行中に致命的なエラーが発生しました: {e}", exc_info=True)
            return {
                "final_answer": "申し訳ありません、要求を処理中に予期せぬ内部エラーが発生しました。",
                "self_criticism": "致命的なエラーにより、自己評価は実行できませんでした。",
                "potential_problems": "システムログを確認してください。",
                "retrieved_info": ""
            }
