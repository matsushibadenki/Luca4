# /app/pipelines/internal_dialogue_pipeline.py
# title: 内省的対話パイプライン
# role: 「心の社会」モデルに基づき、動的に生成された思考エージェント群による内省的な対話を通じて、問題を解決する。

import logging
import time
from typing import Any, Dict

from app.pipelines.base import BasePipeline
from app.models import MasterAgentResponse
from app.internal_dialogue.dialogue_participant_agent import DialogueParticipantAgent
from app.internal_dialogue.consciousness_staging_area import ConsciousnessStagingArea
from app.integrated_information_processing.integrated_information_agent import IntegratedInformationAgent
from app.models import OrchestrationDecision

from app.config import settings

logger = logging.getLogger(__name__)

class InternalDialoguePipeline(BasePipeline):
    """
    動的に生成されたエージェントによる内省的対話を通じて回答を生成するパイプライン。
    """
    def __init__(
        self,
        dialogue_participant_agent: DialogueParticipantAgent,
        consciousness_staging_area: ConsciousnessStagingArea,
        integrated_information_agent: IntegratedInformationAgent,
    ):
        self.dialogue_participant_agent = dialogue_participant_agent
        self.consciousness_staging_area = consciousness_staging_area
        self.integrated_information_agent = integrated_information_agent

    def run(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        """
        パイプラインを実行する。
        """
        start_time = time.time()
        logger.info("--- Internal Dialogue Pipeline START ---")

        # 1. 問題に関連する思考エージェント（ペルソナ）を動的に生成
        participants = self.dialogue_participant_agent.invoke({"query": query})
        if not participants:
            logger.error("対話参加者の生成に失敗しました。")
            return {
                "final_answer": "申し訳ありません、問題について多角的に検討することができませんでした。",
                "self_criticism": "思考の起点となる対話参加者を生成できませんでした。",
                "potential_problems": "LLMが指定したJSON形式でペルソナを生成できなかった可能性があります。",
                "retrieved_info": ""
            }

        # 2. 意識のステージで内省的対話を実行
        max_turns = settings.PIPELINE_SETTINGS["internal_dialogue"]["max_turns"]
        dialogue_summary = self.consciousness_staging_area.run_dialogue(query, participants, max_turns=max_turns)

        # 3. 対話結果を統合して最終回答を生成
        integration_input = {
            "query": query,
            "persona_outputs": dialogue_summary
        }
        final_answer = self.integrated_information_agent.invoke(integration_input)
        
        logger.info(f"--- Internal Dialogue Pipeline END ({(time.time() - start_time):.2f} s) ---")
        
        return {
            "final_answer": final_answer,
            "self_criticism": "内省的対話パイプラインは、動的に生成された複数の視点の議論を経て、統合的な回答を生成しました。",
            "potential_problems": "生成される視点が偏る、あるいは対話が収束しない可能性があります。",
            "retrieved_info": dialogue_summary
        }