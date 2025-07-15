# /app/agents/self_correction_agent.py
# title: 自己修正AIエージェント
# role: 自己改善提案を分析し、システムへの適用を検討・記録する。

import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.agents.base import AIAgent
from app.memory.memory_consolidator import MemoryConsolidator
from app.micro_llm.manager import MicroLLMManager
from app.prompts.manager import PromptManager


logger = logging.getLogger(__name__)

class SelfCorrectionAgent(AIAgent):
    """
    自己改善提案を分析し、その適用を検討・記録するエージェント。
    """
    def __init__(
        self,
        llm: Any,
        memory_consolidator: MemoryConsolidator,
        micro_llm_manager: MicroLLMManager,
        prompt_manager: PromptManager,
    ):
        self.llm = llm
        self.memory_consolidator = memory_consolidator
        self.micro_llm_manager = micro_llm_manager
        self.prompt_manager = prompt_manager
        # このエージェントのプロンプトはPromptManagerから取得
        self.prompt_template = self.prompt_manager.get_prompt("SELF_CORRECTION_AGENT_PROMPT")
        super().__init__(llm=llm, memory_consolidator=memory_consolidator, micro_llm_manager=micro_llm_manager, prompt_manager=prompt_manager)

    def build_chain(self) -> Runnable:
        """
        自己修正の意思決定のためのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm 

    def consider_and_log_application(self, improvement_suggestions: List[Dict[str, Any]]) -> None:
        """
        自己改善提案を検討し、適用を決定した内容をログに記録し、実行する。
        """
        if not improvement_suggestions:
            logger.info("適用すべき自己改善提案がありません。")
            return

        logger.info("自己改善提案の適用を検討・実行中...")
        suggestions_str = "\n".join([str(s) for s in improvement_suggestions])

        try:
            if self._chain is None:
                raise RuntimeError("SelfCorrectionAgent's chain is not initialized.")
            application_decision_summary = self._chain.invoke({"improvement_suggestions": suggestions_str})
            
            if application_decision_summary and "適用すべき提案はありません" not in application_decision_summary:
                self.memory_consolidator.log_autonomous_thought(
                    topic="self_improvement_applied_decision",
                    synthesized_knowledge=f"【自己改善の適用決定】\n決定内容: {application_decision_summary}\n元の提案: {suggestions_str}"
                )
                logger.info(f"自己改善の適用が決定され、ログに記録されました:\n{application_decision_summary}")
                self._execute_improvements(improvement_suggestions)
            else:
                logger.info("自己改善提案の適用は見送られました。")

        except Exception as e:
            logger.error(f"自己修正エージェントによる適用検討中にエラーが発生しました: {e}", exc_info=True)

    def _execute_improvements(self, suggestions: List[Dict[str, Any]]):
        """
        適用可能と判断された改善案を実際に実行する。
        """
        for suggestion in suggestions:
            if not isinstance(suggestion, dict):
                continue
            
            suggestion_type = suggestion.get("type")
            details = suggestion.get("details", {})

            if not isinstance(details, dict):
                continue

            if suggestion_type == "CreateMicroLLM":
                topic = details.get("topic")
                if topic:
                    logger.info(f"改善案に基づき、トピック '{topic}' のマイクロLLM作成サイクルを開始します。")
                    self.micro_llm_manager.run_creation_cycle(topic=topic)
                else:
                    logger.warning(f"CreateMicroLLM提案にトピックが含まれていません: {suggestion}")
            
            elif suggestion_type == "PromptRefinement":
                prompt_key = details.get("target_prompt_key")
                new_prompt = details.get("new_prompt_suggestion")
                if prompt_key and new_prompt:
                    logger.info(f"改善案に基づき、プロンプト '{prompt_key}' の更新を試みます。")
                    success = self.prompt_manager.update_prompt(prompt_key, new_prompt)
                    if success:
                        logger.info(f"プロンプト '{prompt_key}' が正常に更新・保存されました。")
                    else:
                        logger.error(f"プロンプト '{prompt_key}' の更新に失敗しました。")
                else:
                    logger.warning(f"PromptRefinement提案に必要な情報が不足しています: {suggestion}")
            
            else:
                logger.info(f"未対応の改善提案タイプです: {suggestion_type}")