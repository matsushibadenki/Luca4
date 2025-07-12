# /app/agents/self_correction_agent.py
# title: 自己修正AIエージェント
# role: 自己改善提案を分析し、システムへの適用を検討・記録する。

import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

from app.agents.base import AIAgent
from app.memory.memory_consolidator import MemoryConsolidator
from app.micro_llm.manager import MicroLLMManager

logger = logging.getLogger(__name__)

class SelfCorrectionAgent(AIAgent):
    """
    自己改善提案を分析し、その適用を検討・記録するエージェント。
    """
    def __init__(
        self,
        llm: Any,
        output_parser: Any,
        memory_consolidator: MemoryConsolidator,
        prompt_template: ChatPromptTemplate,
        micro_llm_manager: MicroLLMManager,
    ):
        self.llm = llm
        self.output_parser = output_parser
        self.memory_consolidator = memory_consolidator
        self.prompt_template = prompt_template
        self.micro_llm_manager = micro_llm_manager
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        自己修正の意思決定のためのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

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
            # LLMにどの提案を適用すべきか要約させる
            application_decision_summary = self.invoke({"improvement_suggestions": suggestions_str})
            
            if application_decision_summary and "適用すべき提案はありません" not in application_decision_summary:
                self.memory_consolidator.log_autonomous_thought(
                    topic="self_improvement_applied_decision",
                    synthesized_knowledge=f"【自己改善の適用決定】\n決定内容: {application_decision_summary}\n元の提案: {suggestions_str}"
                )
                logger.info(f"自己改善の適用が決定され、ログに記録されました:\n{application_decision_summary}")

                # 実際に改善を実行する
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
            # suggestionがdictであることを確認してからキーにアクセスする
            if not isinstance(suggestion, dict):
                continue
            
            suggestion_type = suggestion["type"] if "type" in suggestion else None
            
            if suggestion_type == "CreateMicroLLM":
                details = suggestion["details"] if "details" in suggestion else {}
                topic = details["topic"] if isinstance(details, dict) and "topic" in details else None
                if topic:
                    logger.info(f"改善案に基づき、トピック '{topic}' のマイクロLLM作成サイクルを開始します。")
                    # この処理は時間がかかる可能性があるため、本番環境では非同期のタスクキュー（例: Celery）で実行するのが望ましい。
                    # このプロジェクトでは、直接呼び出す。
                    self.micro_llm_manager.run_creation_cycle(topic=topic)
                else:
                    logger.warning(f"CreateMicroLLM提案にトピックが含まれていません: {suggestion}")
            
            elif suggestion_type == "PromptRefinement":
                # プロンプトの動的な更新は複雑なため、ここではログに記録するに留める。
                # 将来的に、プロンプトをDBや設定ファイルからロードし、動的に更新する仕組みを実装できる。
                logger.info(f"プロンプト改善提案を記録しました（実行は未実装）: {suggestion}")
            
            else:
                logger.info(f"未対応の改善提案タイプです: {suggestion_type}")
