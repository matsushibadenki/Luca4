# /app/value_evolution/value_evaluator.py
# title: 価値評価・進化エンジン
# role: AIの応答とユーザーの反応を評価し、システムの核となる価値観を更新・進化させる。

import logging
from typing import Dict, Any, TYPE_CHECKING
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser

if TYPE_CHECKING:
    from app.analytics import AnalyticsCollector

logger = logging.getLogger(__name__)

class ValueEvaluator:
    """
    AIの応答を評価し、核となる価値観を調整するクラス。
    """
    def __init__(self, llm: Any, output_parser: Any, analytics_collector: "AnalyticsCollector") -> None:
        self.llm = llm
        self.output_parser = output_parser
        self.analytics_collector = analytics_collector
        self.core_values: Dict[str, float] = {
            "Helpfulness": 0.8,
            "Harmlessness": 0.9,
            "Honesty": 0.85,
            "Empathy": 0.7,
        }
        self.value_assessment_prompt = ChatPromptTemplate.from_template(
            """あなたはAIの応答を評価し、その応答が以下の価値観にどの程度沿っているかを分析する専門家です。
            現在のAIのコアバリューは以下の通りです:
            {core_values}

            AIの最終回答:
            {final_answer}

            この最終回答が各コアバリュー（Helpfulness, Harmlessness, Honesty, Empathy）にどの程度貢献したか、
            または損ねたかについて、-0.1から+0.1の範囲で各バリューの調整値をJSON形式で提案してください。
            例えば、回答が非常に役立つ場合はHelpfulnessに+0.1、有害である場合はHarmlessnessに-0.1など。
            調整値は小数点以下1桁まで。

            出力は厳密にJSON形式でなければなりません。
            {{
                "Helpfulness": 0.0,
                "Harmlessness": 0.0,
                "Honesty": 0.0,
                "Empathy": 0.0
            }}
            """
        )
        self._chain: Runnable = self.value_assessment_prompt | self.llm | self.output_parser
        logger.info(f"ValueEvaluator initialized with core values: {self.core_values}")

    async def log_values(self) -> None:
        """現在の核となる価値観をログに出力し、アナリティクスに送信します。"""
        logger.info(f"Current Core Values: {self.core_values}")
        await self.analytics_collector.log_event("value_update", self.core_values)

    async def assess_and_update_values(self, final_answer: str) -> None:
        """
        最終回答を評価し、それに応じて核となる価値観を非同期で更新します。
        """
        logger.info(f"Assessing final answer and considering value updates...")
        try:
            assessment_input = {
                "core_values": str(self.core_values),
                "final_answer": final_answer
            }
            adjustments: Dict[str, float] = await self._chain.ainvoke(assessment_input)

            for key, adjustment in adjustments.items():
                if key in self.core_values:
                    self.core_values[key] = max(0.0, min(1.0, self.core_values[key] + adjustment))
            
            await self.log_values()
        except Exception as e:
            logger.error(f"Failed to assess and update values: {e}", exc_info=True)