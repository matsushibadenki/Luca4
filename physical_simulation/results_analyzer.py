# /physical_simulation/results_analyzer.py
# title: シミュレーション結果分析エージェント
# role: LLMを使い、物理シミュレーションの結果（ログ、最終状態）を分析し、構造化された「経験」や「洞察」を生成する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from typing import Any, Dict

from app.agents.base import AIAgent

class SimulationEvaluatorAgent(AIAgent):
    """
    シミュレーションログを解釈し、構造化された知識を生成するAIエージェント。
    """
    def __init__(self, llm: Any, output_parser: JsonOutputParser, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        super().__init__()

    def build_chain(self) -> Runnable:
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> Dict[str, Any]:
        """
        シミュレーション結果を分析し、構造化された洞察を返します。
        """
        if not isinstance(input_data, dict):
            raise TypeError("SimulationEvaluatorAgent expects a dictionary as input.")
        
        if "task_description" not in input_data or "final_state" not in input_data:
            raise ValueError("入力データには 'task_description' と 'final_state' が必要です。")
        
        if self._chain is None:
            raise RuntimeError("SimulationEvaluatorAgent's chain is not initialized.")
        
        # 入力データのログが長すぎる場合、主要な部分を切り出す
        input_data["simulation_log"] = str(input_data.get("simulation_log", ""))[:4000]

        result: Dict[str, Any] = self._chain.invoke(input_data)
        return result