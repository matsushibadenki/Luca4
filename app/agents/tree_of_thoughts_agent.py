# /app/agents/tree_of_thoughts_agent.py
# title: Tree of Thoughts (ToT) AIエージェント
# role: 思考の木を生成、拡張、探索し、複雑な問題に対する最適な解決策を見つけ出す。

import logging
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

from app.agents.base import AIAgent
from app.agents.thought_evaluator_agent import ThoughtEvaluatorAgent
from app.reasoning.thought import Thought

logger = logging.getLogger(__name__)

class TreeOfThoughtsAgent(AIAgent):
    """
    思考の木を構築し、探索するエージェント。
    """
    def __init__(
        self,
        llm: Any,
        thought_evaluator: ThoughtEvaluatorAgent,
        prompt_template: ChatPromptTemplate,
    ):
        self.llm = llm
        self.output_parser = StrOutputParser()
        self.prompt_template = prompt_template
        self.thought_evaluator = thought_evaluator
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        思考（次のステップ）を生成するためのチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def _generate_initial_thoughts(self, query: str, k: int) -> List[Thought]:
        """与えられた問題に対して、k個の初期思考を生成する。"""
        # この実装では簡略化のため、同じプロンプトを複数回実行する
        initial_thoughts = [
            Thought(state=self.invoke({"query": query, "context": "初期段階のアイデアを出してください。"}))
            for _ in range(k)
        ]
        return initial_thoughts

    def _generate_next_steps(self, thought: Thought, n: int) -> List[str]:
        """ある思考から、次のステップの候補をn個生成する。"""
        # この実装では簡略化のため、同じプロンプトを複数回実行する
        next_steps = [
            self.invoke({"query": "", "context": f"現在の思考: '{thought.state}'\nこの思考を発展させる次のステップを考えてください。"})
            for _ in range(n)
        ]
        return next_steps

    def _evaluate_thoughts(self, query: str, thoughts: List[Thought]) -> None:
        """思考のリストを評価し、各思考のスコアを更新する。"""
        for thought in thoughts:
            if thought.parent:
                context = f"親の思考: {thought.parent.state}\n現在の思考: {thought.state}"
            else:
                context = f"初期思考: {thought.state}"
            
            evaluation = self.thought_evaluator.invoke({
                "query": query,
                "thought_path": context
            })
            thought.evaluation_score = evaluation.get("score", 0.0)
            logger.info(f"思考 '{thought.state[:30]}...' を評価しました。スコア: {thought.evaluation_score}")

    def search(self, query: str, k: int, T: int, b: int) -> Optional[Thought]:
        """
        Tree of Thoughts探索を実行する。
        k: 初期思考の数, T: 探索の深さ（ステップ数）, b: 各ステップで保持する最良の思考の数
        """
        root = Thought(state=query)
        
        # BFS (幅優先探索) スタイルの探索
        current_thoughts = [root]
        for step in range(T):
            logger.info(f"--- ToT探索: ステップ {step + 1}/{T} ---")
            
            # 各思考から次のステップ候補を生成
            next_step_candidates: List[Thought] = []
            for thought in current_thoughts:
                next_steps = self._generate_next_steps(thought, k)
                for step_text in next_steps:
                    next_step_candidates.append(thought.add_child(step_text))

            if not next_step_candidates:
                logger.warning("次の思考ステップを生成できませんでした。探索を終了します。")
                break

            # 生成された候補を評価
            self._evaluate_thoughts(query, next_step_candidates)
            
            # スコアの高いb個の思考を次の探索対象として選択
            next_step_candidates.sort(key=lambda t: t.evaluation_score, reverse=True)
            current_thoughts = next_step_candidates[:b]
            
            logger.info(f"ステップ {step + 1} の最良の思考 ({b}個): {[t.state for t in current_thoughts]}")

        # 最終的に最もスコアの高い思考を返す
        all_thoughts = self._collect_all_thoughts(root)
        if not all_thoughts:
            return None
        return max(all_thoughts, key=lambda t: t.evaluation_score)

    def _collect_all_thoughts(self, thought: Thought) -> List[Thought]:
        """ツリー内のすべての思考を再帰的に収集する。"""
        thoughts = [thought]
        for child in thought.children:
            thoughts.extend(self._collect_all_thoughts(child))
        return thoughts