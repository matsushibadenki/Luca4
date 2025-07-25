# /app/reasoning/symbolic_verifier.py
# title: 記号的推論検証器
# role: 幾何学の公理や定理に基づき、LLMが生成した仮説の論理的な正しさを検証する。

import logging
from typing import Dict, Any, List, Set, Tuple

logger = logging.getLogger(__name__)

class SymbolicVerifier:
    """
    AlphaGeometryのコンセプトに倣い、記号的な論理ルールに基づいて
    仮説を検証するエンジン。
    この実装は、幾何学の定理証明を簡略化したモデルです。
    """
    def __init__(self):
        # システムの公理や既知の定理を保持する
        # (事実のセット, 導かれる結論)
        self.rules: List[Tuple[Set[str], str]] = [
            # 三段論法のルール
            ({"AならばB", "BならばC"}, "AならばC"),
            ({"点Aと点Bは直線上にある", "点Bと点Cは直線上にある"}, "点A,B,Cは同一直線上にある"),
            ({"三角形ABCは二等辺三角形である", "辺AB = 辺AC"}, "角ABC = 角ACB"),
            ({"角ABC = 90度"}, "三角形ABCは直角三角形である"),
        ]
        logger.info("記号的推論検証器が初期化されました。")

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def verify_and_deduce(self, known_facts: Set[str]) -> Set[str]:
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        """
        既知の事実のセットを受け取り、ルールを適用して導出できる新しい事実をすべて返す。

        Args:
            known_facts (Set[str]): 現在までに証明されている事実のセット。

        Returns:
            Set[str]: 新たに導出された事実のセット。
        """
        newly_deduced_facts: Set[str] = set()
        
        # 新しい事実が導出されなくなるまでルール適用を繰り返す
        while True:
            found_new_fact_in_iteration = False
            current_knowledge = known_facts.union(newly_deduced_facts)

            for premises, conclusion in self.rules:
                # 結論がまだ知られておらず、前提がすべて満たされているか
                if conclusion not in current_knowledge and premises.issubset(current_knowledge):
                    newly_deduced_facts.add(conclusion)
                    found_new_fact_in_iteration = True
                    logger.info(f"新しい事実を演繹: {premises} -> {conclusion}")

            if not found_new_fact_in_iteration:
                break # 新しい発見がなければループを抜ける
        
        return newly_deduced_facts