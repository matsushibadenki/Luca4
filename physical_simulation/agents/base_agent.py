# /physical_simulation/agents/base_agent.py
# title: 強化学習エージェント 抽象基底クラス
# role: すべての強化学習エージェントの基本的な構造とインターフェースを定義する。

from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class BaseRLAgent(ABC):
    """
    全ての強化学習エージェントが継承する抽象基底クラス。
    """

    @abstractmethod
    def select_action(self, state: np.ndarray) -> Any:
        """
        現在の状態(state)に基づき、行動(action)を選択する。
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """
        収集された経験を用いて、エージェントのポリシー（行動方針）を更新する。
        """
        pass