# /physical_simulation/experience_buffer.py
# title: 経験リプレイバッファ
# role: 強化学習エージェントが学習するために、シミュレーションで観測された状態、行動、報酬などを一時的に保存する。

import numpy as np
import torch
from typing import List

class ReplayBuffer:
    """
    強化学習の経験を保存・管理するためのバッファ。
    """
    def __init__(self) -> None:
        self.actions: List[torch.Tensor] = []
        self.states: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.is_terminals: List[bool] = []

    def clear(self) -> None:
        """バッファの内容をすべて消去する。"""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]