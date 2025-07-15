# /physical_simulation/agents/ppo_agent.py
# title: PPO強化学習エージェント
# role: PPO（Proximal Policy Optimization）アルゴリズムに基づき、方策を学習するエージェント。

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from typing import Tuple, List

from physical_simulation.agents.base_agent import BaseRLAgent
from physical_simulation.experience_buffer import ReplayBuffer

# デバイス設定
device = torch.device('cpu')

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

class PPOAgent(BaseRLAgent):
    def __init__(self, state_dim: int, action_dim: int, lr_actor: float, lr_critic: float, gamma: float, K_epochs: int, eps_clip: float):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = ReplayBuffer()

        self.policy = ActorCritic(state_dim, action_dim, 0.1).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim, 0.1).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state_tensor)

        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # バッファにはTensorを直接格納する
        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

        return action.cpu().numpy().flatten(), action_logprob.cpu().numpy()

    def update(self) -> None:
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # モンテカルロ法による報酬の推定
        rewards: List[float] = []
        discounted_reward: float = 0.0 # float型で初期化
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0.0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 報酬の正規化
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-7)

        # バッファから学習データを取得
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

        # Kエポック分ポリシーを最適化
        for _ in range(self.K_epochs):
            # 現在のポリシーで対数確率、状態価値、エントロピーを評価
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # 重要度サンプリング比の計算
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Advantage（アドバンテージ）の計算
            advantages = rewards_tensor - state_values.detach()
            
            # PPOのクリッピング目的関数を計算
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # 最終的な損失を計算（アクター、クリティック、エントロピー）
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards_tensor) - 0.01 * dist_entropy
            
            # 勾配を計算し、オプティマイザで更新
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # 新しいポリシーを古いポリシーにコピー
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 次の学習のためにバッファをクリア
        self.buffer.clear()