# /physical_simulation/environments/block_stacking_env.py
# title: ブロック積み上げ環境 (MuJoCo版)
# role: GymnasiumとMuJoCoを用いて、AIがブロックを積み上げるタスクを学習するための物理シミュレーション環境を定義する。

import os
from typing import Dict, Any, Optional, Tuple

import numpy as np
import mujoco
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.0,
    "azimuth": 135.0,
    "elevation": -35.0,
    "lookat": np.array([0.0, 0.0, 0.2]),
}

class BlockStackingEnv(MujocoEnv, EzPickle):
    """
    ブロック積み上げタスクのためのGymnasium準拠のMuJoCo環境。
    """
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        model_path: str = "block_stacking.xml",
        frame_skip: int = 20,
        render_mode: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        
        fullpath = os.path.join(os.path.dirname(__file__), model_path)
        if not os.path.exists(fullpath):
            raise FileNotFoundError(f"XMLファイルが見つかりません: {fullpath}")

        self.block_names = [f"object{i}" for i in range(5)]
        
        observation_space, action_space = self._construct_spaces()

        MujocoEnv.__init__(
            self,
            model_path=fullpath,
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_id=camera_id,
            camera_name=camera_name,
        )
        
        self.action_space = action_space
        EzPickle.__init__(self, model_path, frame_skip, render_mode, width, height, camera_id, camera_name)

    def _construct_spaces(self) -> Tuple[spaces.Space, spaces.Space]:
        obs_low = np.full(len(self.block_names) * 7, -np.inf, dtype=np.float32)
        obs_high = np.full(len(self.block_names) * 7, np.inf, dtype=np.float32)
        observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        
        act_low = np.array([-1.0] * 4, dtype=np.float32) 
        act_high = np.array([1.0] * 4, dtype=np.float32)
        action_space = spaces.Box(act_low, act_high, dtype=np.float32)
        
        return observation_space, action_space

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.reset_model()
        
        for name in self.block_names:
            qpos_addr = self.data.joint(name).qpos.addr
            qvel_addr = self.data.joint(name).qvel.addr
            
            self.data.qpos[qpos_addr[0]:qpos_addr[0]+3] = np.array([np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), 0.025])
            self.data.qpos[qpos_addr[0]+3:qpos_addr[0]+7] = np.array([1, 0, 0, 0])
            self.data.qvel[qvel_addr[0]:qvel_addr[0]+6] = np.zeros(6)

        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()
            
        return obs, info

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, bool, Dict[str, Any]]:
        block_idx = int(np.round((action[0] + 1) / 2 * (len(self.block_names) - 1)))
        target_pos = action[1:] * 0.5

        qpos_addr = self.data.joint(self.block_names[block_idx]).qpos.addr
        new_pos = self.data.qpos[qpos_addr[0]:qpos_addr[0]+3] + target_pos * 0.1
        self.data.qpos[qpos_addr[0]:qpos_addr[0]+3] = new_pos

        self.do_simulation(np.zeros(self.model.nu), self.frame_skip)
        
        observation = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = False
        info = self._get_info()

        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        obs = []
        for name in self.block_names:
            qpos = self.data.joint(name).qpos
            obs.extend(qpos)
        return np.array(obs, dtype=np.float32)
    
    def _compute_reward(self) -> np.float64:
        z_coords = [self.data.body(name).xpos[2] for name in self.block_names]
        height_reward = sum(z_coords)
        is_stacked = sum(1 for z in z_coords if z > 0.05)
        return np.float64(height_reward + is_stacked)

    def _is_terminated(self) -> bool:
        tower_height = self._get_tower_height()
        if tower_height > 0.2:
            return True
        return False

    def _get_info(self) -> Dict[str, Any]:
        return {"tower_height": np.float64(self._get_tower_height())}
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def _get_tower_height(self) -> float:
        max_z = 0.0
        for name in self.block_names:
            z_pos = self.data.body(name).xpos[2]
            if z_pos > max_z:
                max_z = z_pos
        return max_z