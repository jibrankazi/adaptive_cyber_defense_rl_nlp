import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CyberDefenseEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, seed=1337):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # e.g., {ignore, alert, block}
        self.state = None

    def reset(self, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.state = self.rng.random(8, dtype=np.float32)
        return self.state, {}

    def step(self, action):
        # toy dynamics: reward higher for "block" when latent attack present
        attack = (self.rng.random() < 0.3)
        reward = 1.0 if (attack and action==2) else 0.0
        self.state = self.rng.random(8, dtype=np.float32)
        terminated = False
        truncated  = False
        info = {"attack": attack}
        return self.state, reward, terminated, truncated, info
