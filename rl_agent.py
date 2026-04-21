"""
DQN Agent for Subway Surfers.

State  : NitroGen j_left[0] (2) + buttons[0] (21) = 23-dim float vector
Actions: 0=NOOP, 1=LEFT, 2=RIGHT, 3=UP, 4=DOWN
Reward : +0.1 per step survived, -1.0 on death
Network: Double DQN  (QNet + TargetNet)

Usage:
  agent = DQNAgent()
  agent.load("rl_checkpoint.pt")   # resume training
  state  = agent.extract_state(nitrogen_raw)
  action = agent.select_action(state)
  agent.store(s, a, r, s_next, done)
  loss   = agent.train()
  agent.save("rl_checkpoint.pt")
"""
import os, random
import numpy as np
import torch
import torch.nn as nn
from collections import deque


# ── Action Definitions ───────────────────────────────────────────
ACTION_NAMES = ["NOOP", "LEFT", "RIGHT", "UP", "DOWN"]
ACTION_DICTS = {
    0: {"type": "noop"},
    1: {"type": "swipe", "x1": 0.82, "y1": 0.62, "x2": 0.18, "y2": 0.62, "duration_ms": 120},
    2: {"type": "swipe", "x1": 0.18, "y1": 0.62, "x2": 0.82, "y2": 0.62, "duration_ms": 120},
    3: {"type": "swipe", "x1": 0.50, "y1": 0.87, "x2": 0.50, "y2": 0.37, "duration_ms": 120},
    4: {"type": "swipe", "x1": 0.50, "y1": 0.37, "x2": 0.50, "y2": 0.87, "duration_ms": 120},
}

STATE_DIM  = 23   # j_left(2) + buttons(21)
ACTION_DIM = 5


# ── Q-Network ────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay Buffer ─────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, int(a), float(r), s2, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ── DQN Agent ─────────────────────────────────────────────────────
class DQNAgent:
    def __init__(
        self,
        lr: float           = 1e-3,
        gamma: float        = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float  = 0.05,
        epsilon_decay: float = 0.997,   # reach epsilon 0.05: ~1100 steps
        target_update: int  = 100,      # target network sync interval
        batch_size: int     = 64,
        train_every: int    = 4,        # train every N steps
        device: str         = None,
    ):
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size    = batch_size
        self.train_every   = train_every
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net      = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer()

        self.total_steps  = 0   # environment step count
        self.train_steps  = 0   # training update count
        self.episode_reward = 0.0
        self.episode_count  = 0
        self.best_episode_reward = -float("inf")

    # ── State Extraction ──────────────────────────────────────────
    def extract_state(self, nitrogen_raw) -> np.ndarray:
        """NitroGen dict -> 23-dim float32 state."""
        if isinstance(nitrogen_raw, dict):
            jl = np.array(nitrogen_raw.get("j_left",  [[0.0, 0.0]]), dtype=np.float32)
            bt = np.array(nitrogen_raw.get("buttons", [[0.0] * 21]), dtype=np.float32)
            # use first timestep
            if jl.ndim == 2: jl = jl[0]
            if bt.ndim == 2: bt = bt[0]
        else:
            jl = np.zeros(2,  dtype=np.float32)
            bt = np.zeros(21, dtype=np.float32)

        # align lengths
        jl = np.resize(jl, 2)
        bt = np.resize(bt, 21)
        return np.concatenate([jl, bt])   # (23,)

    @staticmethod
    def zero_state() -> np.ndarray:
        return np.zeros(STATE_DIM, dtype=np.float32)

    # ── Action Selection (epsilon-greedy) ─────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(ACTION_DIM)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(s).argmax(dim=1).item())

    def get_action_dict(self, action_idx: int) -> dict:
        return ACTION_DICTS[action_idx]

    # ── Store Transition ──────────────────────────────────────────
    def store(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)
        self.episode_reward += r
        if done:
            self.episode_count += 1
            if self.episode_reward > self.best_episode_reward:
                self.best_episode_reward = self.episode_reward
            self.episode_reward = 0.0

    # ── Training ──────────────────────────────────────────────────
    def train(self) -> float | None:
        """Sample batch -> Double DQN loss -> update. Returns loss."""
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)

        # current Q values
        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target: select action with online network, evaluate with target network
        with torch.no_grad():
            best_a  = self.q_net(s2).argmax(dim=1, keepdim=True)
            next_q  = self.target_net(s2).gather(1, best_a).squeeze(1)
            target  = r + self.gamma * next_q * (1.0 - d)

        loss = nn.SmoothL1Loss()(q_vals, target)   # Huber loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    # ── Save / Load ───────────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "q_net":        self.q_net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "epsilon":      self.epsilon,
            "total_steps":  self.total_steps,
            "train_steps":  self.train_steps,
            "episode_count": self.episode_count,
            "best_episode_reward": self.best_episode_reward,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon             = ckpt.get("epsilon",       self.epsilon_end)
        self.total_steps         = ckpt.get("total_steps",   0)
        self.train_steps         = ckpt.get("train_steps",   0)
        self.episode_count       = ckpt.get("episode_count", 0)
        self.best_episode_reward = ckpt.get("best_episode_reward", -float("inf"))
        print(f"[RL] Loaded: epsilon={self.epsilon:.3f}  "
              f"env_steps={self.total_steps}  "
              f"train_steps={self.train_steps}  "
              f"episodes={self.episode_count}  "
              f"best_reward={self.best_episode_reward:.2f}")
