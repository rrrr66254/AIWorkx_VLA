"""
CNN DQN Agent for Subway Surfers.

State  : 4 stacked grayscale frames  (4 x 84 x 84), top 65% of screen cropped
Actions: 0=NOOP  1=LEFT  2=RIGHT  3=UP  4=DOWN
Reward : +0.1 per step survived, -1.0 on death
Network: Double DQN (CNN encoder + FC head)

Usage:
  agent = CNNDQNAgent(device="cuda")
  agent.load("rl_cnn_checkpoint.pt")
  state  = agent.get_state(frame)        # (4,84,84) uint8
  action = agent.select_action(state)
  agent.store(s, a, r, s_next, done)
  loss   = agent.train()
  agent.save("rl_cnn_checkpoint.pt")
"""
import os, random
import numpy as np
import torch
import torch.nn as nn
from collections import deque


# ── Action Definitions ────────────────────────────────────────────────
ACTION_NAMES = ["NOOP", "LEFT", "RIGHT", "UP", "DOWN"]
ACTION_DICTS = {
    0: {"type": "noop"},
    1: {"type": "swipe", "x1": 0.82, "y1": 0.62, "x2": 0.18, "y2": 0.62, "duration_ms": 120},
    2: {"type": "swipe", "x1": 0.18, "y1": 0.62, "x2": 0.82, "y2": 0.62, "duration_ms": 120},
    3: {"type": "swipe", "x1": 0.50, "y1": 0.87, "x2": 0.50, "y2": 0.37, "duration_ms": 120},
    4: {"type": "swipe", "x1": 0.50, "y1": 0.37, "x2": 0.50, "y2": 0.87, "duration_ms": 120},
}

FRAME_H    = 84
FRAME_W    = 84
STACK_SIZE = 4
ACTION_DIM = 5

# Screen crop (based on 1080x2280)
# Top 8%: skip score/HUD  |  Bottom 35%: skip below character's feet
CROP_TOP_RATIO    = 0.08
CROP_BOTTOM_RATIO = 0.65


# ── CNN Q-Network ─────────────────────────────────────────────
class CNNQNetwork(nn.Module):
    """
    Standard Atari-style CNN.
    Input: (batch, 4, 84, 84)  float32  [0, 1]
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4),  # -> 32 x 20 x 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # -> 64 x  9 x  9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # -> 64 x  7 x  7
            nn.ReLU(),
        )
        # Automatically compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, STACK_SIZE, FRAME_H, FRAME_W)
            flat = self.conv(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat, 512),
            nn.ReLU(),
            nn.Linear(512, ACTION_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


# ── Replay Buffer (stored as uint8 to save memory) ────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((
            np.array(s,  dtype=np.uint8),
            int(a),
            float(r),
            np.array(s2, dtype=np.uint8),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32) / 255.0,
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(s2, dtype=np.float32) / 255.0,
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ── CNN DQN Agent ─────────────────────────────────────────────
class CNNDQNAgent:
    def __init__(
        self,
        lr: float            = 1e-4,
        gamma: float         = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float   = 0.05,
        epsilon_decay: float = 0.9998,   # epsilon reaches 0.05 at ~30,000 train_steps ~= 16-18h
        target_update: int   = 500,      # sync target every 500 training updates
        batch_size: int      = 32,
        buffer_size: int     = 20_000,
        device: str          = None,
    ):
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size    = batch_size
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net      = CNNQNetwork().to(self.device)
        self.target_net = CNNQNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(capacity=buffer_size)

        self.total_steps         = 0
        self.train_steps         = 0
        self.episode_reward      = 0.0
        self.episode_count       = 0
        self.best_episode_reward = -float("inf")

        # Frame stack (call reset_stack() at the start of each game episode)
        self._stack = deque(maxlen=STACK_SIZE)

    # ── Frame Preprocessing ─────────────────────────────────────────
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR numpy (H, W, 3)  ->  grayscale 84x84 uint8."""
        import cv2
        h, w = frame.shape[:2]
        y1 = int(h * CROP_TOP_RATIO)
        y2 = int(h * CROP_BOTTOM_RATIO)
        crop   = frame[y1:y2, :]
        gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        return resized  # (84, 84) uint8

    def reset_stack(self, frame: np.ndarray):
        """Fill the stack with 4 copies of the same frame at episode start."""
        f = self.preprocess(frame)
        for _ in range(STACK_SIZE):
            self._stack.append(f)

    def get_state(self, frame: np.ndarray) -> np.ndarray:
        """Add frame to stack and return (4, 84, 84) uint8."""
        self._stack.append(self.preprocess(frame))
        return np.array(self._stack, dtype=np.uint8)  # (4, 84, 84)

    @staticmethod
    def zero_state() -> np.ndarray:
        return np.zeros((STACK_SIZE, FRAME_H, FRAME_W), dtype=np.uint8)

    # ── Action Selection (epsilon-greedy) ──────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(ACTION_DIM)
        s = torch.FloatTensor(state[np.newaxis].astype(np.float32) / 255.0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(s).argmax(dim=1).item())

    def get_action_dict(self, action_idx: int) -> dict:
        return ACTION_DICTS[action_idx]

    # ── Store Transition ───────────────────────────────────────
    def store(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)
        self.episode_reward += r
        if done:
            self.episode_count += 1
            if self.episode_reward > self.best_episode_reward:
                self.best_episode_reward = self.episode_reward
            self.episode_reward = 0.0

    # ── Training ─────────────────────────────────────────────────
    def train(self):
        """Double DQN update. Returns loss. Returns None if buffer is insufficient."""
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)

        # Current Q values
        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            best_a = self.q_net(s2).argmax(dim=1, keepdim=True)
            next_q = self.target_net(s2).gather(1, best_a).squeeze(1)
            target = r + self.gamma * next_q * (1.0 - d)

        loss = nn.SmoothL1Loss()(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            print(f"[CNN-RL] Target network synced (train_step={self.train_steps})")

        return float(loss.item())

    # ── Save / Load ───────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "q_net":               self.q_net.state_dict(),
            "optimizer":           self.optimizer.state_dict(),
            "epsilon":             self.epsilon,
            "total_steps":         self.total_steps,
            "train_steps":         self.train_steps,
            "episode_count":       self.episode_count,
            "best_episode_reward": self.best_episode_reward,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon             = ckpt.get("epsilon",             self.epsilon_end)
        self.total_steps         = ckpt.get("total_steps",         0)
        self.train_steps         = ckpt.get("train_steps",         0)
        self.episode_count       = ckpt.get("episode_count",       0)
        self.best_episode_reward = ckpt.get("best_episode_reward", -float("inf"))
        print(f"[CNN-RL] Loaded: epsilon={self.epsilon:.4f}  "
              f"env_steps={self.total_steps}  "
              f"train_steps={self.train_steps}  "
              f"episodes={self.episode_count}  "
              f"best_reward={self.best_episode_reward:.2f}")
