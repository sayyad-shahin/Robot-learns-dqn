"""Double DQN agent: experience replay + target network + epsilon-greedy exploration."""
import numpy as np
from .nn import MLP


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, seed=0):
        self.cap, self.n, self.i = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, d):
        k = self.i
        self.s[k], self.a[k], self.r[k], self.s2[k], self.d[k] = s, a, r, s2, d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch):
        idx = self.rng.integers(0, self.n, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class DQNAgent:
    def __init__(self, obs_dim, n_actions=4, hidden=(128, 128), lr=1e-3, gamma=0.97,
                 buffer_size=50_000, batch_size=64, tau=0.01, seed=0):
        sizes = [obs_dim, *hidden, n_actions]
        self.q = MLP(sizes, lr=lr, seed=seed)
        self.target = MLP(sizes, lr=lr, seed=seed + 1)
        self.target.copy_from(self.q)
        self.buffer = ReplayBuffer(buffer_size, obs_dim, seed)
        self.gamma, self.batch_size, self.tau = gamma, batch_size, tau
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q.forward(obs[None, :])[0]))

    def remember(self, s, a, r, s2, terminal):
        self.buffer.add(s, a, r, s2, float(terminal))

    def learn(self):
        if self.buffer.n < self.batch_size * 5:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        best = np.argmax(self.q.forward(s2), axis=1)                    # online net picks action
        q_next = self.target.forward(s2)[np.arange(len(s)), best]       # target net scores it
        y = r + self.gamma * (1.0 - d) * q_next
        loss = self.q.train_step(s, a, y)
        self.target.soft_update(self.q, self.tau)
        return loss

    # ---- saving / loading ---------------------------------------------
    def save(self, path, env):
        data = {"sizes": np.array(self.q.sizes), "env_size": env.size,
                "env_view": env.view, "env_density": env.density, "gamma": self.gamma}
        for i, (w, b) in enumerate(zip(self.q.W, self.q.b)):
            data[f"W{i}"], data[f"b{i}"] = w, b
        np.savez(path, **data)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        sizes = [int(s) for s in d["sizes"]]
        agent = cls(sizes[0], sizes[-1], hidden=tuple(sizes[1:-1]), gamma=float(d["gamma"]))
        agent.q.W = [d[f"W{i}"] for i in range(len(sizes) - 1)]
        agent.q.b = [d[f"b{i}"] for i in range(len(sizes) - 1)]
        agent.target.copy_from(agent.q)
        env_cfg = {"size": int(d["env_size"]), "view": int(d["env_view"]),
                   "obstacle_density": float(d["env_density"]), "gamma": float(d["gamma"])}
        return agent, env_cfg
