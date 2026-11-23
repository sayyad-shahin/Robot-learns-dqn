"""A small fully-connected neural network with backpropagation and Adam, in pure NumPy."""
import numpy as np


class MLP:
    def __init__(self, sizes, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = [int(s) for s in sizes]
        self.lr = lr
        self.W = [rng.normal(0, np.sqrt(2.0 / i), (i, o)).astype(np.float32)
                  for i, o in zip(self.sizes[:-1], self.sizes[1:])]
        self.b = [np.zeros(o, dtype=np.float32) for o in self.sizes[1:]]
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, x):
        a = np.asarray(x, dtype=np.float32)
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            a = a @ W + b
            if i < last:
                a = np.maximum(a, 0.0)  # ReLU
        return a

    def train_step(self, x, actions, targets, clip_norm=10.0):
        """One gradient step so that Q(x, action) moves toward targets (Huber loss)."""
        acts = [np.asarray(x, dtype=np.float32)]
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(np.maximum(z, 0.0) if i < last else z)
        out = acts[-1]
        batch = len(out)
        idx = np.arange(batch)
        err = out[idx, actions] - targets
        loss = float(np.mean(np.where(np.abs(err) <= 1, 0.5 * err ** 2, np.abs(err) - 0.5)))

        grad = np.zeros_like(out)
        grad[idx, actions] = np.clip(err, -1.0, 1.0) / batch
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ grad
            gb[i] = grad.sum(axis=0)
            if i > 0:
                grad = (grad @ self.W[i].T) * (acts[i] > 0)

        norm = np.sqrt(sum(float((g ** 2).sum()) for g in gW + gb))
        if norm > clip_norm:
            scale = clip_norm / (norm + 1e-8)
            gW = [g * scale for g in gW]
            gb = [g * scale for g in gb]

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for params, grads, m, v in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
            for i in range(len(params)):
                m[i] = b1 * m[i] + (1 - b1) * grads[i]
                v[i] = b2 * v[i] + (1 - b2) * grads[i] ** 2
                params[i] -= self.lr * (m[i] / c1) / (np.sqrt(v[i] / c2) + eps)
        return loss

    def soft_update(self, other, tau):
        for i in range(len(self.W)):
            self.W[i] = (1 - tau) * self.W[i] + tau * other.W[i]
            self.b[i] = (1 - tau) * self.b[i] + tau * other.b[i]

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]
