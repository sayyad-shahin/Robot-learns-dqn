"""Train the robot. Usage:  python train.py            (default 6000 episodes, ~4 min)
                           python train.py --episodes 500 --out models/quick.npz
"""
import argparse, os, time
import matplotlib
matplotlib.use("Agg")  # only saves the plot, no window needed
import matplotlib.pyplot as plt
import numpy as np

from robot_rl import GridRobotEnv, DQNAgent, evaluate, random_policy


def moving_avg(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_history(h, path, window=100):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    r = moving_avg(h["reward"], window)
    ax[0].plot(np.arange(len(r)) + window, r, color="#2a6f97")
    ax[0].set(title="Reward per episode (moving average)", xlabel="Episode", ylabel="Reward")
    s = moving_avg(h["success"], window) * 100
    ax[1].plot(np.arange(len(s)) + window, s, color="#2a9d8f", label="During training (with exploration)")
    ax[1].plot(h["eval_ep"], np.array(h["eval_success"]) * 100, "o-", color="#e76f51", label="Test maps (no exploration)")
    ax[1].set(title="Success rate", xlabel="Episode", ylabel="% of episodes reaching the goal", ylim=(0, 102))
    ax[1].legend(loc="lower right")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=6000)
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--density", type=float, default=0.18)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-frac", type=float, default=0.6, help="fraction of training spent lowering exploration")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/robot_dqn.npz")
    p.add_argument("--plot", default="assets/training_curve.png")
    a = p.parse_args()

    env = GridRobotEnv(a.size, a.density, gamma=a.gamma, seed=a.seed)
    eval_env = GridRobotEnv(a.size, a.density, gamma=a.gamma)
    agent = DQNAgent(env.obs_dim, env.n_actions, lr=a.lr, gamma=a.gamma, seed=a.seed)
    h = {"reward": [], "success": [], "eval_ep": [], "eval_success": []}
    best, t0 = -1.0, time.time()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    print(f"Training for {a.episodes} episodes on {a.size}x{a.size} maps "
          f"({int(a.density * 100)}% obstacles). Network input size: {env.obs_dim}")
    for ep in range(1, a.episodes + 1):
        eps = max(a.eps_end, 1.0 - (1.0 - a.eps_end) * ep / (a.eps_decay_frac * a.episodes))
        obs, total = env.reset(), 0.0
        while True:
            act = agent.act(obs, eps)
            nobs, r, done, truncated, _ = env.step(act)
            agent.remember(obs, act, r, nobs, done)
            agent.learn()
            obs, total = nobs, total + r
            if done or truncated:
                break
        h["reward"].append(total)
        h["success"].append(float(done))

        if ep % 100 == 0:
            print(f"ep {ep:5d} | reward {np.mean(h['reward'][-100:]):6.2f} | "
                  f"success {np.mean(h['success'][-100:]) * 100:5.1f}% | epsilon {eps:.2f} | {time.time() - t0:5.0f}s")
        if ep % a.eval_every == 0 or ep == a.episodes:
            m = evaluate(lambda o: agent.act(o, 0.0), eval_env, episodes=100, seed0=1_000_000)
            h["eval_ep"].append(ep); h["eval_success"].append(m["success_rate"])
            score = m["success_rate"] + 0.1 * m["path_efficiency"]
            tag = ""
            if score > best:
                best = score; agent.save(a.out, env); tag = "  <- new best, saved"
            print(f"   TEST on 100 unseen maps: success {m['success_rate'] * 100:.0f}%, "
                  f"path efficiency {m['path_efficiency']:.2f}{tag}")

    best_agent, _ = DQNAgent.load(a.out)
    final = evaluate(lambda o: best_agent.act(o, 0.0), eval_env, episodes=500)
    base = evaluate(random_policy(a.seed), eval_env, episodes=500)
    plot_history(h, a.plot)
    print("\n=== Final results on 500 unseen maps ===")
    print(f"Trained robot : success {final['success_rate'] * 100:.1f}% | avg steps {final['avg_steps']:.1f} | "
          f"path efficiency {final['path_efficiency']:.2f} | bumps/episode {final['bumps_per_episode']:.2f}")
    print(f"Random robot  : success {base['success_rate'] * 100:.1f}% | bumps/episode {base['bumps_per_episode']:.2f}")
    print(f"Model saved to {a.out}, plot saved to {a.plot}")


if __name__ == "__main__":
    main()
