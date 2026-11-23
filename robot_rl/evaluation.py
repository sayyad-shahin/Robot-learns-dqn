"""Measure how good a policy is on a fixed set of unseen maps."""
import numpy as np


def random_policy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(4))


def evaluate(policy, env, episodes=200, seed0=5_000_000):
    """Run `policy(obs) -> action` on maps seeded seed0..seed0+episodes-1."""
    wins, steps, eff, bumps = 0, [], [], 0
    for i in range(episodes):
        obs = env.reset(seed=seed0 + i)
        optimal = env.optimal
        while True:
            obs, _, done, truncated, info = env.step(policy(obs))
            bumps += info["collided"]
            if done or truncated:
                break
        if done:
            wins += 1
            steps.append(env.steps)
            eff.append(optimal / env.steps)
    return {
        "success_rate": wins / episodes,
        "avg_steps": float(np.mean(steps)) if steps else float("nan"),
        "path_efficiency": float(np.mean(eff)) if eff else 0.0,   # 1.0 = shortest possible path
        "bumps_per_episode": bumps / episodes,
    }
