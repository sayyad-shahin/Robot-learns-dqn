"""Test a trained robot on unseen maps.  Usage: python evaluate.py [--model models/robot_dqn.npz] [--episodes 500]"""
import argparse
from robot_rl import GridRobotEnv, DQNAgent, evaluate, random_policy

p = argparse.ArgumentParser()
p.add_argument("--model", default="models/robot_dqn.npz")
p.add_argument("--episodes", type=int, default=500)
a = p.parse_args()

agent, cfg = DQNAgent.load(a.model)
env = GridRobotEnv(**cfg)
res = evaluate(lambda o: agent.act(o, 0.0), env, a.episodes)
base = evaluate(random_policy(0), env, a.episodes)
print(f"Maps tested: {a.episodes}  ({cfg['size']}x{cfg['size']} grid)")
print(f"Trained robot: success {res['success_rate'] * 100:.1f}% | avg steps {res['avg_steps']:.1f} | "
      f"path efficiency {res['path_efficiency']:.2f} | bumps/episode {res['bumps_per_episode']:.2f}")
print(f"Random robot : success {base['success_rate'] * 100:.1f}% | bumps/episode {base['bumps_per_episode']:.2f}")
