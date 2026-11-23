"""Watch the trained robot navigate new maps.
   python demo.py                       -> opens an animated window
   python demo.py --save assets/demo.gif --no-show
"""
import argparse
import matplotlib

p = argparse.ArgumentParser()
p.add_argument("--model", default="models/robot_dqn.npz")
p.add_argument("--maps", type=int, default=4, help="how many maps to play")
p.add_argument("--seed", type=int, default=2024, help="first map seed")
p.add_argument("--fps", type=int, default=6)
p.add_argument("--save", default=None, help="save the animation as a .gif")
p.add_argument("--no-show", action="store_true", help="do not open a window")
a = p.parse_args()
if a.no_show:
    matplotlib.use("Agg")

from robot_rl import GridRobotEnv, DQNAgent
from robot_rl.viz import record_episode, animate

agent, cfg = DQNAgent.load(a.model)
env = GridRobotEnv(**cfg)
episodes = [record_episode(agent, env, a.seed + i) for i in range(a.maps)]
animate(episodes, env, fps=a.fps, save=a.save, show=not a.no_show)
