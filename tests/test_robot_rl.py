import os, sys, tempfile, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot_rl import GridRobotEnv, DQNAgent, evaluate
from robot_rl.nn import MLP


class TestEnv(unittest.TestCase):
    def test_maps_are_solvable_and_obs_shape(self):
        env = GridRobotEnv(seed=1)
        for s in range(30):
            obs = env.reset(seed=s)
            self.assertGreater(env.optimal, 0)
            self.assertEqual(obs.shape, (env.obs_dim,))

    def test_same_seed_same_map(self):
        env = GridRobotEnv()
        env.reset(seed=5); a = env.grid.copy()
        env.reset(seed=5); self.assertTrue((a == env.grid).all())

    def test_wall_bump_keeps_position(self):
        env = GridRobotEnv(seed=2)
        env.reset(seed=3)
        env.pos = (0, 0); env.grid[0, 0] = 0
        _, r, _, _, info = env.step(0)  # up from the top row
        self.assertTrue(info["collided"]); self.assertEqual(env.pos, (0, 0)); self.assertLess(r, 0)

    def test_following_shortest_path_reaches_goal(self):
        env = GridRobotEnv()
        for s in range(20):
            env.reset(seed=s)
            opt, total = env.optimal, 0.0
            done = False
            while not done:
                r, c = env.pos
                for a, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < env.size and 0 <= nc < env.size and env.dist[nr, nc] == env.dist[r, c] - 1:
                        break
                _, rew, done, _, _ = env.step(a); total += rew
            self.assertEqual(env.steps, opt); self.assertGreater(total, 0)


class TestLearning(unittest.TestCase):
    def test_network_learns_simple_targets(self):
        rng = np.random.default_rng(0)
        net = MLP([4, 32, 2], lr=5e-3)
        x = rng.normal(size=(256, 4)).astype(np.float32)
        a = rng.integers(0, 2, 256)
        y = np.where(a == 0, x[:, 0], -x[:, 1]).astype(np.float32)
        first = net.train_step(x, a, y)
        for _ in range(400):
            last = net.train_step(x, a, y)
        self.assertLess(last, first * 0.2)

    def test_save_load_roundtrip(self):
        env = GridRobotEnv(seed=0)
        agent = DQNAgent(env.obs_dim)
        obs = env.reset(seed=1)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.npz"); agent.save(p, env)
            loaded, cfg = DQNAgent.load(p)
        self.assertEqual(cfg["size"], env.size)
        np.testing.assert_allclose(agent.q.forward(obs[None]), loaded.q.forward(obs[None]), rtol=1e-5)

    def test_evaluate_returns_metrics(self):
        env = GridRobotEnv(seed=0)
        m = evaluate(lambda o: 0, env, episodes=5)
        self.assertIn("success_rate", m)


if __name__ == "__main__":
    unittest.main()
