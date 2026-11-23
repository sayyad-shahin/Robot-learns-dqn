"""The robot's world: a grid with random obstacles and a goal.

The robot cannot see the whole map. It only senses:
  * a small window around itself (which cells are blocked),
  * a short-term memory of cells it already visited in that window,
  * the direction to the goal (like a compass / beacon).
It has to learn from rewards how to use this information to reach the goal.
"""
from collections import deque
import numpy as np

MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # up, down, left, right (row, col)
ACTION_NAMES = ("up", "down", "left", "right")

STEP_COST = -0.02       # small cost for every move: be quick
COLLISION_COST = -0.3   # bumping into a wall or obstacle
GOAL_REWARD = 5.0       # reaching the goal
SHAPING = 0.2           # potential-based hint for moving closer to the goal


class GridRobotEnv:
    def __init__(self, size=10, obstacle_density=0.18, view=5, max_steps=None,
                 gamma=0.97, seed=None):
        if view % 2 == 0:
            raise ValueError("view must be an odd number")
        self.size = size
        self.density = obstacle_density
        self.view = view
        self.half = view // 2
        self.gamma = gamma
        self.max_steps = max_steps or 5 * size
        self.n_actions = 4
        self.obs_dim = view * view * 2 + 2
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _bfs(grid, goal):
        """Shortest path length from every free cell to the goal (-1 = unreachable)."""
        n = grid.shape[0]
        dist = np.full(grid.shape, -1, dtype=np.int32)
        dist[goal] = 0
        queue = deque([goal])
        while queue:
            r, c = queue.popleft()
            for dr, dc in MOVES:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == 0 and dist[nr, nc] < 0:
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))
        return dist

    def reset(self, seed=None):
        """Create a new random map. Passing a seed gives the same map every time."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        n = self.size
        while True:
            grid = (rng.random((n, n)) < self.density).astype(np.int8)
            free = np.argwhere(grid == 0)
            if len(free) < 2:
                continue
            start = tuple(int(v) for v in free[rng.integers(len(free))])
            goal = tuple(int(v) for v in free[rng.integers(len(free))])
            if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < n // 2:
                continue
            dist = self._bfs(grid, goal)
            if dist[start] > 0:  # reachable and not the same cell
                break
        self.grid, self.start, self.goal, self.dist = grid, start, goal, dist
        self.optimal = int(dist[start])
        self._padded = np.pad(grid, self.half, constant_values=1)  # outside = wall
        self.pos = start
        self.visits = np.zeros((n, n), dtype=np.int32)
        self.visits[start] += 1
        self.steps = 0
        self.trail = [start]
        return self._obs()

    def _obs(self):
        r, c = self.pos
        v = self.view
        blocked = self._padded[r:r + v, c:c + v].astype(np.float32)
        visited = np.pad(self.visits, self.half)[r:r + v, c:c + v]
        visited = np.minimum(visited, 3).astype(np.float32) / 3.0
        goal_vec = np.array([(self.goal[0] - r) / self.size,
                             (self.goal[1] - c) / self.size], dtype=np.float32)
        return np.concatenate([blocked.ravel(), visited.ravel(), goal_vec])

    # ---- gym-like API --------------------------------------------------
    def step(self, action):
        r, c = self.pos
        nr, nc = r + MOVES[action][0], c + MOVES[action][1]
        collided = not (0 <= nr < self.size and 0 <= nc < self.size) or bool(self.grid[nr, nc])
        new = (r, c) if collided else (int(nr), int(nc))

        reward = STEP_COST + SHAPING * (self.dist[r, c] - self.gamma * self.dist[new])
        if collided:
            reward += COLLISION_COST
        reached = new == self.goal
        if reached:
            reward += GOAL_REWARD

        self.pos = new
        self.visits[new] += 1
        self.trail.append(new)
        self.steps += 1
        truncated = (self.steps >= self.max_steps) and not reached
        info = {"collided": collided, "success": reached}
        return self._obs(), float(reward), reached, truncated, info
