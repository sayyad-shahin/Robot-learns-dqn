"""Draw and animate the robot with matplotlib."""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle

CMAP = ListedColormap(["#f3f7f8", "#2b3a42"])


def record_episode(agent, env, seed, hold=6):
    """Run one greedy episode and keep a snapshot of every step."""
    env.reset(seed=seed)
    frames = []

    def snap(status):
        frames.append({"grid": env.grid.copy(), "goal": env.goal, "start": env.start,
                       "pos": env.pos, "trail": list(env.trail), "steps": env.steps, "status": status})

    snap("thinking...")
    obs = env.reset(seed=seed)
    snap("thinking...")
    while True:
        obs, _, done, truncated, _ = env.step(agent.act(obs, 0.0))
        if done or truncated:
            break
        snap("exploring")
    status = "Goal reached!" if done else "Out of time"
    for _ in range(hold):
        snap(status)
    return frames


def draw(ax, f, env, title):
    n, half, view = env.size, env.half, env.view
    ax.clear()
    ax.imshow(f["grid"], cmap=CMAP, extent=(0, n, n, 0), vmin=0, vmax=1)
    for k in range(n + 1):
        ax.plot([0, n], [k, k], color="#d5dfe2", lw=0.5)
        ax.plot([k, k], [0, n], color="#d5dfe2", lw=0.5)
    r, c = f["pos"]
    ax.add_patch(Rectangle((c - half, r - half), view, view, fill=False, ec="#f4a261", lw=1.5, ls="--"))
    if len(f["trail"]) > 1:
        ax.plot([p[1] + .5 for p in f["trail"]], [p[0] + .5 for p in f["trail"]], color="#7aa7c7", lw=2, alpha=.9)
    ax.plot(f["start"][1] + .5, f["start"][0] + .5, "s", color="#8d99ae", ms=9)
    ax.plot(f["goal"][1] + .5, f["goal"][0] + .5, "*", color="#2dc653", ms=22, mec="#1b7a35")
    ax.add_patch(Circle((c + .5, r + .5), 0.33, fc="#e63946", ec="white", lw=1.5, zorder=5))
    ax.set_xlim(-0.01, n + 0.01); ax.set_ylim(n + 0.01, -0.01)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title(f"{title}  |  step {f['steps']}  |  {f['status']}", fontsize=10)


def animate(all_frames, env, fps=6, save=None, show=True):
    fig, ax = plt.subplots(figsize=(4.4, 4.6), dpi=70 if save else 100)
    flat = [(i + 1, fr) for i, ep in enumerate(all_frames) for fr in ep]

    def update(k):
        i, fr = flat[k]
        draw(ax, fr, env, f"Map {i}/{len(all_frames)}")

    anim = FuncAnimation(fig, update, frames=len(flat), interval=1000 / fps, repeat=False)
    if save:
        anim.save(save, writer=PillowWriter(fps=fps))
        print(f"Saved animation to {save}")
    if show:
        plt.show()
    plt.close(fig)
