"""
td0_random_walk.py
TD(0) vs constant-alpha Monte Carlo on the 5-state Random Walk
(Sutton & Barto, Reinforcement Learning: An Introduction, 2nd ed., Example 6.2 and Sec. 6.3).

Outputs, written to ./figures next to this file:
  fig1_td0_values.png      TD(0) estimates after 0, 1, 10, 100 episodes vs true values
  fig2_rms_td_vs_mc.png    RMS error vs episodes: TD(0) vs constant-alpha MC, 100 runs
  fig3_batch_td_vs_mc.png  batch TD(0) vs batch MC, 100 runs          (if MAKE_BATCH)
  td0_values.gif           TD(0) estimates evolving over 100 episodes  (if MAKE_GIF)
Requires only numpy and matplotlib.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                          # write files only, no GUI window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

SEED, N_RUNS, N_EPISODES, DPI = 2026, 100, 100, 180
MAKE_BATCH, MAKE_GIF = True, True
CHECKPOINTS = (1, 10, 25, 50, 100)
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Indices 0 and 6 are terminal; 1..5 are A..E. Every episode starts in C. gamma = 1.
N_STATES, START, LEFT, RIGHT = 7, 3, 0, 6
LABELS = ["Left end", "A", "B", "C", "D", "E", "Right end"]
TRUE_V = np.arange(1, 6) / 6.0                 # 1/6, 2/6, ..., 5/6
X = np.arange(5)
TD_COLORS = ["#7aaeea", "#2a78d6", "#1a4f94"]              # blue family = TD
MC_COLORS = ["#f6b08f", "#f08a5d", "#eb6834", "#b8461b"]   # orange family = MC

def generate_episode(rng):
    """Random walk from C until a terminal state; returns all visited states incl. the terminal."""
    s, path = START, [START]
    while s not in (LEFT, RIGHT):
        s += 1 if rng.random() < 0.5 else -1
        path.append(s)
    return path

def reward(s_next):
    return 1.0 if s_next == RIGHT else 0.0     # +1 only when leaving on the right

def init_values():
    v = np.full(N_STATES, 0.5)                 # initial guess V(s) = 0.5 for A..E
    v[LEFT] = v[RIGHT] = 0.0                   # value of a terminal state is 0
    return v

def rms(v):
    return np.sqrt(np.mean((v[1:-1] - TRUE_V) ** 2))

def td0_episode(v, path, alpha):
    """TD(0), online: V(S) <- V(S) + alpha * [R + gamma * V(S') - V(S)]."""
    deltas = []
    for s, s_next in zip(path[:-1], path[1:]):
        delta = reward(s_next) + v[s_next] - v[s]   # TD error with bootstrapped target
        v[s] += alpha * delta
        deltas.append(delta)
    return deltas

def mc_episode(v, path, alpha):
    """Constant-alpha every-visit MC: V(S) <- V(S) + alpha * [G - V(S)], after the episode."""
    g = reward(path[-1])                       # gamma = 1 and only the last reward is non-zero
    for s in path[:-1]:
        v[s] += alpha * (g - v[s])

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print("saved", OUT / name)

# Common random numbers: every method and step size sees the same 100 x 100 episodes.
rng = np.random.default_rng(SEED)
EPISODES = [[generate_episode(rng) for _ in range(N_EPISODES)] for _ in range(N_RUNS)]

# ---------- Figure 1: one run of TD(0) with alpha = 0.1 ----------
v = init_values()
history = [v[1:-1].copy()]
for ep, path in enumerate(EPISODES[0], start=1):
    deltas = td0_episode(v, path, alpha=0.1)
    history.append(v[1:-1].copy())
    if ep == 1:                                # worked example for the slides
        print("Episode 1 :", " -> ".join(LABELS[s] for s in path))
        print("TD errors :", np.round(deltas, 3))
        print("V(A..E)   :", np.round(history[1], 3))

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for k, c, m in zip((0, 1, 10, 100), ["#9cc3f0", "#5b9be6", "#2a78d6", "#1a4f94"], "osD^"):
    ax.plot(X, history[k], marker=m, color=c, lw=2, ms=7,
            label=f"after {k} episode{'' if k == 1 else 's'}")
ax.plot(X, TRUE_V, "k--", lw=2, label="true values")
ax.set(xticks=X, xticklabels=LABELS[1:-1], ylim=(0, 1), xlabel="State",
       ylabel="Estimated value V(s)", title="TD(0) on the 5-state Random Walk (α = 0.1)")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", frameon=False)
save(fig, "fig1_td0_values.png")

# ---------- Figure 2: RMS error of TD(0) vs constant-alpha MC ----------
def rms_curve(update, alpha):
    """RMS error over the 5 states after 0..N_EPISODES episodes, averaged over N_RUNS runs."""
    err = np.zeros((N_RUNS, N_EPISODES + 1))
    for r in range(N_RUNS):
        v = init_values()
        err[r, 0] = rms(v)
        for ep, path in enumerate(EPISODES[r], start=1):
            update(v, path, alpha)
            err[r, ep] = rms(v)
    return err.mean(axis=0)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for name, update, alphas, colors, ls in (("TD(0)", td0_episode, (0.05, 0.10, 0.15), TD_COLORS, "-"),
                                         ("MC", mc_episode, (0.01, 0.02, 0.03, 0.04), MC_COLORS, "--")):
    for a, c in zip(alphas, colors):
        curve = rms_curve(update, a)
        ax.plot(curve, color=c, lw=2, ls=ls, label=f"{name}, α = {a:.2f}")
        print(f"{name:5s} alpha={a:.2f}: " + "  ".join(f"ep{k}={curve[k]:.3f}" for k in CHECKPOINTS))
ax.set(xlim=(0, N_EPISODES), xlabel="Episodes (walks)", ylabel="RMS error, averaged over states",
       title=f"TD(0) vs constant-α MC (average of {N_RUNS} runs)")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend(ncol=2, fontsize=8, frameon=False)
save(fig, "fig2_rms_td_vs_mc.png")

# ---------- Figure 3 (optional): batch TD(0) vs batch MC ----------
# Repeating batch updates with a small alpha until convergence reaches a fixed point,
# computed here directly:
#   batch MC    -> average return observed after each state (best fit to the observed returns)
#   batch TD(0) -> certainty-equivalence estimate: solve V = R_hat + P_hat V for the
#                  Markov model estimated from the same episodes
def batch_curves():
    err_td, err_mc = np.zeros((N_RUNS, N_EPISODES)), np.zeros((N_RUNS, N_EPISODES))
    for r in range(N_RUNS):
        counts = np.zeros((N_STATES, N_STATES))   # transition counts s -> s'
        n, g_sum = np.zeros(N_STATES), np.zeros(N_STATES)
        for ep, path in enumerate(EPISODES[r]):
            g = reward(path[-1])
            for s, s_next in zip(path[:-1], path[1:]):
                counts[s, s_next] += 1
                n[s] += 1
                g_sum[s] += g
            seen = np.flatnonzero(n)              # non-terminal states visited so far
            v_mc, v_td = init_values(), init_values()
            v_mc[seen] = g_sum[seen] / n[seen]
            p = counts[seen] / n[seen, None]      # estimated transition probabilities
            v_td[seen] = np.linalg.solve(np.eye(len(seen)) - p[:, seen], p[:, RIGHT])
            err_td[r, ep], err_mc[r, ep] = rms(v_td), rms(v_mc)
    return err_td.mean(axis=0), err_mc.mean(axis=0)

if MAKE_BATCH:
    td_b, mc_b = batch_curves()
    ep_axis = np.arange(1, N_EPISODES + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ep_axis, td_b, color=TD_COLORS[1], lw=2, label="Batch TD(0)")
    ax.plot(ep_axis, mc_b, color=MC_COLORS[2], lw=2, ls="--", label="Batch MC")
    ax.set(xlim=(1, N_EPISODES), xlabel="Episodes (walks) in the batch",
           ylabel="RMS error, averaged over states",
           title=f"Batch training on the Random Walk (average of {N_RUNS} runs)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    save(fig, "fig3_batch_td_vs_mc.png")

# ---------- Optional GIF: TD(0) estimates over 100 episodes ----------
if MAKE_GIF:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(X, TRUE_V, "k--", lw=2, label="true values")
    (line,) = ax.plot(X, history[0], "o-", color=TD_COLORS[1], lw=2, ms=7, label="TD(0) estimate")
    ax.set(xticks=X, xticklabels=LABELS[1:-1], ylim=(0, 1), xlabel="State", ylabel="V(s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=False)

    def draw(k):
        line.set_ydata(history[k])
        ax.set_title(f"TD(0), α = 0.1, after {k} episodes")
        return (line,)

    anim = FuncAnimation(fig, draw, frames=len(history))
    anim.save(str(OUT / "td0_values.gif"), writer=PillowWriter(fps=8), dpi=100)
    plt.close(fig)
    print("saved", OUT / "td0_values.gif")
