"""5-state random walk (Sutton & Barto 2nd ed., Example 6.2). Standard library only.
States A..E = indices 0..4, start at C (2). Left terminal reward 0, right terminal reward +1, gamma = 1.
True values v_pi = 1/6 .. 5/6. Compares tabular TD(0) with constant-alpha (every-visit) MC."""
import math
import random

TRUE_V = [i / 6 for i in range(1, 6)]


def generate_episode(rng):
    s, trajectory = 2, []
    while 0 <= s <= 4:
        s_next = s + rng.choice((-1, 1))
        r = 1.0 if s_next == 5 else 0.0
        trajectory.append((s, r, s_next))
        s = s_next
    return trajectory


def rms_error(V):
    return math.sqrt(sum((V[i] - TRUE_V[i]) ** 2 for i in range(5)) / 5)


def td0_episode(V, trajectory, alpha, gamma=1.0):
    for s, r, s_next in trajectory:            # online: update after every step
        v_next = V[s_next] if 0 <= s_next <= 4 else 0.0   # V(terminal) = 0
        delta = r + gamma * v_next - V[s]       # TD error
        V[s] += alpha * delta


def mc_episode(V, trajectory, alpha, gamma=1.0):
    G, returns = 0.0, []
    for s, r, _ in reversed(trajectory):        # must wait for episode end
        G = r + gamma * G
        returns.append((s, G))
    for s, G in reversed(returns):
        V[s] += alpha * (G - V[s])


def run(update, alpha, n_episodes=100, n_runs=100, seed=0):
    rng = random.Random(seed)
    curve = [0.0] * (n_episodes + 1)
    for _ in range(n_runs):
        V = [0.5] * 5
        curve[0] += rms_error(V)
        for k in range(1, n_episodes + 1):
            update(V, generate_episode(rng), alpha)
            curve[k] += rms_error(V)
    return [c / n_runs for c in curve]


if __name__ == "__main__":
    checkpoints = [0, 1, 10, 25, 50, 100]
    print("method  alpha  " + "  ".join(f"ep{k:>3}" for k in checkpoints))
    for name, update, alphas in (("TD(0)", td0_episode, (0.05, 0.10, 0.15)),
                                 ("MC", mc_episode, (0.01, 0.02, 0.03, 0.04))):
        for a in alphas:
            c = run(update, a)
            print(f"{name:<6}  {a:<5}  " + "  ".join(f"{c[k]:.3f}" for k in checkpoints))
    # value estimates after 100 episodes, one run, TD(0) alpha=0.1
    rng, V = random.Random(1), [0.5] * 5
    for _ in range(100):
        td0_episode(V, generate_episode(rng), 0.1)
    print("TD(0) a=0.1 after 100 eps:", [round(v, 3) for v in V], "true:", [round(v, 3) for v in TRUE_V])
