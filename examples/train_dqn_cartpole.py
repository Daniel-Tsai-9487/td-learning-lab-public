"""
train_dqn_cartpole.py  (optional bridge demo: "the DQN loss is a TD error")
Trains SB3 DQN on CartPole-v1 on the CPU, logs train/loss to CSV + TensorBoard,
saves / reloads / evaluates the model, and saves a loss + return figure and a short GIF.
Hyperparameters: RL Baselines3 Zoo, hyperparams/dqn.yml, entry "CartPole-v1".
"""
import csv
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

SEED, TOTAL_STEPS = 0, 50_000
ROOT = Path(__file__).resolve().parent
LOG_DIR, MODEL_PATH, FIG_DIR = ROOT / "logs" / "dqn_cartpole", ROOT / "models" / "dqn_cartpole", ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

model = DQN(
    "MlpPolicy", gym.make("CartPole-v1"), learning_rate=2.3e-3, batch_size=64,
    buffer_size=100_000, learning_starts=1_000, gamma=0.99, target_update_interval=10,
    train_freq=256, gradient_steps=128, exploration_fraction=0.16, exploration_final_eps=0.04,
    policy_kwargs=dict(net_arch=[256, 256]), device="cpu", seed=SEED,
)
# drop "tensorboard" from the list if the tensorboard package is not installed
model.set_logger(configure(str(LOG_DIR), ["stdout", "csv", "tensorboard"]))
model.learn(total_timesteps=TOTAL_STEPS, log_interval=10)
model.save(MODEL_PATH)                                   # -> models/dqn_cartpole.zip

model = DQN.load(MODEL_PATH, device="cpu")               # save / load round trip
mean_r, std_r = evaluate_policy(model, Monitor(gym.make("CartPole-v1")),
                                n_eval_episodes=10, deterministic=True)
print(f"Greedy policy, 10 episodes: return {mean_r:.1f} +/- {std_r:.1f} (maximum 500)")

# train/loss = Huber loss of the TD error; plot it next to the episode return
loss_x, loss_y, ret_x, ret_y = [], [], [], []
with open(LOG_DIR / "progress.csv", newline="") as f:
    for row in csv.DictReader(f):
        t = float(row["time/total_timesteps"])
        if row.get("train/loss"):
            loss_x.append(t); loss_y.append(float(row["train/loss"]))
        if row.get("rollout/ep_rew_mean"):
            ret_x.append(t); ret_y.append(float(row["rollout/ep_rew_mean"]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
ax1.plot(loss_x, loss_y, color="#2a78d6", lw=2)
ax1.set(xlabel="Environment steps", ylabel="train/loss (Huber of TD error)", title="DQN loss = TD error")
ax2.plot(ret_x, ret_y, color="#eb6834", lw=2)
ax2.set(xlabel="Environment steps", ylabel="Mean episode return", title="CartPole-v1 return (max 500)")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_dqn_loss_return.png", dpi=180)

# Short GIF of the trained agent (backup visual for the slides)
env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, _ = env.reset(seed=SEED)
frames = []
for _ in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(int(action))
    frames.append(Image.fromarray(env.render()))
    if terminated or truncated:
        break
frames = frames[::2]
frames[0].save(FIG_DIR / "dqn_cartpole.gif", save_all=True, append_images=frames[1:], duration=66, loop=0)
env.close()
