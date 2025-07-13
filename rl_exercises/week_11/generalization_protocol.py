"""
Week 11 Level 1: Generalization Evaluation Protocol
This script uses CARL's CartPole environment with two context features (gravity, pole_length).
It partitions the context space into training and test sets, trains an agent, evaluates performance,
and plots results for in-distribution and out-of-distribution generalization.
"""
import numpy as np
import matplotlib.pyplot as plt
from carl.envs import CARLCartPole
import gym

# Context features to vary
gravity_range_train = np.linspace(9.8, 11.0, 3)  # Training gravity values
pole_length_range_train = np.linspace(0.5, 0.7, 3)  # Training pole length values

gravity_range_test = np.linspace(12.0, 14.0, 3)  # Test gravity values
pole_length_range_test = np.linspace(0.8, 1.0, 3)  # Test pole length values

# Helper to create context dicts
def create_contexts(gravity_range, pole_length_range):
    return [
        {"gravity": g, "pole_length": l}
        for g in gravity_range for l in pole_length_range
    ]

train_contexts = create_contexts(gravity_range_train, pole_length_range_train)
test_contexts = create_contexts(gravity_range_test, pole_length_range_test)

# Simple agent: random policy for demonstration (replace with your agent)
def evaluate_agent(env, n_episodes=5):
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()
            # Gymnasium API returns: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)

# Evaluate on all contexts
def evaluate_on_contexts(contexts, label):
    results = np.zeros((len(gravity_range_train if label=="train" else gravity_range_test),
                       len(pole_length_range_train if label=="train" else pole_length_range_test)))
    for i, g in enumerate(gravity_range_train if label=="train" else gravity_range_test):
        for j, l in enumerate(pole_length_range_train if label=="train" else pole_length_range_test):
            env = CARLCartPole(context={"gravity": g, "pole_length": l})
            avg_reward = evaluate_agent(env)
            results[i, j] = avg_reward
    return results

train_results = evaluate_on_contexts(train_contexts, "train")
test_results = evaluate_on_contexts(test_contexts, "test")

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, results, title, grav, pole in zip(
    axs, [train_results, test_results], ["Train Contexts", "Test Contexts"],
    [gravity_range_train, gravity_range_test], [pole_length_range_train, pole_length_range_test]
):
    im = ax.imshow(results, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Pole Length")
    ax.set_ylabel("Gravity")
    ax.set_xticks(range(len(pole)))
    ax.set_xticklabels([f"{l:.2f}" for l in pole])
    ax.set_yticks(range(len(grav)))
    ax.set_yticklabels([f"{g:.2f}" for g in grav])
    fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("generalization_plot.png")
plt.show()

