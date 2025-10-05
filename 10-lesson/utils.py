import numpy as np
from typing import List, Tuple


def print_policy(
    q_table: np.ndarray, env_shape: tuple, action_symbols: List[str]
) -> None:
    """
    Print the learned policy as a grid showing the best action for each state.

    Args:
        q_table: Q-table with shape (n_states, n_actions)
        env_shape: Shape of the environment grid (rows, cols)
        action_symbols: List of symbols representing each action (e.g., ['←', '↓', '→', '↑'])
    """
    print("\nLearned Policy (best action for each state):")
    print("=" * 50)

    best_actions = np.argmax(q_table, axis=1).reshape(env_shape)

    for i in range(env_shape[0]):
        row = ""
        for j in range(env_shape[1]):
            action_idx = best_actions[i, j]
            row += action_symbols[action_idx] + " "
        print(row)

    print("=" * 50)


def visualize_episode(
    env, agent, max_steps: int = 100, delay: float = 0.3
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Visualize agent's behavior in a single episode with real-time rendering.

    Args:
        env: GridWorld environment
        agent: Trained Q-learning agent
        max_steps: Maximum steps per episode
        delay: Delay between steps in seconds

    Returns:
        Tuple of (trajectory, total_reward)
    """
    state, _ = env.reset()
    trajectory = [env.agent_pos]
    total_reward = 0
    done = False
    steps = 0

    print("\n" + "=" * 50)
    print("VISUALIZING AGENT EPISODE")
    print("=" * 50)

    # Initial render
    env.render_episode_step(action=None, reward=0, delay=delay)

    while not done and steps < max_steps:
        # Select action (greedy, no exploration)
        action = agent.select_action(state, training=False)

        # Take step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        trajectory.append(env.agent_pos)
        steps += 1

        # Render step
        env.render_episode_step(action=action, reward=reward, delay=delay)

        state = next_state

    print(f"\nEpisode finished in {steps} steps with total reward: {total_reward:.2f}")
    print("=" * 50)

    return trajectory, total_reward
