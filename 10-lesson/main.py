import numpy as np
import random
from q_learning import QLearningAgent
from grid_env import GridWorldEnv
from utils import print_policy, visualize_episode


def train_agent(
    env: GridWorldEnv,
    n_episodes: int = 5000,
    max_steps: int = 100,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.999,
    epsilon_min: float = 0.01,
    save_q_table: bool = True,
    q_table_filename: str = None,
) -> QLearningAgent:
    """
    Train Q-learning agent in the specified environment.

    Args:
        env: GridWorld environment
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        learning_rate: Learning rate (alpha)
        discount_factor: Discount factor (gamma)
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay rate
        epsilon_min: Minimum epsilon value
        save_q_table: Whether to save Q-table after training
        q_table_filename: Custom filename for Q-table (default: auto-generated)

    Returns:
        Trained QLearningAgent
    """

    # Initialize agent
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
    )

    # Training metrics
    episode_rewards = []
    episode_lengths = []

    print(f"Starting training for {n_episodes} episodes...")
    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print(f"States: {env.observation_space.n}, Actions: {env.action_space.n}")
    print(f"Obstacles: {np.sum(env.grid == 1)}")
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    print(f"Learning rate: {learning_rate}, Discount factor: {discount_factor}")
    print(
        f"Initial epsilon: {epsilon}, Epsilon decay: {epsilon_decay}, Min epsilon: {epsilon_min}"
    )
    print("=" * 70)

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update Q-table
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        # Decay epsilon after each episode
        agent.decay_epsilon()

        # Store metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Print progress every 500 episodes
        if (episode + 1) % 500 == 0:
            window = min(500, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-window:])
            avg_length = np.mean(episode_lengths[-window:])
            success_rate = sum(1 for r in episode_rewards[-window:] if r > 0) / window
            print(
                f"Episode {episode + 1}/{n_episodes} | "
                f"Avg Reward: {avg_reward:.3f} | "
                f"Avg Length: {avg_length:.1f} | "
                f"Success Rate: {success_rate:.2%} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    # Save Q-table
    if save_q_table:
        if q_table_filename is None:
            # Auto-generate filename with goal position
            n_obstacles = np.sum(env.grid == 1)
            goal_row, goal_col = env.goal_pos
            q_table_filename = f"q_table_grid{env.grid_size}_obs{n_obstacles}_goal{goal_row}_{goal_col}.npy"

        agent.save_q_table(q_table_filename)

    print("\nTraining completed!")
    return agent


def evaluate_agent(
    agent: QLearningAgent,
    env: GridWorldEnv,
    n_episodes: int = 100,
    visualize: bool = True,
) -> None:
    """
    Evaluate trained agent.

    Args:
        agent: Trained Q-learning agent
        env: GridWorld environment
        n_episodes: Number of evaluation episodes
        visualize: Whether to visualize one episode with real-time animation
    """
    total_rewards = []
    successful_episodes = 0
    episode_lengths = []

    print(f"\nEvaluating agent for {n_episodes} episodes...")
    print("=" * 70)

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            # Use greedy policy (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state
            steps += 1

        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        if total_reward > 0:
            successful_episodes += 1

    # Print evaluation results
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = successful_episodes / n_episodes

    print(f"Evaluation Results:")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Average Episode Length: {avg_length:.1f}")
    print(f"Success Rate: {success_rate:.2%} ({successful_episodes}/{n_episodes})")
    print("=" * 70)

    # Print learned policy
    action_symbols = ["↑", "↓", "←", "→"]  # UP, DOWN, LEFT, RIGHT
    print_policy(
        agent.q_table,
        env_shape=(env.grid_size, env.grid_size),
        action_symbols=action_symbols,
    )

    # Visualize one episode with real-time animation
    if visualize:
        print("\nVisualizing agent behavior...")
        trajectory, total_reward = visualize_episode(
            env, agent, max_steps=100, delay=0.5
        )


def main():
    """
    Main function to train and evaluate Q-learning agent in GridWorld.
    """
    print("Q-Learning Agent in GridWorld")
    print("=" * 70)

    # Configuration
    grid_size = 8
    n_obstacles = 0
    start_pos = (0, 0)

    # Generate random goal position (not on start position)
    available_positions = [
        (i, j)
        for i in range(grid_size)
        for j in range(grid_size)
        if (i, j) != start_pos
    ]
    goal_pos = random.choice(available_positions)

    print(
        f"\nCreating {grid_size}x{grid_size} GridWorld with {n_obstacles} obstacles..."
    )
    print(f"Start position: {start_pos}")
    print(f"Goal position: {goal_pos}")

    # Create environment
    env = GridWorldEnv(
        grid_size=grid_size,
        n_obstacles=n_obstacles,
        start_pos=start_pos,
        goal_pos=goal_pos,
        render_mode="human",
    )

    print("\nInitial grid:")
    env.render()

    # Train agent
    print("\n" + "=" * 70)
    agent = train_agent(
        env=env,
        n_episodes=5000,
        max_steps=200,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.01,
    )

    # Evaluate agent
    evaluate_agent(agent, env, n_episodes=100, visualize=True)

    print("\n" + "=" * 70)
    print("All done! Q-table saved:")
    goal_row, goal_col = goal_pos
    q_table_file = (
        f"q_table_grid{grid_size}_obs{n_obstacles}_goal{goal_row}_{goal_col}.npy"
    )
    print(f"  - {q_table_file}")
    print("=" * 70)
    print("\nTo load the Q-table later (for same goal position):")
    print(f"  agent = QLearningAgent(n_states={grid_size * grid_size}, n_actions=4)")
    print(f"  agent.load_q_table('{q_table_file}')")
    print("=" * 70)


if __name__ == "__main__":
    main()
