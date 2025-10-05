import numpy as np


class QLearningAgent:
    """
    Q-learning agent for reinforcement learning in discrete environments.

    Q-learning is a model-free reinforcement learning algorithm that learns
    the value of actions in different states without requiring a model of the environment.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        """
        Initialize Q-learning agent.

        Args:
            n_states: Number of states in the environment
            n_actions: Number of possible actions
            learning_rate: Learning rate (alpha) - how much new information overrides old
            discount_factor: Discount factor (gamma) - importance of future rewards
            epsilon: Initial exploration rate for epsilon-greedy policy
            epsilon_decay: Rate at which epsilon decreases after each episode
            epsilon_min: Minimum value for epsilon
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with zeros
        # Q[state, action] represents expected cumulative reward
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        During training, with probability epsilon, we explore (random action).
        Otherwise, we exploit (choose best known action).

        Args:
            state: Current state
            training: Whether in training mode (uses epsilon-greedy) or evaluation mode (greedy)

        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit: best action based on Q-values
            return np.argmax(self.q_table[state])

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        """
        Update Q-table using the Q-learning update rule (Bellman equation).

        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]

        Where:
        - Q(s,a) is the current Q-value for state s and action a
        - α is the learning rate
        - r is the reward received
        - γ is the discount factor
        - max(Q(s',a')) is the maximum Q-value for the next state s'

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after taking action
            done: Whether episode is finished
        """
        # Current Q-value
        current_q = self.q_table[state, action]

        # Maximum Q-value for next state (0 if episode is done)
        max_next_q = 0 if done else np.max(self.q_table[next_state])

        # Calculate new Q-value using Bellman equation
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        # Update Q-table
        self.q_table[state, action] = new_q

    def decay_epsilon(self) -> None:
        """
        Decay epsilon after each episode to gradually reduce exploration.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_values(self, state: int) -> np.ndarray:
        """
        Get Q-values for all actions in given state.

        Args:
            state: State to get Q-values for

        Returns:
            Array of Q-values for each action
        """
        return self.q_table[state]

    def save_q_table(self, filename: str) -> None:
        """
        Save Q-table to a file.

        Args:
            filename: Path to save the Q-table (should end with .npy)
        """
        np.save(filename, self.q_table)
        print(f"Q-table saved to '{filename}'")

    def load_q_table(self, filename: str) -> None:
        """
        Load Q-table from a file.

        Args:
            filename: Path to the Q-table file (should end with .npy)
        """
        self.q_table = np.load(filename)
        print(f"Q-table loaded from '{filename}'")
