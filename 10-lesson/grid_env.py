import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Optional
import time


class GridWorldEnv(gym.Env):
    """
    Custom Grid World environment for reinforcement learning.

    The agent starts at a starting position and must navigate to a goal position
    while avoiding obstacles. The agent can move in 4 directions: up, down, left, right.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 5,
        n_obstacles: int = 0,
        obstacle_positions: Optional[List[Tuple[int, int]]] = None,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        render_mode: Optional[str] = None,
        step_reward: float = -0.01,
        obstacle_reward: float = -1.0,
        goal_reward: float = 1.0
    ):
        """
        Initialize Grid World environment.

        Args:
            grid_size: Size of the square grid (grid_size x grid_size)
            n_obstacles: Number of random obstacles to place in the grid
            obstacle_positions: Optional list of specific obstacle positions [(row, col), ...]
            start_pos: Starting position (row, col). Default: (0, 0)
            goal_pos: Goal position (row, col). Default: (grid_size-1, grid_size-1)
            render_mode: Rendering mode ('human' or 'rgb_array')
            step_reward: Reward for each step
            obstacle_reward: Penalty for hitting obstacle or wall
            goal_reward: Reward for reaching goal
        """
        super().__init__()

        self.grid_size = grid_size
        self.n_obstacles = n_obstacles
        self.render_mode = render_mode

        # Rewards
        self.step_reward = step_reward
        self.obstacle_reward = obstacle_reward
        self.goal_reward = goal_reward

        # Set start and goal positions
        self.start_pos = start_pos if start_pos is not None else (0, 0)
        self.goal_pos = goal_pos if goal_pos is not None else (grid_size - 1, grid_size - 1)

        # Initialize grid with obstacles
        self.grid = np.zeros((grid_size, grid_size), dtype=int)

        # Place obstacles
        if obstacle_positions is not None:
            # Use specified obstacle positions
            for pos in obstacle_positions:
                if pos != self.start_pos and pos != self.goal_pos:
                    self.grid[pos] = 1
        elif n_obstacles > 0:
            # Generate random obstacles
            self._generate_random_obstacles(n_obstacles)

        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)

        # Observation space: flatten grid position to single integer
        # State is represented as: row * grid_size + col
        self.observation_space = spaces.Discrete(grid_size * grid_size)

        # Current agent position
        self.agent_pos = self.start_pos

    def _generate_random_obstacles(self, n_obstacles: int) -> None:
        """
        Generate random obstacles in the grid, avoiding start and goal positions.

        Args:
            n_obstacles: Number of obstacles to generate
        """
        placed = 0
        max_attempts = n_obstacles * 10  # Prevent infinite loop
        attempts = 0

        while placed < n_obstacles and attempts < max_attempts:
            row = np.random.randint(0, self.grid_size)
            col = np.random.randint(0, self.grid_size)
            pos = (row, col)

            # Don't place obstacle on start or goal
            if pos != self.start_pos and pos != self.goal_pos and self.grid[row, col] == 0:
                self.grid[row, col] = 1
                placed += 1

            attempts += 1

    def _get_state(self) -> int:
        """
        Convert 2D position to 1D state index.

        Returns:
            State as integer
        """
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]

    def _get_position(self, state: int) -> Tuple[int, int]:
        """
        Convert 1D state index to 2D position.

        Args:
            state: State as integer

        Returns:
            Position as (row, col) tuple
        """
        row = state // self.grid_size
        col = state % self.grid_size
        return (row, col)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[int, dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Initial state and info dict
        """
        super().reset(seed=seed)

        # Reset agent to start position
        self.agent_pos = self.start_pos

        return self._get_state(), {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)

        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        # Calculate new position based on action
        row, col = self.agent_pos

        if action == 0:  # UP
            new_pos = (row - 1, col)
        elif action == 1:  # DOWN
            new_pos = (row + 1, col)
        elif action == 2:  # LEFT
            new_pos = (row, col - 1)
        elif action == 3:  # RIGHT
            new_pos = (row, col + 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check if new position is valid
        new_row, new_col = new_pos

        # Check boundaries
        if new_row < 0 or new_row >= self.grid_size or new_col < 0 or new_col >= self.grid_size:
            # Hit wall - stay in place, get penalty
            reward = self.obstacle_reward
            terminated = False
        # Check obstacle
        elif self.grid[new_row, new_col] == 1:
            # Hit obstacle - stay in place, get penalty
            reward = self.obstacle_reward
            terminated = False
        else:
            # Valid move
            self.agent_pos = new_pos
            reward = self.step_reward

            # Check if reached goal
            if self.agent_pos == self.goal_pos:
                reward = self.goal_reward
                terminated = True
            else:
                terminated = False

        truncated = False
        info = {}

        return self._get_state(), reward, terminated, truncated, info

    def render(self):
        """
        Render the environment in human-readable format (ASCII).
        """
        if self.render_mode == "human":
            # Clear screen (works in most terminals)
            print("\033[2J\033[H", end="")

            print("=" * (self.grid_size * 2 + 1))
            for row in range(self.grid_size):
                print("|", end="")
                for col in range(self.grid_size):
                    pos = (row, col)
                    if pos == self.agent_pos:
                        print("A", end=" ")  # Agent
                    elif pos == self.goal_pos:
                        print("G", end=" ")  # Goal
                    elif self.grid[row, col] == 1:
                        print("#", end=" ")  # Obstacle
                    else:
                        print(".", end=" ")  # Empty
                print("|")
            print("=" * (self.grid_size * 2 + 1))
            print(f"Position: {self.agent_pos} | Goal: {self.goal_pos}")
            print()

    def render_episode_step(self, action: Optional[int] = None, reward: float = 0, delay: float = 0.3):
        """
        Render a single step with additional information and delay.

        Args:
            action: Action taken (optional)
            reward: Reward received
            delay: Delay in seconds before rendering next step
        """
        action_names = ["UP ↑", "DOWN ↓", "LEFT ←", "RIGHT →"]

        # Clear screen
        print("\033[2J\033[H", end="")

        print("=" * (self.grid_size * 2 + 1))
        for row in range(self.grid_size):
            print("|", end="")
            for col in range(self.grid_size):
                pos = (row, col)
                if pos == self.agent_pos:
                    print("A", end=" ")  # Agent
                elif pos == self.goal_pos:
                    print("G", end=" ")  # Goal
                elif self.grid[row, col] == 1:
                    print("#", end=" ")  # Obstacle
                else:
                    print(".", end=" ")  # Empty
            print("|")
        print("=" * (self.grid_size * 2 + 1))

        if action is not None:
            print(f"Action: {action_names[action]} | Reward: {reward:.2f}")
        print(f"Position: {self.agent_pos} | Goal: {self.goal_pos}")
        print()

        time.sleep(delay)

    def close(self):
        """
        Clean up resources.
        """
        pass
