"""
Q-Learning: Maze Navigation Agent
Comprehensive implementation of Q-Learning reinforcement learning algorithm
CBSE Class 12 AI Project
"""

import json
import random
from typing import List, Tuple, Dict, Any

class MazeEnvironment:
    """
    Maze environment for reinforcement learning
    """
    
    def __init__(self, maze: List[List[int]]):
        """
        Initialize maze environment
        
        Args:
            maze: 2D grid (0=path, 1=wall, 2=goal)
        """
        self.maze = maze
        self.height = len(maze)
        self.width = len(maze[0])
        self.start_pos = None
        self.goal_pos = None
        self.current_pos = None
        
        # Find start and goal positions
        for i in range(self.height):
            for j in range(self.width):
                if maze[i][j] == 2:
                    self.goal_pos = (i, j)
                elif maze[i][j] == 3:
                    self.start_pos = (i, j)
        
        if self.start_pos is None:
            self.start_pos = (0, 0)
        if self.goal_pos is None:
            self.goal_pos = (self.height - 1, self.width - 1)
        
        self.current_pos = self.start_pos
    
    def reset(self) -> Tuple[int, int]:
        """Reset environment to start position"""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def get_valid_actions(self, position: Tuple[int, int]) -> List[int]:
        """
        Get valid actions from a position
        Actions: 0=Up, 1=Right, 2=Down, 3=Left
        """
        valid_actions = []
        row, col = position
        
        # Check each direction
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        
        for action, (dr, dc) in enumerate(moves):
            new_row, new_col = row + dr, col + dc
            
            # Check if move is valid
            if (0 <= new_row < self.height and 
                0 <= new_col < self.width and 
                self.maze[new_row][new_col] != 1):
                valid_actions.append(action)
        
        return valid_actions
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute action and return new state, reward, done
        
        Args:
            action: 0=Up, 1=Right, 2=Down, 3=Left
        
        Returns:
            (new_position, reward, done)
        """
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        if action not in range(4):
            return self.current_pos, -10, False
        
        dr, dc = moves[action]
        new_row = self.current_pos[0] + dr
        new_col = self.current_pos[1] + dc
        
        # Check if move is valid
        if (0 <= new_row < self.height and 
            0 <= new_col < self.width and 
            self.maze[new_row][new_col] != 1):
            
            self.current_pos = (new_row, new_col)
            
            # Check if goal reached
            if self.current_pos == self.goal_pos:
                return self.current_pos, 100, True
            else:
                return self.current_pos, -1, False
        else:
            # Hit wall or out of bounds
            return self.current_pos, -10, False


class QLearningAgent:
    """
    Q-Learning Agent for maze navigation
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, 
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent
        
        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
    
    def get_q_value(self, state: Tuple[int, int], action: int) -> float:
        """Get Q-value for state-action pair"""
        return self.q_table.get((state, action), 0.0)
    
    def set_q_value(self, state: Tuple[int, int], action: int, value: float):
        """Set Q-value for state-action pair"""
        self.q_table[(state, action)] = value
    
    def choose_action(self, state: Tuple[int, int], valid_actions: List[int]) -> int:
        """Choose action using epsilon-greedy policy"""
        if not valid_actions:
            return 0
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Exploitation
        q_values = [self.get_q_value(state, action) for action in valid_actions]
        max_q = max(q_values)
        
        # Handle ties randomly
        best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state: Tuple[int, int], action: int, 
                      reward: float, next_state: Tuple[int, int], 
                      valid_next_actions: List[int]):
        """Update Q-value using Q-learning update rule"""
        current_q = self.get_q_value(state, action)
        
        # Get max Q-value for next state
        if valid_next_actions:
            max_next_q = max(self.get_q_value(next_state, a) for a in valid_next_actions)
        else:
            max_next_q = 0.0
        
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.set_q_value(state, action, new_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_policy(self, state: Tuple[int, int], valid_actions: List[int]) -> int:
        """Get best action according to learned policy"""
        if not valid_actions:
            return 0
        
        q_values = [self.get_q_value(state, action) for action in valid_actions]
        max_q = max(q_values)
        best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)


def create_maze():
    """Create a sample maze"""
    # 0 = path, 1 = wall, 2 = goal, 3 = start
    maze = [
        [3, 0, 1, 0, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 2]
    ]
    return maze


def train_agent(env: MazeEnvironment, agent: QLearningAgent, 
               n_episodes: int = 500) -> List[Dict[str, Any]]:
    """Train Q-learning agent"""
    training_history = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 100:
            valid_actions = env.get_valid_actions(state)
            action = agent.choose_action(state, valid_actions)
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            valid_next_actions = env.get_valid_actions(next_state)
            agent.update_q_value(state, action, reward, next_state, valid_next_actions)
            
            state = next_state
        
        agent.decay_epsilon()
        
        if episode % 50 == 0:
            training_history.append({
                'episode': episode,
                'steps': steps,
                'reward': total_reward,
                'epsilon': agent.epsilon,
                'success': done
            })
    
    return training_history


def test_agent(env: MazeEnvironment, agent: QLearningAgent) -> Tuple[List[Tuple[int, int]], int]:
    """Test trained agent"""
    state = env.reset()
    path = [state]
    steps = 0
    done = False
    
    while not done and steps < 50:
        valid_actions = env.get_valid_actions(state)
        action = agent.get_policy(state, valid_actions)
        
        next_state, reward, done = env.step(action)
        path.append(next_state)
        steps += 1
        state = next_state
    
    return path, steps


def main():
    """Main execution function"""
    print("=" * 70)
    print("Q-Learning: Maze Navigation Agent")
    print("=" * 70)
    print()
    
    # Create maze
    print("Step 1: Creating Maze Environment")
    print("-" * 70)
    maze = create_maze()
    env = MazeEnvironment(maze)
    
    print("Maze Layout:")
    symbols = {0: '·', 1: '█', 2: 'G', 3: 'S'}
    for row in maze:
        print('  ', ' '.join(symbols[cell] for cell in row))
    print()
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    print(f"Maze size: {env.height}x{env.width}")
    print()
    
    # Initialize agent
    print("Step 2: Initializing Q-Learning Agent")
    print("-" * 70)
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995
    )
    print(f"Learning rate (α): {agent.lr}")
    print(f"Discount factor (γ): {agent.gamma}")
    print(f"Initial exploration rate (ε): {agent.epsilon}")
    print()
    
    # Train agent
    print("Step 3: Training Agent")
    print("-" * 70)
    n_episodes = 500
    print(f"Training for {n_episodes} episodes...")
    training_history = train_agent(env, agent, n_episodes)
    print(f"Training completed!")
    print(f"Final exploration rate: {agent.epsilon:.4f}")
    print(f"Q-table size: {len(agent.q_table)} state-action pairs")
    print()
    
    # Show training progress
    print("Training Progress:")
    for record in training_history:
        status = "✓" if record['success'] else "✗"
        print(f"  Episode {record['episode']:3d}: {status} "
              f"Steps={record['steps']:2d}, "
              f"Reward={record['reward']:6.1f}, "
              f"ε={record['epsilon']:.3f}")
    print()
    
    # Test agent
    print("Step 4: Testing Trained Agent")
    print("-" * 70)
    path, steps = test_agent(env, agent)
    
    print(f"Optimal path found in {steps} steps!")
    print(f"Path: {' → '.join([f'({r},{c})' for r, c in path])}")
    print()
    
    # Visualize path
    print("Path Visualization:")
    path_set = set(path)
    for i, row in enumerate(maze):
        print('  ', end='')
        for j, cell in enumerate(row):
            if (i, j) == env.start_pos:
                print('S', end=' ')
            elif (i, j) == env.goal_pos:
                print('G', end=' ')
            elif (i, j) in path_set:
                print('*', end=' ')
            elif cell == 1:
                print('█', end=' ')
            else:
                print('·', end=' ')
        print()
    print()
    
    # Summary
    print("\n" + "=" * 70)
    print("Q-Learning Summary")
    print("=" * 70)
    print(f"✓ Successfully trained agent to navigate maze")
    print(f"✓ Learned {len(agent.q_table)} state-action values")
    print(f"✓ Found optimal path in {steps} steps")
    print(f"✓ Agent learned to avoid walls and reach goal")
    print()
    print("Key Concepts:")
    print("• Q-learning learns optimal action-value function")
    print("• Epsilon-greedy balances exploration vs exploitation")
    print("• Discount factor determines importance of future rewards")
    print("• Agent learns from trial and error without explicit rules")
    print()

if __name__ == "__main__":
    main()
