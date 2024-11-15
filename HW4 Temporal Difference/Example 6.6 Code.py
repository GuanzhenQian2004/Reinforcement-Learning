import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
from tqdm import tqdm  # For displaying progress bars

# Constants and Parameters
WORLD_HEIGHT = 4
WORLD_WIDTH = 12
EPSILON = 0.1  # Probability for exploration (epsilon-greedy policy)
ALPHA = 0.5  # Step size (learning rate)
GAMMA = 1  # Discount factor for future rewards
ACTIONS = [0, 1, 2, 3]  # Available actions: 0 = Up, 1 = Down, 2 = Left, 3 = Right
START = [3, 0]  # Starting position
GOAL = [3, 11]  # Goal position

# Environment Class
class CliffWalkingEnv:
    """
    Environment for the Cliff Walking problem.
    """
    def __init__(self):
        self.height = WORLD_HEIGHT
        self.width = WORLD_WIDTH
        self.start = START
        self.goal = GOAL

    def step(self, state, action):
        """
        Take a step in the environment given the current state and action.
        Returns the next state and the reward.
        """
        i, j = state
        next_state = self.get_next_state(i, j, action)
        reward = self.get_reward(i, j, action, state)
        return next_state, reward

    def get_next_state(self, i, j, action):
        """
        Determine the next state based on the current state and action.
        """
        if action == 0:  # Up
            return [max(i - 1, 0), j]
        elif action == 1:  # Down
            return [min(i + 1, self.height - 1), j]
        elif action == 2:  # Left
            return [i, max(j - 1, 0)]
        elif action == 3:  # Right
            return [i, min(j + 1, self.width - 1)]
        else:
            raise ValueError("Invalid action")

    def get_reward(self, i, j, action, state):
        """
        Determine the reward for taking an action from the current state.
        Special penalty for falling off the cliff or stepping right from the start.
        """
        if (action == 1 and i == 2 and 1 <= j <= 10) or (action == 3 and state == START):
            return -100  # Penalty for falling off the cliff
        return -1  # Default step penalty

# Agent Class
class Agent:
    """
    Agent that interacts with the Cliff Walking environment using Sarsa or Q-Learning.
    """
    def __init__(self, env):
        self.env = env
        self.q_values = np.zeros((env.height, env.width, len(ACTIONS)))  # Initialize Q-values

    def choose_action(self, state):
        """
        Choose an action based on an epsilon-greedy policy.
        """
        if np.random.binomial(1, EPSILON) == 1:
            return np.random.choice(ACTIONS)  # Explore with probability EPSILON
        values = self.q_values[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])  # Exploit

    def update_q_sarsa(self, state, action, next_state, reward, next_action, step_size, expected=False):
        """
        Update Q-values using the Sarsa or Expected Sarsa update rule.
        """
        if not expected:
            target = self.q_values[next_state[0], next_state[1], next_action]
        else:
            target = self.calculate_expected_value(next_state)
        target *= GAMMA  # Discount future rewards
        self.q_values[state[0], state[1], action] += step_size * (reward + target - self.q_values[state[0], state[1], action])

    def calculate_expected_value(self, state):
        """
        Calculate the expected value for the Expected Sarsa algorithm.
        """
        q_next = self.q_values[state[0], state[1], :]
        best_actions = np.argwhere(q_next == np.max(q_next)).flatten()
        expected_value = 0.0
        for action in ACTIONS:
            if action in best_actions:
                expected_value += ((1.0 - EPSILON) / len(best_actions) + EPSILON / len(ACTIONS)) * q_next[action]
            else:
                expected_value += EPSILON / len(ACTIONS) * q_next[action]
        return expected_value

    def sarsa(self, step_size=ALPHA, expected=False):
        """
        Run one episode of the Sarsa or Expected Sarsa algorithm.
        """
        state = self.env.start
        action = self.choose_action(state)
        rewards = 0.0
        while state != self.env.goal:
            next_state, reward = self.env.step(state, action)
            next_action = self.choose_action(next_state)
            rewards += reward
            self.update_q_sarsa(state, action, next_state, reward, next_action, step_size, expected)
            state = next_state
            action = next_action
        return rewards

    def q_learning(self, step_size=ALPHA):
        """
        Run one episode of the Q-Learning algorithm.
        """
        state = self.env.start
        rewards = 0.0
        while state != self.env.goal:
            action = self.choose_action(state)
            next_state, reward = self.env.step(state, action)
            rewards += reward
            self.q_values[state[0], state[1], action] += step_size * (
                reward + GAMMA * np.max(self.q_values[next_state[0], next_state[1], :]) - self.q_values[state[0], state[1], action]
            )
            state = next_state
        return rewards

# Helper Function for Plotting
def plot_performance(performance, episodes, labels):
    """
    Plot the performance of the algorithms over episodes.
    """
    for method, label in enumerate(labels):
        plt.plot(range(episodes), performance[method, :], label=label)
    plt.xlabel('Episodes')
    plt.ylabel('Reward per Episode')
    plt.ylim(bottom=-100, top=0)
    plt.legend()
    plt.savefig('sarsa_qlearning_plot.png')
    plt.close()

# Main Function for Running the Experiment
def example_6_6():
    """
    Run the Cliff Walking experiment using Sarsa and Q-Learning, and plot the results.
    """
    env = CliffWalkingEnv()
    episodes = 500
    runs = 1000
    performance = np.zeros((2, episodes))
    for _ in tqdm(range(runs), desc="Runs"):
        sarsa_agent = Agent(env)
        q_learning_agent = Agent(env)
        for ep in range(episodes):
            performance[0, ep] += sarsa_agent.sarsa(step_size=0.5)
            performance[1, ep] += q_learning_agent.q_learning(step_size=0.5)
    performance /= runs  # Average performance over all runs
    plot_performance(performance, episodes, ['Sarsa', 'Q-Learning'])

# Entry Point
if __name__ == '__main__':
    example_6_6()
