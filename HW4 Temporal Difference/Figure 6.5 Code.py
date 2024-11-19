import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# Constants for the states and actions in the environment
STATE_A = 0  # Starting state (A)
STATE_B = 1  # Intermediate state (B)
STATE_TERMINAL = 2  # Terminal state
STATE_START = STATE_A  # Initial starting state

ACTION_A_RIGHT = 0  # Action to move right in state A
ACTION_A_LEFT = 1  # Action to move left in state A

EPSILON = 0.1  # Probability for exploration in the epsilon-greedy policy
ALPHA = 0.1  # Learning rate
GAMMA = 1.0  # Discount factor for future rewards

ACTIONS_B = range(0, 10)  # Possible actions in state B

class Environment:
    """Represents the environment with states, transitions, and rewards."""
    def __init__(self):
        self.state_actions = [[ACTION_A_RIGHT, ACTION_A_LEFT], ACTIONS_B]  # Available actions for each state
        self.transitions = [[STATE_TERMINAL, STATE_B], [STATE_TERMINAL] * len(ACTIONS_B)]  # State transitions

    def get_next_state(self, state, action):
        """Returns the next state given the current state and action."""
        return self.transitions[state][action]

    def get_reward(self, state):
        """Returns the reward for taking an action in a given state."""
        return 0 if state == STATE_A else np.random.normal(-0.1, 1)  # Reward for state B is sampled from a normal distribution

class Agent:
    """Represents an agent interacting with the environment using Q-learning algorithms."""
    def __init__(self, environment, use_double_q=False):
        self.env = environment
        self.use_double_q = use_double_q  # Flag to use Double Q-learning
        self.q1 = copy.deepcopy(self.initialize_q_values())  # Initialize Q-values
        self.q2 = copy.deepcopy(self.initialize_q_values()) if use_double_q else None  # Initialize second Q-table for Double Q-learning

    def initialize_q_values(self):
        """Initializes the Q-value table."""
        return [np.zeros(2), np.zeros(len(ACTIONS_B)), np.zeros(1)]  # Q-values for states A, B, and terminal

    def choose_action(self, state, q_values):
        """Chooses an action using the epsilon-greedy policy."""
        if np.random.binomial(1, EPSILON) == 1:
            # With probability EPSILON, choose a random action (exploration)
            return np.random.choice(self.env.state_actions[state])
        else:
            # With probability (1 - EPSILON), choose the action with the highest Q-value (exploitation)
            values = q_values[state]
            return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

    def q_learning_step(self):
        """Performs one episode of Q-learning or Double Q-learning."""
        state = STATE_START  # Start at state A
        left_count = 0  # Counter for tracking left actions in state A
        while state != STATE_TERMINAL:
            # Choose an action based on the Q-values
            if self.use_double_q:
                action = self.choose_action(state, [q1 + q2 for q1, q2 in zip(self.q1, self.q2)])
            else:
                action = self.choose_action(state, self.q1)

            # Track left actions taken in state A
            if state == STATE_A and action == ACTION_A_LEFT:
                left_count += 1

            # Get the reward and next state
            reward = self.env.get_reward(state)
            next_state = self.env.get_next_state(state, action)

            # Update Q-values
            if self.use_double_q:
                self.update_double_q_values(state, action, reward, next_state)
            else:
                self.update_q_values(self.q1, state, action, reward, next_state)

            state = next_state  # Move to the next state
        return left_count

    def update_q_values(self, q_values, state, action, reward, next_state):
        """Updates the Q-value table for regular Q-learning."""
        target = np.max(q_values[next_state])  # Target value for the update
        q_values[state][action] += ALPHA * (reward + GAMMA * target - q_values[state][action])  # Q-learning update rule

    def update_double_q_values(self, state, action, reward, next_state):
        """Updates the Q-value tables for Double Q-learning."""
        # Randomly select which Q-table to update
        if np.random.binomial(1, 0.5) == 1:
            active_q, target_q = self.q1, self.q2
        else:
            active_q, target_q = self.q2, self.q1

        # Choose the best action according to the active Q-table
        best_action = np.random.choice([action for action, value in enumerate(active_q[next_state]) if value == np.max(active_q[next_state])])
        target = target_q[next_state][best_action]  # Calculate the target
        active_q[state][action] += ALPHA * (reward + GAMMA * target - active_q[state][action])  # Double Q-learning update rule

class Experiment:
    """Conducts experiments comparing Q-learning and Double Q-learning."""
    def __init__(self, runs=1000, episodes=300):
        self.runs = runs
        self.episodes = episodes

    def run_experiment(self):
        """Runs the experiment and plots the results."""
        left_counts_q = np.zeros((self.runs, self.episodes))  # Track left actions for Q-learning
        left_counts_double_q = np.zeros((self.runs, self.episodes))  # Track left actions for Double Q-learning
        env = Environment()  # Create the environment

        for run in tqdm(range(self.runs)):  # Loop over multiple runs for statistical reliability
            agent_q = Agent(env, use_double_q=False)  # Q-learning agent
            agent_double_q = Agent(env, use_double_q=True)  # Double Q-learning agent
            for episode in range(self.episodes):  # Loop over episodes
                left_counts_q[run, episode] = agent_q.q_learning_step()
                left_counts_double_q[run, episode] = agent_double_q.q_learning_step()

        self.plot_results(left_counts_q, left_counts_double_q)

    def plot_results(self, left_counts_q, left_counts_double_q):
        """Plots the results of the experiment."""
        mean_left_counts_q = left_counts_q.mean(axis=0)  # Average left actions for Q-learning
        mean_left_counts_double_q = left_counts_double_q.mean(axis=0)  # Average left actions for Double Q-learning

        plt.plot(mean_left_counts_q, label='Q-Learning')
        plt.plot(mean_left_counts_double_q, label='Double Q-Learning')
        plt.plot(np.ones(self.episodes) * 0.05, label='Optimal')  # Optimal reference line

        plt.xlabel('Episodes')
        plt.ylabel('% Left Actions from A')  # Label for y-axis
        plt.yticks(np.linspace(0, 1, 5), ['0%', '25%', '50%', '75%', '100%'])  # Custom y-axis ticks and labels
        plt.legend()
        plt.savefig('figure_6_5.png')  # Save the plot
        plt.close()

if __name__ == '__main__':
    experiment = Experiment()
    experiment.run_experiment()
