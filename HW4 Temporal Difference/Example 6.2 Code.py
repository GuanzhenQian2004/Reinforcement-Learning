import numpy as np
import matplotlib

# Setting up non-interactive Matplotlib backend for saving plots
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================#
# Constants and Initialization#
# ============================#

# Initializing state values
VALUES = np.zeros(7)
VALUES[1:6] = 0.5
VALUES[6] = 1

# True value function for comparison
TRUE_VALUE = np.zeros(7)
TRUE_VALUE[1:6] = np.arange(1, 6) / 6.0
TRUE_VALUE[6] = 1

# Action constants
ACTION_LEFT = 0
ACTION_RIGHT = 1


# ===========================#
# Random Walk Environment    #
# ===========================#
class RandomWalkEnvironment:
    def __init__(self):
        # Define start state and terminal states
        self.start_state = 3
        self.left_terminal_state = 0
        self.right_terminal_state = 6
        self.actions = [ACTION_LEFT, ACTION_RIGHT]
        self.reset()

    def reset(self):
        """Reset the environment to the initial state."""
        self.current_state = self.start_state

    def step(self, action):
        """Take an action and transition to the next state."""
        if action == ACTION_LEFT:
            next_state = self.current_state - 1
        elif action == ACTION_RIGHT:
            next_state = self.current_state + 1
        else:
            raise ValueError("Invalid action")

        reward = 0
        done = False
        # Check if terminal state is reached
        if next_state == self.left_terminal_state or next_state == self.right_terminal_state:
            done = True
        self.current_state = next_state
        return next_state, reward, done


# =============================#
# Value Function Representation#
# =============================#
class ValueFunction:
    def __init__(self, initial_values=None):
        # Initialize values, default values provided for states 1-6
        if initial_values is None:
            self.values = np.zeros(7)
            self.values[1:6] = 0.5
            self.values[6] = 1
        else:
            self.values = initial_values.copy()

    def update(self, state, delta):
        """Update the value of a given state by delta."""
        self.values[state] += delta

    def get_value(self, state):
        """Retrieve the value of a specific state."""
        return self.values[state]


# =========================================#
# Agent Logic for Interacting with the Env #
# =========================================#
class Agent:
    def __init__(self, env, value_function, alpha=0.1):
        # Environment and value function setup
        self.env = env
        self.value_function = value_function
        self.alpha = alpha  # Learning rate

    def choose_action(self):
        """Randomly choose an action."""
        return np.random.choice(self.env.actions)

    def temporal_difference(self, batch=False):
        """Perform TD learning for value estimation."""
        trajectory = []
        rewards = []
        self.env.reset()
        state = self.env.current_state
        trajectory.append(state)

        while True:
            action = self.choose_action()
            next_state, reward, done = self.env.step(action)
            trajectory.append(next_state)
            rewards.append(reward)
            if not batch:
                # Temporal Difference update rule
                delta = self.alpha * (
                        reward + self.value_function.get_value(next_state) - self.value_function.get_value(state)
                )
                self.value_function.update(state, delta)
            if done:
                break
            state = next_state
        return trajectory, rewards

    def monte_carlo(self, batch=False):
        """Perform Monte Carlo learning for value estimation."""
        trajectory = []
        self.env.reset()
        state = self.env.current_state
        trajectory.append(state)

        while True:
            action = self.choose_action()
            next_state, reward, done = self.env.step(action)
            trajectory.append(next_state)
            if done:
                # Set return value based on terminal state
                returns = 1.0 if next_state == self.env.right_terminal_state else 0.0
                break
            state = next_state

        if not batch:
            # Update state values for non-batch MC
            for s in trajectory[:-1]:
                delta = self.alpha * (returns - self.value_function.get_value(s))
                self.value_function.update(s, delta)
        rewards = [returns] * (len(trajectory) - 1)
        return trajectory, rewards


# ===========================#
# State Value Computation    #
# ===========================#
def compute_state_value():
    """Compute and plot estimated state values using TD learning."""
    episodes_to_plot = [0, 1, 10, 100]
    value_function = ValueFunction()
    env = RandomWalkEnvironment()
    agent = Agent(env, value_function)
    plt.figure(1)

    for i in range(1, max(episodes_to_plot) + 1):
        agent.temporal_difference()
        if i in episodes_to_plot:
            # Plot state values at specified episodes
            plt.plot(("A", "B", "C", "D", "E"), value_function.values[1:6], label=f'{i} episodes')

    # Plot true values for comparison
    plt.plot(("A", "B", "C", "D", "E"), TRUE_VALUE[1:6], label='True values')
    plt.xlabel('State')
    plt.ylabel('Estimated Value')
    plt.legend()


# ===========================#
# RMS Error Calculation      #
# ===========================#
def rms_error():
    """Compute and plot RMS error for different learning rates."""
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    episodes = 100 + 1
    runs = 100

    for alpha in td_alphas + mc_alphas:
        total_errors = np.zeros(episodes)
        method = 'TD' if alpha in td_alphas else 'MC'
        linestyle = 'solid' if method == 'TD' else 'dashdot'

        for _ in tqdm(range(runs)):
            value_function = ValueFunction()
            env = RandomWalkEnvironment()
            agent = Agent(env, value_function, alpha=alpha)
            errors = []

            for i in range(episodes):
                error = np.sqrt(np.sum((value_function.values - TRUE_VALUE) ** 2) / 5.0)
                errors.append(error)
                # Perform either TD or MC learning
                agent.temporal_difference() if method == 'TD' else agent.monte_carlo()
            total_errors += np.array(errors)

        total_errors /= runs
        plt.plot(total_errors, linestyle=linestyle, label=f'{method}, $\\alpha$ = {alpha:.02f}')
    plt.xlabel('Walks/Episodes')
    plt.ylabel('Empirical RMS error, averaged over states')
    plt.legend()


# ===========================#
# Batch Updating for Methods #
# ===========================#
def batch_updating(method, episodes, alpha=0.001):
    """Perform batch updating using either TD or MC."""
    runs = 100
    total_errors = np.zeros(episodes)

    for _ in tqdm(range(runs)):
        value_function = ValueFunction()
        value_function.values[1:6] = -1  # Initialize values
        env = RandomWalkEnvironment()
        agent = Agent(env, value_function, alpha=alpha)
        errors = []
        trajectories = []
        rewards_list = []

        for ep in range(episodes):
            trajectory, rewards = agent.temporal_difference(batch=True) if method == 'TD' else agent.monte_carlo(
                batch=True)
            trajectories.append(trajectory)
            rewards_list.append(rewards)

            # Batch update loop
            while True:
                updates = np.zeros(7)
                for trajectory, rewards in zip(trajectories, rewards_list):
                    for i in range(len(trajectory) - 1):
                        state = trajectory[i]
                        target = rewards[i] + value_function.get_value(trajectory[i + 1]) if method == 'TD' else \
                        rewards[i]
                        updates[state] += target - value_function.get_value(state)
                updates *= alpha
                if np.sum(np.abs(updates)) < 1e-3:
                    break
                for s in range(7):
                    value_function.update(s, updates[s])

            error = np.sqrt(np.sum((value_function.values - TRUE_VALUE) ** 2) / 5.0)
            errors.append(error)
        total_errors += np.array(errors)

    total_errors /= runs
    return total_errors


# ===========================#
# Main Function              #
# ===========================#
def example_6_2():
    """Plot state value estimates and RMS error."""
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    compute_state_value()
    plt.subplot(2, 1, 2)
    rms_error()
    plt.tight_layout()
    plt.savefig('example_6_2.png')
    plt.close()


if __name__ == '__main__':
    example_6_2()
