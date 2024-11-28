import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


# Environment Class
class Environment:
    """
    Represents the environment for the n-step TD method.
    Handles the state transitions, rewards, and true value initialization.
    """

    def __init__(self, n_states=19, start_state=10):
        self.n_states = n_states  # Number of states in the environment (excluding terminal states)
        self.start_state = start_state  # Starting state for each episode
        self.end_states = [0, n_states + 1]  # Two terminal states
        self.true_value = np.arange(-20, 22, 2) / 20.0  # True values for all states
        self.true_value[0] = self.true_value[-1] = 0  # Terminal states have a value of 0

    def step(self, state):
        """
        Simulates one step in the environment.
        Chooses the next state randomly and returns the reward based on the transition.

        Args:
            state (int): Current state.

        Returns:
            next_state (int): Next state after taking the step.
            reward (int): Reward associated with the transition.
        """
        if np.random.binomial(1, 0.5) == 1:
            next_state = state + 1  # Move right
        else:
            next_state = state - 1  # Move left

        # Assign reward based on reaching terminal states
        if next_state == 0:
            reward = -1
        elif next_state == self.n_states + 1:
            reward = 1
        else:
            reward = 0  # Intermediate states give no reward

        return next_state, reward


# Agent Class
class Agent:
    """
    Represents the agent that learns using the n-step Temporal Difference (TD) method.
    Interacts with the environment to update state values.
    """

    def __init__(self, env, gamma=1):
        self.env = env  # Reference to the environment
        self.gamma = gamma  # Discount factor for future rewards

    def temporal_difference(self, value, n, alpha):
        """
        Performs the n-step TD update for a single episode.

        Args:
            value (np.ndarray): State value estimates to be updated.
            n (int): Number of steps to look ahead for updates.
            alpha (float): Learning rate for updating state values.
        """
        state = self.env.start_state  # Start from the initial state
        states = [state]  # Track visited states
        rewards = [0]  # Track rewards for visited states
        time = 0  # Time step
        T = float('inf')  # Length of the episode, starts as infinity

        while True:
            time += 1

            # If episode has not ended, take a step in the environment
            if time < T:
                next_state, reward = self.env.step(state)
                states.append(next_state)
                rewards.append(reward)

                # If terminal state is reached, set episode length
                if next_state in self.env.end_states:
                    T = time

            # Time to perform an update
            update_time = time - n
            if update_time >= 0:
                # Compute the return (cumulative discounted reward)
                returns = 0.0
                for t in range(update_time + 1, min(T, update_time + n) + 1):
                    returns += pow(self.gamma, t - update_time - 1) * rewards[t]
                # If not at terminal state, add value estimate of n-steps ahead state
                if update_time + n <= T:
                    returns += pow(self.gamma, n) * value[states[update_time + n]]
                state_to_update = states[update_time]
                # Update state value using the TD error
                if state_to_update not in self.env.end_states:
                    value[state_to_update] += alpha * (returns - value[state_to_update])

            # If the episode has finished, break the loop
            if update_time == T - 1:
                break

            state = next_state  # Move to the next state

    def run_experiment(self, steps, alphas, episodes, runs):
        """
        Runs the n-step TD experiments for various step sizes and learning rates.

        Args:
            steps (np.ndarray): Array of step sizes (n) to test.
            alphas (np.ndarray): Array of learning rates to test.
            episodes (int): Number of episodes for each combination.
            runs (int): Number of independent runs for averaging results.

        Returns:
            errors (np.ndarray): RMS errors for each combination of step size and alpha.
        """
        errors = np.zeros((len(steps), len(alphas)))  # Store RMS errors

        for run in tqdm(range(runs)):  # Repeat for multiple runs to average results
            for step_ind, step in enumerate(steps):
                for alpha_ind, alpha in enumerate(alphas):
                    value = np.zeros(self.env.n_states + 2)  # Initialize state values
                    for ep in range(episodes):
                        self.temporal_difference(value, step, alpha)  # Perform n-step TD
                        # Compute RMS error between estimated and true values
                        errors[step_ind, alpha_ind] += np.sqrt(
                            np.sum(np.power(value - self.env.true_value, 2)) / self.env.n_states
                        )

        # Average errors over all runs and episodes
        errors /= episodes * runs
        return errors


# Utility Functions
def plot_results(errors, steps, alphas):
    """
    Plots the results of the experiment as in Figure 7.2.

    Args:
        errors (np.ndarray): RMS errors for different step sizes and learning rates.
        steps (np.ndarray): Step sizes (n) tested in the experiment.
        alphas (np.ndarray): Learning rates tested in the experiment.
    """
    for i in range(len(steps)):
        plt.plot(alphas, errors[i, :], label=f'n = {steps[i]}')
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])  # Set y-axis limits
    plt.legend()
    plt.savefig('figure_7_2.png')  # Save the plot to a file
    plt.close()


# Main Execution
if __name__ == '__main__':
    # Initialize the environment and agent
    env = Environment()
    agent = Agent(env)

    # Parameters for the experiment
    steps = np.power(2, np.arange(0, 10))  # Step sizes (n)
    alphas = np.arange(0, 1.1, 0.1)  # Learning rates (alpha)
    episodes = 10  # Number of episodes per combination
    runs = 100  # Number of independent runs

    # Run the experiment and collect errors
    errors = agent.run_experiment(steps, alphas, episodes, runs)

    # Plot the results
    plot_results(errors, steps, alphas)
