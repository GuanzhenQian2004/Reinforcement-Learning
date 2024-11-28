import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

class RandomWalkEnvironment:
    """
    Environment for the Random Walk problem.
    """
    def __init__(self, n_states=19, start_state=10):
        self.n_states = n_states
        self.start_state = start_state
        self.end_states = [0, n_states + 1]
        self.reset()

    def reset(self):
        """
        Reset the environment to the start state.
        """
        self.state = self.start_state
        return self.state

    def step(self, action):
        """
        Take a step in the environment.

        Parameters:
            action (int): Action to take (-1 for left, +1 for right).

        Returns:
            next_state (int): The next state after taking the action.
            reward (float): The reward received after taking the action.
            done (bool): True if the episode has ended.
        """
        next_state = self.state + action
        if next_state == self.end_states[0]:
            reward = -1
            done = True
        elif next_state == self.end_states[1]:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.state = next_state
        return next_state, reward, done

class Agent:
    """
    Agent that performs n-step Temporal Difference learning.
    """
    def __init__(self, n_states=19, gamma=1.0):
        self.n_states = n_states
        self.gamma = gamma
        self.value_function = np.zeros(n_states + 2)  # Including terminal states

    def temporal_difference(self, env, n_steps, alpha):
        """
        Perform n-step Temporal Difference learning.

        Parameters:
            env (RandomWalkEnvironment): The environment to interact with.
            n_steps (int): Number of steps 'n' in n-step TD.
            alpha (float): Step size parameter (learning rate).
        """
        state = env.reset()
        states = [state]
        rewards = [0]  # Initialize with a dummy reward
        T = float('inf')  # Time when episode ends
        time = 0  # Current time step

        while True:
            if time < T:
                # Choose action randomly: move left (-1) or right (+1)
                action = np.random.choice([-1, 1])
                next_state, reward, done = env.step(action)

                # Store next state and reward
                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = time + 1  # Episode ends after this time step

            # Time whose estimate is being updated
            tau = time - n_steps + 1
            if tau >= 0:
                # Compute the n-step return
                G = 0.0
                for i in range(tau + 1, min(tau + n_steps, T) + 1):
                    G += (self.gamma ** (i - tau - 1)) * rewards[i]
                if (tau + n_steps) < T:
                    G += (self.gamma ** n_steps) * self.value_function[states[tau + n_steps]]

                state_to_update = states[tau]
                if state_to_update not in env.end_states:
                    # Update the value function
                    self.value_function[state_to_update] += alpha * (G - self.value_function[state_to_update])

            if tau >= T - 1:
                break

            time += 1
            state = states[time]

def figure7_2():
    """
    Reproduce Figure 7.2 from Sutton & Barto's 'Reinforcement Learning: An Introduction'.
    Compare n-step TD methods with different 'n' and 'alpha' values.
    """
    n_states = 19
    n_values = np.power(2, np.arange(0, 10))  # n = 1, 2, 4, ..., 512
    alpha_values = np.arange(0, 1.1, 0.1)     # alpha from 0 to 1 in steps of 0.1
    episodes = 10
    runs = 100

    # True state values from the Bellman equation
    true_value = np.arange(-20, 22, 2) / 20.0
    true_value[0] = true_value[-1] = 0  # Terminal states have zero value

    # Initialize error tracking
    errors = np.zeros((len(n_values), len(alpha_values)))

    # Perform experiments
    for run in tqdm(range(runs)):
        env = RandomWalkEnvironment(n_states=n_states)
        for n_idx, n in enumerate(n_values):
            for alpha_idx, alpha in enumerate(alpha_values):
                agent = Agent(n_states=n_states)
                for episode in range(episodes):
                    # Perform n-step TD learning
                    agent.temporal_difference(env, n, alpha)
                    # Compute RMS error over all non-terminal states
                    errors[n_idx, alpha_idx] += np.sqrt(
                        np.mean((agent.value_function[1:-1] - true_value[1:-1]) ** 2)
                    )

    # Average errors over runs and episodes
    errors /= (runs * episodes)

    # Plot results
    for i, n in enumerate(n_values):
        plt.plot(alpha_values, errors[i, :], label=f'n = {n}')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()

    plt.savefig('professor_figure_7_2.png')
    plt.close()

if __name__ == '__main__':
    figure7_2()
