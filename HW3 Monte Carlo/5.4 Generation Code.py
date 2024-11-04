import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants representing actions
ACTION_BACK = 0
ACTION_END = 1

# Simulation parameters
RUNS = 10
EPISODES = 100000

class Agent:
    def behavior_policy(self):
        """
        Behavior policy: randomly choose ACTION_BACK or ACTION_END with equal probability.
        """
        return np.random.choice([ACTION_BACK, ACTION_END])

    def target_policy(self):
        """
        Target policy: always choose ACTION_BACK.
        """
        return ACTION_BACK

class Environment:
    def step(self, action):
        """
        Takes an action and returns the reward and a boolean indicating if the episode has ended.

        Args:
            action (int): The action taken by the agent.

        Returns:
            reward (int): The reward obtained after taking the action.
            done (bool): True if the episode has ended, False otherwise.
        """
        if action == ACTION_END:
            # Episode ends without reward
            return 0, True
        # With 10% probability, receive a reward and end the episode
        if np.random.rand() < 0.1:
            return 1, True
        else:
            # Continue the episode
            return 0, False

def compute_importance_ratio(trajectory):
    """
    Compute the importance sampling ratio for a trajectory.

    Args:
        trajectory (list): The trajectory of actions taken.

    Returns:
        rho (float): Importance sampling ratio.
    """
    # If any action in the trajectory is ACTION_END, the target policy probability is zero
    if ACTION_END in trajectory:
        return 0
    # Importance ratio is (1/0.5)^len(trajectory) = 2^len(trajectory)
    return 2 ** len(trajectory)

def simulate_episode(agent, env):
    """
    Simulate an episode using the agent and environment.

    Args:
        agent (Agent): The agent instance.
        env (Environment): The environment instance.

    Returns:
        reward (int): The reward obtained in the episode.
        trajectory (list): List of actions taken during the episode.
        rho (float): The importance sampling ratio for the episode.
    """
    trajectory = []
    done = False
    while not done:
        action = agent.behavior_policy()
        trajectory.append(action)
        reward, done = env.step(action)
    rho = compute_importance_ratio(trajectory)
    return reward, trajectory, rho

def run_ordinary_importance_sampling(episodes):
    """
    Perform Ordinary Importance Sampling for a single run.

    Args:
        episodes (int): Number of episodes to simulate.

    Returns:
        np.ndarray: Array of cumulative estimations over episodes.
    """
    rewards = []
    for _ in range(episodes):
        agent = Agent()
        env = Environment()
        reward, trajectory, rho = simulate_episode(agent, env)
        rewards.append(rho * reward)
    cumulative_rewards = np.cumsum(rewards)
    estimations = cumulative_rewards / np.arange(1, episodes + 1)
    return estimations

def run_weighted_importance_sampling(episodes):
    """
    Perform Weighted Importance Sampling for a single run.

    Args:
        episodes (int): Number of episodes to simulate.

    Returns:
        list: List of cumulative estimations over episodes.
    """
    cumulative_reward = 0
    cumulative_rho = 0
    estimations = []
    for _ in range(episodes):
        agent = Agent()
        env = Environment()
        reward, trajectory, rho = simulate_episode(agent, env)
        cumulative_reward += rho * reward
        cumulative_rho += rho
        estimation = cumulative_reward / cumulative_rho if cumulative_rho != 0 else 0
        estimations.append(estimation)
    return estimations

def perform_simulation(runs, episodes):
    """
    Perform the simulation for both Ordinary and Weighted Importance Sampling.

    Args:
        runs (int): Number of runs to simulate.
        episodes (int): Number of episodes per run.

    Returns:
        tuple: Two lists containing estimations from OIS and WIS.
    """
    ois_estimations = []
    wis_estimations = []

    for _ in range(runs):
        ois_estimation = run_ordinary_importance_sampling(episodes)
        wis_estimation = run_weighted_importance_sampling(episodes)
        ois_estimations.append(ois_estimation)
        wis_estimations.append(wis_estimation)

    return ois_estimations, wis_estimations

def plot_estimations(ois_estimations, wis_estimations, episodes):
    """
    Plot the estimations from Ordinary and Weighted Importance Sampling.

    Args:
        ois_estimations (list): Estimations from Ordinary Importance Sampling.
        wis_estimations (list): Estimations from Weighted Importance Sampling.
        episodes (int): Number of episodes.
    """
    plt.figure(figsize=(10, 6))
    x_values = np.arange(1, episodes + 1)
    for estimation in ois_estimations:
        plt.plot(x_values, estimation, color='blue', linewidth=1)
    for estimation in wis_estimations:
        plt.plot(x_values, estimation, color='red', linewidth=1)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Importance Sampling Estimates')
    plt.xscale('log')
    plt.xlim([1, episodes])
    plt.savefig('figure_5_4.png')
    plt.close()

def main():
    ois_estimations, wis_estimations = perform_simulation(RUNS, EPISODES)
    plot_estimations(ois_estimations, wis_estimations, EPISODES)

if __name__ == '__main__':
    main()
