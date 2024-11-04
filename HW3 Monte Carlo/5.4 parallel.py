import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing

ACTION_BACK = 0
ACTION_END = 1

# behavior policy
def behavior_policy():
    return np.random.binomial(1, 0.5)

# target policy
def target_policy():
    return ACTION_BACK

# one turn
def play():
    # track the action for importance ratio
    trajectory = []
    while True:
        action = behavior_policy()
        trajectory.append(action)
        if action == ACTION_END:
            return 0, trajectory
        if np.random.binomial(1, 0.9) == 0:
            return 1, trajectory

def ois_run(episodes):
    rewards = []
    for episode in range(episodes):
        reward, trajectory = play()
        if trajectory[-1] == ACTION_END:
            rho = 0
        else:
            rho = 1.0 / pow(0.5, len(trajectory))
        rewards.append(rho * reward)
    rewards = np.add.accumulate(rewards)
    estimations = rewards / np.arange(1, episodes + 1)
    return estimations

def wis_run(episodes):
    estimations = []
    cumulative_reward = 0
    cumulative_rho = 0
    for episode in range(episodes):
        reward, trajectory = play()
        if trajectory[-1] == ACTION_END:
            rho = 0
        else:
            rho = 1.0 / pow(0.5, len(trajectory))
        cumulative_reward += rho * reward
        cumulative_rho += rho
        if cumulative_rho != 0:
            estimation = cumulative_reward / cumulative_rho
        else:
            estimation = 0
        estimations.append(estimation)
    return estimations

def figure_5_4():
    runs = 10
    episodes = 10 ** 6

    plt.figure(figsize=(10, 6))

    # Prepare arguments for multiprocessing
    args = [episodes] * runs

    # Run OIS and WIS in parallel using multiprocessing
    with multiprocessing.Pool() as pool:
        # Execute OIS runs in parallel
        ois_results = pool.map(ois_run, args)
        # Execute WIS runs in parallel
        wis_results = pool.map(wis_run, args)

    # Plot OIS results
    for estimations in ois_results:
        plt.plot(estimations, color='blue', linewidth=1)

    # Plot WIS results
    for estimations in wis_results:
        plt.plot(estimations, color='red', linewidth=1)

    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Importance Sampling Estimates')
    plt.xscale('log')
    plt.savefig('figure_5_4_parallel.png')
    plt.close()

if __name__ == '__main__':
    figure_5_4()
