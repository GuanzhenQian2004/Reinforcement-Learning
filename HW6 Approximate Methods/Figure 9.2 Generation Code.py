import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# # of states except for terminal states
N_STATES = 1000

# start from a central state
START_STATE = 500

# terminal states
END_STATES = [0, N_STATES + 1]

# possible actions
ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS = [ACTION_LEFT, ACTION_RIGHT]

# maximum stride for an action
STEP_RANGE = 100


class RandomWalkEnvironment:
    def __init__(self, n_states=N_STATES, start_state=START_STATE, end_states=END_STATES, step_range=STEP_RANGE):
        # Initialize the environment with the given number of states, start state, end states, and step range
        self.n_states = n_states
        self.start_state = start_state
        self.end_states = end_states
        self.step_range = step_range
        self.current_state = None

    def reset(self):
        # Reset the environment to the start state and return it
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        # Take a step in the environment given an action
        step = np.random.randint(1, self.step_range + 1) * action
        # Move the agent to the next state
        next_state = self.current_state + step
        # Clip the state so it doesn't go beyond the terminal states
        next_state = max(min(next_state, self.n_states + 1), 0)

        # Assign reward based on whether the next state is terminal or not
        if next_state == 0:
            reward = -1
        elif next_state == self.n_states + 1:
            reward = 1
        else:
            reward = 0

        # Update the current state
        self.current_state = next_state
        # Return the next state and the obtained reward
        return next_state, reward

    def is_terminal(self, state):
        # Check if a given state is terminal (either 0 or N_STATES+1)
        return state in self.end_states

    def get_actions(self):
        # Return all possible actions (left and right)
        return ACTIONS


def compute_true_value():
    # Compute the true state values using dynamic programming
    true_value = np.arange(-1001, 1003, 2) / 1001.0

    # Keep iterating until values converge
    while True:
        old_value = np.copy(true_value)
        # Update each state's value based on the next states' values
        for state in range(1, N_STATES + 1):
            value_sum = 0.0
            # For each action (left or right)
            for action in ACTIONS:
                # For each possible step size
                for step in range(1, STEP_RANGE + 1):
                    actual_step = step * action
                    next_state = state + actual_step
                    # Clip next_state within terminal bounds
                    next_state = max(min(next_state, N_STATES + 1), 0)
                    # Average over all possible actions and steps
                    value_sum += 1.0 / (2 * STEP_RANGE) * true_value[next_state]
            true_value[state] = value_sum

        # Check for convergence
        error = np.sum(np.abs(old_value - true_value))
        if error < 1e-2:
            break

    # Terminal states have true value 0
    true_value[0] = 0
    true_value[-1] = 0

    return true_value


class ValueFunction:
    # This value function approximates the value using state aggregation
    def __init__(self, num_of_groups):
        # Divide the state space into num_of_groups equal groups
        self.num_of_groups = num_of_groups
        self.group_size = N_STATES // num_of_groups
        # Each group has a parameter (theta)
        self.params = np.zeros(num_of_groups)

    def value(self, state):
        # Return value of a given state
        # If the state is terminal, value is 0
        if state in END_STATES:
            return 0
        # Otherwise, find which group the state belongs to
        group_index = (state - 1) // self.group_size
        return self.params[group_index]

    def update(self, delta, state):
        # Update the parameter corresponding to the group's state
        if state in END_STATES:
            return
        group_index = (state - 1) // self.group_size
        self.params[group_index] += delta


class TilingsValueFunction:
    # This value function uses tile coding (multiple tilings) for function approximation
    def __init__(self, numOfTilings, tileWidth, tilingOffset):
        self.numOfTilings = numOfTilings
        self.tileWidth = tileWidth
        self.tilingOffset = tilingOffset

        # Compute the size of each tiling
        self.tilingSize = N_STATES // tileWidth + 1
        # Initialize parameters for each tile in each tiling
        self.params = np.zeros((self.numOfTilings, self.tilingSize))

        # Compute tiling offsets, so each tiling starts at a different position
        self.tilings = np.arange(-tileWidth + 1, 0, tilingOffset)

    def value(self, state):
        # Compute the value by summing over the active tile in each tiling
        if state in END_STATES:
            return 0
        stateValue = 0.0
        for tilingIndex in range(self.numOfTilings):
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            stateValue += self.params[tilingIndex, tileIndex]
        return stateValue

    def update(self, delta, state):
        # Update the parameters of each active tile
        # Each tiling is updated by delta/numOfTilings
        if state in END_STATES:
            return
        delta /= self.numOfTilings
        for tilingIndex in range(self.numOfTilings):
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            self.params[tilingIndex, tileIndex] += delta


POLYNOMIAL_BASES = 0
FOURIER_BASES = 1


class BasesValueFunction:
    # This value function uses polynomial or Fourier bases
    def __init__(self, order, type):
        self.order = order
        self.weights = np.zeros(order + 1)
        self.bases = []

        # Create basis functions
        if type == POLYNOMIAL_BASES:
            # Polynomial bases: s^0, s^1, ..., s^order
            for i in range(0, order + 1):
                self.bases.append(lambda s, i=i: pow(s, i))
        elif type == FOURIER_BASES:
            # Fourier bases: cos(0 * pi * s), cos(1 * pi * s), ...
            for i in range(0, order + 1):
                self.bases.append(lambda s, i=i: np.cos(i * np.pi * s))

    def value(self, state):
        # Map state to [0,1] and evaluate basis functions
        if state in END_STATES:
            return 0
        s = state / float(N_STATES)
        feature = np.asarray([func(s) for func in self.bases])
        # Compute dot product with weights
        return np.dot(self.weights, feature)

    def update(self, delta, state):
        # Update weights by gradient ascent (delta is step size times error)
        if state in END_STATES:
            return
        s = state / float(N_STATES)
        derivative_value = np.asarray([func(s) for func in self.bases])
        self.weights += delta * derivative_value


class Agent:
    def __init__(self, value_function):
        self.value_function = value_function

    def get_action(self, env):
        # Random policy: choose left or right with equal probability
        if np.random.binomial(1, 0.5) == 1:
            return ACTION_RIGHT
        return ACTION_LEFT

    def run_gradient_monte_carlo(self, env, alpha, distribution=None):
        # Run one episode with gradient Monte Carlo updates
        state = env.reset()
        trajectory = [state]
        reward = 0.0

        # Generate an episode until terminal state is reached
        while not env.is_terminal(state):
            action = self.get_action(env)
            next_state, reward = env.step(action)
            trajectory.append(next_state)
            state = next_state

        # Update value function for each visited state
        for s in trajectory[:-1]:
            delta = alpha * (reward - self.value_function.value(s))
            self.value_function.update(delta, s)
            if distribution is not None:
                distribution[s] += 1

    def run_semi_gradient_temporal_difference(self, env, n, alpha):
        state = env.reset()
        states = [state]
        rewards = [0]
        time = 0
        T = float('inf')

        # Follow the n-step TD procedure
        while True:
            time += 1
            if time < T:
                # Get action from the current policy (random)
                action = self.get_action(env)
                # Step in the environment
                next_state, reward = env.step(action)
                states.append(next_state)
                rewards.append(reward)
                if env.is_terminal(next_state):
                    T = time

            update_time = time - n
            if update_time >= 0:
                # Compute the return for n-step TD
                returns = 0.0
                for t in range(update_time + 1, min(T, update_time + n) + 1):
                    returns += rewards[t]
                if update_time + n <= T:
                    returns += self.value_function.value(states[update_time + n])
                state_to_update = states[update_time]
                if not env.is_terminal(state_to_update):
                    # Semi-gradient TD update
                    delta = alpha * (returns - self.value_function.value(state_to_update))
                    self.value_function.update(delta, state_to_update)

            if update_time == T - 1:
                # Episode is over, break out of the loop
                break

            if time < T:
                # Move on to the next state if episode not ended
                state = next_state


def figure_9_2_left(true_value):
    # Generate the left plot of Figure 9.2 (Approximate TD value vs. true value)
    env = RandomWalkEnvironment()
    episodes = int(1e5)
    alpha = 2e-4
    value_function = ValueFunction(10)
    agent = Agent(value_function)

    # Run TD(1-step) for a large number of episodes
    for ep in tqdm(range(episodes)):
        agent.run_semi_gradient_temporal_difference(env, 1, alpha)

    # Extract estimated values for all states
    stateValues = [value_function.value(i) for i in range(1, N_STATES + 1)]
    # Plot the estimated values and the true values
    plt.plot(range(1, N_STATES + 1), stateValues, label='Approximate TD value')
    plt.plot(range(1, N_STATES + 1), true_value[1:-1], label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()


def figure_9_2_right(true_value):
    # Generate the right plot of Figure 9.2 (RMS errors for different n and alpha)
    env = RandomWalkEnvironment()
    steps = np.power(2, np.arange(0, 10))
    # Slightly refined alpha range
    alphas = np.arange(0, 1.1, 0.02)
    episodes = 10
    runs = 100
    errors = np.zeros((len(steps), len(alphas)))

    # We average over multiple runs for better performance estimates
    for run in tqdm(range(runs)):
        for step_ind, step in enumerate(steps):
            for alpha_ind, alpha in enumerate(alphas):
                value_function = ValueFunction(20)
                agent = Agent(value_function)
                for ep in range(episodes):
                    agent.run_semi_gradient_temporal_difference(env, step, alpha)
                    state_value = np.asarray([value_function.value(i) for i in range(1, N_STATES + 1)])
                    # Compute RMS error compared to the true value
                    errors[step_ind, alpha_ind] += np.sqrt(np.sum(np.power(state_value - true_value[1:-1], 2)) / N_STATES)
    errors /= (episodes * runs)

    # Plot the RMS error curves
    for i in range(len(steps)):
        plt.plot(alphas, errors[i, :], label='n = ' + str(steps[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()


def figure_9_2(true_value):
    # Generate the combined figure with two subplots
    plt.figure(figsize=(10, 20))

    # Top subplot
    plt.subplot(2, 1, 1)
    figure_9_2_left(true_value)

    # Bottom subplot
    plt.subplot(2, 1, 2)
    figure_9_2_right(true_value)

    # Save the figure
    plt.savefig('figure_9_2.png')
    plt.close()


if __name__ == '__main__':
    # Compute the true values beforehand
    true_value = compute_true_value()
    # Generate the figure
    figure_9_2(true_value)
