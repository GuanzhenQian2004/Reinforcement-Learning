import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for environments without a display
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Constants
WORLD_SIZE = 4
ACTIONS = [
    np.array([0, -1]),  # Left
    np.array([-1, 0]),  # Up
    np.array([0, 1]),   # Right
    np.array([1, 0])    # Down
]
ACTION_PROB = 0.25  # Equal probability for each action
REWARD = -1         # Uniform step cost
DISCOUNT = 1.0      # Discount factor
OUTPUT_DIR = "state_value_tables"


class GridWorld:
    def __init__(self, size=WORLD_SIZE, reward=REWARD):
        self.size = size
        self.reward = reward
        self.actions = ACTIONS
        self.action_prob = ACTION_PROB

    def is_terminal(self, state):
        x, y = state
        return (x == 0 and y == 0) or (x == self.size - 1 and y == self.size - 1)

    def step(self, state, action):
        if self.is_terminal(state):
            return state, 0

        next_state = np.array(state) + action
        x, y = next_state

        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            next_state = state
        else:
            next_state = next_state.tolist()

        return next_state, self.reward


class PolicyEvaluator:
    def __init__(self, environment, discount=DISCOUNT):
        self.env = environment
        self.discount = discount
        self.state_values = np.zeros((self.env.size, self.env.size))

    def evaluate_policy(self, desired_iterations):
        saved_values = {}
        iteration = 0

        if 0 in desired_iterations:
            saved_values[0] = self.state_values.copy()

        while True:
            delta = self.policy_evaluation_step()
            iteration += 1

            if iteration in desired_iterations:
                saved_values[iteration] = self.state_values.copy()

            if delta < 1e-4:
                saved_values[iteration] = self.state_values.copy()
                print(f'Converged after {iteration} iterations.')
                break

        return saved_values, iteration

    def policy_evaluation_step(self):
        old_state_values = self.state_values.copy()
        delta = 0

        for i in range(self.env.size):
            for j in range(self.env.size):
                if self.env.is_terminal([i, j]):
                    continue
                value = self.compute_state_value([i, j], old_state_values)
                delta = max(delta, abs(value - self.state_values[i, j]))
                self.state_values[i, j] = value
        return delta

    def compute_state_value(self, state, old_state_values):
        value = 0
        for action in self.env.actions:
            next_state, reward = self.env.step(state, action)
            next_i, next_j = next_state
            value += self.env.action_prob * (reward + self.discount * old_state_values[next_i, next_j])
        return value


def draw_table(ax, state_values, iteration_label):
    ax.set_axis_off()
    table = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = state_values.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    for (i, j), val in np.ndenumerate(state_values):
        table.add_cell(i, j, width, height, text=f'{val:.1f}',
                       loc='center', facecolor='white', edgecolor='black')

    ax.add_table(table)
    ax.set_title(f'k = {iteration_label}', fontsize=14)


def save_table_image(state_values, iteration_label):
    fig, ax = plt.subplots(figsize=(4, 4))
    draw_table(ax, state_values, iteration_label)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_path = os.path.join(OUTPUT_DIR, f'state_values_k_{iteration_label}.png')
    plt.savefig(image_path)
    plt.close()
    print(f'Saved state values table for k={iteration_label} to "{image_path}".')


def main():
    desired_iterations = [0, 1, 2, 3, 10]

    environment = GridWorld()
    evaluator = PolicyEvaluator(environment)

    saved_values, convergence_iteration = evaluator.evaluate_policy(desired_iterations)

    for iteration in desired_iterations:
        save_table_image(saved_values[iteration], iteration)

    save_table_image(saved_values[convergence_iteration], f'{convergence_iteration}_converged')


if __name__ == '__main__':
    main()
