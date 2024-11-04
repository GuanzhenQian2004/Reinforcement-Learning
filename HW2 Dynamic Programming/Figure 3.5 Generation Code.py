import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')


class GridWorld:
    def __init__(self, world_size=5, discount=0.9):
        self.world_size = world_size
        self.discount = discount
        self.A_POS = [0, 1]
        self.A_PRIME_POS = [4, 1]
        self.B_POS = [0, 3]
        self.B_PRIME_POS = [2, 3]
        # Left, Up, Right, Down
        self.ACTIONS = [np.array([0, -1]),
                        np.array([-1, 0]),
                        np.array([0, 1]),
                        np.array([1, 0])]
        self.state_space = [
            [i, j] for i in range(self.world_size) for j in range(self.world_size)
        ]

    def step(self, state, action):
        if state == self.A_POS:
            return self.A_PRIME_POS, 10
        if state == self.B_POS:
            return self.B_PRIME_POS, 5

        next_state = (np.array(state) + action).tolist()
        x, y = next_state
        if x < 0 or x >= self.world_size or y < 0 or y >= self.world_size:
            reward = -1.0
            next_state = state
        else:
            reward = 0
        return next_state, reward

    def get_possible_actions(self):
        return self.ACTIONS

    def get_all_states(self):
        return self.state_space


class ValueIterationAgent:
    def __init__(self, env, threshold=1e-4):
        self.env = env
        self.threshold = threshold
        self.values = np.zeros((env.world_size, env.world_size))

    def value_iteration(self):
        iteration = 0
        while True:
            delta = 0
            new_values = np.copy(self.values)
            for state in self.env.get_all_states():
                i, j = state
                max_value = self.compute_state_value(state)
                new_values[i, j] = max_value
                delta = max(delta, abs(self.values[i, j] - max_value))
            self.values = new_values
            iteration += 1
            if self.has_converged(delta):
                break
        return self.values

    def compute_state_value(self, state):
        """
        Computes the maximum value for a given state by considering all possible actions.
        """
        value_list = []
        for action in self.env.get_possible_actions():
            (next_i, next_j), reward = self.env.step(state, action)
            value = reward + self.env.discount * self.values[next_i, next_j]
            value_list.append(value)
        max_value = max(value_list)
        return max_value

    def has_converged(self, delta):
        """
        Checks if the value iteration has converged based on the threshold.
        """
        return delta < self.threshold

    def get_optimal_values(self):
        return self.values


class Visualizer:
    @staticmethod
    def draw_image(values, filename='value_iteration_result.png'):
        fig, ax = plt.subplots()
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])

        nrows, ncols = values.shape
        width, height = 1.0 / ncols, 1.0 / nrows

        # Add cells
        for (i, j), val in np.ndenumerate(values):
            tb.add_cell(i, j, width, height, text=val,
                        loc='center', facecolor='white')

        # Row and column labels...
        for i in range(len(values)):
            tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
                        edgecolor='none', facecolor='none')
            tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center',
                        edgecolor='none', facecolor='none')

        ax.add_table(tb)
        plt.savefig(filename)
        plt.close(fig)


def main():
    # Initialize the environment
    env = GridWorld()

    # Perform value iteration
    agent = ValueIterationAgent(env)
    optimal_values = agent.value_iteration()

    # Visualize the results
    Visualizer.draw_image(np.round(optimal_values, decimals=1), 'prof_figure_3_5.png')


if __name__ == '__main__':
    main()
