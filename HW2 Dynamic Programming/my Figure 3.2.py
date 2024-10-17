import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

class GridWorld:
    def __init__(self, world_size=5, discount=0.9):
        self.world_size = world_size
        self.discount = discount
        self.A_pos = [0, 1]
        self.A_prime_pos = [4, 1]
        self.B_pos = [0, 3]
        self.B_prime_pos = [2, 3]
        self.actions = [
            np.array([0, -1]),  # Left
            np.array([-1, 0]),  # Up
            np.array([0, 1]),   # Right
            np.array([1, 0])    # Down
        ]
        self.action_prob = 0.25

    def step(self, state, action):
        if state == self.A_pos:
            return self.A_prime_pos, 10
        if state == self.B_pos:
            return self.B_prime_pos, 5
        next_state = (np.array(state) + action).tolist()
        x, y = next_state
        if x < 0 or x >= self.world_size or y < 0 or y >= self.world_size:
            return state, -1.0
        return next_state, 0

def compute_state_value(state, value, grid_world):
    v = 0
    for action in grid_world.actions:
        next_state, reward = grid_world.step(state, action)
        next_i, next_j = next_state
        v += grid_world.action_prob * (
            reward + grid_world.discount * value[next_i, next_j]
        )
    return v

def value_iteration(grid_world, threshold=1e-4):
    value = np.zeros((grid_world.world_size, grid_world.world_size))
    while True:
        new_value = np.zeros_like(value)
        for i in range(grid_world.world_size):
            for j in range(grid_world.world_size):
                state = [i, j]
                new_value[i, j] = compute_state_value(state, value, grid_world)
        if np.max(np.abs(value - new_value)) < threshold:
            break
        value = new_value
    return value

def draw_image(value_function, filename='my_figure_3_2.png'):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = value_function.shape
    width, height = 1.0 / ncols, 1.0 / nrows
    for (i, j), val in np.ndenumerate(value_function):
        tb.add_cell(
            i, j, width, height,
            text=round(val, 1),
            loc='center',
            facecolor='white'
        )
    ax.add_table(tb)
    plt.savefig(filename)
    plt.close()

def main():
    grid_world = GridWorld(world_size=5, discount=0.9)
    value = value_iteration(grid_world)
    draw_image(value)

if __name__ == '__main__':
    main()
