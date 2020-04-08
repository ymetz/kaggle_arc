import gym
import numpy as np
import random
import math
from plot_functions import plot_single_image, plot_task


class ReasoningEnv(gym.Env):

    def __init__(self, tasks={}):
        self.dimX, self.dimY = 30, 30
        self.countDim = 10 # number of counts we can save, we expect that 10 are enough for current problems
        self.grid = np.zeros(shape=(self.dimX, self.dimY), dtype=np.int)
        self.count_memory, self.count_memory_idx = np.zeros(shape=(self.countDim,), dtype=np.int), 0
        self.desired_output_grid = np.zeros(shape=(self.dimX, self.dimY), dtype=np.int)
        self.working_output_grid, self.gridDims = np.zeros(shape=(self.dimX, self.dimY), dtype=np.int), \
                                                   (self.dimX, self.dimY)
        # selected elements is a list of grid indices belonging to the selected element, first element as core coord. +
        # following coordinates relative to core coord.
        self.selected_element = [(0,0)]
        # if we iterated all elements of the input grid, continue iteration on working grid
        self.selection_on_working_grid = False
        self.grid_selection_history = np.zeros(shape=(self.dimX, self.dimY), dtype=np.int)
        self.current_task_dims = []

        self.background_color = 0

        self.tasks = tasks
        self.current_task = random.sample(tasks, 1)[0]
        self.current_demo_task = random.sample(self.current_task['train'], 1)[0]
        self.current_test_task = random.sample(self.current_task['test'], 1)[0]

        self.unique_test_colors = []
        self.current_demo_colormaps, self.current_demo_inv_colormaps = [], []
        self.current_test_colormaps, self.current_test_inv_colormaps = [], []

        self.done = False

        self.observation_space = gym.spaces.Box(shape=(self.dimX, self.dimY), low=0, high=8)
        self.action_space = gym.spaces.Discrete(9)

    def reset(self):
        plot_task(self.current_task)
        self.current_demo_task = random.sample(self.current_task['train'], 1)[0]
        self.current_demo_colormaps, self.current_demo_inv_colormaps, unique_test_colors = [], [], []
        self.current_task_dims = []
        for task in self.current_task['train']:
            self.current_task_dims.append((task['input'].shape, task['output'].shape))
            task['input'], task['output'], colmap, inv_colmap, demo_utc = self.squash_colors(
                [np.array(task['input'], dtype=np.int), np.array(task['output'], dtype=np.int)])
            self.current_demo_colormaps.append(colmap)
            self.current_demo_inv_colormaps.append(inv_colmap)
            unique_test_colors.append(demo_utc)
        unique_test_colors = np.unique(unique_test_colors)
        for task in self.current_task['test']:
            task['input'], colmap, inv_colmap = self.squash_colors(
                [np.array(task['input'], dtype=np.int)], single_input=True, test_unique_colors=unique_test_colors)
            self.current_test_colormaps.append(colmap)
            self.current_test_inv_colormaps.append(inv_colmap)
        plot_task(self.current_task)

        self.grid = self.pad_array(self.current_demo_task['input'])
        plot_single_image(self.grid, padded=True)
        self.grid_selection_history = np.logical_or(self.grid == self.background_color, self.grid == 10).astype(int)

        self.desired_output_grid = self.pad_array(self.current_demo_task['output'])
        plot_single_image(self.desired_output_grid, padded=True)
        self.working_output_grid, self.gridDims = self.determine_output_bounds()

        self.done = False

        return self.grid

    def set_current_task(self, task):
        self.current_task = task

    def return_grid(self, test_index):
        def map_func(x):
            return self.current_test_inv_colormaps[test_index][x]

        map_func = np.vectorize(map_func)

        return map_func(self.grid)

    def determine_output_bounds(self):
        if all([demo_dims[0] == demo_dims[1] for demo_dims in self.current_task_dims]):
            return self.pad_array(np.zeros(shape=self.grid.shape)), self.grid.shape
        elif len(set([demo_dims[1] for demo_dims in self.current_task_dims])) == 1:
            return self.pad_array(np.zeros(shape=self.current_task['train'][0]['output'].shape)), \
                   self.current_task['train'][0]['output'].shape
        else:
            return np.zeros(shape=(self.dimX, self.dimY)), (self.dimX, self.dimY)

    def step(self, action):
        """Assumed format:
            action[0]: which basic action to choose
            action[1]: x coordinate if applicable
            action[2]: y coordinate if applicable
        """
        action_mapping = {0: "copy", 1: "recolor", 2: "remove", 3: "count", 4: "move", 5: "mirror",
                          6: "resize_output_grid", 7: "none", 8: "done"}
        action_name = action_mapping[action[0]]

        # we iterate through the elements in our grid (which is part of the input observation for our network)
        self.selected_element = self.get_next_selected_element()
        # the core coord is a random cell (usually top-left) from the selected element, serves as a local coordiante
        # base (0,0) which offset is added to
        core_coord_x, core_coord_y = self.selected_element[0]

        sec_action_input = action[1] # x coordinate - domain -30,+30
        third_action_input = action[2] # y coordinate - domain -30,+30
        fourth_action_input = action[3] # color # domain 0,10

        if action_name == "copy":
            new_core_coord_x, new_core_coord_y = sec_action_input, third_action_input
            for coords in self.selected_element:
                self.working_output_grid[new_core_coord_x+coords[0], new_core_coord_y+coords[1]] = \
                    self.grid[core_coord_x+coords[0], core_coord_y+coords[1]]

        if action_name == "recolor":
            # if x,y == 0,0 -> flood fill the selected element, else treat as offset and recolor single element
            if (core_coord_x, core_coord_y) == (0,0):
                for coords in self.selected_element:
                    self.working_output_grid[new_core_coord_x + coords[0], new_core_coord_y + coords[1]] = \
                        fourth_action_input
            else:
                for coords in self.selected_element:
                    self.working_output_grid[core_coord_x + coords[0], core_coord_y + coords[1]] = \
                        fourth_action_input

        if action_name == "remove":
            for coords in self.selected_element:
                self.working_output_grid[core_coord_x + coords[0], core_coord_y + coords[1]] = self.background_color

        if action_name == "count":
            self.count_memory[self.count_memory_idx] = len(self.selected_element)
            self.count_memory_idx = self.count_memory_idx % self.countDim

        if action_name == "move":
            new_core_coord_x, new_core_coord_y = sec_action_input, third_action_input
            for coords in self.selected_element:
                self.working_output_grid[new_core_coord_x + coords[0], new_core_coord_y + coords[1]] = \
                    self.background_color
            for coords in self.selected_element:
                self.working_output_grid[new_core_coord_x+coords[0], new_core_coord_y+coords[1]] = \
                    self.grid[core_coord_x+coords[0], core_coord_y+coords[1]]

        if action_name == "mirror":
            if (sec_action_input, third_action_input) == (0,1):
                # mirror up
                pass
            elif (sec_action_input, third_action_input) == (0,-1):
                # mirror down
                pass
            elif (sec_action_input, third_action_input) == (-1,0):
                # mirror left
                pass
            elif (sec_action_input, third_action_input) == (1,0):
                # mirror right
                pass
            elif (sec_action_input, third_action_input) == (-1,-1):
                # mirror top-left
                pass
            elif (sec_action_input, third_action_input) == (1,-1):
                # mirror top-right
                pass
            elif (sec_action_input, third_action_input) == (-1,-1):
                # mirror bottom-left
                pass
            elif (sec_action_input, third_action_input) == (-1,-1):
                # mirror bottom-right
                pass

        if action_name == "resize_output_grid":
            self.working_output_grid, self.gridDims = np.zeros(shape=(sec_action_input, third_action_input)), \
                                                      (sec_action_input, third_action_input)

        if action_name == "none":
            pass

        if action_name == "done":
            self.done = True

    def get_next_selected_element(self):
        old_core_coord_x, old_core_coord_y = self.selected_element[0]
        selected_element = []
        non_selected = np.argwhere(self.grid == 0)
        self.selected_element.append(non_selected[0])
        selected_color = self.grid[non_selected[0]]
        for idx_pair in non_selected[1:]:
            if self.grid[idx_pair] == selected_color and ((idx_pair[0]-1, idx_pair[1]) in self.selected_element
                                                          or (idx_pair[0], idx_pair[1]-1) in self.selected_element 
                                                          or (idx_pair[0]-1, idx_pair[1]-1) in self.selected_element
                                                          or (idx_pair[0]+1, idx_pair[1]-1) in self.selected_element):
                self.selected_element.append(idx_pair)
            if idx_pair[0] > self.selected_element[-1][0]+1:
                break

        return self.selected_element

    def valid_primary_actions(self):
        if self.selection_on_working_grid:
            valid_primary_action = [0, 1, 2, 4, 5, 7, 8]
        else:
            valid_primary_action = [0, 3, 6, 7, 8]

    def valid_secondary_actions(self, action):
        if action[0] not in [0, 1, 3, 4, 6]:
            return [], []
        else:
            return [0], [0]

    def render(self):
        plot_task(self.current_task)
        plot_single_image(self.grid)

    def pad_array(self, array):
        x_pad = self.dimX - array.shape[0]
        y_pad = self.dimY - array.shape[1]

        return np.pad(array,
                      ((math.floor(y_pad/2), math.ceil(y_pad/2)), (math.floor(x_pad/2), math.ceil(x_pad/2))),
                      constant_values=10)

    def squash_colors(self, arrays, single_input=False, test_unique_colors=None):
        if not single_input:
            colors_in_example = np.unique(np.concatenate([np.unique(arrays[0]), np.unique(arrays[1])]))
            unique_out_colors = [color for color in colors_in_example if color not in np.unique(arrays[0])]
        else:
            colors_in_example = np.unique(arrays[0])
            unique_out_colors = test_unique_colors

        colormap = {}
        inverse_colormap = {}
        color_index = 0
        for color in colors_in_example:
            if color in unique_out_colors:
                continue
            colormap[color] = color_index
            inverse_colormap[color_index] = color
            color_index += 1
        for color in unique_out_colors: # identify function for colors uniqe to output
            colormap[color] = color
            inverse_colormap[color] = color

        def map_func(x):
            return colormap[x]

        map_func = np.vectorize(map_func)

        if not single_input:
            return map_func(arrays[0]), map_func(arrays[1]), colormap, inverse_colormap, unique_out_colors
        else:
            return map_func(arrays[0]), colormap, inverse_colormap


