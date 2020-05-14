import gym
import numpy as np
import random
import math
from itertools import chain
from kaggle_arc.plot_functions import plot_single_image, plot_task

from copy import deepcopy

action_mapping = {0: "copy", 1: "recolor", 2: "remove", 3: "count", 4: "move", 5: "mirror",
                  6: "resize_output_grid", 7: "none", 8: "done"}


class ReasoningEnv(gym.Env):

    def __init__(self, env_config={}):
        self.grid_height, self.grid_width = 30, 30
        self.countDim = 10  # number of counts we can save, we expect that 10 are enough for current problems
        self.grid = np.zeros(shape=(self.grid_height, self.grid_width), dtype=np.int)
        self.count_memory, self.count_memory_idx = np.zeros(shape=(self.countDim,), dtype=np.int), 0
        self.desired_output_grid = np.zeros(shape=(self.grid_height, self.grid_width), dtype=np.int)
        self.working_output_grid, self.grid_dims = np.zeros(shape=(self.grid_height, self.grid_width), dtype=np.int), \
                                                   (self.grid_height, self.grid_width)
        # selected elements is a list of grid indices belonging to the selected element, first element as core coord. +
        # following coordinates relative to core coord.
        self.selected_element = [(0, 0)]
        # if we iterated all elements of the input grid, continue iteration on working grid
        self.selection_on_working_grid = False
        self.grid_selection_history = np.zeros(shape=(self.grid_height, self.grid_width), dtype=np.int)
        self.current_task_dims, self.output_size_fixed = [], False
        self.copy_input_to_working_grid = False

        self.background_color = 0

        self.running_reward = 0

        self.current_task = None
        self.current_demo_task = None
        self.current_demo_task_index = 0
        self.current_test_task = None
        self.current_test_task_index = 0

        self.unique_test_colors = []
        self.current_demo_colormaps, self.current_demo_inv_colormaps = [], []
        self.current_test_colormaps, self.current_test_inv_colormaps = [], []

        self.done = False

        self.observation_space = gym.spaces.Dict({
            "real_obs": gym.spaces.Tuple(
                [gym.spaces.Box(0, 10, shape=(self.grid_height, self.grid_width), dtype=np.int),
                 gym.spaces.Box(0, 900, shape=(self.countDim,), dtype=np.int)]),
            "action_mask": gym.spaces.Tuple([gym.spaces.Box(0, 1, shape=(9,), dtype=np.int),
                                             gym.spaces.Box(0, 1, shape=(9, 30), dtype=np.int),
                                             gym.spaces.Box(0, 1, shape=(9, 30), dtype=np.int),
                                             gym.spaces.Box(0, 1, shape=(9, 10), dtype=np.int)])
        })
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(9), gym.spaces.Discrete(30), gym.spaces.Discrete(30),
                                              gym.spaces.Discrete(10)])

    def reset(self):
        self.current_demo_task_index = (self.current_demo_task_index + 1) % len(self.current_task['train'])
        self.current_demo_task = self.current_task['train'][self.current_demo_task_index]

        pre_padding_grid_size = self.current_demo_task['input'].shape
        self.grid = self.pad_array(self.current_demo_task['input'])
        self.grid_selection_history = np.logical_or(self.grid == self.background_color, self.grid == 10).astype(int)

        self.desired_output_grid = self.pad_array(self.current_demo_task['output'])
        self.working_output_grid, self.grid_dims = self.determine_output_bounds()
        self.selected_element = self.get_next_selected_element()

        if self.copy_input_to_working_grid:
            self.working_output_grid = np.copy(self.grid)
            self.grid = self.working_output_grid
            self.selection_on_working_grid = True

        self.done = False
        self.running_reward = 0

        a1_mask = self.primary_action_mask()
        a2_mask, a3_mask, a4_mask = self.dependant_action_masks()

        return {"real_obs": tuple([self.grid, self.count_memory]),
                "action_mask": self.observation_space['action_mask'].sample()}

    def set_current_task(self, task):
        self.current_task = task

        self.current_demo_colormaps, self.current_demo_inv_colormaps, unique_test_colors = [], [], []
        self.current_task_dims, in_out_similarities, pre_padding_grid_sizes = [], [], []
        for task in self.current_task['train']:
            self.current_task_dims.append((np.array(task['input']).shape, np.array(task['output']).shape))
            task['input'], task['output'], colmap, inv_colmap, demo_utc = self.squash_colors(
                [np.array(task['input'], dtype=np.int), np.array(task['output'], dtype=np.int)])
            self.current_demo_colormaps.append(colmap)
            self.current_demo_inv_colormaps.append(inv_colmap)
            unique_test_colors.append(demo_utc)
            if task['input'].shape == task['output'].shape:
                # if more than 50% of the input and output grid are equal, and the shapes match, copy the input grid to the
                # working output grid
                in_out_similarities.append(np.count_nonzero(np.equal(task['input'], task['output']) == True))
            else:
                in_out_similarities.append(0)
            pre_padding_grid_sizes.append(task['output'].shape[0] * task['output'].shape[1])

        unique_test_colors = np.unique(chain.from_iterable(unique_test_colors))
        for task in self.current_task['test']:
            task['input'], colmap, inv_colmap = self.squash_colors(
                [np.array(task['input'], dtype=np.int)], single_input=True, test_unique_colors=unique_test_colors)
            self.current_test_colormaps.append(colmap)
            self.current_test_inv_colormaps.append(inv_colmap)

        if all([in_out_similarities[i] > 0.7 * pre_padding_grid_sizes[i]
                for i in range(len(in_out_similarities))]):
            self.copy_input_to_working_grid = True

        self.current_demo_task_index = 0
        self.current_test_task_index = 0

    def get_number_of_demo_test_in_taks(self):
        return len(self.current_task['train']), len(self.current_task['test'])

    def set_next_test_task(self):
        self.current_test_task_index = (self.current_test_task_index + 1) % len(self.current_task['test'])
        self.current_test_task = self.current_task['test'][self.current_test_task_index]

    def return_grid(self, test_index):
        def map_func(x):
            return self.current_test_inv_colormaps[test_index][x]

        map_func = np.vectorize(map_func)

        return map_func(self.grid)

    def determine_output_bounds(self):
        if all([demo_dims[0] == demo_dims[1] for demo_dims in self.current_task_dims]):
            self.output_size_fixed = True
            return self.pad_array(np.zeros(shape=self.current_demo_task['input'].shape)), \
                   self.current_demo_task['input'].shape
        elif len(set([demo_dims[1] for demo_dims in self.current_task_dims])) == 1:
            self.output_size_fixed = True
            return self.pad_array(np.zeros(shape=self.current_task['train'][0]['output'].shape)), \
                   self.current_task['train'][0]['output'].shape
        else:
            self.output_size_fixed = False
            return np.zeros(shape=(self.grid_height, self.grid_width)), (self.grid_height, self.grid_width)

    def get_bounds(self):
        y_pad = self.grid_height - self.grid_dims[0]
        x_pad = self.grid_width - self.grid_dims[1]
        top_bound = math.floor(y_pad / 2)
        bottom_bound = self.grid_height - math.ceil(y_pad / 2)
        left_bound = math.floor(x_pad / 2)
        right_bound = self.grid_width - math.ceil(x_pad / 2)

        return top_bound, bottom_bound, left_bound, right_bound

    def step(self, action):
        """Assumed format:
            action[0]: which basic action to choose
            action[1]: x coordinate if applicable
            action[2]: y coordinate if applicable
            action[3]: color for recolor
        """
        action_name = action_mapping[action[0]]

        reward = 0.0

        a1_mask = self.primary_action_mask()
        a2_mask, a3_mask, a4_mask = self.dependant_action_masks()
        if a1_mask[action[0]] == 0 or a2_mask[action[0], action[1]] == 0 \
                or a3_mask[action[0], action[2]] == 0 or a4_mask[action[0], action[3]] == 0:
            action_name = "none"

        # the core coord is a random cell (usually top-left) from the selected element, serves as a local coordiante
        # base (0,0) which offset is added to

        sec_action_input = action[1]  # row coordinate - domain -30,+30
        third_action_input = action[2]  # col coordinate - domain -30,+30
        fourth_action_input = action[3]  # color # domain 0,10

        if action_name == "copy":
            core_coord_row, core_coord_col = self.selected_element[0]
            new_core_coord_row, new_core_coord_col = sec_action_input - core_coord_row, third_action_input - core_coord_col
            top_bound, bottom_bound, left_bound, right_bound = self.get_bounds()
            for coords in self.selected_element:
                if new_core_coord_row + coords[0] < top_bound or new_core_coord_row + coords[0] >= bottom_bound \
                        or new_core_coord_col + coords[1] < left_bound or new_core_coord_col + coords[1] >= right_bound:
                    continue
                self.working_output_grid[new_core_coord_row + coords[0], new_core_coord_col + coords[1]] = \
                    self.grid[coords[0], coords[1]]

        elif action_name == "recolor":
            # if row,col == 0,0 -> flood fill the selected element, else treat as offset and recolor single element
            if self.selected_element is not None and (sec_action_input, third_action_input) == (0, 0):
                core_coord_row, core_coord_col = self.selected_element[0]
                new_core_coord_row, new_core_coord_col = sec_action_input - core_coord_row, third_action_input - core_coord_col
                for coords in self.selected_element:
                    self.working_output_grid[new_core_coord_row + coords[0], new_core_coord_col + coords[1]] = \
                        fourth_action_input
            else:
                self.working_output_grid[sec_action_input, third_action_input] = fourth_action_input

        elif action_name == "remove":
            for coords in self.selected_element:
                self.working_output_grid[coords[0], coords[1]] = self.background_color

        elif action_name == "count":
            self.count_memory[self.count_memory_idx] = len(self.selected_element)
            self.count_memory_idx = (self.count_memory_idx + 1) % self.countDim

        elif action_name == "move":
            core_coord_row, core_coord_col = self.selected_element[0]
            element_color = self.working_output_grid[self.selected_element[0][0], self.selected_element[0][1]]
            new_core_coord_row, new_core_coord_col = sec_action_input - core_coord_row, third_action_input - core_coord_col
            top_bound, bottom_bound, left_bound, right_bound = self.get_bounds()
            for coords in self.selected_element:
                self.working_output_grid[coords[0], coords[1]] = \
                    self.background_color
            for coords in self.selected_element:
                if new_core_coord_row + coords[0] < top_bound or new_core_coord_row + coords[0] >= bottom_bound \
                        or new_core_coord_row + coords[1] < left_bound or new_core_coord_row + coords[1] >= right_bound:
                    continue
                self.working_output_grid[new_core_coord_row + coords[0], new_core_coord_row + coords[1]] = element_color

        elif action_name == "mirror":
            if (sec_action_input, third_action_input) == (-1, 0):
                # mirror up
                pass
            elif (sec_action_input, third_action_input) == (1, 0):
                # mirror down
                pass
            elif (sec_action_input, third_action_input) == (0, -1):
                # mirror left
                pass
            elif (sec_action_input, third_action_input) == (0, 1):
                # mirror right
                pass
            elif (sec_action_input, third_action_input) == (-1, -1):
                # mirror top-left
                pass
            elif (sec_action_input, third_action_input) == (-1, 1):
                # mirror top-right
                pass
            elif (sec_action_input, third_action_input) == (1, -1):
                # mirror bottom-left
                pass
            elif (sec_action_input, third_action_input) == (1, 1):
                # mirror bottom-right
                pass

        elif action_name == "resize_output_grid":
            self.working_output_grid, self.grid_dims = self.pad_array(
                np.zeros(shape=(sec_action_input, third_action_input))), \
                                                       (sec_action_input, third_action_input)

        elif action_name == "none":
            pass

        elif action_name == "done":
            if np.array_equal(self.working_output_grid, self.desired_output_grid):
                reward = 10.0
                print("succesfull done")
            else:
                reward = -10.0
                print("faulty done")
            self.done = True

        # we iterate through the elements in our grid (which is part of the input observation for our network)
        self.selected_element = self.get_next_selected_element()
        if self.selected_element is None and not self.selection_on_working_grid:
            self.grid = self.working_output_grid
            self.selected_element = self.get_next_selected_element()
            self.grid_selection_history = np.logical_or(self.grid == self.background_color, self.grid == 10).astype(int)
            self.selection_on_working_grid = True

        a1_mask = self.primary_action_mask()
        a2_mask, a3_mask, a4_mask = self.dependant_action_masks()

        if np.array_equal(self.working_output_grid, self.desired_output_grid):
            print("step solved")

        if self.done:
            print("env: done")

        self.running_reward += reward
        score = self.running_reward if self.done else 0

        return {"real_obs": tuple([self.grid, self.count_memory]),
                "action_mask": self.observation_space['action_mask'].sample()}, \
               score, \
               self.done, \
               {}

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        obs = np.array(list(self.env.unwrapped.state))
        return {}

    def get_state(self):
        return deepcopy(self.env), self.running_reward

    def get_next_selected_element(self):
        # plot_single_image(self.grid)
        selected_element, expand_set = [], set()
        non_selected = np.argwhere(self.grid_selection_history == 0)
        if non_selected.shape[0] == 0:  # and self.selection_on_working_grid:
            return None
        selected_element.append(non_selected[0])
        expand_set.add((non_selected[0][0], non_selected[0][1]))
        selected_color = self.grid[non_selected[0][0], non_selected[0][1]]
        self.grid[non_selected[0][0], non_selected[0][1]] = self.background_color
        while not not expand_set:
            to_be_expanded = expand_set.pop()
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    new_coord = (to_be_expanded[0] + i, to_be_expanded[1] + j)
                    if new_coord[0] < 0 or new_coord[0] >= self.grid_width or new_coord[1] < 0 or new_coord[
                        1] >= self.grid_height:
                        continue
                    if self.grid_selection_history[new_coord[0]][new_coord[1]] == 0 and self.grid[new_coord[0]][
                        new_coord[1]] == selected_color:
                        expand_set.add(new_coord)
                        self.grid_selection_history[new_coord[0]][new_coord[1]] = 1
                        selected_element.append(new_coord)

        for elem_idx_pair in selected_element:
            # if not self.selection_on_working_grid:
            # self.grid[elem_idx_pair[0], elem_idx_pair[1]] = self.background_color # remove elements from working grid
            self.grid_selection_history[elem_idx_pair[0], elem_idx_pair[1]] = 1

        return selected_element

    def primary_action_mask(self):
        if np.array_equal(self.working_output_grid, self.desired_output_grid):
            print("solved")
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])  # allowed: [8]
        if not self.output_size_fixed:
            resize_action_mask = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
        else:
            resize_action_mask = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1])

        if self.selected_element is None:
            return np.array([0, 1, 0, 0, 0, 0, 0, 0, 1])  # allowed: [1, 8]
        elif self.selection_on_working_grid:
            return np.array([1, 1, 1, 1, 1, 0, 0, 1, 1]) + resize_action_mask  # allowed: [0, 1, 2, 3, 4, 5, (6), 7, 8]
        else:
            return np.array([1, 0, 0, 1, 0, 0, 0, 1, 1]) + resize_action_mask  # allowed: [0, 3, 6, 7, 8]

    def dependant_action_masks(self):
        a2_mask, a3_mask, a4_mask = np.zeros(shape=(9, 30)), np.zeros(shape=(9, 30)), np.zeros(shape=(9, 10)),
        # if 2:remove, 7:none, 8:done, no followup action requred
        top_bound, bottom_bound, left_bound, right_bound = self.get_bounds()
        for primary_action in [0, 1, 3, 4]:
            np.put(a2_mask[primary_action, :], np.arange(top_bound, bottom_bound), 1)
            np.put(a3_mask[primary_action, :], np.arange(left_bound, right_bound), 1)
            np.put(a4_mask[primary_action, :], [self.background_color], 1)
            # primary action == 5
            np.put(a2_mask[5, :], np.arange(top_bound, bottom_bound), 1)
            np.put(a3_mask[5, :], np.arange(left_bound, right_bound), 1)
            np.put(a4_mask[5, :], [self.background_color], 1)
            # primary_action == 6:
            a2_mask[primary_action, :]
            a2_mask[6, :] = np.ones(shape=(30,))
            a3_mask[6, :] = np.ones(shape=(30,))
            a4_mask[6, :] = np.ones(shape=(10,))
        return a2_mask, a3_mask, a4_mask

    def render(self):
        plot_single_image(self.grid, padded=True)
        plot_single_image(self.working_output_grid, padded=True)

    def pad_array(self, array):
        y_pad = self.grid_height - array.shape[0]
        x_pad = self.grid_width - array.shape[1]

        return np.pad(array,
                      ((math.floor(y_pad / 2), math.ceil(y_pad / 2)), (math.floor(x_pad / 2), math.ceil(x_pad / 2))),
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
        for u_color in unique_out_colors:  # identify function for colors unique to output
            colormap[u_color] = u_color
            inverse_colormap[u_color] = u_color

        def map_func(x):
            return colormap[x]

        map_func = np.vectorize(map_func)

        if not single_input:
            return map_func(arrays[0]), map_func(arrays[1]), colormap, inverse_colormap, unique_out_colors
        else:
            return map_func(arrays[0]), colormap, inverse_colormap
