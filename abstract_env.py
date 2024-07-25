import gymnasium as gym
import numpy as np
import tkinter as tk
import math
from itertools import chain
from matplotlib import colors

from copy import deepcopy

action_mapping = {0: "copy", 1: "recolor", 2: "remove", 3: "count", 4: "move", 5: "mirror",
                  6: "resize_output_grid", 7: "none", 8: "done"}

original_cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

padded_cmap= colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#F5F5F5', '#FFD700'])

class ReasoningEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode: str|None = None):

        
        self.grid_height, self.grid_width = 30, 30
        self.countDim = 10  # number of counts we can save, we expect that 10 are enough for current problems
        self.grid = np.zeros(shape=(self.grid_height, self.grid_width), dtype=np.int32)
        self.selection_mask = np.zeros(shape=(self.grid_height, self.grid_width), dtype=np.int32)
        self.count_memory, self.count_memory_idx = np.zeros(shape=(self.countDim,), dtype=np.int32), 0
        self.desired_output_grid = np.zeros(shape=(self.grid_height, self.grid_width), dtype=np.int32)
        self.working_output_grid, self.grid_dims = np.zeros(shape=(self.grid_height, self.grid_width), dtype=np.int32), \
                                                   (self.grid_height, self.grid_width)
        # selected elements is a list of grid indices belonging to the selected element, first element as core coord. +
        # following coordinates relative to core coord.
        self.selected_element = [(0, 0)]
        # if we iterated all elements of the input grid, continue iteration on working grid
        self.selection_on_working_grid = False
        self.grid_selection_history = np.zeros(shape=(self.grid_height, self.grid_width), dtype=np.int32)
        self.current_task_dims, self.output_size_fixed = [], False
        self.copy_input_to_working_grid = False

        self.background_color = 0

        self.running_reward = 0

        self.render_mode = render_mode

        self.current_task = None
        self.current_demo_task = None
        self.current_demo_task_index = 0
        self.current_test_task = None
        self.current_test_task_index = 0

        self.current_demo_colormaps, self.current_demo_inv_colormaps = [], []
        self.current_test_colormaps, self.current_test_inv_colormaps = [], []

        self.done = False

        self.window = None
        self.canvas = None

        self.observation_space = gym.spaces.Dict({
            # Observation has
            # - 3 channels for working grid, selection mask and output image
            # - 10 channels for the colors, 1 for background and one for the selection mask
            "real_obs": gym.spaces.Tuple(
                [gym.spaces.Box(0, 10, shape=(self.grid_height, self.grid_width, 3), dtype=np.int32),
                 gym.spaces.Box(0, 900, shape=(self.countDim,), dtype=np.int32)]),
            "action_mask": gym.spaces.Tuple([gym.spaces.Box(0, 1, shape=(19,), dtype=np.int32),
                                             gym.spaces.Box(0, 1, shape=(9, 30), dtype=np.int32),
                                             gym.spaces.Box(0, 1, shape=(9, 30), dtype=np.int32)])
        })
        # Action Space
        #   0 - copy: (30x, 30y) - 0 means current object, else bounding box
        #   1 - recolor_0: recolor selected element with color 0
        #   2 - recolor_1: recolor selected element with color 1
        #   3 - recolor_2: recolor ...
        #   4 - recolor_3: recolor ...
        #   5 - recolor_4: recolor ...
        #   6 - recolor_5: recolor ...
        #   7 - recolor_6: recolor ...
        #   8 - recolor_7: recolor ...
        #   9 - recolor_8: recolor ...
        #   10 - recolor_9: recolor ...
        #   11 - draw: (30x, 30y) - draw at specified coordinate with "special color", if not recolored defaults
        #   to first color
        #   12 - remove: - means removing the highlighted element equiv. to setting to background color
        #   13 - count: counting the tiles in the currently selected element
        #   14 - move: (30x, 30y) - move to x/y coordinate
        #   15 - mirror: (top, right, bottom, left) - axis, mirrors in place (in bounding box)
        #   16 - resize_output_grid: (30x, 30y) - hard coded for most examples
        #   17 - none: do nothing / skip highlighted element
        #   18 - done: - action to be submitted when done
        self.action_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(19), gym.spaces.Discrete(30), gym.spaces.Discrete(30), ])

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
        a2_mask, a3_mask = self.dependent_action_masks()

        if self.render_mode == "human":
            self.render()

        return {"real_obs": tuple([np.dstack([self.grid, self.selection_mask, self.working_output_grid]),
                                   self.count_memory]),
                "action_mask": tuple([a1_mask, a2_mask, a3_mask])}, {}

    def set_current_task(self, task):
        self.current_task = task

        self.current_demo_colormaps, self.current_demo_inv_colormaps, unique_test_colors = [], [], []
        self.current_task_dims, in_out_similarities, pre_padding_grid_sizes = [], [], []
        for task in self.current_task['train']:
            self.current_task_dims.append((np.array(task['input']).shape, np.array(task['output']).shape))
            task['input'], task['output'], colmap, inv_colmap, demo_utc = self.squash_colors(
                [np.array(task['input'], dtype=np.int32), np.array(task['output'], dtype=np.int32)])
            self.current_demo_colormaps.append(colmap)
            self.current_demo_inv_colormaps.append(inv_colmap)
            unique_test_colors.append(demo_utc)
            if task['input'].shape == task['output'].shape:
                # if more than 50% of the input and output grid are equal, and the shapes match,
                # copy the input grid to the working output grid
                in_out_similarities.append(np.count_nonzero(np.equal(task['input'], task['output']) == True))
            else:
                in_out_similarities.append(0)
            pre_padding_grid_sizes.append(task['output'].shape[0] * task['output'].shape[1])

        unique_test_colors = np.unique(chain.from_iterable(unique_test_colors))
        for task in self.current_task['test']:
            task['input'], colmap, inv_colmap = self.squash_colors(
                [np.array(task['input'], dtype=np.int32)], single_input=True, test_unique_colors=unique_test_colors)
            self.current_test_colormaps.append(colmap)
            self.current_test_inv_colormaps.append(inv_colmap)

        if all([in_out_similarities[i] > 0.7 * pre_padding_grid_sizes[i]
                for i in range(len(in_out_similarities))]):
            self.copy_input_to_working_grid = True

        self.current_demo_task_index = 0
        self.current_test_task_index = 0

    def get_number_of_demo_test_in_taks(self):
        return len(self.current_task['train']), len(self.current_task['test'])

    @staticmethod
    def grid_to_one_hot_channel_output(grid: np.array, nr_of_values):
        return_array = np.zeros(shape=grid.shape + (nr_of_values,))

        for channel in range(nr_of_values):
            # only set the specific
            return_array[:, :, channel] = np.where(grid == channel, grid, 0)

        return return_array

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
        """

        reward = 0.0

        a1_mask = self.primary_action_mask()
        a2_mask, a3_mask = self.dependent_action_masks()

        # the core coord is a random cell (usually top-left) from the selected element, serves as a local coordiante
        # base (0,0) which offset is added to

        sec_action_input = action[1]  # row coordinate - domain -30,+30
        third_action_input = action[2]  # col coordinate - domain -30,+30

        # copy
        if action[0] == 0:
            core_coord_row, core_coord_col = self.selected_element[0]
            new_core_coord_row, new_core_coord_col = sec_action_input - core_coord_row, \
                                                     third_action_input - core_coord_col
            top_bound, bottom_bound, left_bound, right_bound = self.get_bounds()
            for coords in self.selected_element:
                if new_core_coord_row + coords[0] < top_bound or new_core_coord_row + coords[0] >= bottom_bound \
                        or new_core_coord_col + coords[1] < left_bound or new_core_coord_col + coords[1] >= right_bound:
                    continue
                self.working_output_grid[new_core_coord_row + coords[0], new_core_coord_col + coords[1]] = \
                    self.grid[coords[0], coords[1]]

        elif 10 >= action[0] >= 1:  # recolor
            # fill the selected element
            color = action[0] - 1
            if self.selected_element is not None:
                for coords in self.selected_element:
                    self.working_output_grid[coords[0], coords[1]] = color

        elif action[0] == 11:  # draw at specified position with special color (10)
            # 10 is the special color reserved for drawing
            self.working_output_grid[sec_action_input, third_action_input] = 10

        elif action[0] == 12:  # remove
            for coords in self.selected_element:
                self.working_output_grid[coords[0], coords[1]] = self.background_color

        elif action[0] == 13:  # count
            self.count_memory[self.count_memory_idx] = len(self.selected_element)
            self.count_memory_idx = (self.count_memory_idx + 1) % self.countDim

        elif action[0] == 14:  # move
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

        elif action[0] == 15:
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

        elif action[0] == 16:  # resize output working grid
            self.working_output_grid, self.grid_dims = self.pad_array(
                np.zeros(shape=(sec_action_input, third_action_input))), \
                                                       (sec_action_input, third_action_input)

        elif action[0] == 17:  # none
            pass

        elif action[0] == 18:  # done
            if np.array_equal(self.working_output_grid, self.desired_output_grid):
                reward = 10.0
                print("succesfull done")
            else:
                reward = -10.0
            self.done = True

        # we iterate through the elements in our grid (which is part of the input observation for our network)
        self.selected_element = self.get_next_selected_element()
        if self.selected_element is None and not self.selection_on_working_grid:
            self.grid = self.working_output_grid
            self.selected_element = self.get_next_selected_element()
            self.grid_selection_history = np.logical_or(self.grid == self.background_color, self.grid == 10).astype(int)
            self.selection_on_working_grid = True

        a1_mask = self.primary_action_mask()
        a2_mask, a3_mask = self.dependent_action_masks()

        self.running_reward += reward
        score = self.running_reward if self.done else 0

        if self.render_mode == "human":
            self.render()

        return {"real_obs": tuple([np.dstack([self.grid, self.selection_mask, self.working_output_grid]),
                                   self.count_memory]), "action_mask": tuple([a1_mask, a2_mask, a3_mask])}, \
               score, self.done, False, {}

    # def set_state(self, state):
    #     self.running_reward = state[1]
    #     self.env = deepcopy(state[0])
    #
    # def get_state(self):
    #     return deepcopy(self.env), self.running_reward

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
                    if new_coord[0] < 0 or new_coord[0] >= self.grid_width or new_coord[1] < 0 \
                            or new_coord[1] >= self.grid_height:
                        continue
                    if self.grid_selection_history[new_coord[0]][new_coord[1]] == 0 \
                            and self.grid[new_coord[0]][new_coord[1]] == selected_color:
                        expand_set.add(new_coord)
                        self.grid_selection_history[new_coord[0]][new_coord[1]] = 1
                        selected_element.append(new_coord)

        for elem_idx_pair in selected_element:
            # if not self.selection_on_working_grid:
            # self.grid[elem_idx_pair[0], elem_idx_pair[1]] = self.background_color # remove elements from working grid
            self.grid_selection_history[elem_idx_pair[0], elem_idx_pair[1]] = 1

        self.selection_mask = np.zeros(shape=(self.grid_height, self.grid_width), dtype=np.int32)
        np.put(self.selection_mask, selected_element, 1)


        return selected_element

    def primary_action_mask(self):
        return_action_mask = np.zeros(shape=(19,))
        if np.array_equal(self.working_output_grid, self.desired_output_grid):
            print("solved")
            np.put(return_action_mask, 18, 1)  # allowed: done
            return return_action_mask
        if not self.output_size_fixed:
            np.put(return_action_mask, 16, 1)  # allowed: resize_grid

        if self.selected_element is None:
            np.put(return_action_mask, [11, 18], 1)  # allowed: draw, done
        elif self.selection_on_working_grid:
            # only allow recoloring to
            allowed_colors = list(self.current_demo_inv_colormaps[self.current_demo_task_index].keys())
            np.put(return_action_mask, [0, 1, 11, 12, 13, 14, 17, 18] + allowed_colors, 1)
        else:
            np.put(return_action_mask, [0, 17, 18], 1)  # allowed: [0, 3, 6, 7, 8]

        return return_action_mask

    def dependent_action_masks(self):
        a2_mask, a3_mask = np.zeros(shape=(19, 30)), np.zeros(shape=(19, 30))
        # if 1-10, 12:remove, 17:none, 18:done, no followup action allowe
        top_bound, bottom_bound, left_bound, right_bound = self.get_bounds()
        for primary_action in [0, 11, 13, 14]:
            np.put(a2_mask[primary_action, :], np.arange(top_bound, bottom_bound), 1)
            np.put(a3_mask[primary_action, :], np.arange(left_bound, right_bound), 1)
        # primary_action == 16
        a2_mask[6, :] = np.ones(shape=(30,))
        a3_mask[6, :] = np.ones(shape=(30,))
        return a2_mask, a3_mask

    def render(self):
        #plot_single_image(self.grid, padded=True)
        #plot_single_image(self.working_output_grid, padded=True)

        if self.window is None:
            self.window = tk.Tk()
            self.window.title("Reasoning Environment")
            self.canvas = tk.Canvas(self.window, width=600, height=600)
            self.canvas.pack()

        self.canvas.delete("all")
        tile_size = 20
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                color = self.working_output_grid[i, j]
                self.canvas.create_rectangle(j * tile_size, i * tile_size, (j + 1) * tile_size, (i + 1) * tile_size,
                                             fill=colors.to_hex(padded_cmap.colors[color.astype(np.int16)]), outline="black")

        if self.render_mode == 'rgb_array':
            self.window.update()
            self.window.after(1000)
            self.window.update_idletasks()
            self.window.update()
            self.canvas.postscript(file="test.ps", colormode='color')
            self.window.update_idletasks()
            self.window.update()
        elif self.render_mode == "human":
            self.window.update_idletasks()
            self.window.update()

    def pad_array(self, array):
        y_pad = self.grid_height - array.shape[0]
        x_pad = self.grid_width - array.shape[1]

        # 10 is the special color for drawing, 11 the padding background color
        return np.pad(array,
                      ((math.floor(y_pad / 2), math.ceil(y_pad / 2)), (math.floor(x_pad / 2), math.ceil(x_pad / 2))),
                      constant_values=11)

    def squash_colors(self, arrays, single_input=False, test_unique_colors=None):
        if not single_input:
            colors_in_example = np.unique(np.concatenate([np.unique(arrays[0]), np.unique(arrays[1])]))
            unique_out_colors = [color for color in colors_in_example if color not in np.unique(arrays[0])]
        else:
            colors_in_example = np.unique(arrays[0])
            unique_out_colors = test_unique_colors

        colormap = {}
        inverse_colormap = {}
        color_index = 1
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
