import numpy as np
import pandas as pd
from abstract_env import ReasoningEnv
from typing import List
import matplotlib.pyplot as plt
import random
import time

import os
import json
from pathlib import Path
from plot_functions import plot_task, plot_single_image

import numpy as np

data_path = Path('/home/yannick/Documents/abstract_reasoning/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

# initialize network (auto regressive policy)

# input: 30x30 grid

# Action Space
#   0 - copy: (30x, 30y) - 0 means current object, else bounding box
#   1 - recolor: (30x, 30y, 10color) - means coordiante & color
#   2 - remove: - means removing the highlighted element equiv. to setting to background color
#   3 - count: (30x, 30y) - 0 means counting the current element
#   4 - move: (30x, 30y) - move to x/y coordinate
#   5 - mirror: (top, right, bottom, left) - axis, mirrors in place (in bounding box)
#   6 - resize_output_grid: (30x, 30y) - hard coded for most examples
#   7 - none: do nothing / skip highlighted element
#   8 - done: - action to be submitted when done

tasks = []
for file in os.listdir(training_path):
    with open(os.path.join(training_path, file), 'r') as f:
        tasks.append(json.load(f))

training_iterations = 50
env = ReasoningEnv()
env.set_current_task(tasks[0])
env.reset()
env.step([1,2,3,4])

exit()

available_primary_actions = [0,1,2,3,4,5,6,7,8]
available_secondary_actions = np.arange(-29, 30,)
available_third_actions = np.arange(-29, 30)
available_fourth_actions = np.arange(0,)

start_time = time.time()
for iteration in range(training_iterations):
    iter_task_list = tasks
    random.shuffle(iter_task_list)

    for i, task in enumerate(iter_task_list):
        env.set_current_task(task)
        for _ in range(5):
            obs = env.reset()
            for _ in range(500):
                primary_action_mask = env.primary_action_mask()
                random_action = random.choice(list(set(available_primary_actions)-set(primary_action_mask)))
                sec_th_fth_masks = env.dependant_action_masks()
                scd_act = random.choice(list(set()-set(sec_th_fth_masks[0]))) if len(sec_th_fth_masks[0][random_action]) < len(available_secondary_actions) else None
                thi_act = random.choice(list(set(available_third_actions)-set(sec_th_fth_masks[1]))) if len(sec_th_fth_masks[1][random_action]) < len(available_third_actions) else Noneavailable_secondary_actions
                fth_act = random.choice(list(set(available_fourth_actions)-set(sec_th_fth_masks[2]))) if len(sec_th_fth_masks[2][random_action]) < len(available_fourth_actions) else None
                # print("actions", random_action, scd_act, thi_act, fth_act)
                obs, rew, done, _ = env.step([random_action, scd_act, thi_act, fth_act])
                if env.done:
                    obs = env.reset()
        print(i)
    break

print("FINISHED in {}s".format(time.time()-start_time))


