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
#   11 - draw: (30x, 30y) - draw at specified coordiante with "special color", if not recolored defaults to first color
#   12 - remove: - means removing the highlighted element equiv. to setting to background color
#   13 - count: (30x, 30y) - 0 means counting the current element
#   14 - move: (30x, 30y) - move to x/y coordinate
#   15 - mirror: (top, right, bottom, left) - axis, mirrors in place (in bounding box)
#   16 - resize_output_grid: (30x, 30y) - hard coded for most examples
#   17 - none: do nothing / skip highlighted element
#   18 - done: - action to be submitted when done

tasks = []
for file in os.listdir(training_path):
    with open(os.path.join(training_path, file), 'r') as f:
        tasks.append(json.load(f))

training_iterations = 50
env = ReasoningEnv()
env.set_current_task(tasks[0])
env.reset()

for iteration in range(training_iterations):
    iter_task_list = tasks
    random.shuffle(iter_task_list)

    for i, task in enumerate(iter_task_list):
        start_time = time.time()
        env.set_current_task(task)
        for _ in range(5):
            obs = env.reset()
            for _ in range(500):
                primary_action_mask = obs["action_mask"][0]
                primary_action = np.random.choice(np.argwhere(primary_action_mask == 1).flatten())
                secondary_action_mask, third_action_mask = obs["action_mask"][1], obs["action_mask"][1]
                if np.count_nonzero(secondary_action_mask) > 0 and np.count_nonzero(third_action_mask) > 0:
                    secondary_action = np.random.choice(np.argwhere(secondary_action_mask == 1).flatten())
                    third_action = np.random.choice(np.argwhere(third_action_mask == 1).flatten())
                obs, rew, done, _ = env.step([primary_action, secondary_action, third_action])
                if env.done:
                    obs = env.reset()
        print(i, " time for task:", time.time()-start_time, "s")
    break

print("FINISHED in {}s".format(time.time()-start_time))


