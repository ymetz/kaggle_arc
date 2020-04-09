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

data_path = Path('C:\\Users\\Yannick\\PycharmProjects\\abstract_reasoning\\kaggle_arc')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

# initialize network (auto regressive policy)

# input: 30x30 grid

# Action Space
#   copy: (30x, 30y) - 0 means current object, else bounding box
#   recolor: (30x, 30y, 10color) - means coordiante & color
#   remove: - means removing the highlighted element equiv. to setting to background color
#   count: (30x, 30y) - 0 means counting the current element, else color in bounding box
#   move: (30x, 30y) - move to x/y coordinate
#   mirror: (top, right, bottom, left) - axis, mirrors in place (in bounding box)
#   resize_output_grid: (30x, 30y) - hard coded for most examples
#   none: do nothing / skip highlighted element
#   done: - action to be submitted when done

tasks = []
for file in os.listdir(training_path):
    with open(os.path.join(training_path, file), 'r') as f:
        tasks.append(json.load(f))

training_iterations = 50
env = ReasoningEnv(tasks=tasks)

start_time = time.time()
for iteration in range(training_iterations):
    iter_task_list = tasks
    random.shuffle(iter_task_list)

    for i, task in enumerate(iter_task_list):
        env.set_current_task(task)
        for _ in range(5):
            obs = env.reset()
            for _ in range(500):
                elem = env.get_next_selected_element()
                if env.done_current_demo:
                    obs = env.reset()
        print(i)

    break

    print("====> FINISHED TRAINING ITERATION ", iteration)

print("FINISHED in {}s".format(time.time()-start_time))


