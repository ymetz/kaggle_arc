import numpy as np
import pandas as pd
from abstract_env import ReasoningEnv
from typing import List
import matplotlib.pyplot as plt

import os
import json
from pathlib import Path
from plot_functions import plot_task, plot_single_image

import numpy as np

data_path = Path('.')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

# max_shape = (0,0)
# num_larger = 0
# file = ""
# num_total = 0
# for file in os.listdir(training_path):
#     with open(os.path.join(training_path, file), 'r') as f:
#         task = json.load(f)
#     for task in task['train']:
#         shape = np.array(task['input']).shape
#         if shape[0] > 24 or shape[1] > 24:
#             num_larger += 1
#         if shape[0] > max_shape[0] or shape[1] > max_shape[1]:
#             max_shape = shape
#             file = file
#         num_total += 1
# print(shape, file, num_total, num_larger)

# unique_colors = []
# dif_to_out = 0
# for file in os.listdir(training_path):
#     with open(os.path.join(training_path, file), 'r') as f:
#         task = json.load(f)
#     task_cols = []
#     plot_task_var = False
#     for demo in task['train']:
#         unique_elems_in = np.unique(np.array(demo['input']))
#         unique_elems_out = np.unique(np.array(demo['output']))
#         cols = [col for col in unique_elems_out if col not in unique_elems_in]
#         if len(cols) > 0:
#             task_cols.append(cols)
#             plot_task_var = True
#     if plot_task_var:
#         print(task_cols)
#         plot_task(task)
#         unique_colors.extend([unique_elems_in.shape[0],unique_elems_out.shape[0]])
# print(np.unique(np.array(unique_colors)))

tasks = []
for file in os.listdir(training_path):
    with open(os.path.join(training_path, file), 'r') as f:
        tasks.append(json.load(f))


env = ReasoningEnv(tasks=tasks)
obs = env.reset()
# env.render()


def create_submission():
    submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')

    def flattener(pred):
        str_pred = str([row for row in pred])
        str_pred = str_pred.replace(', ', '')
        str_pred = str_pred.replace('[[', '|')
        str_pred = str_pred.replace('][', '|')
        str_pred = str_pred.replace(']]', '|')
        return str_pred

    for output_id in submission.index:
        task_id = output_id.split('_')[0]
        pair_id = int(output_id.split('_')[1])
        f = str(test_path / str(task_id + '.json'))
        with open(f, 'r') as read_file:
            task = json.load(read_file)
        # skipping over the training examples, since this will be naive predictions
        # we will use the test input grid as the base, and make some modifications
        data = task['test'][pair_id]['input']  # test pair input
        # for the first guess, predict that output is unchanged
        pred_1 = flattener(data)
        # for the second guess, change all 0s to 5s
        data = [[5 if i == 0 else i for i in j] for j in data]
        pred_2 = flattener(data)
        # for the last gues, change everything to 0
        data = [[0 for i in j] for j in data]
        pred_3 = flattener(data)
        # concatenate and add to the submission output
        pred = pred_1 + ' ' + pred_2 + ' ' + pred_3 + ' '
        submission.loc[output_id, 'output'] = pred

    submission.to_csv('submission.csv')

create_submission()






