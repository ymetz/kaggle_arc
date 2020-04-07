import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from typing import List

original_cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

padded_cmap= colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#F5F5F5'])


def plot_task(task, padded=False):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    if padded:
        this_cmap = padded_cmap
        norm = colors.Normalize(vmin=0, vmax=10)
    else:
        this_cmap = original_cmap
        norm = colors.Normalize(vmin=0, vmax=9)
    num_of_train_tasks = len(task['train'])
    num_of_test_tasks = len(task['test'])
    fig, axs = plt.subplots(2, num_of_train_tasks+num_of_test_tasks, figsize=(12, 12))
    for i, train_task in enumerate(task['train']):
        axs[0][i].imshow(train_task['input'], cmap=this_cmap, norm=norm)
        axs[0][i].axis('off')
        axs[0][i].set_title('Train Input '+str(i))
        axs[1][i].imshow(train_task['output'], cmap=this_cmap, norm=norm)
        axs[1][i].axis('off')
        axs[1][i].set_title('Train Output ' + str(i))
    for i, test_task in enumerate(task['test']):
        j = i + num_of_train_tasks
        axs[0][j].imshow(task['test'][0]['input'], cmap=this_cmap, norm=norm)
        axs[0][j].axis('off')
        axs[0][j].set_title('Test Input '+str(i))
        axs[1][j].imshow(task['test'][0]['output'], cmap=this_cmap, norm=norm)
        axs[1][j].axis('off')
        axs[1][j].set_title('Test Output '+str(i))
    plt.tight_layout()
    plt.show()


def plot_single_image(array, padded=False):
    if padded:
        this_cmap = padded_cmap
        norm = colors.Normalize(vmin=0, vmax=10)
    else:
        this_cmap = original_cmap
        norm = colors.Normalize(vmin=0, vmax=9)
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    axs.imshow(array, cmap=this_cmap, norm=norm)
    axs.axis('off')
    plt.tight_layout()
    plt.show()
