import collections
import math

import numpy as np

class Node:

    def __init__(self, action, obs, done, reward, state, parent=None):
        self.env = parent.env
        self.action = action

        self.is_expanded = False
        self.parent = parent
        self.children = {}

        self.action_space_size = self.env.action_space
        self.child_total_value = np.zeros(
            [self.action_space_size], dtype=np.float32 # Q
        )
        self.policy_values = np.zeros(
            [self.action_space_size], dtype=np.float32 # P
        )
        self.valid_actions = obs["action_mask"].astype(np.bool)

        self.reward = reward
        self.done = done
        self.state = state

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action]

    @property
    def total_value(self):
        return self.parent.child_total_visits[self.action]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_visits[self.action]

    def best_action(self):
        child_score = self.child_total_value / (1 + self.child_number_visits)
        masked_child_score = child_score
        masked_child_score[~self.valid_actions] = -np.inf
        return np.argmax(masked_child_score)

    def  select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def get_child(self, action):
        if action not in self.children:
            self.env.set_state(self.state)
            obs, reward, done, _ = self.env.step(action)
            next_state = self.env.step()
            self.children[action] = Node(
                state=next_state,
                action=action,
                parent=self,
                reward=reward,
                done=done,
                obs=obs,
                mcts=self.mcts)
        return self.children[action]

    def backup(self, value):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value
            current = current.parent

class RootParentNode:
    def _init_(self, env):
        self.parent = None
        self.children = {}
        self.p_children = None
        self.env = env

class TreeSearch:
    def __init__(self, model, tree_search_param):
        self.model = model
        self.temperature = tree_search_param["temperature"]
        self.dir_epsilon = tree_search_param["dirichlet_epsilon"]
        self.num_sims = tree_search_param["num_simulations"]
        self.exploit = tree_search_param["argmax_tree_policy"]
        self.add_dirichlet_noise = tree_search_param["add_dirichlet_noise"]
        self.c_puct = tree_search_param["puct_coefficient"]

    def compute_action(self, node):
        for _ in range(self.num_sims):
            leaf  = node.select()
            if leaf.done:
                 value = leaf.reward
            else:
                child_priors, value = self.model.compute_priors_and_value(leaf.obs)
                if self.add_dirichlet_noise:
                    child_priors = (1- self.dir_epsilon) * child_priors
                    child_priors += self.dir_epsilon * np.random.dirichlet([self.dir_noise] * child_priors.size)

                    leaf.expand(child_priors)
                leaf.backup(value)

        tree_policy = node.child_number_visits / node.number_visits
        tree_policy = tree_policy / np.max(tree_policy)
        tree_policy = np.power(tree_policy, self.temperature)
        tree_policy = tree_policy / np.sum(tree_policy)

        if self.exploit: # for inference
            action = np.argmax(tree_policy)
        else:
            action = np.random.choice(
                np.arange(node.action_space_size), p=tree_policy)
        return tree_policy, action, node.children[action]




