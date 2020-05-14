import gym
from gym.spaces import Discrete, Tuple
import random
from pathlib import Path
import os
import json
import time
import numpy as np

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.tuple_actions import TupleActions
from ray.rllib.utils import try_import_tf
from ray.rllib.agents.ppo import PPOTrainer
import argparse

from abstract_env import ReasoningEnv

tf = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument("--stop", type=int, default=200)
parser.add_argument("--num_cpus", type=int, default=0)


class ReasoningAutoregressiveOutput(ActionDistribution):
    """Action Distribution P(a1, a2, a3) = P(a1) * P(a2 | a1) * (P(a3 | a1, a2)"""

    @staticmethod
    def required_model_output_shape(self, model_config):
        return 20  # controls model output feature vector size

    def deterministic_sample(self):
        # first, sample a1
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        # sample a2 conditioned on a1
        a2_dist = self._a2_distribution(a1)
        a2 = a1_dist.deterministic_sample()

        a3_dist = self._a3_distribution(a1, a2)
        a3 = a3_dist.deterministic_sample()

        a4_dist = self._a4_distribution(a1, a2, a3)
        a4 = a4_dist.deterministic_sample()

        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2) + a3_dist.logp(a3) + a4_dist.logp(a4)

        # return the action tuple
        return TupleActions([a1, a2, a3, a4])

    def sample(self):
        # first, sample a1
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        # sample a2 conditioned on a1
        a2_dist = self._a2_distribution(a1)
        a2 = a1_dist.sample()

        a3_dist = self._a3_distribution(a1, a2)
        a3 = a3_dist.sample()

        a4_dist = self._a4_distribution(a1, a2, a3)
        a4 = a4_dist.sample()

        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2) + a3_dist.logp(a3) + a4_dist.logp(a4)

        # return the action tuple
        return TupleActions([a1, a2, a3, a4])

    def logp(self, actions):
        a1, a2, a3, a4 = actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3]
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        a2_vec = tf.expand_dims(tf.cast(a2, tf.float32), 1)
        a3_vec = tf.expand_dims(tf.cast(a3, tf.float32), 1)
        a1_logits, a2_logits, a3_logits, a4_logits = self.model.action_model([self.inputs, a1_vec, a2_vec, a3_vec])
        return (
                Categorical(a1_logits).logp(a1) + Categorical(a2_logits).logp(a2) + Categorical(a3_logits).logp(a3)
                + Categorical(a4_logits).logp(a4)
        )

    def sampled_action_logp(self):
        return tf.exp(self._action_logp)

    def entropy(self):
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())
        a3_dist = self._a3_distribution(a1_dist.sample(), a2_dist.sample())
        a4_dist = self._a4_distribution(a1_dist.sample(), a2_dist.sample(), a3_dist.sample())
        return a1_dist.entropy() + a2_dist.entropy() + a3_dist.entropy() + a4_dist.entropy()

    def kl(self, other):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()
        a3_terms = self._a3_distribution(a1, a2).kl(other._a3_distribution(a1, a2))
        a3_dist = self._a3_distribution(a1, a2)
        a3 = a3_dist.sample()
        a4_terms = self._a4_distribution(a1, a2, a3).kl(other._a4_distribution(a1, a2, a3))

        return a1_terms + a2_terms + a3_terms + a4_terms

    def _a1_distribution(self):
        BATCH = tf.shape(self.inputs)[0]
        a1_logits, _, _, _ = self.model.action_model(
            [self.inputs, tf.zeros((BATCH, 1)), tf.zeros((BATCH, 1)), tf.zeros((BATCH, 1))])
        a1_logits = a1_logits * self.model.a1_mask
        a1_dist = Categorical(a1_logits)
        return a1_dist

    def _a2_distribution(self, a1):
        BATCH = tf.shape(self.inputs)[0]
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        _, a2_logits, _, _ = self.model.action_model(
            [self.inputs, a1_vec, tf.zeros((BATCH, 1)), tf.zeros((BATCH, 1))])
        # a2_logits = a2_logits + self.model.a2_mask[tf.keras.backend.eval(a1)[0]]
        a2_dist = Categorical(a2_logits)
        return a2_dist

    def _a3_distribution(self, a1, a2):
        BATCH = tf.shape(self.inputs)[0]
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        a2_vec = tf.expand_dims(tf.cast(a2, tf.float32), 1)
        a1_logits, _, a3_logits, _ = self.model.action_model([self.inputs, a1_vec, a2_vec, tf.zeros((BATCH, 1))])
        max_a1_action = tf.math.argmax(a1_logits, axis=1)
        # a3_logits = a3_logits + self.model.a3_mask[max_a1_action]
        a3_dist = Categorical(a3_logits)
        return a3_dist

    def _a4_distribution(self, a1, a2, a3):
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        a2_vec = tf.expand_dims(tf.cast(a2, tf.float32), 1)
        a3_vec = tf.expand_dims(tf.cast(a3, tf.float32), 1)
        a1_logits, _, _, a4_logits = self.model.action_model([self.inputs, a1_vec, a2_vec, a3_vec])
        max_a1_action = tf.math.argmax(a1_logits, axis=1)
        # a4_logits = a4_logits + self.model.a4_mask[max_a1_action]
        a4_dist = Categorical(a4_logits)
        return a4_dist


class AutoregressiveActionsModel(TFModelV2):
    """Implements the '.action_model' branch requried above"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, true_obs_shape=(910,)):
        super(AutoregressiveActionsModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        # Inputs
        obs_input = tf.keras.layers.Input(shape=true_obs_shape, name="obs_input")
        a1_input = tf.keras.layers.Input(shape=(1,), name="a1_input")
        a2_input = tf.keras.layers.Input(shape=(1,), name="a2_input")
        a3_input = tf.keras.layers.Input(shape=(1,), name="a3_input")
        ctx_input = tf.keras.layers.Input(shape=(num_outputs,), name="ctx_input")

        self.a1_mask = None
        self.a2_mask = None
        self.a3_mask = None
        self.a4_mask = None
        # self.register_variables([self.a1_mask, self.a2_mask, self.a3_mask, self.a4_mask])

        # Output of the model (normally 'logits' but for an autoregressive dist this is more like a context/feature
        # layer encoding the obs)
        context = tf.keras.layers.Dense(
            num_outputs,
            name="hidden",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(obs_input)

        # V(s)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(context)

        # P(a1 | obs)
        a1_logits = tf.keras.layers.Dense(
            9,
            name="a1_logits",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(ctx_input)

        # P(a2 | a1)
        a2_context = tf.keras.layers.Concatenate(axis=1)([ctx_input, a1_input])
        a2_hidden = tf.keras.layers.Dense(
            40,
            name="a2_hidden",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(a2_context)
        a2_logits = tf.keras.layers.Dense(
            30,
            name="a2_logits",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(a2_hidden)

        # P(a3 | a1, a2)
        a3_context = tf.keras.layers.Concatenate(axis=1)([ctx_input, a1_input, a2_input])
        a3_hidden = tf.keras.layers.Dense(
            40,
            name="a3_hidden",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(a3_context)
        a3_logits = tf.keras.layers.Dense(
            30,
            name="a3_logits",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(a3_hidden)

        # P(a4 | a1, a2, a3)
        a4_context = tf.keras.layers.Concatenate(axis=1)([ctx_input, a1_input, a2_input, a3_input])
        a4_hidden = tf.keras.layers.Dense(
            40,
            name="a4_hidden",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(a4_context)
        a4_logits = tf.keras.layers.Dense(
            10,
            name="a4_logits",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(a4_hidden)

        # Base layers
        self.base_model = tf.keras.Model(obs_input, [context, value_out])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

        # Autoregressive action sampler
        self.action_model = tf.keras.Model([ctx_input, a1_input, a2_input, a3_input],
                                           [a1_logits, a2_logits, a3_logits, a4_logits])
        self.action_model.summary()
        self.register_variables(self.action_model.variables)

    def forward(self, input_dict, state, seq_lens):
        context, self._value_out = self.base_model(input_dict["obs_flat"][:, :910])
        self.a1_mask, self.a2_mask, self.a3_mask, self.a4_mask = input_dict["obs"]["action_mask"][0], \
                                                                 input_dict["obs"]["action_mask"][1], \
                                                                 input_dict["obs"]["action_mask"][2], \
                                                                 input_dict["obs"]["action_mask"][3],
        return context, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


if __name__ == "__main__":
    args = parser.parse_args()
    data_path = Path('.')
    training_path = data_path / 'training'
    evaluation_path = data_path / 'evaluation'
    test_path = data_path / 'test'

    ray.init(num_cpus=args.num_cpus or None)
    ModelCatalog.register_custom_model("autoregressive_model",
                                       AutoregressiveActionsModel)
    ModelCatalog.register_custom_action_dist("reasoning_autoreg_output",
                                             ReasoningAutoregressiveOutput)

    def train(config, reporter):
        in_tasks = config.pop("tasks")
        trainer = PPOTrainer(config=config, env=ReasoningEnv)
        while True:
            distributed_task = random.choice(in_tasks)
            trainer.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.set_current_task(distributed_task)
                )
            )
            result = trainer.train()
            print(result)
            reporter(**result)


    tasks = []
    for file in os.listdir(training_path):
        with open(os.path.join(training_path, file), 'r') as f:
            tasks.append(json.load(f))

    tune.run(
        train,
        resources_per_trial={'gpu': 1},
        config={
            "env": ReasoningEnv,
            "env_config": {"tasks": tasks},
            "gamma": 0.9,
            "num_gpus": 1,
            "num_envs_per_worker": 128,
            "num_workers": 10,
            "tasks": tasks
        },
    )
