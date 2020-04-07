import gym
from gym.spaces import Discrete, Tuple
import random

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.tuple_actions import TupleActions
from ray.rllib.utils import try_import_tf
import argparse

from abstract_env import ReasoningEnv

tf = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")  # try PG, PPO, IMPALA
parser.add_argument("--stop", type=int, default=200)
parser.add_argument("--num-cpus", type=int, default=0)


class ReasoningAutoregressiveOutput(ActionDistribution):
    """Action Distribution P(a1, a2, a3) = P(a1) * P(a2 | a1) * (P(a3 | a1, a2)"""

    @staticmethod
    def required_model_output_shape(self, model_config):
        return 16

    def deterministic_sample(self):
        # first, sample a1
        a1_dist = self._a1_distrubtion()
        a1 = a1_dist.determinstic_sample()

        # sample a2 conditioned on a1
        a2_dist = self._a2_distribution(a1)
        a2 = a1_dist.determinstic_sample()

        a3_dist = self._a3_distribution(a1, a2)
        a3 = a3_dist.deterministic_sample()
        self._action_logp = a1_dist(a1) + a2_dist.logp(a2) + a3_dist.logp(a3)

        # return the action tuple
        return TupleActions([a1, a2, a3])

    def deterministic_sample(self):
        # first, sample a1
        a1_dist = self._a1_distrubtion()
        a1 = a1_dist.sample()

        # sample a2 conditioned on a1
        a2_dist = self._a2_distribution(a1)
        a2 = a1_dist.sample()

        a3_dist = self._a3_distribution(a1, a2)
        a3 = a3_dist.sample()
        self._action_logp = a1_dist(a1) + a2_dist.logp(a2) + a3_dist.logp(a3)

        # return the action tuple
        return TupleActions([a1, a2, a3])

    def logp(self, actions):
        a1, a2, a3 = actions[:, 0], actions[:, 1], actions[:, 2]
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        a2_vec = tf.expand_dims(tf.cast(a2, tf.float32), 1)
        a1_logits, a2_logits, a3_logits = self.model.action_model([self.inputs, a1_vec, a2_vec])
        return (
            Categorical(a1_logits).logp(a1) + Categorical(a2_logits).logp(a2) + Categorical(a3_logits).logp(a2)
        )

    def sample_action_logp(self):
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())
        a3_dist = self._a3_distribution(a1_dist.sample(), a2_dist.sample())
        return a1_dist.entropy() + a2_dist.entropy() + a3_dist.entropy()

    def kl(self, other):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))
        a3_terms = self._a3.distribution(a1, a2).kl(other._a3_distribution(a1, a2))

        return a1_terms + a2_terms + a3_terms

    def _a1_distribution(self):
        BATCH = tf.shape(self.inputs)[0]
        a1_logits, _ = self.model.action_model([self.inputs, tf.zeros(BATCH, 1)])
        a1_dist = Categorical(a1_logits)
        return a1_dist

    def _a2_distribution(self, a1):
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        _, a2_logits = self.model.action_model([self.inputs, a1_vec])
        a2_dist = Categorical(a2_logits)
        return a2_dist

    def _a3_distribution(self, a1, a2):
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        a2_vec = tf.expand_dims(tf.cast(a2, tf.float32), 1)
        _, a3_logits = self.model.action_model([self.inputs, a1_vec, a2_vec])
        a3_dist = Categorical(a3_logits)
        return a3_dist


class AutoregressiveActionsModel(TFModelV2):
    """Implements the '.action_model' branch requried above"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(AutoregressiveActionsModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        if action_space != Tuple([Discrete(10), Discrete(30), Discrete(30)]):
            raise ValueError(
                "This model only supports tje [10, 30, 30] action space"
            )

        # Inputs
        obs_input = tf.keras.layers.Input(
            shape=obs_space.shape, name="obs_input")
        a1_input = tf.keras.layers.Input(shape=(1, ), name="a1_input")
        a2_input = tf.keras.layers.Input(shape=(1, ), name="a2_input")
        ctx_input = tf.keras.layers.Input( shape=(num_outputs, ), name="ctx_input")

        # Output of th model (normally 'logits' but for an autoregressive dist this is more like a context/feature
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
            kernel_intializer=normc_initializer(0.01))(context)

        # P(a1 | obs)
        a1_logits = tf.keras.layers.Dense(
            10,
            name="a1_logits",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(context)

        # P(a2 | a1)
        a2_context = a1_input
        a2_hidden = tf.keras.layers.Dense(
            16,
            name="a2_hidden",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(a2_context)
        a2_logits = tf.keras.layers.Dense(
            30,
            name="a2_logits",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(a2_hidden)

        # P(a3 | a1, a2)
        a3_context = tf.concat([a1_input, a2_input], 0)
        a3_hidden = tf.keras.layers.Dense(
            16,
            name="a3_hidden",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(a3_context)
        a3_logits = tf.keras.layers.Dense(
            2,
            name="a3_logits",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(a3_hidden)

        # Base layers
        self.base_model = tf.keras.Model(obs_input, [context, value_out])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

        # Autoregressive action sampler
        self.action_model = tf.keras.Model([ctx_input, a1_input, a2_input],
                                         [a1_logits, a2_logits, a3_logits])
        self.action_model.summary()
        self.register_variables(self.action_model.variables)

    def forward(self, input_dict, state, seq_lens):
        context, self._value_out = self.base_model(input_dict["obs"])
        return context, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cups or None)
    ModelCatalog.register_custom_model("autoregressive_model",
                                       AutoregressiveActionsModel)
    ModelCatalog.register_custom_model("reasoning_autoreg_output",
                                       ReasoningAutoregressiveOutput)

    tune.run(
        args.run,
        stop={"epsiode_reward_mean": args.stop},
        config={
            "env": ReasoningEnv,
            "gamma": 0.5,
            "num_gpus": 0,
            "model": {
                "custom_model": "autoregressive_model",
                "custom_action_dist": "reasoning_autoreg_output"
            }
        }
    )


