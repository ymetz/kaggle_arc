import torch as th

from monet_pytorch import transformer

QUESTION_VOCAB_SIZE = 82
ANSWER_VOCAB_SIZE = 22

MAX_QUESTION_LENGTH = 20
MAX_CHOICE_LENGTH = 12

NUM_CHOICES = 4
EMBED_DIM = 16

PRETRAINED_MODEL_CONFIG = dict(
    use_relative_positions=True,
    shuffle_objects=True,
    transformer_layers=28,
    head_size=128,
    num_heads=10,
    embed_dim=EMBED_DIM
)

def append_ids(tensor, id_vector, dim):
    id_vector = th.FloatTensor(id_vector)
    for a in range(len(tensor.shape)):
        if a != dim:
            id_vector = th.unsqueeze(id_vector, dim=a)
    tiling_vector = [s if i != dim else 1 for i, s in enumerate(tensor.shape)]
    id_tensor = th.tile(id_vector, tiling_vector)
    print(tensor.shape, id_tensor.shape)
    return th.cat((tensor, id_tensor), dim=dim)

class ClevrerTransformerModel(object):
    """Model from Ding et al. 2020 (https://arxiv.org/abs/2012.08508)."""

    def __init__(self, use_relative_positions, shuffle_objects, transformer_layers, num_heads, head_size, embed_dim, input_shape):
        self._embed_dim = embed_dim
        self._embed = th.nn.Embedding(QUESTION_VOCAB_SIZE, embed_dim - 2)
        self._shuffle_objects = shuffle_objects
        self._memory_transformer = transformer.TransformerTower(
            value_size=embed_dim+2,
            num_heads=num_heads,
            num_layers=transformer_layers,
            use_relative_positions=use_relative_positions,
            causal=False,
            input_shape=input_shape,
            input_dtype=th.float32,
            state_prototype=transformer.AttentionState()
        )

        self._final_layer_mc= th.nn.Sequential(
            th.nn.Linear(embed_dim, head_size),
            th.nn.ReLU,
            th.nn.Linear(head_size, 1)
        )
        self._final_layer_descriptive = th.nn.Sequential(
            th.nn.Linear(embed_dim, head_size),
            th.nn.ReLU,
            th.nn.Linear(head_size, ANSWER_VOCAB_SIZE)
        )

        self._dummy = th.zeros(embed_dim+2, dtype=th.float32)
        self._infill_linear = th.nn.Linear(embed_dim+2, embed_dim+2)
        self._mask_embedding = th.zeros(embed_dim+2, dtype=th.float32)

    def apply_transformers(self, lang_embedding, vision_embedding):
        """Applies transformer to language and vision input"""

        def _unroll(tensor):
            """Unroll the time dimension into the object dimensions"""
            return th.reshape(tensor, (tensor.shape[0], -1, tensor.shape[3]))

        words = append_ids(lang_embedding, (1, 0), dim=2)
        dummy_word = th.tile(self._dummy[None, None, :], (words.shape[0], 1, 1))
        vision_embedding = append_ids(vision_embedding, (0, 1), dim=3)
        vision_over_time = _unroll(vision_embedding)
        transformer_input = th.cat((dummy_word, words, vision_over_time), dim=1)

        output, _ = self._memory_transformer(transformer_input)

        return output[:, 0, :]

    def apply_model_descriptive(self, inputs):
        """Applies model to CEVRER descriptive questions

            Args:
                inputs: dict of form: {
                "question": int32 tensor of shape (batch, MAX_QUESTION_LENGTH)
                "monet_lantents": float32 tensor of shape (batch, frames, 8, 16)
            }
            Returns:
                Tensor of shape (batch, ANSWER_VOCAB_SIZE) representing logits for
                each possible answer word
        """
        question = inputs["question"]

        question_embedding = self._embed(question)
        question_embedding = append_ids(question_embedding, (0, 1), 2)
        choices_embedding = self._embed(
            th.zeros((question.shape[0], MAX_CHOICE_LENGTH), th.int64)
        )
        choices_embedding = append_ids(choices_embedding, (0, 1), 2)
        lang_embedding = th.cat((question_embedding, choices_embedding), dim=1)

        vision_embedding = inputs["vision_latents"]

        if self._shuffle_objects:
            vision_embedding = th.permute(vision_embedding, (2, 1, 0, 3))
            rand_indx = th.randperm(vision_embedding.shape[0])
            vision_embedding = vision_embedding[rand_indx]
            vision_embedding = th.permute(vision_embedding, (2, 1, 0, 3))

        output = self.apply_transformers(lang_embedding, vision_embedding)
        output = self._final_layer_descriptive(output)

        return output

    def apply_model_mc(self, inputs):
        """Applies model to CLEVRER multiple-choice questions.
        Args:
          inputs: dict of form: {
            "question": tf.int32 tensor of shape [batch, MAX_QUESTION_LENGTH],
            "choices": tf.int32 tensor of shape [batch, 4, MAX_CHOICE_LENGTH],
            "monet_latents": tf.float32 tensor of shape [batch, frames, 8, 16],
          }
        Returns:
          Tensor of shape [batch, 4], representing logits for each choice
        """
        question = inputs["question"]
        choices = inputs["choices"]

        question_embedding = self._embed(question)
        question_embedding = append_ids(question_embedding, (1, 0), 2)

        choices_embedding = self._embed(choices)
        choices_embedding = append_ids(choices_embedding, (0, 1), 3)

        lang_embedding = th.cat(th.tile(question_embedding[:, None], (1, choices_embedding.shape[1], 1, 1)), choices_embedding, dim=2)

        vision_embedding = inputs["vision_latents"]

        if self._shuffle_objects:
            vision_embedding = th.permute(vision_embedding, (2, 1, 0, 3))
            rand_indx = th.randperm(vision_embedding.shape[0])
            vision_embedding = vision_embedding[rand_indx]
            vision_embedding = th.permute(vision_embedding, (2, 1, 0, 3))

        output_per_choice = []
        for c in range(NUM_CHOICES):
            output = self.apply_transformers(lang_embedding[:, c, :, :], vision_embedding)
            output_per_choice.append(output)

        output = th.stack(output_per_choice, dim=1)
        output = th.squeeze(self._final_layer_mc(output), axis=2)

        return output

    @staticmethod
    def get_input_shape(inputs, embed_dim):
        question = inputs["question"]

        question_embedding = th.zeros(1, embed_dim - 2, dtype=th.float32)
        question_embedding = append_ids(question_embedding, (0, 1), 2)
        choices_embedding = th.zeros(1, embed_dim - 2, dtype=th.float32)
        choices_embedding = append_ids(choices_embedding, (0, 1), 2)
        lang_embedding = th.cat((question_embedding, choices_embedding), dim=1)
        vision_embedding = inputs["vision_latents"]

        def _unroll(tensor):
            """Unroll the time dimension into the object dimensions"""
            return th.reshape(tensor, (tensor.shape[0], -1, tensor.shape[3]))

        words = append_ids(lang_embedding, (1, 0), dim=2)
        dummy_word = th.tile(th.zeros(embed_dim+2, dtype=th.float32)[None, None, :], (words.shape[0], 1, 1))
        vision_embedding = append_ids(vision_embedding, (0, 1), dim=3)
        vision_over_time = _unroll(vision_embedding)
        transformer_input = th.cat((dummy_word, words, vision_over_time), dim=1)

        return transformer_input.shape





