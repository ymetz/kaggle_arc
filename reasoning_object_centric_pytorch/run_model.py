import json
from absl import app
from absl import flags
import numpy as np
import torch as th

from monet_pytorch import model as modellib

BATCH_SIZE = 1
NUM_FRAMES = 25
NUM_OBJECTS = 8

_BASE_DIR = flags.DEFINE_string(
    "base_dir", "./clevrer_monet_latents",
    "Directory containing checkpoints and MONet latents.")
_SCENE_IDX = flags.DEFINE_integer(
    "scene_idx", 1000, "Scene index of CLEVRER video.")


def load_monet_latents(base_dir, scene_index):
    filename = f"{base_dir}/train/{scene_index}.npz"
    with open(filename, "rb") as f:
        return np.load(f)


def _split_string(s):
    """Splits string to words and standardize alphabet."""
    return s.lower().replace("?", "").split()


def _pad(array, length):
    """Pad an array to desired length."""
    return np.pad(array, [(0, length - array.shape[0])], mode="constant")


def encode_sentence(token_map, sentence, pad_length):
    """Encode CLEVRER question/choice sentences as sequence of token ids."""
    ret = np.array(
        [token_map["question_vocab"][w] for w in _split_string(sentence)],
        np.int32)
    return _pad(ret, pad_length)


def encode_choices(token_map, choices):
    """Encode CLEVRER choices."""
    arrays = [encode_sentence(token_map, choice["choice"],
                              modellib.MAX_CHOICE_LENGTH)
              for choice in choices]
    return _pad(np.stack(arrays, axis=0), modellib.NUM_CHOICES)


def main(unused_argv):
    base_dir = _BASE_DIR.value
    with open(f"{base_dir}/vocab.json", "rb") as f:
        token_map = json.load(f)

    reverse_answer_lookup = {v: k for k, v in token_map["answer_vocab"].items()}

    with open(f"{base_dir}/train.json", "rb") as f:
        questions_data = json.load(f)

    inputs_descriptive = {
        "monet_latents": th.FloatTensor(BATCH_SIZE, NUM_FRAMES, NUM_OBJECTS, modellib.EMBED_DIM),
        "question": th.IntTensor(BATCH_SIZE, modellib.MAX_QUESTION_LENGTH),
    }

    inputs_mc = {
        "monet_latents": th.FloatTensor(BATCH_SIZE, NUM_FRAMES, NUM_OBJECTS, modellib.EMBED_DIM),
        "question": th.IntTensor(BATCH_SIZE, modellib.MAX_QUESTION_LENGTH),
        "choices": th.IntTensor(
            BATCH_SIZE, modellib.NUM_CHOICES,
                       modellib.MAX_CHOICE_LENGTH),
    }


    # Get initial shapes
    shape = modellib.ClevrerTransformerModel.get_input_shape(inputs_descriptive,
                                                             embed_dim=modellib.PRETRAINED_MODEL_CONFIG["embed_dim"])

    model = modellib.ClevrerTransformerModel(**modellib.PRETRAINED_MODEL_CONFIG, input_shape=shape)

    def eval_descriptive(monet_latents, question_json):
        # CLEVRER provides videos with 128 frames. In our model, we subsample 25
        # frames (as was done in Yi et al (2020)).
        # For training, we randomize the choice of 25 frames, and for evaluation, we
        # sample the 25 frames as evenly as possible.
        # We do that by doing strided sampling of the frames.
        stride, rem = divmod(monet_latents.shape[0], NUM_FRAMES)
        monet_latents = monet_latents[None, :-rem:stride]
        assert monet_latents.shape[1] == NUM_FRAMES
        question = encode_sentence(token_map, question_json["question"],
                                   modellib.MAX_QUESTION_LENGTH)
        batched_question = np.expand_dims(question, axis=0)
        logits = model.apply_model_descriptive({"monet_latents":monet_latents, "question": batched_question })
        descriptive_answer = np.argmax(logits)
        return reverse_answer_lookup[descriptive_answer]

    def eval_mc(monet_latents, question_json):
        stride, rem = divmod(monet_latents.shape[0], NUM_FRAMES)
        monet_latents = monet_latents[None, :-rem:stride]
        assert monet_latents.shape[1] == NUM_FRAMES
        question = encode_sentence(
            token_map, question_json["question"], modellib.MAX_QUESTION_LENGTH)
        choices = encode_choices(
            token_map, question_json["choices"])

        mc_answer = model.apply_model_mc({"monet_latents":monet_latents, "question": np.expand_dims(question, axis=0), "choices": np.expand_dims(choices, axis=0) })
        return mc_answer >= 0

    sample_scene_idx = _SCENE_IDX.value
    question_json = questions_data[sample_scene_idx]["questions"][0]
    print("Descriptive Question: ", question_json["question"])
    print("Model Answer: ",
          eval_descriptive(load_monet_latents(base_dir, sample_scene_idx),
                           question_json))
    print("True Answer: ", question_json["answer"])

    question_json = questions_data[sample_scene_idx]["questions"][-1]
    print("Multiple-Choice Question: ", question_json["question"])
    for i, choice_json in enumerate(question_json["choices"]):
        print(f"{i + 1}) {choice_json['choice']}")
    print("Model Answer: ",
          eval_mc(load_monet_latents(base_dir, sample_scene_idx), question_json))
    print("True Answer: ",
          [choice_json["answer"] for choice_json in question_json["choices"]])


if __name__ == "__main__":
    app.run(main)
