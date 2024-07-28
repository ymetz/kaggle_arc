import collections
from typing import Sequence, Optional, Union, Dict, Tuple, List

import torch as th
import torch.nn as nn
import torch.nn.init
from torch.nn import functional as F
import numpy as np

AttentionState = collections.namedtuple('AttentionState',
                                        ('queries', 'keys', 'values', 'logits',
                                         'weights', 'embeddings', 'read_words'))

CompressedMemoryState = collections.namedtuple(
    'CompressedMemoryState', ('episodic_memory', 'compressed_memory', 'index'))

def rel_shift(position_logits: th.Tensor):
    """
    Shifting of logits for relative attention
    """

    if len(position_logits.shape) != 4:
        raise ValueError("Expected 4D position logits")

    input_shape = position_logits.shape
    batch_size = position_logits.shape[0]
    num_heads = input_shape[1]
    t1 = input_shape[2]
    t2 = input_shape[3]

    to_pad = th.zeros((batch_size, num_heads, t1, 1))
    position_logits = th.cat((to_pad, position_logits), dim=-1)
    # Reshape trick to shift input
    position_logits = th.reshape(position_logits,
                                 (batch_size, num_heads, t2 + 1, t1))
    # Remove extra time dimension and reshape
    position_logits = position_logits[:, :, 1:]
    position_logits = th.reshape(position_logits, input_shape)
    return position_logits

def _concat_and_slice(prev_memory: th.Tensor, new_memory: th.Tensor):
    original_memory_size = prev_memory.shape[1]
    concat_memory = th.cat([prev_memory, new_memory], dim=1)
    memory = concat_memory[:, -original_memory_size:]
    return memory, concat_memory

def simple_attention(queries: th.Tensor, keys: th.Tensor, values: th.Tensor):
    logits = th.matmul(queries, keys)
    weights = th.softmax(logits)
    return th.matmul(weights, values)

class ResidualDropout(nn.Module):
    """
    Wrapper class that applies residual connections, dropout and layer norm.
    By default applies a relu to the module output before the other operations.
    """

    def __init__(self,
                 layer: nn.Module,
                 dropout_rate: float,
                 layer_norm: str = "input",
                 embedding_dim: int = 128,
                 is_training = True
                 ):
        self._module = layer
        self._dropout_rate = dropout_rate
        self._layer_norm = layer_norm
        super(ResidualDropout, self).__init__()

        self.layer_norm_in = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(p=self._dropout_rate)

        self.layer_norm_out = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Apply Lin. Norm + dropout
        if self._layer_norm in ("both", "input"):
            norm_input = self.layer_norm_in(x)
        else:
            norm_input = x

        module_output = self._module(norm_input)
        module_state = None

        # If module outputs multiple items, assumes (output, state) tuple.
        if isinstance(module_output, tuple):
            module_output, module_state = module_output

        model_out = self.dropout(module_output)

        # Res. Connection
        out = model_out + x

        if self._layer_norm in ("both", "output"):
            output = self.layer_norm_out(out)
        else:
            output = out

        if module_state is None:
            return output
        else:
            return output, module_state


def future_mask(chunk_size: int, dtype):
    """
    Creates attention mask to ensure an element i cannot attend to j > i
    :param chunk_size:
    :param dtype:
    :return:
    """
    square = th.ones((chunk_size, chunk_size), dtype=dtype)
    # Create upper diagonal matrix and remove diagonal enties (allow self-attn)
    mask = th.triu(square, diagonal=False)
    mask = -1e6 * th.reshape(mask, (1, 1, chunk_size, chunk_size))
    return mask

def _memory_size(state: Union[CompressedMemoryState, th.Tensor]):
    if isinstance(state, CompressedMemoryState):
        return (state.episodic_memory.shape[1] + state.compressed_memory.shape[1])
    else:
        return state.shape[1]

def create_mask(input_shape, input_dtype, state, equal_window):
    """
    Creates mask for future sequence positions.
    :param inputs: inputs tensor of shape [B, N, D]
    :param state: optional tensor of shape [B, M, D], CompressedMemoryState or
        a list where the entry corresponds to the ith layer's state
    :param equal_window: if True, then each activation has an equally-sized attention
    window of length 'M'. This only makes sense if fa state is given
    :return: Float tensor of shape [1, 1, N, N+M], to be summed with logits
    """
    chunk_size = input_shape
    dtype = input_dtype
    mask = future_mask(chunk_size, dtype)
    if state is not None:
        if isinstance(state, (tuple, list)):
            largest_memory_layer = np.argmax([_memory_size(s) for s in state])
            state = state[largest_memory_layer]
        mem_size = _memory_size(state)
        mask = th.cat(
            [th.zeros((1, 1, chunk_size, mem_size), dtype=dtype), mask],
            dim=3
        )

    if equal_window:
        attn_mask = th.ones([chunk_size, chunk_size], dtype=dtype)
        mask_dia = th.diag(attn_mask)
        mask_l = th.tril(attn_mask)
        start_mask = th.reshape(mask_l - mask_dia, [1,1,chunk_size,chunk_size]) * 1e6
        mask = th.cat(
           [ mask[:,:,:,:chunk_size] + start_mask, mask[:,:,:,chunk_size:]]
        )
    return mask

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

def weight_init(module, init_std):
    for block in module:
        try:
            for m in module[block]:
                normal_init(m, 0.0, init_std)
        except:
            normal_init(block)

def default_mlp(input_size: int, hidden_sizes: List[int], activate_final: bool = False, init_std: float = 2., **kwargs):
    """
    Standard batch-applied MLP for transformer modules
    :param hidden_size:
    :param activate_final:
    :param init_std:
    :param kwargs:
    :return:
    """
    mlp = nn.Sequential()
    mlp.add_module("ff_block", nn.Linear(input_size, hidden_sizes[0]))
    mlp.add_module("dropout", nn.Dropout(p=0.5))
    mlp.add_module("activation_fn", nn.ReLU())
    for i, hidden in enumerate(hidden_sizes[1:]):
        mlp.add_module("ff_block", nn.Linear(input_size, hidden))
        mlp.add_module("dropout", nn.Dropout(p=0.5))
        if i == len(hidden_sizes) - 2:
            # last layer
            if activate_final:
                mlp.add_module("activation_fn", nn.ReLU())
        else:
            mlp.add_module("activation_fn", nn.ReLU())

    weight_init(weight_init, init_std)

    return mlp

def get_position_encodings(
        sequence_length: int,
        hidden_size: int,
        clamp_value: float,
        max_timescale: float=10000.,
        min_timescale: float =2.0
):
    """Creates sinusoidal encodings of shape [1, N + M, D]"""
    # Note: when not using relative position encodings, min_timescale must be 2.0
    # and hidden_size must be an even number. Otherwise ,the dimension do not match.
    pos_seq = th.range(sequence_length -1, -1, -1.0)
    if clamp_value > 0:
        pos_seq = th.minimum(pos_seq, clamp_value)
    freqs = th.range(0, hidden_size, min_timescale)
    inv_freq = 1 / (max_timescale**(freqs / hidden_size))
    sinusoid_inp = th.einsum("i,j->ij", pos_seq, inv_freq)
    pos_emb = th.cat((th.sin(sinusoid_inp), th.cos(sinusoid_inp)), dim=-1)
    pos_emb = th.unsqueeze(pos_emb, 0)

    output_dim = pos_emb.shape[-1]
    if output_dim != hidden_size:
        raise ValueError(
            "position embedding dimension ({}) does not match that of the input ({})."
            .format(output_dim, hidden_size)
        )
    return pos_emb

class MultiheadAttention(nn.Module):

    def __init__(self, value_size: int, key_size: int, num_heads: int, mask: th.Tensor = None,
                 scaling: bool=True, positional_encodings: th.Tensor = None, use_relative_encodings: bool=False,
                 dropout_p: float = 0.5, init_std: float = 2.0):
        """
        Create a Multihead Attention Block
        :param value_size:
        :param key_size:
        :param num_heads:
        :param mask:
        :param scaling:
        :param positional_encodings:
        :param use_relative_encodings:
        :param init_std:
        """
        super(MultiheadAttention, self).__init__()
        self._value_size = value_size
        self._key_size = key_size
        self._sizes = {
            "value": self._value_size,
            "key": self._key_size,
            "query": self._key_size,
            "relative_keys": self._key_size,
            "relative_keys_0": self._key_size
        }
        self._num_heads = num_heads
        self._mask = mask
        self._scaling = scaling
        self._positional_encodings = positional_encodings
        self._use_relative_positions = use_relative_encodings
        self._dropout_p = dropout_p
        self._init_std = init_std

        embedding_size = self._value_size * self._num_heads

        class MultiHeadLinear(nn.Module):

            def __init__(self, num_heads, init_std):
                super(MultiHeadLinear, self).__init__()

                self.w = None
                self.num_heads = num_heads
                self._init_std = init_std

            def forward(self, inputs, name):
                input_size = inputs.shape[-1]

                if self.w is None:
                    self.w = th.empty(input_size, self.num_heads, input_size)
                    torch.nn.init.normal_(self.w, std=self._init_std)
                out = th.einsum("bij,jhk->khik", inputs, self.w)

                return out


        self.q_linear = MultiHeadLinear(self._num_heads, self._init_std)
        self.k_linear = MultiHeadLinear(self._num_heads, self._init_std)
        self.v_linear = MultiHeadLinear(self._num_heads, self._init_std)

        self.r_w_bias = th.empty(1, self._num_heads, 1, self._key_size)

        if self._use_relative_positions:
            self.relative_key_linears = [
                MultiHeadLinear(self._num_heads, self._init_std) for _ in range(len(self._positional_encodings))
            ]
            self.r_r_biases = [
                th.empty(1, self._num_heads, 1, self._key_size) for _ in range(len(self._positional_encodings))
            ]

        self.weight_softmax = th.nn.Softmax()
        self.weight_dropout = th.nn.Dropout()

        self.multihead_attention = th.nn.MultiheadAttention(
            embedding_size,
            num_heads=self._num_heads,
            dropout_p=self._dropout_p
        )

        self.merge_linear = th.nn.Linear(self._key_size, self._key_size)

    def forward(self, inputs, query_inputs=None, state=None, key_value_inputs=None):

        if key_value_inputs is not None and state is not None:
            raise ValueError('Only one of the key_value_input and state is needed.')
        embedding_size = self._value_size * self._num_heads

        q_inputs = inputs if query_inputs is None else query_inputs
        q_size = q_inputs.shape[1]

        if key_value_inputs is not None:
            k_inputs = key_value_inputs
            v_inputs = k_inputs
        elif state is None:
            if isinstance(state, CompressedMemoryState):
                state_memory_list = [state.compressed_memory, state.episodic_memory]
            else:
                state_memory_list = [state]

            k_inputs = th.cat(state_memory_list + [inputs], dim=1)
            v_inputs = k_inputs
        else:
            k_inputs = inputs
            v_inputs = inputs

        batch_size = inputs.shape[0]
        chunk_size = inputs.shape[1]
        att_size = k_inputs.shape[1]

        if self._positional_encodings and not self._use_relative_positions:
            if len(self._positional_encodings) != 1:
                raise ValueError(
                    'Absolute positional encodings only supported for 1 memory. '
                    'Found %i.' % len(self._positional_encodings))
            key_positions, query_positions = self._positional_encodings[0]
            k_inputs += key_positions
            q_inputs += query_positions

        q = self.q_linear(q_inputs)
        k = self.k_linear(k_inputs)
        v = self.v_linear(v_inputs)

        if self._scaling:
            q *= self._key_size**-0.5

        if self._use_relative_positions:
            content_logits = th.matmul(q+self.r_w_bias)
            all_relative_logits = []

            # Loop over multiple positional encodings
            for i, positional_encoding in enumerate(self._positional_encodings):
                key_positions, query_positions = positional_encoding
                if key_positions.shape[-1] != att_size:
                    key_positions = key_positions[:, -att_size:] # Cropy to layer mem size
                is_final = i == len(self._positional_encodings) - 1

                relative_keys = self.relative_key_linears[i](key_positions)
                r_r_bias = self.r_r_biases[i]

                relative_keys = th.tile(relative_keys, (batch_size, 1, 1, 1))
                relative_logits = th.matmul(q + r_r_bias)
                relative_logits = rel_shift(relative_logits)

                if not is_final:
                    relative_logits = relative_logits[:,:,:,:,:-chunk_size]
                all_relative_logits.append(relative_logits)
            all_relative_logits = th.cat(all_relative_logits, 3)
            logits = content_logits + all_relative_logits
        else:
            logits = th.matmul(q, k)
            content_logits = logits

        if self._mask is not None:
            if self._mask.shape[-1] != att_size:
                mask = self._mask[:, :, :, -att_size:]
            else:
                mask = self._mask
            logits += mask

        weights = self.weight_softmax(logits)
        weights = self.weight_dropout(weights)

        output_transpose = th.einsum("bhij,bhjk->bihk", weights, v)

        # [B, L, H, V] -> [B, L, HV]
        attended_inputs = th.reshape(output_transpose, (batch_size, q_size, embedding_size))

        # Apply final mlp to mix information between heads
        output = self.merge_linear(
            attended_inputs.view(attended_inputs.shape[0]*attended_inputs[1], *attended_inputs.shape[2:])
        ).view(*attended_inputs.shape)

        attention_state = AttentionState(
            queries=q,
            keys=k,
            values=v,
            weights=weights,
            logits=logits,
            embeddings=inputs,
            read_words=output
        )

        return output, attention_state


class TransformerTower(nn.Module):

    def __init__(self,
                 value_size: int,
                 num_heads: int,
                 num_layers: int,
                 causal: bool =True,
                 key_size=None,
                 shared_attention=False,
                 output_size=None,
                 mlp_hidden_sizes=tuple([1024]),
                 dropout_rate=0.1,
                 use_relative_positions=True,
                 clamp_time_range=0,
                 same_attention_length=False,
                 layer_norm="input",
                 state_prototype = None,
                 input_shape: tuple() = (1,),
                 input_dtype: type = th.float):

        """
        Initializes TransformerTower.
        Args:
          value_size: dimensionality of values per-head.
          num_heads: number of attention heads.
          num_layers: number of transformer blocks, where each block contains a
            multi-head attention layer and an MLP.
          causal: if True, applies a causal mask.
          key_size: optional dimensionality of key size. If unspecified then it is
            set to `value_size`.
          shared_attention: if True, attention params are shared across all layers.
          output_size: if set, the desired output dimensionality. By default the
            output size is `value_size` x `num_heads`.
          mlp_hidden_sizes: tuple containing dimensionality of mlp layer(s). If
            multiple values are specified, the mlp contains multiple layers for each
            transformer block.
          dropout_rate: dropout rate applied to hidden activations, attention, and
            positional encodings.
          use_relative_positions: if False, applies absolute positional encodings.
            If true, uses relative positional encodings from Dai et al. 2019.
          clamp_time_range: clamps max temporal positional encoding if specified.
          same_attention_length: if True, attention is masked to ensure each
            position in the sequence contains the same length of attention.
          layer_norm: Where to apply layer-norm in Transformer block. Can be one of
            'input' (Vaswani et al. 2017), 'output', or 'both'.
          name: name of variable scope.
        """
        super(TransformerTower, self).__init__()

        self._causal = causal
        self._mask = None

        if key_size is None:
            key_size = value_size
        self._key_size = key_size
        self._value_size = value_size
        self._shared_attention = shared_attention
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._output_size = output_size
        self._embedding_size = self._value_size * self._num_heads
        self._mlp_hidden_sizes = list(mlp_hidden_sizes) + [self._embedding_size]
        self._multihead_attention = None
        self._object_embeddings = None
        self._dropout_rate = dropout_rate
        self._positional_encodings = None
        self._use_relative_positions = use_relative_positions
        self._clamp_time_range = use_relative_positions
        self._same_attention_length  = same_attention_length
        self._layer_norm = layer_norm
        self._attention_modules = []
        self._object_mlps = []

        self.input_embedding_mlp = None

        self.key_pos_dropout = th.nn.Dropout(p=self._dropout_rate)

        if state_prototype is None:
            memory_sizes = [0]
        elif isinstance(state_prototype[0], CompressedMemoryState):
            cm_mem_size = max(_memory_size(s.compressed_memory) for s in state_prototype)
            em_mem_size = max(_memory_size(s.episodic_memory) for s in state_prototype)
            memory_sizes = [cm_mem_size, em_mem_size]
        else:
            memory_sizes = [max([_memory_size(s) for s in state_prototype])]
        chunk_size = input_shape[1]
        self._positional_encodings = []
        # Creates positional encodings for different memory types
        for i, memory_size in enumerate(memory_sizes):
            seq_len = chunk_size + memory_size
            key_positions = get_position_encodings(
                sequence_length=seq_len,
                hidden_size=input_shape[2],
                clamp_value=self._clamp_time_range
            )
            key_positions = self.key_pos_dropout(key_positions)
            key_positions = key_positions.to(dtype=input_dtype)
            query_positions = key_positions[:, :-chunk_size, :]
            self._positional_encodings.append((key_positions, query_positions))

        if self._causal:
            self._mask = create_mask(input_shape, input_dtype, state_prototype, self._same_attention_length)

        for i in range(self._num_layers):
            attention_module = MultiheadAttention(
                value_size=value_size,
                key_size=key_size,
                num_heads=num_heads,
                mask=self._mask,
                positional_encodings=self._positional_encodings,
                use_relative_encodings=self._use_relative_positions,
                init_std=2. / np.sqrt(self._num_layers)
            )
            _multihead_attention_module = ResidualDropout(
                attention_module, self._dropout_rate, layer_norm=self._layer_norm
            )
            mlp = default_mlp(
                self._mlp_hidden_sizes, init_std=2. / np.sqrt(self._num_layers)
            )
            object_mlp = ResidualDropout(
                mlp, self._dropout_rate, layer_norm=self._num_layers
            )

            self._attention_modules.append(attention_module)
            self._object_mlps.append(object_mlp)

        self.merge_linear = th.nn.Linear(self._output_size, self._output_size)


    def forward(self, inputs, state=None, condition=None, final_layer_key_value_inputs=None):
        """Calculates multi-layer self attention and mlp transformation.
        Args:
          inputs: Tensor of shape [batch_size, num_steps, dim_size].
          state: optional list of length num_layers of tensors of shape
            [batch_size, memory_size, dim_size].
          condition: optional tensor to condition on. The shape is shape
            [batch_size, dim_size].
          is_training: If true, dropout is applied.
          final_layer_key_value_inputs: optional Tensor to be used as the key and
            value for the final multi-head attention layer of shape
            [batch_size, num_steps, dim_size]. Useful when the tower is a Seq2Seq
            decoder and it can attend to encoder outputs.
        Returns:
          output: tensor of shape [batch_size, num_steps, output_dim_size].
          state: list of length `num_layers` containing AttentionState tuples.
        """
        if final_layer_key_value_inputs is not None and state is not None and len(state) == (self._num_layers - 1):
            raise ValueError("When the final_layer_key_value_input is set, exclude the state of the last layer")

        if condition is not None:
            condition_tile = th.tile(th.unsqueeze(condition, 1), (1, inputs.shape[1], 1))
            inputs = th.cat((inputs, condition_tile), -1)

        # Map inputs to be of "embeddings_size" dimensions
        if inputs.shape[-1] != self._embedding_size:
            if self.input_embedding_mlp is None:
                self.input_embedding_mlp = default_mlp((self._embedding_size), activate_final=True)\

            inputs = self.input_embedding_mlp(inputs)

        layer_i_inputs= inputs
        attention_states = []
        key_value_inputs = None

        for i in range(self._num_layers):

            state_i = None if state is None else state[i]
            if i == (self._num_layers - 1) and final_layer_key_value_inputs is not None:
                # When the final_layer_key_value_inputs is set, the final layerr of attention will use it as a the key &
                # value, thus no need for state
                key_value_inputs = final_layer_key_value_inputs
                state_i = None

            attention_outputs, attention_state = self._attention_modules[i](
                layer_i_inputs,
                state=state_i,
                key_value_inputs=key_value_inputs
            )
            attention_state.append(attention_state)
            # Feed forward with residuals
            output = self._object_mlps[i](
                attention_outputs,
            )
            layer_i_inputs = output

        if self._output_size is not None:
            output = self.merge_linear(
                output.view(output.shape[0] * output[1], *output.shape[2:])
            ).view(*output.shape)

        return output, attention_states

    def attention_module(self, i):
        """Returns the i-th layer attention module"""
        return self._attention_modules[i]



