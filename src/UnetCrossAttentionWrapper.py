import dataclasses

from diffusers import UNet2DConditionModel
from diffusers.models.cross_attention import CrossAttention
import torch


class UnetCrossAttentionWrapper(UNet2DConditionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        attentions = [(name, x) for (name, x) in self.named_modules() if type(x) == CrossAttention]
        cross_attention = [(name, x) for (name, x) in attentions if x.to_v.in_features != x.to_v.out_features]
        self.cross_attentions = cross_attention
        for name, x in cross_attention:
            original_scores_func = x.get_attention_scores
            x.get_attention_scores = lambda query, key, attention_mask: self.wrap_attention_scores(original_scores_func,
                                                                                                   query, key,
                                                                                                   attention_mask)

    def wrap_attention_scores(self, original_scores_func, query, key, attention_mask):
        scores = original_scores_func(query, key, attention_mask)
        n_patches = scores.shape[1]

        if self.start_index == '$':
            scores[:, :, 1:] = 0.
            scores[:, :, 0] = 1.
            return scores / scores.sum(-1).unsqueeze(-1)

        # first hald does not look at chunk
        scores[:, :n_patches // 2, self.start_index:self.end_index] = 0

        # second hald looks only at chunks
        scores[:, n_patches // 2:, :self.start_index] = 0
        scores[:, n_patches // 2:, self.end_index:] = 0

        # renormalize attentions
        scores = scores / scores.sum(-1).unsqueeze(-1)

        return scores


def set_attn_processors(unet, start_index, end_index):
    attn_procs = {}
    cross_att_count = 0

    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = CustomCrossAttnProcessor(name, start_index, end_index)

    unet.set_attn_processor(attn_procs)


# @dataclasses.dataclass
class CustomCrossAttnProcessor:
    def __init__(self, name, start_index, end_index):
        self.name = name
        self.start_index = start_index
        self.end_index = end_index

    # def __init__(self, attnstore, place_in_unet):
    #     super().__init__()
    #     self.attnstore = attnstore
    #     self.place_in_unet = place_in_unet

    def alter_attention(self, attention_probs):
        n_patches = attention_probs.shape[1]

        if self.start_index == '$':
            attention_probs[:, :, :] = 0.
            # attention_probs[:, :, 0] = 1.
            return attention_probs
            # return attention_probs / attention_probs.sum(-1).unsqueeze(-1)

        # first hald does not look at chunk
        attention_probs[:, :n_patches // 2, self.start_index:self.end_index] = 0.

        # second hald looks only at chunks
        attention_probs[:, n_patches // 2:, :self.start_index] = 0.
        attention_probs[:, n_patches // 2:, self.end_index:] = 0.

        # renormalize attentions
        return attention_probs / attention_probs.sum(-1).unsqueeze(-1)

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if is_cross:
            attention_probs = self.alter_attention(attention_probs)
        # self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
