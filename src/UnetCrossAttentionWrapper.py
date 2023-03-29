from diffusers import UNet2DConditionModel
from diffusers.models.cross_attention import CrossAttention


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
        return scores
