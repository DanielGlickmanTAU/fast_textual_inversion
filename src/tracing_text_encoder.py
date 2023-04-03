import torch
import torch.nn


def find_str_indicies_in_input_ids_(tokenizer, input_ids, str):
    list_to_remove = tokenizer(str, add_special_tokens=False, return_tensors="pt")['input_ids']
    list_to_remove = list_to_remove.to(input_ids.device)
    # Find the starting index of the sublist to remove
    if list_to_remove.shape[-1] > 1:
        list_to_remove = list_to_remove.squeeze()
    input_ids = input_ids.squeeze()
    start_index = (input_ids == list_to_remove[0]).nonzero(as_tuple=False)

    # Check if the sublist matches the list_to_remove
    for start in start_index:
        end_index = start + len(list_to_remove)
        if (input_ids[start: end_index].tolist() == list_to_remove.tolist()) or len(list_to_remove) == 1:
            return start, end_index
    raise ValueError('str not found')


class TracingTextEncoder(torch.nn.Module):
    def __init__(self, text_encoder, tokenizer, mode='cross', left_side=None, right_side=None):
        super().__init__()
        self.right_side = right_side
        self.left_side = left_side
        self.mode = mode
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        try:
            # need this for stable diffusion flow
            self.config = text_encoder.config
            self.dtype = text_encoder.dtype
            self.device = text_encoder.device
        except:
            pass

    def find_str_indicies_in_input_ids(self, input_ids, str):
        tokenizer = self.tokenizer
        return find_str_indicies_in_input_ids_(tokenizer, input_ids, str)

    def remove_sublist(self, input_ids, start_index, end_index):
        input_ids = torch.cat([input_ids[:start_index], input_ids[end_index:]])
        return input_ids

    def remove_str(self, input_ids, str):
        start_index, end_index = self.find_str_indicies_in_input_ids(input_ids, str)
        return self.remove_sublist(input_ids, start_index, end_index)

    def get_ids_to_merge_back(self, left_ids, right_ids):
        return torch.cat((left_ids != self.tokenizer.eos_token_id, right_ids != self.tokenizer.bos_token_id))

    def get_non_eos_ids(self, left_ids, right_ids):
        right_real_tokes = torch.logical_and(self.tokenizer.eos_token_id != right_ids,
                                             self.tokenizer.bos_token_id != right_ids)
        return torch.cat((left_ids != self.tokenizer.eos_token_id, right_real_tokes))

    def forward(self, input_ids, attention_mask):
        if not self.mode or self.mode == 'None':
            return self.text_encoder(input_ids, attention_mask)
        # just case, all ids are eos
        if (input_ids == self.tokenizer.eos_token_id).float().mean() > 0.9:
            # return [self.text_encoder(input_ids[:, :1], attention_mask)[0].repeat((1, self.num_embeddings, 1))]
            return self.text_encoder(input_ids, attention_mask=attention_mask)

        if self.mode == 'no_eos':
            return self.text_encoder((input_ids[input_ids != self.tokenizer.eos_token_id]).unsqueeze(0),
                                     attention_mask=attention_mask)

        if self.mode == 'eos':
            return self.text_encoder((input_ids[input_ids == self.tokenizer.eos_token_id]).unsqueeze(0),
                                     attention_mask=attention_mask)

        if self.mode == 'cross':
            left_ids = self.remove_str(input_ids.squeeze(0), self.right_side)
            right_ids = self.remove_str(input_ids.squeeze(0), self.left_side)
            non_eos_ids = self.get_non_eos_ids(left_ids, right_ids)

            out_left = self.text_encoder(left_ids.unsqueeze(0), attention_mask=attention_mask, )[0]
            out_right = self.text_encoder(right_ids.unsqueeze(0), attention_mask=attention_mask, )[0]

            out = torch.cat((out_left.squeeze(), out_right.squeeze()))

            eos_left = out_left[:, (self.tokenizer.eos_token_id == left_ids)]
            eos_right = out_right[:, (self.tokenizer.eos_token_id == right_ids)]
            all_eos = torch.cat((eos_left, eos_right), dim=1).squeeze(0)
            assert all_eos.shape[0] > 1
            all_eos_shuffled = all_eos[torch.randperm(all_eos.shape[0])]
            #
            out_start = out[non_eos_ids]
            eos_padding = all_eos_shuffled[:77 - len(out_start)]
            out = torch.cat((out_start, eos_padding))

            # merge_back_mask = self.get_ids_to_merge_back(left_ids, right_ids)
            # out = out[merge_back_mask]
            self.num_embeddings = out.shape[0]

        if self.mode == 'causal':
            out = self.text_encoder(input_ids, attention_mask=attention_mask)[0]

            input_ids = input_ids.squeeze(0)
            start_remove1, end_remove1 = self.find_str_indicies_in_input_ids(input_ids, self.left_side)
            ids_tmp = self.remove_str(input_ids, self.left_side)
            # start_remove2, end_remove2 = self.find_str_indicies_in_input_ids(ids_tmp, 'blue')
            out = self.remove_sublist(out, start_remove1, end_remove1)
            # out = self.remove_sublist(out, start_remove2, end_remove2)
            return [out]

        return [out.unsqueeze(0)]
