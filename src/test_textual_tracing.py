from unittest import TestCase

from transformers import CLIPTokenizer, CLIPTextModel

from src.tracing_text_encoder import TracingTextEncoder


class TestTracingTextEncoder(TestCase):
    def test_cross(self):
        prompt = 'a woman in a red shirt and a man in a blue shirt'
        tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', cache_dir='~/cache',
                                                  subfolder="tokenizer")
        input_ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # clip = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-v1-5', cache_dir='~/cache',
        #                                      subfolder="text_encoder", revision=None)
        def dummy_encoder(inputs, attention_mask):
            return inputs

        encoder = TracingTextEncoder(dummy_encoder, tokenizer)
        input_ids = input_ids['input_ids']

        input_ids = input_ids.squeeze()
        ids_without_women = encoder.remove_str(input_ids, 'a woman in a red shirt')
        self.assertEqual(tokenizer.decode(ids_without_women, skip_special_tokens=True), 'and a man in a blue shirt')

        ids_without_men = encoder.remove_str(input_ids, 'and a man in a blue shirt')
        self.assertEqual(tokenizer.decode(ids_without_men, skip_special_tokens=True), 'a woman in a red shirt')

        processed_text = encoder(input_ids.squeeze(), None)
        processed_text = processed_text[0].squeeze()
        self.assertEqual(tokenizer.decode(processed_text, skip_special_tokens=True), prompt)

    def test_causal(self):
        prompt = 'a woman in a red shirt and a man in a blue shirt'
        tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', cache_dir='~/cache',
                                                  subfolder="tokenizer")
        input_ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        def dummy_encoder(inputs, attention_mask):
            return inputs.unsqueeze(0)

        encoder = TracingTextEncoder(dummy_encoder, tokenizer, mode='causal')
        input_ids = input_ids['input_ids']

        input_ids = input_ids.squeeze()

        processed_text = encoder(input_ids.squeeze(), None)
        processed_text = processed_text[0].squeeze()
        self.assertEqual(tokenizer.decode(processed_text, skip_special_tokens=True), 'a man in a shirt')
