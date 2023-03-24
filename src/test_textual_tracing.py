from unittest import TestCase

from transformers import CLIPTokenizer

from src.textual_tracing import TracingTextEncoder


class TestTracingTextEncoder(TestCase):
    def test_forward(self):
        prompt = 'a woman in a red shirt and a men in a blue shirt'
        tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', cache_dir='~/cache',
                                                  subfolder="tokenizer")
        input_ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        encoder = TracingTextEncoder(None, tokenizer)
        input_ids = input_ids['input_ids']

        input_ids = input_ids.squeeze()
        ids_without_women = encoder.remove_str(input_ids, 'a woman in a red shirt')
        self.assertEqual(tokenizer.decode(ids_without_women, skip_special_tokens=True), 'and a men in a blue shirt')

        ids_without_men = encoder.remove_str(input_ids, 'and a men in a blue shirt')
        self.assertEqual(tokenizer.decode(ids_without_men, skip_special_tokens=True), 'a woman in a red shirt')
        # encoder(input_ids['input_ids'], None)
