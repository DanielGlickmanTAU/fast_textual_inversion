from unittest import TestCase

from src.data.concepts_datasets import get_datasetstate_with_k_random_indices_with_label, get_images_from_dataset_state


class Test(TestCase):
    def test_get_images_from_dataset_state(self):
        state = get_datasetstate_with_k_random_indices_with_label('food', label=1, k=5, split='test')
        images1 = get_images_from_dataset_state(state)
        images2 = get_images_from_dataset_state(state)
        self.assertEqual(images1, images2)
