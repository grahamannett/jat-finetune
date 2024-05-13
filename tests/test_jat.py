import unittest
import torch
import transformers
import datasets
from jat_finetune.jat.jat_dataset_utils import JAT_DATASET_CONFIG_NAMES


class TestJATModel(unittest.TestCase):
    def setUp(self):
        self.model_name = "jat-project/jat"
        self.dataset_path = "jat-project/jat-dataset"
        self.dataset_names = JAT_DATASET_CONFIG_NAMES
        self.dataset_split = "train"

    def test_model_loading(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto"
        )
        self.assertIsInstance(model, transformers.AutoModelForCausalLM)

    def test_processor_loading(self):
        processor = transformers.AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.assertIsInstance(processor, transformers.AutoProcessor)

    def test_dataset_loading(self):
        dataset = datasets.load_dataset(self.dataset_path, name=self.dataset_names[0], split=self.dataset_split)
        self.assertIsInstance(dataset, datasets.Dataset)
        self.assertGreater(len(dataset), 0)

    @unittest.skip("Skip unless interested in all datasets")
    def test_all_dataset_loading(self):
        for dataset_name in self.dataset_names:
            with self.subTest(dataset_name=dataset_name):
                dataset = datasets.load_dataset(self.dataset_path, name=dataset_name, split=self.dataset_split)
                self.assertIsInstance(dataset, datasets.Dataset)
                self.assertGreater(len(dataset), 0)

    def test_model_inference(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto"
        )
        processor = transformers.AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        for dataset_name in self.dataset_names:
            with self.subTest(dataset_name=dataset_name):
                dataset = datasets.load_dataset(self.dataset_path, name=dataset_name, split=self.dataset_split)
                sample = dataset[0]
                encoded_sample = processor(**{k: [v] for k, v in sample.items()}, return_tensors="pt")
                encoded_sample.to(model.device)

                with torch.no_grad():
                    output = model(**encoded_sample)

                self.assertIsInstance(
                    output,
                    transformers.modeling_outputs.CausalLMOutputWithCrossAttentions,
                )


if __name__ == "__main__":
    unittest.main()
