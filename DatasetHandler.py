from Logger import Logger
from datasets import load_dataset

class DatasetHandler:
    def __init__(self, dataset_name, tokenizer, max_length=128):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_dataset(self):
        Logger.log("Loading dataset from Hugging Face...")
        dataset = load_dataset(self.dataset_name)

        if "validation" not in dataset:
            Logger.log("Validation split not found. Splitting train dataset into train and validation...")
            dataset = dataset["train"].train_test_split(test_size=0.119)
            print("[dataset]", dataset)

        return dataset

    def split_dataset(self, dataset, test_size=0.2):
        if "validation" not in dataset:
            Logger.log("Validation split not found. Splitting train dataset into train and validation...")
            dataset = dataset["train"].train_test_split(test_size=test_size)
        return dataset
        
    def preprocess(self, dataset):
        Logger.log("Simplifying dataset and tokenizing...")

        def simplify_and_tokenize_function(examples):
            if "text" not in examples or "category" not in examples:
                raise ValueError("Dataset must contain 'text' and 'category' columns.")
            simplified_texts = [" ".join(text.split()[:5]) for text in examples["text"]]
            tokenized = self.tokenizer(
                simplified_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            # Map category to labels and match the sequence length
            tokenized["labels"] = [
                label if isinstance(label, int) else 0 for label in examples["category"]
            ]
            tokenized["labels"] = tokenized["input_ids"]  # Align labels with input_ids
            return tokenized

        tokenized_dataset = dataset.map(simplify_and_tokenize_function, batched=True)
        return tokenized_dataset
