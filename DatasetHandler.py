from Logger import Logger
from datasets import load_dataset
import re 

class DatasetHandler:

    def __init__(self, dataset_name, tokenizer, max_length=128):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_dataset(self):
        Logger.log("Loading dataset from Hugging Face...")
        dataset = load_dataset(self.dataset_name)
        Logger.log("Dataset loaded successfully.")
        return dataset

    def split_dataset(self, dataset, test_size=0.113):
        Logger.log("Splitting dataset into train and test...")

        # Dataset'in train-test bölünmesi
        train_test_splitted_data = dataset['train'].train_test_split(test_size=test_size, seed=42)

        Logger.log("Dataset split successfully.")
        Logger.log(f"Train dataset size: {len(train_test_splitted_data['train'])}")
        Logger.log(f"Test dataset size: {len(train_test_splitted_data['test'])}")   
                
        print(f"Train dataset size: {len(train_test_splitted_data['train'])}")
        print(f"Test dataset size: {len(train_test_splitted_data['test'])}")

        return train_test_splitted_data

    def preprocess(self, dataset):
        Logger.log("Simplifying dataset and tokenizing...")

        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        Logger.log("Tokenization completed.")
        return tokenized_dataset


if __name__ == "__main__":

    DATASET_NAME = "savasy/ttc4900"
    MODEL_NAME = "TURKCELL/Turkcell-LLM-7b-v1"
    MAX_LENGTH = 128
  
    # Dataset handling
    dataset_handler = DatasetHandler(DATASET_NAME, AutoTokenizer.from_pretrained(MODEL_NAME), MAX_LENGTH)
    raw_dataset = dataset_handler.load_dataset()

    print("Raw dataset : ", raw_dataset)
    split_dataset = dataset_handler.split_dataset(raw_dataset, test_size=0.113)
    print("Dataset splitted .. ", split_dataset)
    tokenized_dataset = dataset_handler.preprocess(split_dataset),
    print("Dataset tokenized  .. ", tokenized_dataset)
