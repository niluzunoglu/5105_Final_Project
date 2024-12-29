from Logger import Logger
from DatasetHandler import DatasetHandler
from ModelTrainer import ModelTrainer
from peft import LoraConfig 
from transformers import TrainingArguments, AutoTokenizer

MODELS = ["TURKCELL/Turkcell-LLM-7b-v1",
          "NovusResearch/Novus-7b-tr_v1",
          "Orbina/Orbita-v0.1",
          "sambanovasystems/SambaLingo-Turkish-Chat",
          "ytu-ce-cosmos/turkish-gpt2-large"]

DATASET = "savasy/ttc4900"

if __name__ == "__main__":
    # Logging setup
    Logger.setup_logging()

    # Configuration
    MODEL_NAME = MODELS[0] 
    DATASET_NAME = DATASET  # Dataset from Hugging Face
    OUTPUT_DIR = "NLP_FINAL_PROJECT/outputs"
    MAX_LENGTH = 128
    TRAIN_EPOCHS = 1
    BATCH_SIZE = 4

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Training arguments
    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        learning_rate=1e-4,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=TRAIN_EPOCHS,
        save_steps=10,
        save_total_limit=2,
        fp16=True,
        logging_dir="NLP_FINAL_PROJECT/logs"
    )

    # Dataset handling
    dataset_handler = DatasetHandler(DATASET_NAME, AutoTokenizer.from_pretrained(MODEL_NAME), MAX_LENGTH)
    raw_dataset = dataset_handler.load_dataset()
    split_dataset = dataset_handler.split_dataset(raw_dataset)
    tokenized_dataset = dataset_handler.preprocess(split_dataset)

    # Train-test split
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]

    # Model training
    trainer = ModelTrainer(MODEL_NAME, OUTPUT_DIR, lora_config, train_args)
    model, tokenizer = trainer.load_model()

    print("Sample Input IDs:", train_dataset[0]["input_ids"])
    print("Sample Labels:", train_dataset[0]["labels"])

    trainer.train(model, tokenizer, train_dataset, eval_dataset)

    # Model evaluation
    trainer.evaluate(model, tokenizer, eval_dataset)
