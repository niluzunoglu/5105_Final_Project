from Logger import Logger
from DatasetHandler import DatasetHandler
from ModelTrainer import ModelTrainer

if __name__ == "__main__":
    # Logging setup
    Logger.setup_logging()

    # Configuration
    MODEL_NAME = "NovusResearch/Novus-7b-tr_v1"
    DATASET_NAME = "savasy/ttc4900"  # Dataset from Hugging Face
    OUTPUT_DIR = "./fine_tuned_model"
    MAX_LENGTH = 128
    TRAIN_EPOCHS = 3
    #BATCH_SIZE = 4

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
        logging_dir="./logs"
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
    trainer.train(model, tokenizer, train_dataset, eval_dataset)

    # Model evaluation
    trainer.evaluate(model, tokenizer, eval_dataset)
