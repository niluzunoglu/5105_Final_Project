from Logger import Logger
from DatasetHandler import DatasetHandler
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

class ModelTrainer:
    def __init__(self, model_name, output_dir, lora_config=None, train_args=None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.lora_config = lora_config
        self.train_args = train_args

    def load_model(self):
        Logger.log("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_8bit=True,
            device_map="auto"
        )
        
        if self.lora_config:
            Logger.log("Applying LoRA configuration...")
            model = get_peft_model(model, self.lora_config)
            model.print_trainable_parameters()
        
        return model, tokenizer

    def train(self, model, tokenizer, train_dataset, eval_dataset):
        Logger.log("Starting training...")

        trainer = Trainer(
            model=model,
            args=self.train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )

        trainer.train()
        Logger.log("Training completed.")
        
        Logger.log("Saving model...")
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        Logger.log(f"Model saved to {self.output_dir}")

    def evaluate(self, model, tokenizer, eval_dataset):
        Logger.log("Starting evaluation...")

        trainer = Trainer(
            model=model,
            args=self.train_args,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )

        eval_results = trainer.evaluate()
        Logger.log("Evaluation results:")
        for key, value in eval_results.items():
            Logger.log(f"{key}: {value}")
        return eval_results
