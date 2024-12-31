import os 
import pandas as pd 
import re 

from Logger import Logger
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
class ModelTrainer:
  
    def __init__(self, model_name, output_dir="NLP_FINAL_PROJECT/outputs", train_args=None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.train_args = train_args
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self):
        Logger.log("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_8bit=True,
            device_map="auto"
        )
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
            print(f"{key}: {value}")

        return eval_results
