import os 
import pandas as pd 
import re 

from Logger import Logger
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

class ModelTrainer:
    def __init__(self, model_name, output_dir="NLP_FINAL_PROJECT/outputs", lora_config=None, train_args=None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.lora_config = lora_config
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

    def clean_text(self,text):
      # Geçersiz karakterleri temizle
      text = re.sub(r'[^\x20-\x7E]', '', text)  # ASCII olmayan karakterleri kaldır
      return text.strip()


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

        # Save evaluation results to an Excel file
        Logger.log("Saving evaluation results to Excel...")

        rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

        results_data = {
            "Text": [],
            "Label": [],
            "Model Output": [],
            "ROUGE-1": [],
            "ROUGE-L": [],
            "BLEU": []
        }

        for example in eval_dataset:
            text = self.clean_text(example["text"])
            label = self.clean_text(str(example["labels"]))
            model_output = self.clean_text(
                tokenizer.decode(
                    model.generate(
                        tokenizer.encode(text, return_tensors="pt", truncation=True).to(model.device)
                    )[0],
                    skip_special_tokens=True
                )
            )

            rouge_scores = rouge.score(label, model_output)
            bleu_score = sentence_bleu([label.split()], model_output.split())

            results_data["Text"].append(text)
            results_data["Label"].append(label)
            results_data["Model Output"].append(model_output)
            results_data["ROUGE-1"].append(rouge_scores["rouge1"].fmeasure)
            results_data["ROUGE-L"].append(rouge_scores["rougeL"].fmeasure)
            results_data["BLEU"].append(bleu_score)

        df = pd.DataFrame(results_data)
        excel_path = os.path.join(self.output_dir, f"evaluation_results_{self.model_name.replace('/', '_')}.xlsx")
        df.to_excel(excel_path, index=False)
        Logger.log(f"Evaluation results saved to {excel_path}")

        return eval_results
