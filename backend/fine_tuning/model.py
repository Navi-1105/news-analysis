from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import TextDataset, DataCollatorForLanguageModeling
import torch
from typing import List, Dict
import os

class NewsModel:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add special tokens
        special_tokens = {
            'pad_token': '<PAD>',
            'bos_token': '<BOS>',
            'eos_token': '<EOS>'
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_dataset(self, texts: List[str], output_dir: str):
        """
        Prepare dataset for fine-tuning
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write texts to file
        with open(os.path.join(output_dir, "train.txt"), "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")

        # Create dataset
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=os.path.join(output_dir, "train.txt"),
            block_size=128
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        return train_dataset, data_collator

    def fine_tune(self, train_dataset, data_collator, output_dir: str):
        """
        Fine-tune the model
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=500,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate a response using the fine-tuned model
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def load_model(self, model_path: str):
        """
        Load a fine-tuned model
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path) 