⚖️ Legal Summarization using Gemma 2B-IT
Fine-tuned the Gemma 2B-IT model for legal document summarization. This project enables the generation of concise, context-aware summaries for complex legal texts using instruction-tuned data, built with Hugging Face Transformers and PEFT.

📌 Table of Contents
🧠 Overview

📁 Dataset Format

🛠️ Setup

🚀 Training & Fine-tuning

🔍 Inference Example

🧾 Model Card

📦 Hugging Face Upload

📄 License

🧠 Overview
This repository presents a fine-tuned version of Google's Gemma 2B-IT LLM to perform legal summarization, making long and complex legal documents easier to comprehend.

🔑 Core Features
Legal domain-specific summarization

Fine-tuned using instruction-tuning approach

Leverages PEFT for efficient training

Outputs concise, readable summaries of legal content

📁 Dataset Format
The dataset is structured as JSONL with the following fields:

json
Copy
Edit
{
  "instruction": "Summarize the following legal case...",
  "context": "This is the full legal document text...",
  "response": "This case is about...",
  "category": "contract_law" 
}
instruction: Task prompt for the model

context: Full legal text (input)

response: Target summary (output)

category: Type of legal content (optional)

🛠️ Setup
bash
Copy
Edit
# Clone the repo
git clone https://github.com/yourusername/legal-summarization-gemma2b.git
cd legal-summarization-gemma2b

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt
📦 Requirements (included in requirements.txt)
text
Copy
Edit
transformers
peft
datasets
accelerate
bitsandbytes
sentencepiece
scipy
🚀 Training & Fine-tuning
We use Hugging Face's PEFT + LoRA for memory-efficient training.

python
Copy
Edit
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

# Load dataset
dataset = load_dataset("path_to_your_dataset")

# Tokenization
def tokenize_fn(example):
    inputs = f"{example['instruction']}\n{example['context']}"
    outputs = example['response']
    return tokenizer(inputs, text_target=outputs, truncation=True, max_length=1024)

tokenized_dataset = dataset.map(tokenize_fn)

# LoRA configuration
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    logging_dir="./logs",
    save_total_limit=2,
    num_train_epochs=3,
    evaluation_strategy="no",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()
🔍 Inference Example
python
Copy
Edit
from transformers import pipeline

pipe = pipeline("text2text-generation", model="yourusername/legal-summarizer-gemma2b", tokenizer="yourusername/legal-summarizer-gemma2b")

instruction = "Summarize the following legal clause:"
context = "According to the Indian Contract Act of 1872, any agreement made under coercion..."

prompt = f"{instruction}\n{context}"

summary = pipe(prompt, max_length=256, do_sample=True)
print(summary[0]['generated_text'])
🧾 Model Card
Model: Fine-tuned google/gemma-2b-it
Domain: Legal NLP
Technique: Instruction Tuning + LoRA
Use Case: Contract summaries, case briefs, legal compliance

Trained On:

Custom legal dataset with instructions, gold summaries, and model outputs

Legal categories: contract law, criminal law, civil procedures, etc.

📦 Hugging Face Upload (Optional)
You can push your fine-tuned model using:

bash
Copy
Edit
huggingface-cli login

from huggingface_hub import HfApi, create_repo, upload_folder
api = HfApi()
create_repo("legal-summarizer-gemma2b", private=False)

upload_folder(
  repo_id="yourusername/legal-summarizer-gemma2b",
  folder_path="./results",
  commit_message="Upload fine-tuned model"
)

Sample outputs:
![pic_1 output](https://github.com/user-attachments/assets/1e998229-9c67-422d-8b09-3e3003845ef2)
![pic_2 output](https://github.com/user-attachments/assets/6570b17a-6555-4db1-8bbe-25cdcd7bf20a)
![pic_3 output](https://github.com/user-attachments/assets/280cd367-d9c0-4555-aeae-d4c4fdb14fe2)
Credits: @Sreecharan03
