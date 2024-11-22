import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForLanguageModeling
from huggingface_hub import login

login(token="")

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-1.1-2b-it",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    token="",
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it", token="")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("text", data_files="/kaggle/input/fine-tune/fine_tune.txt")

dataset = dataset["train"].train_test_split(test_size=0.1)


def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )


tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    save_steps=10_000,
    save_total_limit=2,
    fp16=True,
    logging_steps=10,
    learning_rate=2e-4,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./qloRA_model")
