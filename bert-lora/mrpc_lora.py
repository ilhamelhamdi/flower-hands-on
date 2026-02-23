import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import evaluate
import numpy as np
import pandas as pd
import time 
from peft import LoraConfig, get_peft_model, TaskType

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.append(project_root)   
from common.resource_monitor import ResourceMonitor

torch.set_num_threads(8)

raw_dataset = load_dataset("nyu-mll/glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

def encode(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")
dataset = raw_dataset.map(encode, batched=True)
dataset = dataset.map(
    lambda examples: {"labels": examples["label"]}, batched=True)
train_subset = dataset["train"].shuffle(seed=42).select(range(1000))
eval_subset = dataset["validation"].shuffle(seed=42).select(range(100))


lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none"
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"]
    }


training_args = TrainingArguments(
    output_dir="./lora_mrpc_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    use_cpu=True,
    logging_steps=10
)

monitor = ResourceMonitor(interval=2)
monitor.start()

start_time = time.time()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=eval_subset,
    compute_metrics=compute_metrics
)

# Start Trainings
train_output = trainer.train()
total_time = time.time() - start_time

monitor.stop()
monitor.join()

df = pd.DataFrame(monitor.log)
stats = {
    "Total Time (s)": total_time,
    "Avg RAM (MB)": df["ram_mb"].mean(),
    "Peak RAM (MB)": df["ram_mb"].max(),
    "Avg CPU %": df["cpu_percent"].mean(),
    "Peak CPU %": df["cpu_percent"].max(),
    "F1 Score": trainer.evaluate()["eval_f1"],
    "Accuracy": trainer.evaluate()["eval_accuracy"]
}

print("\n--- RESEARCH METRICS ---")
print(pd.Series(stats))
df.to_csv("training_resource_log.csv", index=False)