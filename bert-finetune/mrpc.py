import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import evaluate
import numpy as np

torch.set_num_threads(4)

raw_dataset = load_dataset("nyu-mll/glue", "mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def encode(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")


dataset = raw_dataset.map(encode, batched=True)
dataset = dataset.map(
    lambda examples: {"labels": examples["label"]}, batched=True)

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

train_subset = dataset["train"].shuffle(seed=42).select(range(150))
eval_subset = dataset["validation"].shuffle(seed=42).select(range(20))


training_args = TrainingArguments(
    output_dir="./mrpc_results",
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=eval_subset,
    compute_metrics=compute_metrics
)

# Start Trainings
trainer.train()

results = trainer.evaluate()
print("\nFinal Evaluation Results:", results)
