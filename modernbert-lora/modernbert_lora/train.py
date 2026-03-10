from common.trainer_callback import EpochTimerCallback
from common.resource_monitor import ResourceMonitor
import sys
import os
from datetime import datetime
import time
import pandas as pd
from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np
import wandb
from modernbert_lora.utils import load_config, to_dict
from modernbert_lora.dataset import get_preprocessed_data
from modernbert_lora.models import get_model



def train(task_name, cli_args=None):
    config = load_config(cli_args=cli_args)
    ds_config = to_dict(config.datasets[task_name])

    # Create output directory given current timestamp
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    # Initialize wandb run
    run_name = f"{task_name}-{timestamp}{('-' + config.args.run_name) if config.args.run_name else ''}"
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=run_name,
        config=config
    )

    # Load Data
    train_set, val_set, tokenizer, data_collator = get_preprocessed_data(
        config.model_id, ds_config)

    # Setup Metrics
    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return acc_metric.compute(predictions=predictions, references=labels)

    # Load Model
    num_labels = len(train_set.features["label"].names)
    model = get_model(config.model_id, to_dict(config.lora), num_labels)

    # Training Arguments from Config
    training_args = TrainingArguments(
        output_dir=save_path,
        eval_strategy="epoch",
        **(to_dict(config.args))
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EpochTimerCallback()],
    )

    # Resource Monitoring and Timing
    monitor = ResourceMonitor(interval=2)
    monitor.start()
    start_time = time.time()

    train_output = trainer.train()

    # Calculate your Research Metrics
    monitor.stop()
    df = pd.DataFrame(monitor.log)
    stats = {
        "research/total_time": time.time() - start_time,
        "research/peak_ram_mb": df["ram_mb"].max(),
        "research/avg_cpu_percent": df["cpu_percent"].mean(),
    }

    wandb.log(stats)
    wandb.finish()


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "sst2"
    overrides = sys.argv[2:]
    train(task, cli_args=overrides)
