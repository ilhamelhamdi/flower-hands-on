"""flowertune-llm: A Flower / FlowerTune app."""

import os
from datetime import datetime

import wandb
import pandas as pd
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from omegaconf import DictConfig
from transformers import TrainingArguments, Trainer
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from flowertune_tinybert_lora.utils import replace_keys
from flowertune_tinybert_lora.models import get_model
from flowertune_tinybert_lora.dataset import get_encoding_func_and_data_collator, compute_metrics

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Initialize Weights & Biases
    wandb_run = initialize_wandb(cfg)

    # Get initial model weights
    init_model = get_model(cfg.model)
    arrays = ArrayRecord(get_peft_model_state_dict(init_model))

    # Prepare validation set for evaluation function
    val_set, data_collator = get_validation_set_and_data_collator(cfg)

    # Define strategy
    strategy = FedAvg(
        fraction_train=cfg.strategy.fraction_train,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
    )

    # Start strategy, run FedAvg for `num_rounds`
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"save_path": save_path}),
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(
            cfg, val_set, data_collator, save_path
        ),
    )

    wandb_run.finish()


def get_validation_set_and_data_collator(cfg):
    """Load validation set for evaluation."""
    from datasets import load_dataset

    raw_val_set = load_dataset(
        cfg.dataset.name, cfg.dataset.subset, split="validation")
    encoding_func, data_collator = get_encoding_func_and_data_collator(
        cfg.model.name)
    val_set = raw_val_set.map(encoding_func, batched=True)
    return val_set, data_collator


def get_evaluate_fn(cfg, validation_set, data_collator, save_path):
    """Return an evaluation function for saving global model ."""

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        # Initialize model with current aggregated weights
        model = get_model(cfg.model)
        set_peft_model_state_dict(model, arrays.to_torch_state_dict())

        # Save model
        if server_round != 0 and (
            server_round == cfg.num_server_rounds or server_round % cfg.train.save_every_round == 0
        ):
            model.save_pretrained(f"{save_path}/peft_{server_round}")

        # Evaluate model on validation set
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"{save_path}/eval", per_device_eval_batch_size=32),
            eval_dataset=validation_set,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
        metrics = trainer.evaluate()

        wandb.log(
            {
                "server_round": server_round,
                "eval_loss": metrics["eval_loss"],
                "eval_accuracy": metrics["eval_accuracy"],
                "eval_f1": metrics["eval_f1"],
            }
        )

        # Save to local CSV
        res = {"round": [server_round], "accuracy": [
            metrics["eval_accuracy"]], "loss": [metrics["eval_loss"]]}
        df = pd.DataFrame(res)
        df.to_csv(f"{save_path}/results.csv", mode='a', index=False,
                  header=not os.path.exists(f"{save_path}/results.csv"))

        return MetricRecord(metrics)

    return evaluate


def initialize_wandb(cfg):
    """Initialize Weights & Biases for experiment tracking."""
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=cfg,
    )
