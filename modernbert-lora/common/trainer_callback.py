from transformers import TrainerCallback
import time
import wandb


class EpochTimerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        print(
            f"\n--- Epoch {state.epoch:.0f} Duration: {epoch_time:.2f} seconds ---")

        wandb.log({
            "epoch/duration_sec": epoch_time,
            "epoch/number": state.epoch
        })
