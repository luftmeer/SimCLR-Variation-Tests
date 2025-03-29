import os
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import wandb
import warnings

class TrainingMonitor:
    def __init__(self, save_dir, plot_every=100, maxlen=500, enabled=True, rank=0):
        self.save_dir = os.path.join(save_dir, "training_logs")
        os.makedirs(self.save_dir, exist_ok=True)
        self.plot_every = plot_every
        self.rank = rank
        self.enabled = enabled

        self.gradient_norms = deque(maxlen=maxlen)
        self.losses = deque(maxlen=maxlen)
        self.lrs = deque(maxlen=maxlen)
        self.batch_indices = deque(maxlen=maxlen)

        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "tensorboard")) if self.enabled and self.rank == 0 else None
        self.use_wandb = enabled and wandb.run is not None

    def log(self, model, loss_value, optimizer, batch_idx, epoch=None):
        if not self.enabled or self.rank != 0:
            return

        # Compute gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        # Warn if gradient norm is too large
        if grad_norm > 1000:
            warnings.warn(f"Large gradient norm detected: {grad_norm:.2f} at batch {batch_idx}")

        # Get learning rate (assumes 1 param group)
        lr = optimizer.param_groups[0]['lr']

        # Append to buffers
        step = batch_idx + (epoch * 100000 if epoch is not None else 0)
        self.gradient_norms.append(grad_norm)
        self.losses.append(loss_value)
        self.lrs.append(lr)
        self.batch_indices.append(step)

        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("Loss", loss_value, step)
            self.tb_writer.add_scalar("Gradient Norm", grad_norm, step)
            self.tb_writer.add_scalar("Learning Rate", lr, step)

        # Log to Weights & Biases
        if self.use_wandb:
            wandb.log({
                "loss": loss_value,
                "gradient_norm": grad_norm,
                "learning_rate": lr,
                "batch": batch_idx,
                "epoch": epoch
            })

        # Plot and save
        if batch_idx % self.plot_every == 0:
            self._plot(batch_idx, epoch)
            self._save_csv()

    def _plot(self, batch_idx, epoch):
        x = list(self.batch_indices)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.set_title(f"Training Monitor — Epoch {epoch}, Batch {batch_idx}")
        ax1.plot(x, self.gradient_norms, label="Grad Norm", color="tab:blue")
        ax1.set_ylabel("Gradient Norm", color="tab:blue")
        ax1.tick_params(axis='y', labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(x, self.losses, label="Loss", color="tab:red")
        ax2.set_ylabel("Loss", color="tab:red")
        ax2.tick_params(axis='y', labelcolor="tab:red")

        plt.xlabel("Batch")
        fig.tight_layout()

        filename = f"monitor_epoch{epoch}_batch{batch_idx}.png"
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def _save_csv(self):
        df = pd.DataFrame({
            "batch": list(self.batch_indices),
            "loss": list(self.losses),
            "grad_norm": list(self.gradient_norms),
            "lr": list(self.lrs),
        })
        df.to_csv(os.path.join(self.save_dir, "training_log.csv"), index=False)
