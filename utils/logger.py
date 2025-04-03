import os
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import warnings
import time
import yaml
import threading
import torch
import zipfile
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


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
        self.low_grad_counts = deque(maxlen=maxlen)
        self.batch_indices = deque(maxlen=maxlen)

        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "tensorboard")) if self.enabled and self.rank == 0 else None

    def log(self, model, loss_value, optimizer, batch_idx, epoch=None, logits=None):
        if not self.enabled or self.rank != 0:
            return

        # Compute gradient norm and low gradient count
        grad_norm = 0.0
        very_low_grad_count = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
                update = (p.grad * optimizer.param_groups[0]['lr']).abs().mean().item()
                if update < 1e-5:
                    very_low_grad_count += 1

        grad_norm = grad_norm ** 0.5

        # Warnings
        if grad_norm > 1000:
            warnings.warn(f"Large gradient norm detected: {grad_norm:.2f} at batch {batch_idx}")
        if grad_norm < 0.1:
            warnings.warn(f"Gradient norm very low ({grad_norm:.4f}) at batch {batch_idx}. Potential vanishing gradients.")
        if very_low_grad_count > 0:
            warnings.warn(f"{very_low_grad_count} parameters have very small update magnitudes at batch {batch_idx}.")

        # Learning rate
        lr = optimizer.param_groups[0]['lr']

        # Append to buffers
        step = batch_idx + (epoch * 100000 if epoch is not None else 0)
        self.gradient_norms.append(grad_norm)
        self.losses.append(loss_value)
        self.lrs.append(lr)
        self.low_grad_counts.append(very_low_grad_count)
        self.batch_indices.append(step)

        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("Loss", loss_value, step)
            self.tb_writer.add_scalar("Gradient Norm", grad_norm, step)
            self.tb_writer.add_scalar("Learning Rate", lr, step)
            self.tb_writer.add_scalar("Low Grad Count", very_low_grad_count, step)

        # Log logits if available
        if logits is not None:
            log_min = logits.min().item()
            log_max = logits.max().item()
            log_mean = logits.mean().item()

            if self.tb_writer:
                self.tb_writer.add_scalar("Logits/Min", log_min, step)
                self.tb_writer.add_scalar("Logits/Max", log_max, step)
                self.tb_writer.add_scalar("Logits/Mean", log_mean, step)

            if batch_idx % self.plot_every == 0:
                print(f"[Logits] Batch {batch_idx} | min: {log_min:.4f}, max: {log_max:.4f}, mean: {log_mean:.4f}")

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

        # Optional: overlay low grad counts
        fig2, ax3 = plt.subplots(figsize=(8, 4))
        ax3.set_title("Very Low Gradient Count")
        ax3.plot(x, self.low_grad_counts, label="Low Grad Count", color="tab:purple")
        ax3.set_xlabel("Batch")
        ax3.set_ylabel("Count")
        ax3.grid(True)

        fig.tight_layout()
        fig2.tight_layout()

        filename1 = f"monitor_epoch{epoch}_batch{batch_idx}.png"
        filename2 = f"low_grad_epoch{epoch}_batch{batch_idx}.png"
        fig.savefig(os.path.join(self.save_dir, filename1))
        fig2.savefig(os.path.join(self.save_dir, filename2))
        plt.close(fig)
        plt.close(fig2)

    def _save_csv(self):
        df = pd.DataFrame({
            "batch": list(self.batch_indices),
            "loss": list(self.losses),
            "grad_norm": list(self.gradient_norms),
            "lr": list(self.lrs),
            "low_grad_count": list(self.low_grad_counts),
        })
        df.to_csv(os.path.join(self.save_dir, "training_log.csv"), index=False)

class LinearEvaluationMonitor:
    def __init__(self, save_dir, class_names=None):
        self.save_dir = os.path.join(save_dir, "linear_eval_logs")
        os.makedirs(self.save_dir, exist_ok=True)
        self.class_names = class_names
        self.data = defaultdict(list)
        self.tb_writer = SummaryWriter(log_dir=self.save_dir)
        
    def log_model_weights(self, model, epoch):
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() > 1:
                self.tb_writer.add_histogram(f"Weights/{name}", param.data.cpu().numpy(), epoch)
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    total_norm += grad_norm ** 2
                    self.tb_writer.add_histogram(f"Gradients/{name}", param.grad.data.cpu().numpy(), epoch)
                    self.tb_writer.add_scalar(f"GradNorms/{name}", grad_norm, epoch)
        total_norm = total_norm ** 0.5
        self.tb_writer.add_scalar("GradNorms/Total", total_norm, epoch)

    def detect_anomalies(self, model, epoch):
        nan_found = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_found = True
                print(f"[Anomaly] NaN detected in gradients of {name} at epoch {epoch}")
                self.tb_writer.add_text("Anomaly", f"NaN in {name} at epoch {epoch}", epoch)
        if not nan_found:
            self.tb_writer.add_text("Anomaly", "No NaNs detected in gradients", epoch)

    def log_metrics(self, epoch, top1, top5, loss, lr, eval_time, per_class_acc, model=None, features=None, labels=None):
        self.data['epoch'].append(epoch)
        self.data['top1'].append(top1)
        self.data['top5'].append(top5)
        self.data['loss'].append(loss)
        self.data['lr'].append(lr)
        self.data['eval_time'].append(eval_time)
        self.data['timestamp'].append(time.strftime('%Y-%m-%d_%H-%M-%S'))

        if model is not None:
            frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.data['frozen_params'].append(frozen)
            self.data['trainable_params'].append(trainable)

            self.tb_writer.add_scalar('Params/Frozen', frozen, epoch)
            self.tb_writer.add_scalar('Params/Trainable', trainable, epoch)
            self.log_model_weights(model, epoch)
            self.detect_anomalies(model, epoch)

        if features is not None and labels is not None:
            self.tb_writer.add_embedding(features, metadata=labels, tag=f"embeddings/epoch_{epoch}")

        if per_class_acc is not None:
            for idx, acc in enumerate(per_class_acc):
                label = self.class_names[idx] if self.class_names else str(idx)
                self.data[f"acc_{label}"].append(acc)
                self.tb_writer.add_scalar(f"PerClassAccuracy/{label}", acc, epoch)
                
        self.tb_writer.add_scalar('Accuracy/Top1', top1, epoch)
        self.tb_writer.add_scalar('Accuracy/Top5', top5, epoch)
        self.tb_writer.add_scalar('Loss', loss, epoch)
        self.tb_writer.add_scalar('LearningRate', lr, epoch)
        self.tb_writer.add_scalar('EvalTime', eval_time, epoch)

        self.log_tsne_async(features, labels, epoch)
        self._save_yaml(epoch)
        self._save_csv()
        self.summarize_all_epochs()

    def _save_yaml(self, epoch):
        yaml_path = os.path.join(self.save_dir, f"eval_epoch_{epoch}.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump({k: v[-1] for k, v in self.data.items()}, f)

    def _save_csv(self):
        df = pd.DataFrame(self.data)
        df.to_csv(os.path.join(self.save_dir, "linear_eval_metrics.csv"), index=False)

    def log_tsne_async(self, features, labels, epoch):
        thread = threading.Thread(target=self._compute_and_plot_tsne, args=(features, labels, epoch))
        thread.start()

    def _compute_and_plot_tsne(self, features, labels, epoch):
        print("[t-SNE] Computing 2D projection...", flush=True)
        tsne = TSNE(n_components=2, init='pca', random_state=42)
        reduced = tsne.fit_transform(features.cpu().numpy())

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels.cpu(), cmap='tab10', alpha=0.6)
        if self.class_names:
            legend = plt.legend(handles=scatter.legend_elements()[0], labels=self.class_names, loc='best')
            plt.gca().add_artist(legend)

        plt.title(f"t-SNE projection (Epoch {epoch})")
        plt.tight_layout()
        tsne_path = os.path.join(self.save_dir, f"tsne_epoch_{epoch}.png")
        plt.savefig(tsne_path)
        plt.close()
        print(f"[t-SNE] Saved to {tsne_path}", flush=True)

    def log_confusion_matrix(self, y_true=None, y_pred=None, cm_tensor=None, epoch=0):
        print("[Confusion Matrix] Generating plot...", flush=True)
        if cm_tensor is not None:
            cm = cm_tensor.cpu().numpy()
        elif y_true is not None and y_pred is not None:
            cm = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names) if self.class_names else max(y_true)+1))
        else:
            raise ValueError("Provide either y_true and y_pred or cm_tensor")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        plt.title(f"Confusion Matrix (Epoch {epoch})")
        plt.tight_layout()
        cm_path = os.path.join(self.save_dir, f"confusion_matrix_epoch_{epoch}.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"[Confusion Matrix] Saved to {cm_path}", flush=True)

    def summarize_all_epochs(self):
        csv_path = os.path.join(self.save_dir, "linear_eval_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['top1'] = df['top1'].astype(float)
            df['top5'] = df['top5'].astype(float)
            summary_path = os.path.join(self.save_dir, "summary_report.txt")
            with open(summary_path, 'w') as f:
                f.write("Linear Evaluation Summary Report\n")
                f.write("===============================\n")
                f.write(f"Total Epochs: {len(df)}\n")
                f.write(f"Best Top-1 Accuracy: {df['top1'].max():.2f}% at Epoch {df.loc[df['top1'].idxmax(), 'epoch']}\n")
                f.write(f"Best Top-5 Accuracy: {df['top5'].max():.2f}% at Epoch {df.loc[df['top5'].idxmax(), 'epoch']}\n")
                f.write(f"Average Evaluation Time: {df['eval_time'].mean():.2f}s\n")
                if 'frozen_params' in df.columns:
                    f.write(f"Frozen Parameters: {df['frozen_params'].iloc[-1]}\n")
                    f.write(f"Trainable Parameters: {df['trainable_params'].iloc[-1]}\n")

            # Accuracy curves
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['epoch'], df['top1'], label='Top-1 Accuracy')
            ax.plot(df['epoch'], df['top5'], label='Top-5 Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Top-1 and Top-5 Accuracy Over Epochs')
            ax.legend()
            plt.grid(True)
            plt.tight_layout()
            acc_plot_path = os.path.join(self.save_dir, "accuracy_over_epochs.png")
            plt.savefig(acc_plot_path)
            plt.close()
            print(f"[Summary] Accuracy plot saved to {acc_plot_path}", flush=True)

            # Loss curve
            if 'loss' in df:
                plt.figure(figsize=(10, 6))
                plt.plot(df['epoch'], df['loss'], label='Loss', color='tab:red')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Loss Over Epochs')
                plt.grid(True)
                plt.tight_layout()
                loss_plot_path = os.path.join(self.save_dir, "loss_over_epochs.png")
                plt.savefig(loss_plot_path)
                plt.close()
                print(f"[Summary] Loss curve saved to {loss_plot_path}", flush=True)

            # Per-class accuracy heatmap
            acc_cols = [col for col in df.columns if col.startswith('acc_')]
            if acc_cols:
                acc_matrix = df[acc_cols].T  # shape: [classes, epochs]
                plt.figure(figsize=(12, 6))
                im = plt.imshow(acc_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
                plt.colorbar(im, label='Accuracy')
                plt.yticks(range(len(acc_cols)), acc_cols)
                plt.xticks(range(len(df['epoch'])), df['epoch'], rotation=90)
                plt.xlabel('Epoch')
                plt.title('Per-Class Accuracy Heatmap')
                plt.tight_layout()
                heatmap_path = os.path.join(self.save_dir, "per_class_accuracy_heatmap.png")
                plt.savefig(heatmap_path)
                plt.close()
                print(f"[Summary] Per-class accuracy heatmap saved to {heatmap_path}", flush=True)

            # Create archive
            zip_path = os.path.join(self.save_dir, "linear_eval_logs.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(self.save_dir):
                    for file in files:
                        if file != os.path.basename(zip_path):
                            full_path = os.path.join(root, file)
                            arcname = os.path.relpath(full_path, self.save_dir)
                            zipf.write(full_path, arcname)
            print(f"[Summary] Logs archived at {zip_path}", flush=True)
        else:
            print("[Summary] No metrics CSV found to summarize.", flush=True)