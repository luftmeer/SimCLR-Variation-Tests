import os
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import warnings
import time
import yaml
import torch
import zipfile
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import numpy as np
import math


class TrainingMonitor:
    def __init__(self, save_dir,  batch_size, n_augments=2, plot_every=100, maxlen=500, enabled=True, rank=0):
        self.save_dir = os.path.join(save_dir, "training_logs")
        os.makedirs(self.save_dir, exist_ok=True)
        self.plot_every = plot_every
        self.rank = rank
        self.enabled = enabled
        self.n_augments = n_augments

        self.losses = deque(maxlen=maxlen)
        self.lrs = deque(maxlen=maxlen)
        self.low_grad_counts = deque(maxlen=maxlen)
        self.batch_indices = deque(maxlen=maxlen)
        
        self.combs = math.comb(self.n_augments, 2)
        self.comb_losses = [] if self.combs == 1 else [[] for _ in range(self.combs)]
        self.positive_samples = [] if self.combs == 1 else [[] for _ in range(self.combs)]
        self.negative_samples = [] if self.combs == 1 else [[] for _ in range(self.combs)]
        self.batch_size = batch_size

        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "tensorboard")) if self.enabled and self.rank == 0 else None

    def log_epoch(self, epoch, lr, final_similarity=None):
        if not self.enabled or self.rank != 0:
            return

        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar("Epoch/Learning_Rate", lr, epoch)
        
        # Combine tensors
        pos_neg_dict = {}
        if self.combs == 1:
            positives = torch.cat(self.positive_samples)
            negatives = torch.cat(self.negative_samples).view(-1)
            avg_loss = np.mean(self.comb_losses)
            
            # Stats
            mean_pos = positives.mean().item()
            mean_neg = negatives.mean().item()
            margin = mean_pos - mean_neg
            std_pos = positives.std().item()
            std_neg = negatives.std().item()
            
            pos_neg_dict['average_loss'] = avg_loss
            pos_neg_dict['mean_positive'] = mean_pos
            pos_neg_dict['mean_negative'] = mean_neg
            pos_neg_dict['contrast_margin'] = margin
            pos_neg_dict['std_positive'] = std_pos
            pos_neg_dict['std_negative'] = std_neg
            
            if self.tb_writer:
                self.tb_writer.add_scalar("Epoch/Loss", avg_loss, epoch)
                self.tb_writer.add_scalar("Epoch/Mean_Positive", mean_pos, epoch)
                self.tb_writer.add_scalar("Epoch/Mean_Negative", mean_neg, epoch)
                self.tb_writer.add_scalar("Epoch/Contrast_Margin", margin, epoch)
                
                self.tb_writer.add_histogram("Epoch/Positive Similarities", positives, epoch)
                self.tb_writer.add_histogram("Epoch/Negative Similarities", negatives, epoch)
                
        else:
            for comb in range(self.combs):
                positives = torch.cat(self.positive_samples[comb])
                negatives = torch.cat(self.negative_samples[comb]).view(-1)
                avg_loss = np.mean(self.comb_losses[comb])
                
                # Stats
                mean_pos = positives.mean().item()
                mean_neg = negatives.mean().item()
                margin = mean_pos - mean_neg
                std_pos = positives.std().item()
                std_neg = negatives.std().item()
                
                pos_neg_dict[f'average_loss_{comb}'] = avg_loss
                pos_neg_dict[f'mean_positive_{comb}'] = mean_pos
                pos_neg_dict[f'mean_negative_{comb}'] = mean_neg
                pos_neg_dict[f'contrast_margin_{comb}'] = margin
                pos_neg_dict[f'std_positive_{comb}'] = std_pos
                pos_neg_dict[f'std_negative_{comb}'] = std_neg
                
                if self.tb_writer:
                    self.tb_writer.add_scalar(f"Epoch/Loss_{comb}", avg_loss, epoch)
                    self.tb_writer.add_scalar(f"Epoch/Mean_Positive_{comb}", mean_pos, epoch)
                    self.tb_writer.add_scalar(f"Epoch/Mean_Negative_{comb}", mean_neg, epoch)
                    self.tb_writer.add_scalar(f"Epoch/Contrast_Margin_{comb}", margin, epoch)
                    
                    self.tb_writer.add_histogram(f"Epoch/Positive Similarities_{comb}", positives, epoch)
                    self.tb_writer.add_histogram(f"Epoch/Negative Similarities_{comb}", negatives, epoch)   
        
            if final_similarity is not None and final_similarity.shape[0] <= 128:
                self.tb_writer.add_histogram("Epoch/Final_Similarity_Row0", final_similarity[0], epoch)

        # CSV export
        row = {"epoch": epoch, "lr": lr}
        row.update(pos_neg_dict)
        self._append_epoch_to_csv(row)

        # Optional: console summary
        print(f"\n[Epoch {epoch}] Logging Summary:")
        if self.combs == 1:
            print(f"  Loss: {avg_loss:.4f} | Pos: {mean_pos:.4f} ± {std_pos:.4f} | "
                f"Neg: {mean_neg:.4f} ± {std_neg:.4f} | Margin: {margin:.4f}")
        else:
            for comb in range(self.combs):
                avg_loss = pos_neg_dict[f'average_loss_{comb}']
                mean_pos = pos_neg_dict[f'mean_positive_{comb}']
                mean_neg = pos_neg_dict[f'mean_negative_{comb}']
                std_pos = pos_neg_dict[f'std_positive_{comb}']
                std_neg = pos_neg_dict[f'std_negative_{comb}']
                margin = pos_neg_dict[f'contrast_margin_{comb}']

                print(f"  Combo {comb:>2}: Loss {avg_loss:.4f} | Pos: {mean_pos:.4f} ± {std_pos:.4f} | "
                    f"Neg: {mean_neg:.4f} ± {std_neg:.4f} | Margin: {margin:.4f}")
        
        # Cleanup
        if self.combs == 1:
            self.positive_samples.clear()
            self.negative_samples.clear()
            self.comb_losses.clear()
        else:
            for lst in (self.positive_samples, self.negative_samples, self.comb_losses):
                for sublist in lst:
                    sublist.clear()
    
    def _append_epoch_to_csv(self, row_dict):
        csv_path = os.path.join(self.save_dir, "epoch_log.csv")
        df_new = pd.DataFrame([row_dict])

        if not os.path.exists(csv_path):
            df_new.to_csv(csv_path, index=False)
        else:
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
    
    def log_tsne_embeddings(self, embeddings, projections, labels, epoch):
        """
        Generate and save t-SNE plots and data for embeddings and projections.
        """
        if not self.enabled or self.rank != 0:
            return
        for i in range(len(embeddings)):
            emb = embeddings[i]
            proj = projections[i]

            if isinstance(emb, list): emb = torch.cat(emb, dim=0)
            if isinstance(proj, list): proj = torch.cat(proj, dim=0)

            emb_np = emb.numpy()
            proj_np = proj.numpy()
            label_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
            # Run t-SNE
            tsne_e = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto').fit_transform(emb_np)
            tsne_p = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto').fit_transform(proj_np)

            # Save plots
            self._save_tsne_plot(tsne_e, label_np, f"tsne/tsne_embeddings_epoch{epoch}_augmentation{i}.png", title="t-SNE: Encoder Embeddings")
            self._save_tsne_plot(tsne_p, label_np, f"tsne/tsne_projections_epoch{epoch}_augmentation{i}.png", title="t-SNE: Projected Features")

            # Save raw 2D coords
            df = pd.DataFrame({
                "x_embed": tsne_e[:, 0],
                "y_embed": tsne_e[:, 1],
                "x_proj": tsne_p[:, 0],
                "y_proj": tsne_p[:, 1],
                "label": labels
            })
            df.to_csv(os.path.join(self.save_dir, f"tsne/tsne_coords_epoch{epoch}_augmentation{i}.csv"), index=False)

            print(f"[t-SNE] Saved encoder and projection plots for epoch {epoch} of Augmentation {i}")
    
    def log_gradient(self, model, optimizer, batch_idx, epoch):
        if not self.enabled or self.rank != 0:
            return

        # Compute gradient norm and low gradient count
        grad_norm = 0.0
        very_low_grad_count = 0
        for _, p in model.named_parameters():
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

        # Append to buffers
        step = batch_idx + (epoch * self.batch_size if epoch is not None else 0)
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("Gradient Norm", grad_norm, step)
            self.tb_writer.add_scalar("Low Grad Count", very_low_grad_count, step)
    
    def log_logits(self, batch_idx, epoch, comb_nr, logits=None):
        if not self.enabled or self.rank != 0 or isinstance(logits, type(None)):
            return
        
        step = batch_idx + (epoch * self.batch_size if epoch is not None else 0)
        # Log logits if available
        if logits is not None:
            log_min = logits.min().item()
            log_max = logits.max().item()
            log_mean = logits.mean().item()

            if self.tb_writer:
                self.tb_writer.add_scalar(f"Logits/Min_{comb_nr}", log_min, step)
                self.tb_writer.add_scalar(f"Logits/Max_{comb_nr}", log_max, step)
                self.tb_writer.add_scalar(f"Logits/Mean_{comb_nr}", log_mean, step)
    
    
    def _save_tsne_plot(self, tsne_coords, labels, filename, title="t-SNE Plot"):
        plt.figure(figsize=(7, 6))
        sns.scatterplot(x=tsne_coords[:, 0], y=tsne_coords[:, 1], hue=labels, palette='tab10', s=10, alpha=0.7, linewidth=0)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Label", fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

        
    def log_pos_neg_samples(self, positive, negative, comb_nr):
        if not self.enabled or self.rank != 0:
            return
        
        if self.combs == 1:
            self.positive_samples.append(positive)
            self.negative_samples.append(negative.view(-1))
        else:
            self.positive_samples[comb_nr].append(positive)
            self.negative_samples[comb_nr].append(negative.view(-1))
    
    def log_losses(self, loss, comb_nr):
        if not self.enabled or self.rank != 0:
            return
        
        if self.combs == 1:
            self.comb_losses.append(loss)
        else:
            self.comb_losses[comb_nr].append(loss)
    
    def _plot_similarity_distributions(self, epoch):
        if not self.positive_samples or not self.negative_samples:
            return
        if self.combs == 1:
            pos = torch.cat(self.positive_samples).numpy()
            neg = torch.cat(self.negative_samples).numpy()

            plt.figure(figsize=(8, 5))
            sns.histplot(pos, bins=50, color='green', label='Positives', stat='density', kde=True)
            sns.histplot(neg, bins=50, color='red', label='Negatives', stat='density', kde=True)
            plt.title(f"Similarity Distributions — Epoch {epoch}")
            plt.xlabel("Cosine Similarity / τ")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()

            filename = f"similarity_dist_epoch{epoch}.png"
            plt.savefig(os.path.join(self.save_dir, filename))
            plt.close()

            print(f"[Similarity Plot] Saved {filename}")
            
        else:
            for comb in range(self.combs):
                pos = torch.cat(self.positive_samples[comb]).numpy()
                neg = torch.cat(self.negative_samples[comb]).numpy()

                plt.figure(figsize=(8, 5))
                sns.histplot(pos, bins=50, color='green', label='Positives', stat='density', kde=True)
                sns.histplot(neg, bins=50, color='red', label='Negatives', stat='density', kde=True)
                plt.title(f"Similarity Distributions — Epoch {epoch}, Combination {comb}")
                plt.xlabel("Cosine Similarity / τ")
                plt.ylabel("Density")
                plt.legend()
                plt.tight_layout()

                filename = f"similarity_dist_epoch{epoch}_comb{comb}.png"
                plt.savefig(os.path.join(self.save_dir, filename))
                plt.close()

                print(f"[Similarity Plot] Saved {filename}")
    
    def _save_similarity_csv(self, epoch):
        if not self.positive_samples or not self.negative_samples:
            return
        
        if self.combs == 1:
            pos = torch.cat(self.positive_samples).numpy()
            neg = torch.cat(self.negative_samples).numpy()
            df = pd.DataFrame({
                "similarity": np.concatenate([pos, neg]),
                "type": ["positive"] * len(pos) + ["negative"] * len(neg)
            })
            df.to_csv(os.path.join(self.save_dir, f"similarities_epoch{epoch}.csv"), index=False)
        else:
            for comb in range(self.combs):
                pos = torch.cat(self.positive_samples[comb]).numpy()
                neg = torch.cat(self.negative_samples[comb]).numpy()
                df = pd.DataFrame({
                    "similarity": np.concatenate([pos, neg]),
                    "type": ["positive"] * len(pos) + ["negative"] * len(neg)
                })
                df.to_csv(os.path.join(self.save_dir, f"similarities_epoch{epoch}_comb{comb}.csv"), index=False)
            
class LinearEvaluationMonitor:
    def __init__(self, save_dir, cpt_epoch: int, class_names=None):
        self.save_dir = os.path.join(save_dir, "linear_eval_logs", str(cpt_epoch))
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

    def log_metrics(self, epoch, top1, top5, loss, lr, eval_time, per_class_acc, model=None, features=None, labels=None, logits=None, prefix: str="train"):
        tag = lambda name: f"{prefix}/{name}" if prefix else name

        self.data['epoch'].append(epoch)
        self.data['top1'].append(top1)
        self.data['top5'].append(top5)
        self.data['loss'].append(loss)
        self.data['lr'].append(lr)
        self.data['eval_time'].append(eval_time)
        self.data['timestamp'].append(time.strftime('%Y-%m-%d_%H-%M-%S'))
        self.data['method'].append(prefix)

        if model is not None:
            frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.data['frozen_params'].append(frozen)
            self.data['trainable_params'].append(trainable)

            self.tb_writer.add_scalar(tag('Params/Frozen'), frozen, epoch)
            self.tb_writer.add_scalar(tag('Params/Trainable'), trainable, epoch)
            self.log_model_weights(model, epoch)
            self.detect_anomalies(model, epoch)

        """        
        if features is not None and labels is not None and (epoch == 0 or (epoch+1) % 10 == 0 or (epoch+1)==101):
            metadata = [", ".join(self.class_names[l.item()]) if isinstance(self.class_names[l.item()], tuple) else str(self.class_names[l.item()]) for l in labels]
            self.tb_writer.add_embedding(features, metadata=metadata, tag=f"{prefix}_features_embeddings/epoch_{epoch}")
        
        if logits is not None and labels is not None and (epoch == 0 or (epoch+1) % 10 == 0 or (epoch+1)==101):
            metadata = [", ".join(self.class_names[l.item()]) if isinstance(self.class_names[l.item()], tuple) else str(self.class_names[l.item()]) for l in labels]
            self.tb_writer.add_embedding(logits, metadata=metadata, tag=f"{prefix}_logits_embeddings/epoch_{epoch}")
        """
        
        if per_class_acc is not None:
            for idx, acc in enumerate(per_class_acc):
                label = self.class_names[idx] if self.class_names else str(idx)
                self.data[f"acc_{label}"].append(acc.item())
                self.tb_writer.add_scalar(tag(f"PerClassAccuracy/{label}"), acc, epoch)

        self.tb_writer.add_scalar(tag('Accuracy/Top1'), top1, epoch)
        self.tb_writer.add_scalar(tag('Accuracy/Top5'), top5, epoch)
        self.tb_writer.add_scalar(tag('Loss'), loss, epoch)
        self.tb_writer.add_scalar(tag('LearningRate'), lr, epoch)
        self.tb_writer.add_scalar(tag('EvalTime'), eval_time, epoch)

        self.log_tsne(features, labels, epoch, prefix, logits=False)
        self.log_tsne(logits, labels, epoch, prefix, logits=True)
        self._save_yaml(epoch, prefix)
        self._save_csv()
        self.summarize_all_epochs(prefix)

    def _save_yaml(self, epoch, prefix=""):
        tag = lambda name: f"{prefix}_{name}" if prefix else name
        yaml_path = os.path.join(self.save_dir, f"eval_{tag(f'epoch_{epoch}')}.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump({k: v[-1] for k, v in self.data.items() if tag("") in k}, f)

    def _save_csv(self):
        df = pd.DataFrame(self.data)
        df.to_csv(os.path.join(self.save_dir, "linear_eval_metrics.csv"), index=False)
        
    def log_tsne(self, features, labels, epoch, prefix="train", logits=False):
        print(f"[t-SNE] Computing 2D projection... ({'logits' if logits else 'features'})", flush=True)
        tsne = TSNE(n_components=2, init='pca', random_state=42)
        reduced = tsne.fit_transform(features.detach().cpu().numpy() if logits else features.cpu().numpy())

        plt.figure(figsize=(12, 6))
        ax = plt.gca()

        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels.cpu(), cmap='tab10', alpha=0.6)
        
        if self.class_names:
            legend = ax.legend(
                handles=scatter.legend_elements()[0],
                labels=self.class_names,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.
            )

        plt.title(f"t-SNE projection (Epoch {epoch})")
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on right for legend

        tsne_path = os.path.join(
            self.save_dir,
            f"{prefix}_tsne_{'logits' if logits else 'features'}_epoch_{epoch}.png" if prefix else f"tsne_epoch_{epoch}.png"
        )
        plt.savefig(tsne_path, bbox_inches='tight')
        plt.close()
        print(f"[t-SNE] Saved to {tsne_path}", flush=True)


    def log_confusion_matrix(self, y_true=None, y_pred=None, cm_tensor=None, epoch=0, prefix: str='train'):
        tag = lambda name: f"{prefix}_{name}" if prefix else name
        print("[Confusion Matrix] Generating plot...", flush=True)
        if cm_tensor is not None:
            cm = cm_tensor.cpu().numpy()
        elif y_true is not None and y_pred is not None:
            cm = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names) if self.class_names else max(y_true)+1))
        else:
            raise ValueError("Provide either y_true and y_pred or cm_tensor")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        fig, ax = plt.subplots(figsize=(16, 16))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        plt.title(f"Confusion Matrix (Epoch {epoch})")
        #plt.tight_layout()
        cm_path = os.path.join(self.save_dir, tag(f"confusion_matrix_epoch_{epoch}.png"))
        plt.savefig(cm_path)
        plt.close()
        print(f"[Confusion Matrix] Saved to {cm_path}", flush=True)

    def summarize_all_epochs(self, prefix: str='train'):
        csv_path = os.path.join(self.save_dir, "linear_eval_metrics.csv")
        if os.path.exists(csv_path):
            full_df = pd.read_csv(csv_path)
            df = full_df[full_df['method']==prefix]
            df['top1'] = df['top1'].astype(float)
            df['top5'] = df['top5'].astype(float)
            summary_path = os.path.join(self.save_dir, "summary_report.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Linear Evaluation Summary Report ({prefix})\n")
                f.write("============================================\n")
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
            acc_plot_path = os.path.join(self.save_dir, f"{prefix}_accuracy_over_epochs.png")
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
                loss_plot_path = os.path.join(self.save_dir, f"{prefix}_loss_over_epochs.png")
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
                heatmap_path = os.path.join(self.save_dir, f"{prefix}_per_class_accuracy_heatmap.png")
                plt.savefig(heatmap_path)
                plt.close()
                print(f"[Summary] Per-class accuracy heatmap saved to {heatmap_path}", flush=True)

        else:
            print("[Summary] No metrics CSV found to summarize.", flush=True)