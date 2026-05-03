import os
from typing import Dict, Optional


class Logger:
    """Lightweight logging wrapper supporting W&B and TensorBoard."""

    def __init__(self, use_wandb=False, use_tensorboard=False,
                 wandb_project=None, wandb_run_name=None, output_dir="outputs/"):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if use_wandb:
            import wandb
            wandb.init(project=wandb_project, name=wandb_run_name)

        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=output_dir)

    def log(self, metrics: Dict, step: int):
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

        if self.use_tensorboard:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)

        # Always print
        metrics_str = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in metrics.items())
        print(f"step={step}  {metrics_str}")

    def finish(self):
        if self.use_wandb:
            import wandb
            wandb.finish()
        if self.use_tensorboard:
            self.writer.close()
