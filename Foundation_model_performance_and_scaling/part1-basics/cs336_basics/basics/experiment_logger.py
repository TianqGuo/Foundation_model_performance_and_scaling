#!/usr/bin/env python3
"""
Experiment Logger - Integrates with existing training infrastructure

This module provides a wrapper around the ExperimentTracker that integrates
seamlessly with the existing training loop in cs336_basics/train.py
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from .experiment_tracker import ExperimentTracker, ExperimentConfig


class ExperimentLogger:
    """
    High-level experiment logger that wraps ExperimentTracker and adds visualization.

    This class integrates with the existing training loop and provides:
    - Automatic metric tracking
    - Loss curve plotting
    - Integration with Python logging
    - Experiment comparison utilities
    """

    def __init__(
        self,
        experiment_name: str,
        config: ExperimentConfig,
        output_dir: Path,
        python_logger: Optional[logging.Logger] = None,
        resume: bool = False,
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration
            output_dir: Directory to save experiment artifacts
            python_logger: Optional Python logger for integration
            resume: Whether resuming from a previous run
        """
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            config=config,
            output_dir=output_dir,
            resume=resume,
        )

        self.logger = python_logger or logging.getLogger(__name__)
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)

        # Track best validation loss
        self.best_val_loss = float('inf')
        self.best_val_step = 0

    def log_training_step(
        self,
        step: int,
        train_loss: float,
        learning_rate: float,
        tokens_per_sec: Optional[float] = None,
    ) -> None:
        """
        Log a training step.

        Args:
            step: Current gradient step
            train_loss: Training loss value
            learning_rate: Current learning rate
            tokens_per_sec: Training throughput
        """
        self.tracker.log_step(
            step=step,
            train_loss=train_loss,
            learning_rate=learning_rate,
            tokens_per_sec=tokens_per_sec,
        )

    def log_validation_step(
        self,
        step: int,
        val_loss: float,
        val_perplexity: float,
    ) -> None:
        """
        Log a validation step.

        Args:
            step: Current gradient step
            val_loss: Validation loss value
            val_perplexity: Validation perplexity
        """
        # Log to tracker
        self.tracker.log_step(
            step=step,
            val_loss=val_loss,
            val_perplexity=val_perplexity,
        )

        # Track best validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_step = step
            self.logger.info(
                f"New best validation loss: {val_loss:.4f} at step {step}"
            )

    def plot_loss_curves(self, save_path: Optional[Path] = None) -> None:
        """
        Plot training and validation loss curves.

        Args:
            save_path: Optional path to save the plot (default: output_dir/loss_curves.png)
        """
        if save_path is None:
            save_path = self.output_dir / "loss_curves.png"

        train_data = self.tracker.get_train_losses()
        val_data = self.tracker.get_val_losses()

        if not train_data:
            self.logger.warning("No training data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot by gradient steps
        train_steps = [d[0] for d in train_data]
        train_losses = [d[2] for d in train_data]
        ax1.plot(train_steps, train_losses, label='Train Loss', alpha=0.7)

        if val_data:
            val_steps = [d[0] for d in val_data]
            val_losses = [d[2] for d in val_data]
            ax1.plot(val_steps, val_losses, label='Val Loss', marker='o', markersize=3)

        ax1.set_xlabel('Gradient Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{self.experiment_name} - Loss vs Steps')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot by wallclock time
        train_times = [d[1] / 60 for d in train_data]  # Convert to minutes
        ax2.plot(train_times, train_losses, label='Train Loss', alpha=0.7)

        if val_data:
            val_times = [d[1] / 60 for d in val_data]
            ax2.plot(val_times, val_losses, label='Val Loss', marker='o', markersize=3)

        ax2.set_xlabel('Wallclock Time (minutes)')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'{self.experiment_name} - Loss vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Loss curves saved to {save_path}")

    def plot_learning_rate_schedule(self, save_path: Optional[Path] = None) -> None:
        """
        Plot learning rate schedule.

        Args:
            save_path: Optional path to save the plot
        """
        if save_path is None:
            save_path = self.output_dir / "lr_schedule.png"

        # Extract learning rates
        data = [(m.step, m.learning_rate) for m in self.tracker.metrics if m.learning_rate is not None]

        if not data:
            self.logger.warning("No learning rate data to plot")
            return

        steps, lrs = zip(*data)

        plt.figure(figsize=(10, 5))
        plt.plot(steps, lrs)
        plt.xlabel('Gradient Steps')
        plt.ylabel('Learning Rate')
        plt.title(f'{self.experiment_name} - Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Learning rate schedule saved to {save_path}")

    def finalize(self) -> None:
        """Finalize experiment and save all artifacts."""
        # Plot curves
        try:
            self.plot_loss_curves()
            self.plot_learning_rate_schedule()
        except Exception as e:
            self.logger.warning(f"Failed to generate plots: {e}")

        # Save tracker data
        self.tracker.finalize()

        # Log summary
        self.logger.info(f"Experiment '{self.experiment_name}' finalized")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f} at step {self.best_val_step}")


def compare_experiments(
    experiment_dirs: list[Path],
    output_path: Path,
    experiment_names: Optional[list[str]] = None,
) -> None:
    """
    Compare multiple experiments by plotting their loss curves together.

    Args:
        experiment_dirs: List of experiment output directories
        output_path: Path to save comparison plot
        experiment_names: Optional custom names for experiments
    """
    if experiment_names is None:
        experiment_names = [d.name for d in experiment_dirs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_dirs)))

    for i, (exp_dir, name) in enumerate(zip(experiment_dirs, experiment_names)):
        metrics_csv = exp_dir / "metrics.csv"
        if not metrics_csv.exists():
            print(f"Warning: {metrics_csv} not found")
            continue

        import csv
        steps, times, train_losses, val_losses = [], [], [], []

        with open(metrics_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['train_loss']:
                    steps.append(int(row['step']))
                    times.append(float(row['elapsed_time']))
                    train_losses.append(float(row['train_loss']))

                if row['val_loss']:
                    val_losses.append((int(row['step']), float(row['elapsed_time']), float(row['val_loss'])))

        # Plot by steps
        ax1.plot(steps, train_losses, label=f'{name} (train)', color=colors[i], alpha=0.7)
        if val_losses:
            val_steps, val_times, val_loss_vals = zip(*val_losses)
            ax1.plot(val_steps, val_loss_vals, label=f'{name} (val)',
                    color=colors[i], marker='o', markersize=3, linestyle='--')

        # Plot by time
        ax2.plot(times, train_losses, label=f'{name} (train)', color=colors[i], alpha=0.7)
        if val_losses:
            ax2.plot(val_times, val_loss_vals, label=f'{name} (val)',
                    color=colors[i], marker='o', markersize=3, linestyle='--')

    ax1.set_xlabel('Gradient Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Experiment Comparison - Loss vs Steps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Wallclock Time (minutes)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Experiment Comparison - Loss vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved to {output_path}")
