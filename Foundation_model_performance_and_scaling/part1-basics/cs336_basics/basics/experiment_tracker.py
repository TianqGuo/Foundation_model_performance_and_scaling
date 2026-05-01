#!/usr/bin/env python3
"""
Experiment Tracking Infrastructure for CS336 Assignment 1

This module provides comprehensive experiment tracking with:
- Loss curves (training and validation)
- Gradient steps and wallclock time tracking
- Hyperparameter logging
- CSV and JSON export for analysis
- Integration with existing training infrastructure
"""

import json
import csv
import time
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    # Experiment metadata
    experiment_name: str
    experiment_id: str
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Model hyperparameters
    vocab_size: int = 10000
    context_length: int = 512
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    rope_theta: float = 10000.0

    # Training hyperparameters
    batch_size: int = 32
    max_iterations: int = 10000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0

    # Data configuration
    dataset: str = "TinyStories"
    train_data_path: str = ""
    val_data_path: str = ""

    # System configuration
    device: Optional[str] = None
    dtype: str = "float32"
    seed: int = 42

    # Logging intervals
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 1000

    # Optional sweep metadata (e.g., processed_tokens)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, filepath: Path) -> None:
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: Path) -> 'ExperimentConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class MetricPoint:
    """Single metric measurement point."""
    step: int
    wallclock_time: float
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_perplexity: Optional[float] = None
    learning_rate: Optional[float] = None
    tokens_per_sec: Optional[float] = None


class ExperimentTracker:
    """
    Tracks experiment metrics and provides structured logging.

    Features:
    - Tracks loss curves with gradient steps and wallclock time
    - Exports to CSV and JSON for easy analysis
    - Computes statistics (min, max, mean, final)
    - Generates experiment summaries
    """

    def __init__(
        self,
        experiment_name: str,
        config: ExperimentConfig,
        output_dir: Path,
        resume: bool = False,
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration
            output_dir: Directory to save experiment artifacts
            resume: Whether resuming from a previous run
        """
        self.experiment_name = experiment_name
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracking data
        self.metrics: List[MetricPoint] = []
        self.start_time = time.time()
        self.experiment_start_wallclock = time.time()

        # Save configuration
        if not resume:
            self.config.to_json(self.output_dir / "config.json")

        # Initialize CSV files
        self.metrics_csv_path = self.output_dir / "metrics.csv"
        if not resume:
            self._init_metrics_csv()

        print(f"Experiment tracker initialized: {experiment_name}")
        print(f"Output directory: {self.output_dir}")

    def _init_metrics_csv(self) -> None:
        """Initialize CSV file with headers."""
        headers = [
            'step', 'wallclock_time', 'elapsed_time',
            'train_loss', 'val_loss', 'val_perplexity',
            'learning_rate', 'tokens_per_sec'
        ]
        with open(self.metrics_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def log_step(
        self,
        step: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_perplexity: Optional[float] = None,
        learning_rate: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
    ) -> None:
        """
        Log metrics for a single step.

        Args:
            step: Current gradient step
            train_loss: Training loss value
            val_loss: Validation loss value
            val_perplexity: Validation perplexity
            learning_rate: Current learning rate
            tokens_per_sec: Training throughput
        """
        current_time = time.time()
        wallclock_time = current_time - self.experiment_start_wallclock

        # Create metric point
        metric = MetricPoint(
            step=step,
            wallclock_time=wallclock_time,
            train_loss=train_loss,
            val_loss=val_loss,
            val_perplexity=val_perplexity,
            learning_rate=learning_rate,
            tokens_per_sec=tokens_per_sec,
        )
        self.metrics.append(metric)

        # Append to CSV
        with open(self.metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                wallclock_time,
                wallclock_time / 60,  # elapsed_time in minutes
                train_loss if train_loss is not None else '',
                val_loss if val_loss is not None else '',
                val_perplexity if val_perplexity is not None else '',
                learning_rate if learning_rate is not None else '',
                tokens_per_sec if tokens_per_sec is not None else '',
            ])

    def get_train_losses(self) -> List[tuple]:
        """Get list of (step, wallclock_time, train_loss) tuples."""
        return [
            (m.step, m.wallclock_time, m.train_loss)
            for m in self.metrics
            if m.train_loss is not None
        ]

    def get_val_losses(self) -> List[tuple]:
        """Get list of (step, wallclock_time, val_loss) tuples."""
        return [
            (m.step, m.wallclock_time, m.val_loss)
            for m in self.metrics
            if m.val_loss is not None
        ]

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics for the experiment."""
        train_losses = [m.train_loss for m in self.metrics if m.train_loss is not None]
        val_losses = [m.val_loss for m in self.metrics if m.val_loss is not None]

        stats = {
            'total_steps': len(self.metrics),
            'total_wallclock_time': self.metrics[-1].wallclock_time if self.metrics else 0,
            'total_wallclock_hours': (self.metrics[-1].wallclock_time / 3600) if self.metrics else 0,
        }

        if train_losses:
            stats['train_loss'] = {
                'min': float(np.min(train_losses)),
                'max': float(np.max(train_losses)),
                'mean': float(np.mean(train_losses)),
                'final': float(train_losses[-1]),
                'initial': float(train_losses[0]),
            }

        if val_losses:
            stats['val_loss'] = {
                'min': float(np.min(val_losses)),
                'max': float(np.max(val_losses)),
                'mean': float(np.mean(val_losses)),
                'final': float(val_losses[-1]),
                'best': float(np.min(val_losses)),
            }

        return stats

    def save_summary(self) -> None:
        """Save experiment summary to JSON."""
        summary = {
            'experiment_name': self.experiment_name,
            'config': self.config.to_dict(),
            'statistics': self.compute_statistics(),
            'metrics_count': len(self.metrics),
        }

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Experiment summary saved to {summary_path}")

    def save_metrics_json(self) -> None:
        """Save all metrics to JSON for detailed analysis."""
        metrics_data = [
            {
                'step': m.step,
                'wallclock_time': m.wallclock_time,
                'train_loss': m.train_loss,
                'val_loss': m.val_loss,
                'val_perplexity': m.val_perplexity,
                'learning_rate': m.learning_rate,
                'tokens_per_sec': m.tokens_per_sec,
            }
            for m in self.metrics
        ]

        json_path = self.output_dir / "metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        print(f"Metrics JSON saved to {json_path}")

    def finalize(self) -> None:
        """Finalize experiment tracking and save all artifacts."""
        self.save_summary()
        self.save_metrics_json()

        # Print final statistics
        stats = self.compute_statistics()
        print("\n" + "="*80)
        print(f"EXPERIMENT COMPLETE: {self.experiment_name}")
        print("="*80)
        print(f"Total steps: {stats['total_steps']}")
        print(f"Total time: {stats['total_wallclock_hours']:.2f} hours")

        if 'train_loss' in stats:
            print(f"\nTraining Loss:")
            print(f"  Initial: {stats['train_loss']['initial']:.4f}")
            print(f"  Final: {stats['train_loss']['final']:.4f}")
            print(f"  Min: {stats['train_loss']['min']:.4f}")

        if 'val_loss' in stats:
            print(f"\nValidation Loss:")
            print(f"  Final: {stats['val_loss']['final']:.4f}")
            print(f"  Best: {stats['val_loss']['best']:.4f}")

        print(f"\nArtifacts saved to: {self.output_dir}")
        print("="*80 + "\n")


def create_experiment_config(
    experiment_name: str,
    description: str = "",
    **kwargs
) -> ExperimentConfig:
    """
    Helper function to create experiment configuration.

    Args:
        experiment_name: Name of the experiment
        description: Description of what is being tested
        **kwargs: Additional configuration parameters

    Returns:
        ExperimentConfig instance
    """
    import uuid
    experiment_id = str(uuid.uuid4())[:8]

    exp_fields = {f.name for f in fields(ExperimentConfig)}

    normalized_kwargs: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

    for key, value in kwargs.items():
        if key in exp_fields:
            if key == "metadata":
                if not isinstance(value, dict):
                    raise TypeError("ExperimentConfig metadata must be a dict")
                metadata.update(value)
            else:
                normalized_kwargs[key] = value
        else:
            metadata[key] = value

    if metadata:
        normalized_kwargs["metadata"] = {**normalized_kwargs.get("metadata", {}), **metadata}

    return ExperimentConfig(
        experiment_name=experiment_name,
        experiment_id=experiment_id,
        description=description,
        **normalized_kwargs
    )
