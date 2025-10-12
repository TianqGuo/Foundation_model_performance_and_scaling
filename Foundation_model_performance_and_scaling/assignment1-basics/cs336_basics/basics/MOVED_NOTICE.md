# Structure Change Notice

## New Organization

The experiment tracking infrastructure has been reorganized for better modularity:

### Before
```
assignment1-basics/
└── assignment_experiments/
    ├── experiment_tracker.py
    ├── experiment_logger.py
    ├── run_experiment.py
    ├── analyze_experiments.py
    └── configs/
```

### After
```
assignment1-basics/
└── cs336_basics/
    ├── basics/                      # Infrastructure (tracking, logging, viz)
    │   ├── experiment_tracker.py
    │   ├── experiment_logger.py
    │   ├── run_experiment.py
    │   ├── analyze_experiments.py
    │   ├── verify_setup.py
    │   ├── example_usage.py
    │   ├── configs/
    │   ├── README.md
    │   ├── QUICK_START.md
    │   └── EXPERIMENT_LOG.md
    │
    └── assignment_experiments/      # Actual experiments
        ├── __init__.py
        └── README.md
```

## Updated Commands

All commands now use the `cs336_basics.basics` module:

### Run Experiment
```bash
# Old
uv run python -m assignment_experiments.run_experiment --name exp1 --config ...

# New
uv run python -m cs336_basics.basics.run_experiment --name exp1 --config ...
```

### Analyze Experiments
```bash
# Old
uv run python -m assignment_experiments.analyze_experiments list

# New
uv run python -m cs336_basics.basics.analyze_experiments list
```

### Verify Setup
```bash
# Old
uv run python assignment_experiments/verify_setup.py

# New
uv run python cs336_basics/basics/verify_setup.py
```

### Example Usage
```bash
# Old
uv run python assignment_experiments/example_usage.py

# New
uv run python cs336_basics/basics/example_usage.py
```

## Import Paths

When using the infrastructure programmatically:

```python
# Old
from assignment_experiments.experiment_tracker import ExperimentTracker
from assignment_experiments.experiment_logger import ExperimentLogger

# New
from cs336_basics.basics.experiment_tracker import ExperimentTracker
from cs336_basics.basics.experiment_logger import ExperimentLogger

# Or use the package-level imports
from cs336_basics.basics import ExperimentTracker, ExperimentLogger
```

## Configuration Files

Configs are now at:
- `cs336_basics/basics/configs/tinystories_17m_baseline.json`

## Documentation

All documentation has been updated:
- `cs336_basics/basics/README.md` - Full guide
- `cs336_basics/basics/QUICK_START.md` - Quick start
- `cs336_basics/basics/EXPERIMENT_LOG.md` - Experiment log template

## Why the Change?

This organization provides better separation:

1. **`cs336_basics/basics/`**: Reusable infrastructure
   - Experiment tracking
   - Logging and visualization
   - Analysis tools
   - Can be reused across assignments

2. **`cs336_basics/assignment_experiments/`**: Assignment-specific experiments
   - Baseline experiments
   - Ablation studies
   - Specific to Assignment 1, Section 7
   - Clean separation of concerns

This makes the codebase more maintainable and the infrastructure more reusable.
