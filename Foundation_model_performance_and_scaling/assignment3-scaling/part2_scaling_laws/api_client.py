"""
Training API Client

Client for querying the CS336 scaling laws training API.
Provides methods to query training losses, check FLOPs usage, and retrieve run history.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


class TrainingAPIClient:
    """Client for the CS336 training API."""

    BASE_URL = "http://hyperturing.stanford.edu:8000"

    # Valid hyperparameter ranges from the API spec
    VALID_RANGES = {
        'd_model': (64, 1024),
        'num_layers': (2, 24),
        'num_heads': (2, 16),
        'batch_size': {128, 256},
        'learning_rate': (1e-4, 1e-3),
        'train_flops': {
            int(1e13), int(3e13), int(6e13),
            int(1e14), int(3e14), int(6e14),
            int(1e15), int(3e15), int(6e15),
            int(1e16), int(3e16), int(6e16),
            int(1e17), int(3e17), int(6e17),
            int(1e18),
        }
    }

    def __init__(self, api_key: str, cache_file: Optional[str] = None):
        """
        Initialize API client.

        Args:
            api_key: Your SSH public key (without newlines)
            cache_file: Optional path to cache file for storing run history
        """
        self.api_key = api_key
        self.cache_file = Path(cache_file) if cache_file else None
        self._run_cache = {}

        if self.cache_file and self.cache_file.exists():
            self._load_cache()

    def _load_cache(self):
        """Load cached runs from file."""
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                self._run_cache = {self._config_key(run): run for run in data.get('runs', [])}
            print(f"Loaded {len(self._run_cache)} cached runs from {self.cache_file}")
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            self._run_cache = {}

    def _save_cache(self):
        """Save run cache to file."""
        if not self.cache_file:
            return

        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'runs': list(self._run_cache.values()),
                    'total_cached': len(self._run_cache)
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    @staticmethod
    def _config_key(config: Dict) -> str:
        """Create a unique key for a configuration."""
        key_params = ['d_model', 'num_layers', 'num_heads', 'batch_size',
                      'learning_rate', 'train_flops']
        return json.dumps({k: config[k] for k in key_params if k in config}, sort_keys=True)

    def validate_config(self, config: Dict) -> tuple[bool, Optional[str]]:
        """
        Validate configuration parameters.

        Returns:
            (is_valid, error_message)
        """
        # Check d_model
        d_model = config.get('d_model')
        if not (self.VALID_RANGES['d_model'][0] <= d_model <= self.VALID_RANGES['d_model'][1]):
            return False, f"d_model must be in [{self.VALID_RANGES['d_model'][0]}, {self.VALID_RANGES['d_model'][1]}]"

        # Check num_layers
        num_layers = config.get('num_layers')
        if not (self.VALID_RANGES['num_layers'][0] <= num_layers <= self.VALID_RANGES['num_layers'][1]):
            return False, f"num_layers must be in [{self.VALID_RANGES['num_layers'][0]}, {self.VALID_RANGES['num_layers'][1]}]"

        # Check num_heads
        num_heads = config.get('num_heads')
        if not (self.VALID_RANGES['num_heads'][0] <= num_heads <= self.VALID_RANGES['num_heads'][1]):
            return False, f"num_heads must be in [{self.VALID_RANGES['num_heads'][0]}, {self.VALID_RANGES['num_heads'][1]}]"

        # Check batch_size
        batch_size = config.get('batch_size')
        if batch_size not in self.VALID_RANGES['batch_size']:
            return False, f"batch_size must be one of {self.VALID_RANGES['batch_size']}"

        # Check learning_rate
        lr = config.get('learning_rate')
        if not (self.VALID_RANGES['learning_rate'][0] <= lr <= self.VALID_RANGES['learning_rate'][1]):
            return False, f"learning_rate must be in [{self.VALID_RANGES['learning_rate'][0]}, {self.VALID_RANGES['learning_rate'][1]}]"

        # Check train_flops
        train_flops = config.get('train_flops')
        if train_flops not in self.VALID_RANGES['train_flops']:
            return False, f"train_flops must be one of {sorted(self.VALID_RANGES['train_flops'])}"

        # Check if num_heads divides d_model
        if d_model % num_heads != 0:
            return False, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        return True, None

    def query_loss(self, config: Dict, use_cache: bool = True) -> Dict:
        """
        Query training loss for a configuration.

        Args:
            config: Dictionary with keys: d_model, num_layers, num_heads,
                    batch_size, learning_rate, train_flops
            use_cache: Whether to use cached results if available

        Returns:
            Dictionary with 'loss' and 'total_flops_used' keys
        """
        # Check cache first
        config_key = self._config_key(config)
        if use_cache and config_key in self._run_cache:
            cached = self._run_cache[config_key]
            print(f"  [CACHED] Using cached result for config")
            return {
                'loss': cached['loss'],
                'total_flops_used': cached.get('total_flops_used', 0),
                'cached': True
            }

        # Validate config
        is_valid, error = self.validate_config(config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error}")

        # Prepare request
        params = {
            'd_model': config['d_model'],
            'num_layers': config['num_layers'],
            'num_heads': config['num_heads'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'train_flops': int(config['train_flops']),
            'api_key': self.api_key
        }

        # Make request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.BASE_URL}/loss", params=params, timeout=30)

                if response.status_code == 200:
                    result = response.json()

                    # Cache the result
                    cached_run = {**config, 'loss': result['loss'],
                                  'total_flops_used': result.get('total_flops_used', 0)}
                    self._run_cache[config_key] = cached_run
                    self._save_cache()

                    return result
                else:
                    error_msg = response.json().get('message', 'Unknown error')
                    raise RuntimeError(f"API error: {error_msg}")

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"  Request timeout, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(2)
                else:
                    raise RuntimeError("Request timed out after multiple retries")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Request failed: {e}")

        raise RuntimeError("Failed to query API after multiple retries")

    def get_total_flops_used(self) -> float:
        """Get total FLOPs used by this API key."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/total_flops_used",
                params={'api_key': self.api_key},
                timeout=10
            )

            if response.status_code == 200:
                return float(response.json())
            else:
                error_msg = response.json().get('message', 'Unknown error')
                raise RuntimeError(f"API error: {error_msg}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get FLOPs usage: {e}")

    def get_previous_runs(self) -> List[Dict]:
        """Get all previous runs from the API."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/previous_runs",
                params={'api_key': self.api_key},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return data.get('previous_runs', [])
            else:
                error_msg = response.json().get('message', 'Unknown error')
                raise RuntimeError(f"API error: {error_msg}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get previous runs: {e}")

    def sync_cache_with_api(self):
        """Sync local cache with API's previous runs."""
        print("Syncing cache with API...")
        try:
            previous_runs = self.get_previous_runs()

            for run in previous_runs:
                config_key = self._config_key(run)
                self._run_cache[config_key] = run

            self._save_cache()
            print(f"Synced {len(previous_runs)} runs from API")

        except Exception as e:
            print(f"Warning: Could not sync with API: {e}")

    @staticmethod
    def estimate_model_parameters(d_model: int, num_layers: int) -> int:
        """
        Estimate non-embedding parameters for a model configuration.

        Formula: 12 * num_layers * d_model^2
        """
        return 12 * num_layers * (d_model ** 2)

    @staticmethod
    def estimate_tokens_from_flops(train_flops: float, num_params: int) -> float:
        """
        Estimate number of tokens from FLOPs and parameters.

        Formula: D = C / (6N)
        """
        return train_flops / (6 * num_params)


def load_api_key(key_file: str = "../api_key.txt") -> str:
    """Load API key from file."""
    key_path = Path(key_file)
    if not key_path.exists():
        raise FileNotFoundError(
            f"API key file not found: {key_file}\n"
            f"Please create a file with your SSH public key (no newlines)"
        )

    with open(key_path, 'r') as f:
        api_key = f.read().strip().replace('\n', '').replace('\r', '')

    return api_key