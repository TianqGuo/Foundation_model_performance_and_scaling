#!/usr/bin/env python3
"""
Test API Connection

Verifies that the API is accessible and the API key works.
"""

import argparse
import sys
from pathlib import Path

import requests

from api_client import TrainingAPIClient, load_api_key


def test_api_connection(api_key_file: str = "../api_key.txt"):
    """Test API connection and key."""

    print("=" * 80)
    print("API CONNECTION TEST")
    print("=" * 80)

    # Test 1: Check API documentation endpoint
    print("\n1. Testing API documentation endpoint...")
    try:
        response = requests.get("http://hyperturing.stanford.edu:8000/docs", timeout=5)
        if response.status_code == 200:
            print("   ✓ API documentation accessible")
        else:
            print(f"   ✗ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print("   ✗ Connection timeout - are you on Stanford VPN?")
        return False
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Connection failed: {e}")
        return False

    # Test 2: Load API key
    print("\n2. Loading API key...")
    try:
        api_key = load_api_key(api_key_file)
        print(f"   ✓ API key loaded from {api_key_file}")
        print(f"   Key length: {len(api_key)} characters")
    except FileNotFoundError as e:
        print(f"   ✗ {e}")
        print("\n   Please create the API key file:")
        print(f"     echo 'your-ssh-public-key' > {api_key_file}")
        return False

    # Test 3: Initialize client
    print("\n3. Initializing API client...")
    try:
        client = TrainingAPIClient(api_key)
        print("   ✓ Client initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize client: {e}")
        return False

    # Test 4: Check total FLOPs used
    print("\n4. Checking current FLOPs usage...")
    try:
        total_flops = client.get_total_flops_used()
        print(f"   ✓ Total FLOPs used: {total_flops:.2e}")

        budget = 2e18
        remaining = budget - total_flops
        utilization = (total_flops / budget) * 100

        print(f"   Budget: {budget:.2e} FLOPs")
        print(f"   Remaining: {remaining:.2e} FLOPs ({100-utilization:.1f}%)")
        print(f"   Used: {utilization:.1f}%")

        if total_flops >= budget:
            print("   ⚠ WARNING: Budget limit reached!")
    except Exception as e:
        print(f"   ✗ Failed to check FLOPs usage: {e}")
        print("   This might indicate an invalid API key")
        return False

    # Test 5: Validate a sample configuration
    print("\n5. Validating sample configuration...")
    sample_config = {
        'd_model': 512,
        'num_layers': 12,
        'num_heads': 8,
        'batch_size': 256,
        'learning_rate': 3e-4,
        'train_flops': int(1e15)
    }

    is_valid, error = client.validate_config(sample_config)
    if is_valid:
        print("   ✓ Sample configuration is valid")
    else:
        print(f"   ✗ Sample configuration invalid: {error}")
        return False

    # Test 6: Try querying previous runs
    print("\n6. Retrieving previous runs...")
    try:
        previous_runs = client.get_previous_runs()
        print(f"   ✓ Retrieved {len(previous_runs)} previous runs")

        if len(previous_runs) > 0:
            print("\n   Recent runs:")
            for i, run in enumerate(previous_runs[:3], 1):
                print(f"     {i}. d_model={run['d_model']}, "
                      f"num_layers={run['num_layers']}, "
                      f"loss={run['loss']:.4f}")
            if len(previous_runs) > 3:
                print(f"     ... and {len(previous_runs) - 3} more")
    except Exception as e:
        print(f"   ✗ Failed to retrieve previous runs: {e}")
        # This is not a critical failure
        pass

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    print("\nYou're ready to run experiments!")
    print("Next steps:")
    print("  1. Preview strategy: ./run_part2.sh --dry-run")
    print("  2. Run experiments: ./run_part2.sh")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(description='Test API connection')
    parser.add_argument(
        '--api-key-file',
        type=str,
        default='../api_key.txt',
        help='Path to API key file'
    )

    args = parser.parse_args()

    success = test_api_connection(args.api_key_file)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()