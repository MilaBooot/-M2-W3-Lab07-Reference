"""
src/data/utils.py
-----------------
Shared utility functions used across all pipeline scripts.
"""

import yaml
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _PROJECT_ROOT  # expose for use in notebooks and scripts

def load_config(path="configs/config.yaml"):
    """Load YAML configuration file from the project root."""
    with open(_PROJECT_ROOT / path, "r") as f:
        return yaml.safe_load(f)


def ordinal_days(val):
    """
    Convert ordinal day-range strings to numeric midpoints.

    The dataset stores days-since-policy as string ranges like '1 to 7'
    or 'more than 30'. These are meaningless to a numeric model as strings.
    We map each range to its approximate midpoint so the model can reason
    about proximity in time.

    Mapping:
        'none'          ->  0   (no gap — policy and incident same day)
        '1 to 7'        ->  4   (midpoint of 1–7)
        '8 to 15'       -> 11   (midpoint of 8–15)
        '15 to 30'      -> 22   (midpoint of 15–30)
        'more than 30'  -> 35   (conservative estimate for 30+ days)

    Why this matters:
        A claim filed 1–7 days after a policy starts is a strong fraud signal.
        Without this conversion, the model cannot detect that pattern.
    """
    mapping = {
        "none":         0,
        "1 to 7":       4,
        "8 to 15":      11,
        "15 to 30":     22,
        "more than 30": 35,
    }
    return mapping.get(str(val).strip().lower(), 0)


def ordinal_past_claims(val):
    """
    Convert PastNumberOfClaims string categories to numeric midpoints.

        'none'        ->  0
        '1'           ->  1
        '2 to 4'      ->  3
        'more than 4' ->  5
    """
    mapping = {
        "none":         0,
        "1":            1,
        "2 to 4":       3,
        "more than 4":  5,
    }
    return mapping.get(str(val).strip().lower(), 0)


def ordinal_num_supplements(val):
    """
    Convert NumberOfSuppliments to numeric midpoints.

        'none'        ->  0
        '1 to 2'      ->  1
        '3 to 5'      ->  4
        'more than 5' ->  6
    """
    mapping = {
        "none":         0,
        "1 to 2":       1,
        "3 to 5":       4,
        "more than 5":  6,
    }
    return mapping.get(str(val).strip().lower(), 0)
