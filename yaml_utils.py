"""
YAML Serialization Utilities

This module provides utilities for serializing Python objects (particularly dataclasses)
to YAML format, with special handling for None values, collections, and exclusion flags.
"""

from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Union


def clean_none_values(obj: Any) -> Any:
    """
    Recursively clean None values from data structures and convert tuples to lists.

    Args:
        obj: The object to clean

    Returns:
        The cleaned object with None values removed and tuples converted to lists
    """
    if isinstance(obj, dict):
        return {k: clean_none_values(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, (list, tuple)):
        return [clean_none_values(item) for item in obj]
    return obj


def exclude_from_dict(obj: Any) -> bool:
    """
    Check if an object should be excluded from dictionary conversion.

    Args:
        obj: The object to check

    Returns:
        bool: True if the object should be excluded, False otherwise
    """
    return hasattr(obj, "metadata") and obj.metadata.get("exclude_from_dict", False)


def custom_asdict(obj: Any) -> Union[Dict, List, Any]:
    """
    Recursively converts a dataclass instance to a dictionary, respecting exclusion flags.

    This function handles:
    - Dataclass instances (converted to dictionaries)
    - Lists and tuples (processed recursively)
    - Dictionaries (processed recursively)
    - Other types (returned as-is)

    Args:
        obj: The object to convert

    Returns:
        The converted object as a dictionary or other appropriate type
    """

    def _asdict_inner(obj: Any) -> Any:
        if is_dataclass(obj):
            return {
                f.name: _asdict_inner(getattr(obj, f.name))
                for f in fields(obj)
                if not exclude_from_dict(f)
            }
        elif isinstance(obj, (list, tuple)):
            return [_asdict_inner(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: _asdict_inner(value) for key, value in obj.items()}
        return obj

    return _asdict_inner(obj)


def save_to_yaml(obj: Any, filename: str) -> None:
    """
    Save an object to a YAML file, handling special cases like None values and exclusions.

    This function:
    1. Converts the object to a dictionary (if it's a dataclass)
    2. Cleans None values and converts tuples to lists
    3. Saves the result to a YAML file

    Args:
        obj: The object to save
        filename: Path where to save the YAML file
    """
    import yaml

    with open(filename, "w") as f:
        yaml.dump(
            clean_none_values(custom_asdict(obj)),
            f,
            default_flow_style=False,
            sort_keys=False,
        )
