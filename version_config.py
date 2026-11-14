# version_config.py

"""
Version configuration for the video analytics pipeline.
Specifies required package versions and provides version checking utilities.
"""

import sys
import warnings
from typing import Dict, Optional

import pandas as pd
import plotly
from packaging import version

REQUIRED_VERSIONS = {
    "python": ">=3.12,<3.14",
    "plotly": ">=5.18.0,<6.0.0",
    "pandas": ">=2.2.0,<3.0.0",
}


def check_package_version(
    package_name: str, current_version: str, required_version: str
) -> bool:
    """
    Check if the current package version meets requirements.
    Returns True if version is compatible, False otherwise.
    """
    from packaging import specifiers

    spec = specifiers.SpecifierSet(required_version)
    return version.parse(current_version) in spec


def get_current_versions() -> Dict[str, str]:
    """Get current versions of all dependencies"""
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "plotly": plotly.__version__,
        "pandas": pd.__version__,
    }


def verify_environment(raise_on_error: bool = False) -> Optional[Dict[str, str]]:
    """
    Verify all package versions meet requirements.
    Returns dict of incompatible packages and their versions if any found.
    """
    current_versions = get_current_versions()
    incompatible = {}

    for package, required in REQUIRED_VERSIONS.items():
        if not check_package_version(package, current_versions[package], required):
            incompatible[package] = {
                "current": current_versions[package],
                "required": required,
            }
            warnings.warn(
                f"{package} version {current_versions[package]} is incompatible. "
                f"Required: {required}",
                RuntimeWarning,
            )

    if incompatible and raise_on_error:
        versions_str = "\n".join(
            f"  {pkg}: found {info['current']}, requires {info['required']}"
            for pkg, info in incompatible.items()
        )
        raise RuntimeError(
            f"Incompatible package versions found:\n{versions_str}\n"
            "Please install compatible versions using requirements.txt"
        )

    return incompatible if incompatible else None
