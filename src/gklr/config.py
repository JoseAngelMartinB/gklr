"""GKLR Config module."""
from typing import Optional, Dict, Any

import os
import sys
import multiprocessing
import numpy as np


from .logger import *

__all__ = ["Config"]


def init_environment_variables(num_cores: Optional[int] = None):
    """Initializes the environment variables."""
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)


class Config:
    def __init__(self):
        """Class object to store config and hyperparameters."""
        from gklr import __version__
        self.info = {
            "python_version": sys.version,
            "GKLR_version": __version__,
            "directory": os.getcwd(),
        }
        self.hyperparameters = {
            "num_cores": multiprocessing.cpu_count(),
            "kernel": "rbf",
            "kernel_params": {"gamma": 1.0},
            "nystrom": False,
            "compression": None,
        }
        init_environment_variables(self.hyperparameters["num_cores"])

    def __str__(self):
        rval = f"\nGKLR hyperparameters:\n---------------\n"
        for key, val in self.hyperparameters.items():
            rval += f" - {key:<24}: {val}\n"
        rval += "\n"
        return rval

    def __getitem__(self, name: str) -> Any:
        if name in self.hyperparameters:
            return self.hyperparameters[name]
        else:
            return None

    def __setitem__(self, name: str, val: Any) -> None:
        if name in self.hyperparameters:
            self.hyperparameters[name] = val
        else:
            raise NameError(f"Hyperparameter {name} is not a valid option.")

    def __call__(self) -> Dict[str, Any]:
        return self.hyperparameters

    def set_hyperparameter(self, key: str, value: Any):
        """Helper method to set the hyperparameters of GKLR.
        
        Args:
            key: The hyperparameter to set.
            value: The value to set the hyperparameter to.
        """
        self.hyperparameters[key] = value
        logger_debug(f"Set hyperparameter {key} = {value}")

    def remove_hyperparameter(self, key: str):
        """Helper method to remove a hyperparameter from GKLR.
        
        Args:
            key: The hyperparameter to remove.
        """
        if key in self.hyperparameters:
            del self.hyperparameters[key]
            logger_debug(f"Removed hyperparameter {key}")
        else:
            msg = f"Hyperparameter {key} not found"
            logger_error(msg)
            raise ValueError(msg)

    def check_values(self):
        """Checks validity of hyperparameter values."""
        assert isinstance(self["num_cores"], (int, np.integer))
        assert isinstance(self["kernel_params"]["gamma"], float)
        assert isinstance(self["nystrom"], bool)
        assert (self["compression"] is None) or (isinstance(self["compression"], float))
        # TODO: Assert kernel
