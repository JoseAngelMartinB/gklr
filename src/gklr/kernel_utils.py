import numpy as np
from sklearn.metrics import pairwise

# Define constants
DEFAULT_DTYPE = np.float64
VALID_PMLE_METHODS = [None, "Tikhonov"]
DEFAULT_NYSTROM_COMPRESSION = 0.1
SCIPY_OPTIMIZATION_METHODS = ["Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "COBYLA", "SLSQP", "trust-constr", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"]
CUSTOM_OPTIMIZATION_METHODS = ["SGD", "momentumSGD", "adam"]

# Create a dictionary relating the kernel type parameter to the class from sklearn.gaussian_process.kernels that
# implements that kernel.
kernel_type_to_class = {"rbf": pairwise.rbf_kernel,
                        }

valid_kernel_list = kernel_type_to_class.keys()
valid_kernel_params = ["gamma"]

def convert_size_bytes_to_human_readable(size_in_bytes):
    """ Convert the size from bytes to other units like KB, MB or GB.
    
    Args:
        size_in_bytes: Size in bytes.
        
    Returns:
        A string with the size in bytes, KB, MB or GB."""
    if size_in_bytes < 1024:
        return (size_in_bytes, "Bytes")
    elif size_in_bytes < (1024*1024):
        return (np.round(size_in_bytes/1024, 2), "KB")
    elif size_in_bytes < (1024*1024*1024):
        return (np.round(size_in_bytes/(1024*1024), 2), "MB")
    else:
        return (np.round(size_in_bytes/(1024*1024*1024), 2), "GB")

def elapsed_time_to_str(elapsed_time_sec: float) -> str:
    """Convert the elapsed time in seconds to a string with the appropriate units.
    
    Args:
        elapsed_time_sec: Elapsed time in seconds

    Returns:
        A string with the elapsed time in seconds or minutes.
    """
    if elapsed_time_sec > 60:
        return f"{elapsed_time_sec/60:.2f} minutes"
    else:
        return f"{elapsed_time_sec:.2f} seconds"
