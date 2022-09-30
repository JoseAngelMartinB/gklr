import numpy as np
from sklearn.metrics import pairwise

# Define constants
DEFAULT_DTYPE = np.float64
VALID_PMLE_METHODS = [None, "Tikhonov"]
DEFAULT_NYSTROM_COMPRESSION = 0.1

# Create a dictionary relating the kernel type parameter to the class from sklearn.gaussian_process.kernels that
# implements that kernel.
kernel_type_to_class = {"rbf": pairwise.rbf_kernel,
                        }

valid_kernel_list = kernel_type_to_class.keys()


def convert_size_bytes_to_human_readable(size_in_bytes):
   """ Convert the size from bytes to other units like KB, MB or GB"""
   if size_in_bytes < 1024:
       return (size_in_bytes, "Bytes")
   elif size_in_bytes < (1024*1024):
       return (np.round(size_in_bytes/1024, 2), "KB")
   elif size_in_bytes < (1024*1024*1024):
       return (np.round(size_in_bytes/(1024*1024), 2), "MB")
   else:
       return (np.round(size_in_bytes/(1024*1024*1024), 2), "GB")

def elapsed_time_to_str(elapsed_time_sec):
    if elapsed_time_sec > 60:
        return("{time:.2f} minutes".format(time=elapsed_time_sec/60))
    else:
        return("{time:.2f} seconds".format(time=elapsed_time_sec))
