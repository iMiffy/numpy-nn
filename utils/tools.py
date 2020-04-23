import numpy as np

def rel_error(x, y):
    if not np.array_equal(np.isnan(x), np.isnan(y)):
        return np.nan
    rel_error = np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y)))
    rel_error = np.sum(np.nan_to_num(rel_error)) / np.sum(~np.isnan(rel_error))

    return rel_error

def warn(*args, **kwargs):
    pass