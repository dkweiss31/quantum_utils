import re
import glob
import os
import h5py
import pathos
import numpy as np
import itertools


def generate_file_path(extension, file_name, path):
    # Ensure the path exists.
    os.makedirs(path, exist_ok=True)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for file_name_ in glob.glob(os.path.join(path, "*")):
        if f"_{file_name}.{extension}" in file_name_:
            numeric_prefix = int(
                re.match(r"(\d+)_", os.path.basename(file_name_)).group(1)
            )
            max_numeric_prefix = max(numeric_prefix, max_numeric_prefix)

    # Generate the file path.
    file_path = os.path.join(
        path, f"{str(max_numeric_prefix + 1).zfill(5)}_{file_name}.{extension}"
    )
    return file_path


def extract_info_from_h5(filepath):
    data_dict = {}
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            data_dict[key] = f[key][()]
        param_dict = dict(f.attrs.items())
    return data_dict, param_dict


def append_to_h5(filepath, data_dict):
    with h5py.File(filepath, "a") as f:
        for key, val in data_dict.items():
            f[key].resize(f[key].shape[0] + 1, axis=0)
            f[key][-1] = val


def write_to_h5_multi(filepath, data_dict, param_dict):
    with h5py.File(filepath, "a") as f:
        for key, val in data_dict.items():
            f.create_dataset(key, data=[val], chunks=True, maxshape=(None, *val.shape))
        for kwarg in param_dict.keys():
            try:
                f.attrs[kwarg] = param_dict[kwarg]
            except TypeError:
                f.attrs[kwarg] = str(param_dict[kwarg])


def write_to_h5(filepath, data_dict, param_dict):
    with h5py.File(filepath, "w") as f:
        for key, val in data_dict.items():
            f.create_dataset(key, data=val)
        for kwarg in param_dict.keys():
            try:
                f.attrs[kwarg] = param_dict[kwarg]
            except TypeError:
                f.attrs[kwarg] = str(param_dict[kwarg])


def update_data_in_h5(filepath, data_dict):
    with h5py.File(filepath, "a") as f:
        for key, val in data_dict.items():
            if key in f:
                del f[key]
            f.create_dataset(key, data=val)


def parallel_map(num_cpus, func, parameters):
    if num_cpus == 1:
        return map(func, parameters)

    with pathos.pools.ProcessPool(nodes=num_cpus) as pool:
        result = pool.map(func, parameters)
    return result


def get_map(num_cpus: int = 1):
    if num_cpus == 1:
        return map
    return pathos.pools.ProcessPool(nodes=num_cpus).map


def param_map(f, parameters, map_fun=map, dtype=object):
    """
    Code due to Peter Groszkowski
    Maps function `f` over the product of the parameters given in `parameters`
    (which is assumed to have a list-of-lists-like structure). The returned data
    is an ndarray with dimensions set by the input data in `parameters`.

    e.g.:

    input_data_a=['1','2','3']
    input_data_b=['a','b']

    def f(a, b):
        return "{}{}".format(a,b)

    data=param_map(f, [input_data_a, input_data_b], map_fun=map)

    print(data.shape) #gives (3,2)
    print(data[0,0]) #gives '1a'


    Parameters
    ----------
    f:  function
        Function that is to be applied to each element of an array
    parameters: iterable of iterables (i.e. list of lists)


    Returns
    -------
    ndarray with f applied to products of input parameters

    """
    dims_list = [len(i) for i in parameters]
    total_dim = np.prod(dims_list)

    # all the possible combinations of input parameters
    parameters_prod = tuple(itertools.product(*parameters))

    # We want to force a 1d numpy array of size total_dim,
    # regardless what 'data' is, even if it's iterable
    # (list, sequence, etc), but by default, the np.array()
    # constructor will try to create new array dimensions
    # from objects that can be indexed (sequences, list, etc).
    data = np.empty(total_dim, dtype=dtype)
    for i, d in enumerate(map_fun(f, parameters_prod)):
        data[i] = d

    return np.reshape(data, dims_list)
