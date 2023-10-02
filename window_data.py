import numpy as np
import torch

def _get_labelled_window(windowed_data, horizon: int):
    """Create labels for windowed dataset
    Input: [0, 1, 2, 3, 4, 5] and horizon=1
    Output: ([0, 1, 2, 3, 4], [5])

    Parameters
    ----------
    data : array
        time series to be labelled
    horizon : int
        the horizon to predict
    """
    return windowed_data[:, :-horizon], windowed_data[:, -horizon:]

def timeseries_dataset_from_array(data, window_size, horizon, stride=1, label_indices: list=None):
    # Adapted from https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    # and https://www.mlq.ai/time-series-tensorflow-windows-horizons/
    """Creates windows and labels
    Input data must have format [time, ..., features], where ... can be e.g. lat and lon.
    Outputs have format [batch, time, ..., feature]. Number of features can vary depending on label_indices.

    Returns
    -------
    tuple(array, array)
        Windows and labels with shapes [batch, time, ..., feature]
    """
    # Create window of the specific size. Add horizon for to include labels.
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)
    # Create the timesteps. subtract window_size and horizon to get equal length windows and subtract 1 to account for
    # 0-indexing
    time_step = np.expand_dims(
        np.arange(data.shape[0] - (window_size + horizon - 1), step=stride), axis=0
    ).T

    # Create the window indexex
    window_indexes = window_step + time_step

    # Get the windows from the data in [batch, time, ..., features]
    windowed_data = data[window_indexes]

    # Split windows and labels
    windows, labels = _get_labelled_window(windowed_data, horizon)

    # Select only the labels we need
    if label_indices is not None:
        assert (
            type(label_indices) == list
        ), f"label_indices needs to be list[int], but is {type(label_indices)}"
        labels = labels[..., label_indices]

    return windows, labels


data = np.random.normal(size=(100, 10)) # 100 timesteps with 10 inputs
features, labels = timeseries_dataset_from_array(data, 10, 1) # get inputs and targets in batches with 10 timestep inputs and predict the next timestep t+1
dataset = torch.utils.data.TensorDataset(torch.tensor(features), torch.tensor(labels)) # insert into tensor dataset
data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True) # insert dataset into data loader

input_data, target_data = next(iter(data_loader))