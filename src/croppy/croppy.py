from typing import Union

import numpy as np

def crop_to_shape(
    array: np.ndarray, shape: tuple[int, ...], return_slices:bool=False
) -> np.ndarray|tuple[np.ndarray, tuple[slice,...]]:
    if len(shape) != array.ndim:
        raise ValueError(
            "The shape argument must have the same number of dimensions as the array"
        )
    target_shape = np.array(shape)
    array_shape = np.array(array.shape)

    if np.any(target_shape > array_shape):
        raise ValueError(
            "The target shape must be smaller than, or equal to the provided array shape"
        )

    delta = array_shape - target_shape
    crop_start = delta // 2
    crop_end = array_shape - (delta - crop_start)

    slices = tuple(np.s_[start:end] for start, end in zip(crop_start, crop_end))
    cropped_array = array[slices]
    if return_slices:
        return cropped_array, slices
    else:
        return cropped_array


def crop_roi(
    array: np.ndarray,
    roi_mask: np.ndarray,
    axes: Union[None, int, tuple[int, ...]],
    return_slices: bool = False,
) -> np.ndarray|tuple[np.ndarray, tuple[slice,...]]:
    if array.shape != roi_mask.shape:
        raise ValueError("The array and roi_mask must have the same shape")

    if axes is None:
        axes_indices = np.arange(array.ndim)
    else:
        axes_indices = np.atleast_1d(axes)
        axes_indices = axes_indices % array.ndim

    roi_indices = np.argwhere(roi_mask).T

    slices = _full_array_slices(array)
    for dimension in axes_indices:
        start = roi_indices[dimension].min()
        end = roi_indices[dimension].max() + 1
        slices[dimension] = np.s_[start:end]
    slices = tuple(slices)
    cropped_array = array[slices]
    if return_slices:
        return cropped_array, slices
    else:
        return cropped_array

def _full_array_slices(array: np.ndarray) -> list[slice]:
    return [np.s_[:]] * array.ndim