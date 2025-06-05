from typing import Union

import numpy as np


def crop_to_shape(
    array: np.ndarray, shape: tuple[int, ...], return_slices: bool = False
) -> np.ndarray | tuple[np.ndarray, tuple[slice, ...]]:
    """
    Crop a NumPy array to the specified target shape, centered within the original array.

    The function performs a centered crop of the input array to match the given shape.
    It ensures that the target shape is compatible with the array's dimensions and
    is not larger than the array in any axis.

    Parameters:
        array (np.ndarray): The input array to crop.
        shape (tuple[int, ...]): The desired shape to crop the array to. Must match the number of dimensions of `array`.
        return_slices (bool, optional): If True, also return the slice objects used to crop the array. Defaults to False.

    Returns:
        np.ndarray | tuple[np.ndarray, tuple[slice, ...]]:
            - If `return_slices` is False: the cropped array.
            - If `return_slices` is True: a tuple containing the cropped array and the slice objects.

    Raises:
        ValueError: If `shape` has a different number of dimensions than `array`.
        ValueError: If any dimension in `shape` is larger than the corresponding dimension in `array`.

    Example:
        >>> arr = np.arange(100).reshape((10, 10))
        >>> crop_to_shape(arr, (6, 6)).shape
        (6, 6)
    """
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
) -> np.ndarray | tuple[np.ndarray, tuple[slice, ...]]:
    """
    Crop an array to the minimal bounding box that contains the region of interest (ROI),
    optionally limited to specific axes.

    The function identifies the smallest sub-array that includes all `True` values
    in the `roi_mask` and crops the input `array` accordingly. Optionally, cropping
    can be restricted to specific axes. If no axes are specified, all dimensions are considered.

    Parameters:
        array (np.ndarray): The input array to crop.
        roi_mask (np.ndarray): A boolean array of the same shape as `array` indicating the region of interest.
        axes (int | tuple[int, ...] | None): The axes along which to apply cropping.
            If None, cropping is applied across all axes.
        return_slices (bool, optional): If True, also return the slice objects used to crop the array. Defaults to False.

    Returns:
        np.ndarray | tuple[np.ndarray, tuple[slice, ...]]:
            - If `return_slices` is False: the cropped array.
            - If `return_slices` is True: a tuple containing the cropped array and the slice objects.

    Raises:
        ValueError: If `array` and `roi_mask` do not have the same shape.

    Example:
        >>> arr = np.zeros((5, 5))
        >>> arr[1:4, 2:5] = 1
        >>> mask = arr.astype(bool)
        >>> cropped, slices = crop_roi(arr, mask, axes=(0, 1), return_slices=True)
        >>> cropped.shape
        (3, 3)
    """
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
