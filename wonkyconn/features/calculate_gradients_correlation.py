import glob
from pathlib import Path
from typing import Iterable, List, Tuple

import nibabel as nib
import numpy as np
from brainspace.gradient import GradientMaps  # type: ignore[import-not-found]
from nilearn import image  # type: ignore[import-not-found]
from nilearn.maskers import NiftiLabelsMasker  # type: ignore[import-not-found]
from scipy import stats

from ..base import ConnectivityMatrix


def remove_nan_from_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    col_mask = ~np.all(np.isnan(matrix), axis=0)

    kept_idx = np.where(col_mask)[0]
    conn_clean = matrix[np.ix_(kept_idx, kept_idx)]

    return conn_clean, kept_idx


def remove_nan_roi_atlas(atlas: nib.Nifti1Image, kept_idx: np.ndarray) -> nib.Nifti1Image:
    """
    Remove ROIs from an atlas that are not present in a connectivity matrix.

    Parameters
    ----------
    atlas_path : str
        Path to the original NIfTI atlas.
    conn_matrix : np.ndarray
        Square connectivity matrix corresponding to the atlas.
    save_path : str, optional
        Path to save the masked atlas. If None, the image is returned but not saved.

    Returns
    -------
    nib.Nifti1Image
        New atlas image with only ROIs present in conn_matrix.
    """

    # Load atlas
    atlas_data = atlas.get_fdata()

    roi_labels = sorted([int(x) for x in np.unique(atlas_data) if x != 0])

    # Now remove labels that were removed from the conn matrix
    kept_labels = [roi_labels[i] for i in kept_idx]

    # Make a mask image that keeps only kept_labels
    keep_mask = np.isin(atlas_data, kept_labels)
    kept_atlas_data = atlas_data.copy()
    kept_atlas_data[~keep_mask] = 0
    return nib.Nifti1Image(kept_atlas_data, atlas.affine, atlas.header), kept_labels


def overlapping_atlas_with_mask(subject_atlas: nib.Nifti1Image, group_mask: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Create a new atlas that only contains the regions that overlap with the group gradient mask.

    Parameters
    ----------
    subject_atlas : nib.Nifti1Image
        The subject's atlas in NIfTI format.
    group_mask : nib.Nifti1Image
        The group gradient mask in NIfTI format.

    Returns
    -------
    nib.Nifti1Image
        A new atlas that only contains the regions that overlap with the group gradient mask.
    """

    mask_gradient_resampled = image.resample_to_img(
        group_mask, subject_atlas, interpolation="nearest", copy_header=True, force_resample=True
    )

    # Get arrays
    atlas_data = subject_atlas.get_fdata()
    mask_data = mask_gradient_resampled.get_fdata() > 0  # binarized mask

    masked_atlas_data = np.where(mask_data, atlas_data, 0)

    return nib.Nifti1Image(masked_atlas_data, affine=subject_atlas.affine, header=subject_atlas.header)


def clean_matrix_from_atlas(matrix: np.ndarray, atlas: nib.Nifti1Image) -> np.ndarray:
    """
    Remove rows/columns from a connectivity matrix for regions not present in the atlas.

    Parameters
    ----------
    matrix : np.ndarray
        Connectivity matrix
    atlas : nib.Nifti1Image
        Masked atlas

    Returns
    -------
    np.ndarray
        Connectivity matrix limited to regions present in the atlas.
    """

    atlas_data = atlas.get_fdata()
    roi_labels = sorted(np.unique(atlas_data[atlas_data > 0]))

    # Get indices corresponding to these labels (assuming atlas labels are 1-based)
    indices_to_keep = [int(lab - 1) for lab in roi_labels]

    if max(indices_to_keep) >= matrix.shape[0]:
        return matrix
    else:
        return matrix[np.ix_(indices_to_keep, indices_to_keep)]


def extract_gradients(
    connectivity_matrices: Iterable[ConnectivityMatrix], atlas: nib.Nifti1Image
) -> tuple[List[np.ndarray], np.ndarray]:
    """
    Calculate the gradients for each individual and load group-level gradients
    from Margulies et al., 2016 for alignment.

    Parameters:
    - connectivity_matrices (np.array): The connectivity matrix containing the correlation values
    - atlas (Atlas): The Atlas object.

    Returns:
    - ind_aligned_gradient (np.array): The aligned individual gradients.
    - group_gradients (np.array): The group-level gradients.
    """
    # Path to the repo root (assuming this script is inside wonkyconn/)
    repo_root = Path(__file__).resolve().parent.parent

    path_gradients = repo_root / "data" / "gradients"
    gradient_mask = nib.load(path_gradients / "gradientmask_cortical_subcortical.nii.gz")

    # Load all group gradient maps
    gradient_files = sorted(glob.glob(str(path_gradients / "templates" / "gradient*_cortical_subcortical.nii.gz")))

    gradients = []
    for connectivity_matrix in connectivity_matrices:
        # Remove NaN from matrix
        matrix = np.asarray(connectivity_matrix.load(), dtype=np.float64)
        conn_clean, kept_idx = remove_nan_from_matrix(matrix)

        atlas_mask_without_nan, _ = remove_nan_roi_atlas(atlas, kept_idx)
        masked_atlas = overlapping_atlas_with_mask(atlas_mask_without_nan, gradient_mask)
        masked_matrix = clean_matrix_from_atlas(conn_clean, masked_atlas)

        masker = NiftiLabelsMasker(labels_img=masked_atlas, mask_img=gradient_mask)

        group_gradients = []
        for fname in gradient_files:
            grad_img = nib.load(fname)
            grad_vals = masker.fit_transform(grad_img)  # shape (1, n_regions)
            group_gradients.append(grad_vals.squeeze())

        group_gradients_np = np.vstack(group_gradients).T  # shape (n_regions, n_components)

        # Compute individual gradients
        gm = GradientMaps(n_components=5, alignment="procrustes", kernel="normalized_angle")
        ind_gradient = gm.fit(masked_matrix, reference=group_gradients_np)

        gradients.append(ind_gradient.aligned_)

    return gradients, group_gradients_np


def calculate_group_gradients_similarity(correlations: List[float]) -> float:
    """
    Calculate the mean of a all participant's correlation values.

    Parameters:
    - correlations (list): A list of correlation values.

    Returns:
    - float: The mean of the correlation values.
    """
    return float(np.mean(correlations))


def calculate_gradients_similarity(gradients: List[np.ndarray], group_gradients: np.ndarray) -> float:
    """
    Calculate the Spearman's correlation between the individual gradients and the reference
    group-level gradients from Margulies et al., 2016. Then apply Fishers R-to-Z transformation
    and average across all gradient components.

    Parameters:
    - ind_aligned_gradient (np.array): The aligned individual gradients.
    - group_gradients (np.array): The group-level gradients.

    Returns:
    - correlarions (list): The list of Spearman correlation values for each gradient component.
    """

    # Spearman correlation per component
    correlations = []
    for gradient in gradients:
        correlations_per_subject = []
        for i in range(group_gradients.shape[1]):
            rho, _ = stats.spearmanr(gradient[:, i], group_gradients[:, i])
            correlations_per_subject.append(rho)

        correlations.append(np.mean(np.arctanh(correlations_per_subject)))  # Fisher R-to-Z transformation and average

    return calculate_group_gradients_similarity(correlations)
