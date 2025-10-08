import numpy as np
import nibabel as nib
import glob

from scipy import stats

from brainspace.gradient import GradientMaps  # type: ignore[import-not-found]
from brainspace.gradient import alignment
from nilearn.maskers import NiftiLabelsMasker  # type: ignore[import-not-found]

from typing import Tuple, List


def remove_nan_from_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    col_mask = ~np.all(np.isnan(matrix), axis=0)

    kept_idx = np.where(col_mask)[0]
    removed_idx = np.where(~col_mask)[0]
    conn_clean = matrix[np.ix_(kept_idx, kept_idx)]

    return conn_clean, kept_idx, removed_idx


def overlapping_mask(subject_atlas: nib.Nifti1Image, group_mask: nib.Nifti1Image, matrix: np.ndarray) -> nib.Nifti1Image:
    """
    TODO: not used yet, will be implemented in the future or align atlas with gradient mask
    """


def mask_atlas_to_matrix(atlas: nib.Nifti1Image, kept_idx: np.ndarray) -> tuple[nib.Nifti1Image, np.ndarray]:
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
    kept_atlas_img = nib.Nifti1Image(kept_atlas_data, atlas.affine, atlas.header)
    return kept_atlas_img


def extract_gradients(ind_matrix: np.ndarray, atlas: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
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
    path_gradients = "wonkyconn/data/gradients"

    # Remove NaN from matrix
    conn_clean, kept_idx, removed_idx = remove_nan_from_matrix(ind_matrix)

    # Then remove the region that do not overlap between atlas and matrix
    atlas_mask_without_nan = mask_atlas_to_matrix(atlas, kept_idx)
    masker = NiftiLabelsMasker(labels_img=atlas_mask_without_nan, standardize=False)

    # Load all group gradient maps
    gradient_files = sorted(glob.glob(f"{path_gradients}/templates/gradient*_cortical_subcortical.nii.gz"))

    group_gradients = []
    print("before the loop")
    for fname in gradient_files:
        grad_img = nib.load(fname)
        grad_vals = masker.fit_transform(grad_img)  # shape (1, n_regions)
        group_gradients.append(grad_vals.squeeze())
    print("after the loop")

    group_gradients_np = np.vstack(group_gradients).T  # shape (n_regions, n_components)

    # Compute individual gradients
    gm = GradientMaps(n_components=5)
    ind_gradient = gm.fit(conn_clean)

    # align (Procrustes)
    ind_aligned_gradient = alignment.procrustes(ind_gradient.gradients_, group_gradients_np)

    return ind_aligned_gradient, group_gradients_np


def calculate_gradients_similarity(ind_aligned_gradient: np.ndarray, group_gradients: np.ndarray) -> float:
    """
    Calculate the Spearman's correlation between the individual gradients and the reference
    group-level gradients from Margulies et al., 2016. Then apply Fishers R-to-Z transformation
    and average across all gradient components.

    Parameters:
    - ind_aligned_gradient (np.array): The aligned individual gradients.
    - group_gradients (np.array): The group-level gradients.

    Returns:
    - corrs (list): The list of Spearman correlation values for each gradient component.
    """
    corrs = []

    for i in range(group_gradients.shape[1]):
        rho, _ = stats.spearmanr(ind_aligned_gradient[:, i], group_gradients[:, i])
        corrs.append(rho)

    return np.mean(np.arctanh(corrs))


def calculate_group_gradients_similarity(correlations: List[float]) -> float:
    """
    Calculate the mean of a list of correlation values.

    Parameters:
    - correlations (list): A list of correlation values.

    Returns:
    - float: The mean of the correlation values.
    """
    return float(np.mean(correlations))
