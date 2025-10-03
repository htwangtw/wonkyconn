import numpy as np
import nibabel as nib
import glob

from scipy import stats

from brainspace.gradient import GradientMaps  # type: ignore[import-not-found]
from brainspace.gradient import alignment
from nilearn.maskers import NiftiLabelsMasker  # type: ignore[import-not-found]


def remove_nan_from_matrix(matrix: np.ndarray) -> np.ndarray:
    col_mask = ~np.all(np.isnan(matrix), axis=0)
    row_mask = ~np.all(np.isnan(matrix), axis=1)

    # Apply masks
    cleaned_matrix = matrix[np.ix_(row_mask, col_mask)]
    print("clean matrix", cleaned_matrix.shape)

    return cleaned_matrix


def remove_cerebellum_from_atlas(atlas: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Remove cerebellar regions from an atlas.
    Parameters:
        - atlas (nib.Nifti1Image): The original atlas image.
    Returns:
        - nib.Nifti1Image: The atlas image with cerebellar regions removed.
    """
    atlas_data = atlas.get_fdata()

    # Identify cerebellar labels
    cerebellum_labels = np.arange(418, 435)  # TODO Change this
    atlas_data_no_cerebellum = np.where(np.isin(atlas_data, cerebellum_labels), 0, atlas_data)

    masked_atlas_img = nib.Nifti1Image(atlas_data_no_cerebellum, atlas.affine, atlas.header)
    return masked_atlas_img


def mask_atlas_to_matrix(atlas: nib.Nifti1Image, conn_matrix: np.ndarray) -> tuple[nib.Nifti1Image, np.ndarray]:
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

    labels_original = np.unique(atlas_data)
    labels_original = labels_original[labels_original > 0]

    # Determine which labels to keep based on matrix size
    n_rois_matrix = conn_matrix.shape[0]
    labels_to_keep = labels_original[:n_rois_matrix]

    # Mask atlas
    atlas_data_masked = np.where(np.isin(atlas_data, labels_to_keep), atlas_data, 0)

    # Create new NIfTI
    atlas_mask_without_nan = nib.Nifti1Image(atlas_data_masked, atlas.affine, atlas.header)

    # Remove region with non overlap in matrix
    labels_original = np.unique(atlas_mask_without_nan.get_fdata().astype(int))
    labels_masked = np.unique(atlas_data_masked)

    labels_original = labels_original[labels_original > 0]
    labels_masked = labels_masked[labels_masked > 0]

    missing_labels = set(labels_original) - set(labels_masked)
    missing = sorted(list(missing_labels))

    # Drop rows/cols
    corr_matrix_masked = np.delete(conn_matrix, missing, axis=0)
    corr_matrix_masked = np.delete(corr_matrix_masked, missing, axis=1)

    return atlas_mask_without_nan, corr_matrix_masked


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
    ind_matrix = remove_nan_from_matrix(ind_matrix)

    # First remove the cerebellum
    # TODO: use mask instead of removing by hand
    masked_atlas = remove_cerebellum_from_atlas(atlas)

    # Then remove the region
    atlas_mask_without_nan, corr_matrix_masked = mask_atlas_to_matrix(masked_atlas, ind_matrix)
    print(corr_matrix_masked.shape)

    masker = NiftiLabelsMasker(labels_img=atlas_mask_without_nan, standardize=False).fit()

    # Load all group gradient maps
    gradient_files = sorted(glob.glob(f"{path_gradients}/templates/gradient*_cortical_subcortical.nii.gz"))

    group_gradients = []
    for fname in gradient_files:
        grad_img = nib.load(fname)
        grad_vals = masker.transform(grad_img)  # shape (1, n_regions)
        group_gradients.append(grad_vals.squeeze())

    group_gradients_np = np.vstack(group_gradients).T  # shape (n_regions, n_components)

    # Compute individual gradients
    gm = GradientMaps(n_components=5)
    ind_gradient = gm.fit(corr_matrix_masked)

    # align (Procrustes)
    ind_aligned_gradient = alignment.procrustes(ind_gradient.gradients_, group_gradients_np)

    return ind_aligned_gradient, group_gradients_np


def calculate_gradients_similarity(ind_aligned_gradient: np.ndarray, group_gradients: np.ndarray) -> float:
    """
    Calculate the Spearman's correlation between the individual gradients and the reference
    group-level gradients from Margulies et al., 2016.

    Parameters:
    - ind_aligned_gradient (np.array): The aligned individual gradients.
    - group_gradients (np.array): The group-level gradients.

    Returns:
    - corrs (list): The list of Spearman correlation values for each gradient component.
    """
    # Spearman correlation per component
    corrs = []
    for i in range(group_gradients.shape[1]):
        rho, _ = stats.spearmanr(ind_aligned_gradient[:, i], group_gradients[:, i])
        corrs.append(rho)

    correlation_mean = np.mean(corrs)
    return correlation_mean
