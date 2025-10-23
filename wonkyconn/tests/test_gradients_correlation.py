from pathlib import Path

import nibabel as nib
import numpy as np

from wonkyconn.features import calculate_gradients_correlation


def create_fake_connectivity(n_regions=434, n_subjects=1):
    # random symmetric connectivity matrix with diagonal=1
    matrix = np.random.uniform(-1, 1, size=(n_regions, n_regions))
    conn_matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(conn_matrix, 1)

    return conn_matrix


def test_gradients():
    repo_root = Path(__file__).resolve().parent.parent

    path = repo_root / "data" / "gradients"
    atlas = nib.load(str(path / "atlas" / "atlas-Schaefer2018Combined_dseg.nii.gz"))

    conn_matrix = create_fake_connectivity(n_regions=434, n_subjects=3)

    random_gradient, group_gradients = calculate_gradients_correlation.extract_gradients(conn_matrix, atlas=atlas)

    # Calculate similarity for random gradient
    random_individual_spearman = calculate_gradients_correlation.calculate_gradients_similarity(
        random_gradient, group_gradients
    )
    random_similarity = calculate_gradients_correlation.calculate_group_gradients_similarity(random_individual_spearman)

    # Calculate similarity for template gradients (should be high value)
    template_gradient_similarity = calculate_gradients_correlation.calculate_gradients_similarity(
        group_gradients, group_gradients
    )

    assert template_gradient_similarity > 0.99
    assert random_similarity < template_gradient_similarity
