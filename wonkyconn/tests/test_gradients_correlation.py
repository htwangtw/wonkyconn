import numpy as np
from wonkyconn.features import calculate_gradients_correlation
import nibabel as nib


def create_fake_connectivity(n_regions=434):
    # random symmetric connectivity matrix with diagonal=1
    matrix = np.random.uniform(-1, 1, size=(n_regions, n_regions))
    conn_matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(conn_matrix, 1)

    return conn_matrix


def test_extract_gradients():
    conn_matrix = create_fake_connectivity()

    atlas = nib.load("wonkyconn/data/test_data/atlas/atlas-Schaefer2018Combined_dseg.nii.gz")

    ind_aligned_gradient, group_gradients = calculate_gradients_correlation.extract_gradients(conn_matrix, atlas=atlas)

    assert group_gradients is not None
    assert ind_aligned_gradient is not None


def test_calculate_gradients_similarity():
    conn_matrix = create_fake_connectivity()
    atlas = nib.load("wonkyconn/data/test_data/atlas/atlas-Schaefer2018Combined_dseg.nii.gz")

    ind_aligned_gradient, group_gradients = calculate_gradients_correlation.extract_gradients(conn_matrix, atlas=atlas)

    correlation = calculate_gradients_correlation.calculate_gradients_similarity(ind_aligned_gradient, group_gradients)

    assert correlation is not None
