import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from wonkyconn.base import ConnectivityMatrix
from wonkyconn.features import calculate_gradients_correlation


def create_fake_connectivity(n_regions=434, n_subjects=5) -> tuple[Path, list[ConnectivityMatrix]]:
    """
    Create fake connectivity matrices in a temporary directory and return both
    the directory path and the ConnectivityMatrix objects.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="fake_connectivity_"))

    connectivity_matrices = []
    for subj_idx in range(n_subjects):
        # Generate symmetric random matrix
        matrix = np.random.uniform(-1, 1, size=(n_regions, n_regions))
        conn_matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(conn_matrix, 1)

        # Save as TSV
        path = temp_dir / f"sub-{subj_idx + 1}_connectivity.tsv"
        np.savetxt(path, conn_matrix, delimiter="\t", fmt="%.6f")

        # Create metadata
        metadata = {
            "subject_id": f"sub-{subj_idx + 1}",
            "n_regions": n_regions,
            "description": "Fake symmetric connectivity matrix",
        }

        connectivity_matrices.append(ConnectivityMatrix(path=path, metadata=metadata))

    return connectivity_matrices


def test_gradients():
    repo_root = Path(__file__).resolve().parent.parent

    path = repo_root / "data" / "gradients"
    atlas = nib.load(str(path / "atlas" / "atlas-Schaefer2018Combined_dseg.nii.gz"))

    connectivity_matrices = create_fake_connectivity(n_regions=434, n_subjects=3)

    random_gradient, group_gradients = calculate_gradients_correlation.extract_gradients(connectivity_matrices, atlas=atlas)

    # Calculate similarity for random gradient
    random_similarity = calculate_gradients_correlation.calculate_gradients_similarity(random_gradient, group_gradients)

    # Calculate similarity for template gradients (should be high value)
    template_gradient_similarity = calculate_gradients_correlation.calculate_gradients_similarity(
        [group_gradients], group_gradients
    )

    assert template_gradient_similarity > 0.99
    assert random_similarity < template_gradient_similarity
