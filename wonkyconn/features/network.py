"""Network similarity measures."""

from __future__ import (
    annotations,
)

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..base import ConnectivityMatrix


def single_subject_within_network_connectivity(
    connectivity_matrix: ConnectivityMatrix,
    region_membership: pd.DataFrame,
    yeo_network_index: int = 7,
) -> pd.DataFrame:
    """_summary_

    Args:
        connectivity_matrix (ConnectivityMatrix): _description_
        region_membership (pd.DataFrame): _description_
        yeo_network_index (int, optional): _description_. Defaults to 7.

    Returns:
        pd.DataFrame: _description_
    """

    yeo_network_labels = range(region_membership.shape[0])
    # get roi that only belong to one given network
    roi_index_single_network_membership = region_membership.index[region_membership.T.sum() == 1]
    roi_index_single_and_dmn = roi_index_single_network_membership[
        region_membership.loc[roi_index_single_network_membership, f"yeo7-{yeo_network_index}"] == 1
    ]
    roi_index = roi_index_single_and_dmn - 1  # convert to zero based index

    # calculate similarity of the thresholded individual level seed based connectivity with the binary mask
    subj_average_connectivity_within_network = []
    subj_variance_connectivity_within_network = []
    subj_corr_wtih_network = []
    for idx_roi in roi_index:
        seed_based_map = connectivity_matrix.load()[int(idx_roi), :]
        # isolate the parcal per network
        networks = [region_membership[f"yeo7-{int(n + 1)}"].values * seed_based_map for n in yeo_network_labels]
        networks = np.asarray(networks)
        networks[networks == 0] = np.nan
        mean_within_network_connection = np.nanmean(networks, axis=1)
        std_within_network_connection = np.nanstd(networks, axis=1)
        correlation_with_given_network = np.corrcoef(seed_based_map, region_membership[f"yeo7-{yeo_network_index}"].to_numpy())

        subj_average_connectivity_within_network.append(mean_within_network_connection)
        subj_variance_connectivity_within_network.append(std_within_network_connection)
        subj_corr_wtih_network.append(correlation_with_given_network[1, 0])

    # summarise of the given subject
    subj_average_connectivity_within_network = pd.DataFrame(subj_average_connectivity_within_network).mean()
    subj_variance_connectivity_within_network = pd.DataFrame(subj_variance_connectivity_within_network).mean()
    subj_corr_wtih_network = pd.DataFrame(subj_corr_wtih_network).mean()
    return subj_average_connectivity_within_network, subj_variance_connectivity_within_network, subj_corr_wtih_network


def network_similarity(
    connectivity_matrices: list[ConnectivityMatrix], region_membership: pd.DataFrame
) -> Tuple[np.float64, np.float64]:
    """_summary_

    Args:
        connectivity_matrices (list[ConnectivityMatrix]): _description_
        region_membership (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    average_connectivity_within_network, std_connectivity_within_network, corr_wtih_dmn = [], [], []
    for cm in connectivity_matrices:
        mean, std, corr = single_subject_within_network_connectivity(cm, region_membership, yeo_network_index=7)
        average_connectivity_within_network.append(mean)
        std_connectivity_within_network.append(std)
        corr_wtih_dmn.append(corr)

    # average_connectivity_within_network = pd.DataFrame(
    #     average_connectivity_within_network, columns=[f"mean_{r}" for r in region_membership.columns]
    # )
    # std_connectivity_within_network = pd.DataFrame(
    #     std_connectivity_within_network, columns=[f"sd_{r}" for r in region_membership.columns]
    # )
    # corr_wtih_dmn = pd.DataFrame(corr_wtih_dmn, columns=["corr_with_dmn"])
    summary = pd.DataFrame(
        [average_connectivity_within_network, std_connectivity_within_network, corr_wtih_dmn],
        columns=[f"mean_{r}" for r in region_membership.columns]
        + [f"sd_{r}" for r in region_membership.columns]
        + ["corr_with_dmn"],
    )

    # calculate distance
    summary["mean-diff_dmn_visual"] = summary["mean_yeo7-7"] - summary["mean_yeo7-1"]
    summary["mean-diff_dmn_fpn"] = summary["mean_yeo7-7"] - summary["mean_yeo7-6"]

    mean_corr_with_dmn = summary["corr_with_dmn"].mean()
    t_stats_dmn_vis_fpn, _ = stats.ttest_rel(summary["mean-diff_dmn_visual"], summary["mean-diff_dmn_fpn"])
    return mean_corr_with_dmn, t_stats_dmn_vis_fpn
