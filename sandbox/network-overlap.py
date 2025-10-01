import pandas as pd
import numpy as np
from nilearn.datasets import (
    fetch_development_fmri,
    fetch_atlas_yeo_2011,
    fetch_atlas_schaefer_2018,
    fetch_atlas_basc_multiscale_2015
)
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.image import load_img, math_img, resample_to_img, iter_img
from nilearn.interfaces.fmriprep import load_confounds_strategy


dataset = fetch_development_fmri(reduce_confounds=False)
yeo_networks = fetch_atlas_yeo_2011(n_networks=7, thickness='thick')
schaefer400 = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
# mist444 = fetch_atlas_basc_multiscale_2015(resolution=400, version='asym')

# extract schaefer 400
confounds, sample_masks = load_confounds_strategy(dataset.func, denoise_strategy='simple', motion='basic')
atlas_masker = NiftiLabelsMasker(schaefer400.maps, labels=schaefer400.labels).fit()
connectome = ConnectivityMeasure(kind='correlation')

connectomes_prepro = []
for img, c, s in zip(dataset.func, confounds, sample_masks):
    ts = atlas_masker.transform(img, confounds=c, sample_mask=s)
    conn = connectome.fit_transform(ts)
    del ts
    connectomes_prepro.append(conn)

# create a dictionary to summarise yeo network seed converage
yeo7_nii = load_img(yeo_networks.maps)
yeo7_nii = list(iter_img(yeo7_nii))[0]  # for some reason there's a fourth dimention
network_labels = np.unique(yeo7_nii.dataobj)[1:]
atlas_nii = load_img(schaefer400.maps)
region_labels = np.unique(atlas_nii.dataobj)[1:]

yeo7_nii = resample_to_img(yeo7_nii, schaefer400.maps, interpolation='nearest')
region_membership = pd.DataFrame(0, index=region_labels, columns=[f'yeo7-{int(n)}' for n in network_labels])
network_masker = NiftiLabelsMasker(yeo7_nii).fit()

for n in network_labels:
    cur_region = math_img(f'img=={n}', img=yeo7_nii)
    masker = NiftiMasker(cur_region)
    atlas_parcel_in_network = masker.fit_transform(schaefer400.maps)
    atlas_parcel_in_network = np.unique(atlas_parcel_in_network)[1:]
    region_membership.loc[atlas_parcel_in_network, f'yeo7-{int(n)}'] = 1

# approach 1: calculate similarity of the thresholded individual level seed based connectivity with the binary mask
# for i, r in enumerate(region_labels):


# approach 2: calculate similarity of the thresholded group level seed based connectivity with the binary mask
group_connectome = np.mean(np.array(connectomes_prepro), axis=0).squeeze()
average_connectivity_within_network = []

for i, r in enumerate(region_labels):
    seed_based_map = group_connectome[i, :]
    seed_based_map[seed_based_map==0] = np.nan
    networks = [region_membership[f'yeo7-{int(n)}'].values * seed_based_map for n in network_labels]
    average_connectivity_within_network.append(np.mean(np.array(networks), axis=1))


i = 0
seed_map = atlas_masker.inverse_transform(group_connectome[i, :])
cur_region = math_img('img==7', img=yeo7_nii)
cur_region = resample_to_img(cur_region, seed_map, interpolation='nearest')

plt.figure()
display = plot_stat_map(seed_map, transparency_range=[0, 0.3], transparency=seed_map, cmap='cold_hot', symmetric_cbar=True, title=schaefer400.labels[i+1])
display.add_contours(cur_region, color='k')
plt.savefig('example_not_dmn.png')


i = 399
seed_map = atlas_masker.inverse_transform(group_connectome[i, :])
plt.figure()
display = plot_stat_map(seed_map, transparency_range=[0, 0.3], transparency=seed_map, cmap='cold_hot', symmetric_cbar=True, title=schaefer400.labels[i+1])
display.add_contours(cur_region, color='k')
plt.savefig('example_dmn.png')