import os

import nibabel as nib
from joblib import Parallel, delayed
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Spacingd,
)
from tqdm import tqdm

root_dir = "../"
data_dir = os.path.join(root_dir, "data")
flare_dir = os.path.join(data_dir, "FLARE")
scan_dir = os.path.join(flare_dir, "scans")

z_thickness = []

pixdim = (1.5, 1.5, 3.0)
transform = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),
    ]
)

out_dir = os.path.join(data_dir, f"FLARE_{','.join(map(str, pixdim))}", "scans")
os.makedirs(out_dir, exist_ok=True)


def preprocess(data):
    scan_name = data["name_img"]

    try:
        data = transform(data)
    except Exception as e:
        return
    image = data["image"].squeeze()
    affine = data["image_meta_dict"]["affine"]

    out_path = os.path.join(out_dir, f"{scan_name}.nii.gz")
    nii = nib.Nifti1Image(image, affine)
    nib.save(nii, out_path)


scan_names = sorted(filter(lambda x: x.endswith(".nii.gz"), os.listdir(scan_dir)))
data_dicts = [
    {"image": os.path.join(scan_dir, scan_name), "name_img": scan_name.split(".")[0]}
    for scan_name in scan_names
]

Parallel(n_jobs=32)(delayed(preprocess)(data) for data in tqdm(data_dicts))
