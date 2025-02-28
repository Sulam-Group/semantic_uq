import os

import nibabel as nib
from tqdm import tqdm

workdir = "../"
data_dir = os.path.join(workdir, "data")
flare_dir = os.path.join(data_dir, "FLARE")

scans = list(filter(lambda x: x.endswith(".nii.gz"), os.listdir(flare_dir)))
for scan_name in tqdm(scans):
    scan_path = os.path.join(flare_dir, scan_name)
    try:
        scan = nib.load(scan_path)
        scan.get_fdata()
    except:
        print(f"Exception occurred while loading {scan_name}")
        with open(os.path.join(flare_dir, "bad_scans.txt"), "w") as f:
            f.write(scan_name)
