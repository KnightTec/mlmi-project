import os
import nibabel as nib

# Specify the folder path
folder_path = './IXI-T2'

# Get a list of all NIfTI files in the folder
nii_files = [file for file in os.listdir(folder_path) if file.endswith('.nii.gz')]

# Iterate over each NIfTI file and print its shape
for nii_file in nii_files:
    current_nii_file = os.path.join(folder_path, nii_file)
    current_nii_data = nib.load(current_nii_file)
    current_shape = current_nii_data.get_fdata().shape

    print(f"File: {nii_file} - Shape: {current_shape}")
