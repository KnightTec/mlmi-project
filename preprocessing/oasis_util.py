import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def visualize_structural_mri_session(session_directory):
    file_format = ".nii.gz"

    file_paths = []
    for current_root, dirs, files in os.walk(session_directory):
        for file in files:
            if file.endswith(file_format):
                file_paths.append((os.path.join(current_root, file), file))

    file_paths.sort(key=lambda tup: tup[1])

    modalities = []
    shapes = []
    imgs = []
    for file_path, file in file_paths:
        nii_data = nib.load(file_path)

        image_data = nii_data.get_fdata()
        data_shape = image_data.shape

        half = int(image_data.shape[2] / 3 * 1.5)
        mri_sample_slice = image_data.astype(np.float32)[:, :, half]
        imgs.append(mri_sample_slice)

        modality = ""
        if "T1w" in file:
            modality = "T1w"
        elif "TSE_T2w" in file:
            modality = "TSE-T2w"
        elif "T2w" in file:
            modality = "T2w"
        elif "T2star" in file:
            modality = "T2*"
        elif "FLAIR" in file:
            modality = "FLAIR"
        elif "angio" in file:
            modality = "TOF MRA"
        
        modalities.append(modality)
        shapes.append(data_shape)

    fig, ax = plt.subplots(nrows=1, ncols=len(modalities), figsize=(15, 15))
    for i, img in enumerate(imgs):
        ax[i].imshow(img.T, cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(f"{modalities[i]} {shapes[i]}")  # Set title for each modality image

    plt.tight_layout()            
    plt.show()