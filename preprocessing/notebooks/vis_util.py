import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import random

def select_fixed_random_indices(input_list, num_indices=32, seed=42):
    random.seed(seed)  # Set the random seed for reproducibility
    if num_indices > len(input_list):
        raise ValueError("Number of indices to select exceeds the length of the input list.")
    
    # Use random.sample() to select random indices without duplicates
    random_indices = random.sample(range(len(input_list)), num_indices)
    
    return random_indices

def visualize_random_miriad(miriad_dir, axis=2, slice_fraction=0.5):
    file_formats = (".nii.gz", ".nii")

    file_paths = []
    for current_root, dirs, files in os.walk(miriad_dir):
        for file in files:
            if file.endswith(file_formats):
                file_paths.append((os.path.join(current_root, file), file))

    file_paths.sort(key=lambda tup: tup[1])
    file_paths = file_paths[:32]

    #file_paths = select_fixed_random_indices(file_paths)

    imgs = []
    for file_path, file in file_paths:
        nii_data = nib.load(file_path)

        image_data = nii_data.get_fdata()
        data_shape = image_data.shape
        
        if axis == 0:
            half = int(image_data.shape[0] * slice_fraction)
            mri_sample_slice = image_data.astype(np.float32)[half, :, :]
        elif axis == 1:
            half = int(image_data.shape[1] * slice_fraction)
            mri_sample_slice = image_data.astype(np.float32)[:, half, :]
        elif axis == 2:
            half = int(image_data.shape[2] * slice_fraction)
            mri_sample_slice = image_data.astype(np.float32)[:, :, half]
        else:
            raise ValueError("Invalid axis value!") 
        
        imgs.append(mri_sample_slice)

    fig, ax = plt.subplots(nrows=4, ncols=8, figsize=(80, 160))
    for x in range(4):
        for y in range(8):
            img = imgs[x*4+y]
            ax[x][y].imshow(img.T, cmap='gray')
            ax[x][y].axis('off')

    plt.tight_layout()            
    plt.show()



def visualize_structural_mri_session(session_directory, axis=2, slice_fraction=0.5, val_percentile=None, limit=None):
    file_formats = (".nii.gz", ".nii")

    file_paths = []
    for current_root, dirs, files in os.walk(session_directory):
        for file in files:
            if file.endswith(file_formats):
                file_paths.append((os.path.join(current_root, file), file))

    file_paths.sort(key=lambda tup: tup[1])

    if not limit:
        limit = 100
    
    modalities = []
    shapes = []
    imgs = []
    counter = 0
    for file_path, file in file_paths:
        counter += 1
        if counter > limit:
            break

        modality = "T1"
        if "t1ce" in file:
            modality = "T1CE"
        elif "T1w" in file or "t1" in file or "T1" in file:
            modality = "T1"
        elif "T2star" in file:
            modality = "T2*"
        elif "TSE_T2w" in file:
            modality = "TSE-T2"
        elif "T2w" in file or "t2" in file or "T2" in file:
            modality = "T2"
        elif "FLAIR" in file or "flair" in file:
            modality = "FLAIR"
        elif "angio" in file:
            modality = "TOF MRA"
        elif "PD" in file:
            modality = "PD"
        elif "MRA" in file:
            modality = "MRA"
        elif "seg" in file:
            continue

        nii_data = nib.load(file_path)

        image_data = nii_data.get_fdata()
        data_shape = image_data.shape

        if val_percentile and modality == "MRA":
            threshold_value = np.percentile(image_data, val_percentile)
            image_data[image_data > threshold_value] = threshold_value
        
        if axis == 0:
            half = int(image_data.shape[0] * slice_fraction)
            mri_sample_slice = image_data.astype(np.float32)[half, :, :]
        elif axis == 1:
            half = int(image_data.shape[1] * slice_fraction)
            mri_sample_slice = image_data.astype(np.float32)[:, half, :]
        elif axis == 2:
            half = int(image_data.shape[2] * slice_fraction)
            mri_sample_slice = image_data.astype(np.float32)[:, :, half]
        else:
            raise ValueError("Invalid axis value!") 
        
        imgs.append(mri_sample_slice)


        modalities.append(modality)
        shapes.append(data_shape)

    constant_height = 5  # constant height for each subplot
    custom_fontsize = 10
    fig, ax = plt.subplots(nrows=1, ncols=len(modalities), figsize=(2 * len(modalities), constant_height))
    for i, img in enumerate(imgs):
        ax[i].imshow(img.T, cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(f"{modalities[i]} {shapes[i]}",fontsize=custom_fontsize)  # Set title for each modality image

    plt.tight_layout()            
    plt.show()

def visualize_structural_mri_session_data(mri_scans: list, modality_names: list[str], axis=2):
    modalities = modality_names
    shapes = []
    imgs = []

    for scan in mri_scans:
        image_data = scan
        data_shape = image_data.shape

        if axis == 0:
            half = int(image_data.shape[0] / 2)
            mri_sample_slice = image_data.astype(np.float32)[half, :, :]
        elif axis == 1:
            half = int(image_data.shape[1] / 2)
            mri_sample_slice = image_data.astype(np.float32)[:, half, :]
        elif axis == 2:
            half = int(image_data.shape[2] / 2)
            mri_sample_slice = image_data.astype(np.float32)[:, :, half]
        else:
            raise ValueError("Invalid axis value!")

        imgs.append(mri_sample_slice)
        shapes.append(data_shape)

    fig, ax = plt.subplots(nrows=1, ncols=len(modalities), figsize=(15, 15))
    for i, img in enumerate(imgs):
        ax[i].imshow(img.T, cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(f"{modalities[i]} {shapes[i]}")  # Set title for each modality image

    plt.tight_layout()            
    plt.show()

def visualize_axial_slice(file_path, fraction=0.5):
    """
    Visualize the central axial slice of a .nii file.
    
    Parameters:
        file_path (str): Path to the .nii file
    """
    
    # Load the .nii file
    img = nib.load(file_path)
    
    # Get the image data as a numpy array
    img_data = img.get_fdata()
    img_data = img_data[:,:,:,0]
    
    # Determine the index of the central slice along the axial axis (axis 2)
    central_slice_idx = int(img_data.shape[2] * fraction)
    
    # Extract the central axial slice
    central_slice = img_data[:, :, central_slice_idx]
    
    # Display the central slice
    plt.imshow(central_slice.T, cmap="gray", origin="lower")
    plt.title(f"SRI-24 T1 Template {img_data.shape}")
    plt.axis("off")
    plt.show()

# Example usage
# visualize_central_axial_slice("path/to/your/file.nii")
