import numpy as np
import nibabel as nib

data = np.random.normal(0, 1, (256, 256)).astype(np.float32)

# Create an identity affine matrix. This is a 4x4 matrix that doesn't change
# the spatial coordinates when applied to the data.
affine = np.eye(4)

# Convert the numpy array into a Nifti1Image
img = nib.Nifti1Image(data, affine)

# Optionally, if you want to save this image to a file:
img.to_filename('gauss_2d_256.nii.gz')
